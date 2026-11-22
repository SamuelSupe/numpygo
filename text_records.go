package np

import (
	"bufio"
	"io"
	"strings"
)

// ReadTextRows is the delimiter/quote/comment reader shared by structured
// text constructors. The returned column map refers to original columns.
func ReadTextRows(r io.Reader, opt TextOptions) ([][]string, []int, error) {
	if opt.SkipRows < 0 || opt.MaxRows < 0 {
		return nil, nil, ErrValue
	}
	comment := opt.Comment
	if comment == "" {
		comment = "#"
	}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 4096), 64*1024*1024)
	var rows [][]string
	var columns []int
	line := 0
	for scanner.Scan() {
		line++
		if line <= opt.SkipRows {
			continue
		}
		f, e := splitTextLine(scanner.Text(), opt.Delimiter, comment, opt.QuoteChar)
		if e != nil {
			return nil, nil, e
		}
		if len(f) == 0 {
			continue
		}
		c := make([]int, len(f))
		for i := range c {
			c[i] = i
		}
		if opt.UseCols != nil {
			selected := make([]string, len(opt.UseCols))
			c = make([]int, len(opt.UseCols))
			for i, k := range opt.UseCols {
				if k < 0 {
					k += len(f)
				}
				if k < 0 || k >= len(f) {
					return nil, nil, ErrIndex
				}
				selected[i], c[i] = f[k], k
			}
			f = selected
		}
		if columns == nil {
			columns = c
		} else if len(columns) != len(c) {
			return nil, nil, ErrShape
		}
		for i := range f {
			f[i] = strings.TrimSpace(f[i])
		}
		rows = append(rows, f)
		if opt.MaxRows > 0 && len(rows) == opt.MaxRows {
			break
		}
	}
	if e := scanner.Err(); e != nil {
		return nil, nil, e
	}
	return rows, columns, nil
}

// LoadStructuredText loads column-oriented records with a declared schema.
// Fixed subarray leaves consume one input column per scalar.
func LoadStructuredText(r io.Reader, schema StructuredSchema, opt TextOptions, allowMissing bool) (*Array, error) {
	rows, columns, e := ReadTextRows(r, opt)
	if e != nil {
		return nil, e
	}
	if opt.Unpack || opt.NDMin < 0 || opt.NDMin > 2 {
		return nil, ErrValue
	}
	out, e := ZerosRecords(schema, len(rows))
	if e != nil {
		return nil, e
	}
	type field struct {
		a   *Array
		rel int
	}
	var fields []field
	var collect func(*Array, int, int) error
	collect = func(a *Array, rank, rel int) error {
		n, e := shapeSizeWidth(1, a.shape[rank:])
		if e != nil {
			return e
		}
		if a.DType() != RecordDType {
			for i := 0; i < n; i++ {
				fields = append(fields, field{a, rel + flatOffset(i, a.shape[rank:], a.strides[rank:])})
			}
			return nil
		}
		s, ok := a.StructuredSchema().(interface{ Names() []string })
		if !ok {
			return ErrDType
		}
		for i := 0; i < n; i++ {
			offset := rel + flatOffset(i, a.shape[rank:], a.strides[rank:])
			for _, name := range s.Names() {
				f, e := a.Field(name)
				if e != nil {
					return e
				}
				if e = collect(f, a.NDim(), offset); e != nil {
					return e
				}
			}
		}
		return nil
	}
	if e = collect(out, 1, 0); e != nil {
		return nil, e
	}
	for i, row := range rows {
		if len(row) != len(fields) {
			return nil, ErrShape
		}
		for j, f := range fields {
			s := row[j]
			var v any = s
			converter := opt.Converters[columns[j]]
			if converter == nil {
				converter = opt.Converter
			}
			missing := s == ""
			for _, m := range opt.MissingValues {
				missing = missing || m == s
			}
			if allowMissing && missing {
				v = opt.FillingValue
				if v == nil {
					switch d := f.a.DType(); {
					case d.IsFloat():
						v = "nan"
					case d.IsComplex():
						v = "nan"
					case d.IsInteger():
						v = int64(-1)
					case d == Bool:
						v = false
					case d.IsTemporal():
						v = "NaT"
					default:
						v = ""
					}
				}
			} else if converter != nil {
				v, e = converter(s)
				if e != nil {
					return nil, e
				}
			} else if f.a.DType().IsNativeNumeric() || f.a.DType().IsExtended() {
				v, e = parseNumber(s, f.a.DType())
				if e != nil {
					return nil, ErrValue
				}
			}
			value, e := f.a.convertValue(v)
			if e != nil {
				return nil, e
			}
			f.a.setElement(f.a.offset+i*f.a.strides[0]+f.rel, value)
		}
	}
	if opt.NDMin == 0 && len(rows) == 1 {
		return out.Reshape()
	}
	if opt.NDMin == 2 {
		return out.Reshape(1, len(rows))
	}
	return out, nil
}
