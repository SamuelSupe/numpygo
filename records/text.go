package records

import (
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/SamuelSupe/numpygo"
)

func LoadText(r io.Reader, schema *Schema, opt np.TextOptions) (*Array, error) {
	return wrap(np.LoadStructuredText(r, schema, opt, false))
}

type NamedTextOptions struct {
	Text   np.TextOptions
	Names  []string
	Header bool
}

// GenFromText infers a packed record dtype from named text columns. A
// boolean, integer, real, complex or fixed-width Unicode column is selected
// from all its nonmissing tokens.
func GenFromText(r io.Reader, opt NamedTextOptions) (*Array, error) {
	o := opt.Text
	if opt.Header && o.MaxRows > 0 {
		o.MaxRows++
	}
	rows, columns, e := np.ReadTextRows(r, o)
	if e != nil {
		return nil, e
	}
	names := opt.Names
	if opt.Header {
		if len(rows) == 0 {
			return nil, np.ErrShape
		}
		names = rows[0]
		rows = rows[1:]
	}
	n := len(columns)
	if names == nil {
		names = make([]string, n)
		for i := range names {
			names[i] = "f" + strconv.Itoa(i)
		}
	}
	if len(names) != n {
		return nil, np.ErrShape
	}
	specs := make([]FieldSpec, n)
	values := make([][]any, len(rows))
	for i := range values {
		values[i] = make([]any, n)
	}
	missing := func(s string) bool {
		if s == "" {
			return true
		}
		for _, v := range o.MissingValues {
			if s == v {
				return true
			}
		}
		return false
	}
	parsers := []func(string) (any, error){
		func(s string) (any, error) {
			switch strings.ToLower(s) {
			case "true":
				return true, nil
			case "false":
				return false, nil
			}
			return nil, np.ErrValue
		},
		func(s string) (any, error) { return strconv.ParseInt(s, 10, 64) },
		func(s string) (any, error) { return strconv.ParseFloat(s, 64) },
		func(s string) (any, error) {
			return strconv.ParseComplex(strings.ReplaceAll(strings.ReplaceAll(s, "j", "i"), "J", "i"), 128)
		},
		func(s string) (any, error) { return s, nil },
	}
	for c, name := range names {
		k, width := 0, 0
		for _, row := range rows {
			s := row[c]
			if missing(s) {
				continue
			}
			width = max(width, utf8.RuneCountInString(s))
			for k < 4 {
				if _, e := parsers[k](s); e == nil {
					break
				}
				k++
			}
		}
		var typ Type
		if k == 4 {
			typ = FixedUnicode(width, nil)
		} else {
			typ = Numeric([]np.DType{np.Bool, np.Int64, np.Float64, np.Complex128}[k], nil)
		}
		specs[c] = FieldSpec{Name: name, Type: typ}
		for i, row := range rows {
			s := row[c]
			var v any
			if missing(s) {
				v = o.FillingValue
				if v == nil {
					v = []any{false, int64(-1), math.NaN(), complex(math.NaN(), 0), ""}[k]
				}
			} else {
				v, e = parsers[k](s)
				if e != nil {
					return nil, e
				}
			}
			values[i][c] = v
		}
	}
	s, e := NewSchema(specs, false, 0)
	if e != nil {
		return nil, e
	}
	a, e := FromRows(s, values, len(rows))
	if e != nil {
		return nil, e
	}
	if o.NDMin == 0 && len(rows) == 1 {
		return wrap(a.nd.Reshape())
	}
	if o.NDMin == 2 {
		return wrap(a.nd.Reshape(1, len(rows)))
	}
	return a, nil
}
