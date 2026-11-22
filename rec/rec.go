// Package rec provides typed equivalents of NumPy's record-array
// constructors using the unified records/ndarray storage.
package rec

import (
	"encoding/binary"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/records"
	"io"
	"slices"
	"strconv"
	"strings"
)

type Options struct {
	Schema         *records.Schema
	Formats        []string
	Names          []string
	Titles         []any
	Aligned        bool
	ByteOrder      binary.ByteOrder
	Shape          []int // nil infers constructor shape; explicit empty slice is scalar
	Offset         int
	Strides        []int  // byte strides for RecArray
	Order          string // C or F for RecArray
	NoCopy         bool   // only for Array from existing array
	WritableBuffer bool   // explicit opt-in for FromString's immutable-buffer contract
}

func parseFormat(s string) (records.Type, error) {
	s = strings.TrimSpace(s)
	var shape []int
	if strings.HasPrefix(s, "(") {
		i := strings.IndexByte(s, ')')
		if i < 0 {
			return records.Type{}, np.ErrValue
		}
		for _, v := range strings.Split(s[1:i], ",") {
			v = strings.TrimSpace(v)
			if v == "" {
				continue
			}
			n, e := strconv.Atoi(v)
			if e != nil || n < 0 {
				return records.Type{}, np.ErrShape
			}
			shape = append(shape, n)
		}
		s = strings.TrimSpace(s[i+1:])
	} else {
		i := 0
		for i < len(s) && s[i] >= '0' && s[i] <= '9' {
			i++
		}
		if i > 0 {
			n, e := strconv.Atoi(s[:i])
			if e != nil {
				return records.Type{}, np.ErrShape
			}
			shape = []int{n}
			s = s[i:]
		}
	}
	d, e := dtypes.Parse(s)
	if e != nil {
		return records.Type{}, e
	}
	t, e := d.RecordType()
	if e != nil {
		return records.Type{}, e
	}
	t.Shape = shape
	return t, nil
}
func schemaFromTypes(types []records.Type, opt Options) (*records.Schema, error) {
	if len(types) == 0 || len(opt.Names) > len(types) || len(opt.Titles) > len(types) {
		return nil, np.ErrShape
	}
	spec := make([]records.FieldSpec, len(types))
	for i, t := range types {
		name := "f" + strconv.Itoa(i)
		if i < len(opt.Names) && strings.TrimSpace(opt.Names[i]) != "" {
			name = strings.TrimSpace(opt.Names[i])
		}
		var title any
		if i < len(opt.Titles) {
			title = opt.Titles[i]
			if s, ok := title.(string); ok {
				title = strings.TrimSpace(s)
			}
		}
		if opt.ByteOrder != nil {
			t.Order = opt.ByteOrder
		}
		spec[i] = records.FieldSpec{Name: name, Title: title, Type: t}
	}
	s, e := records.NewSchema(spec, opt.Aligned, 0)
	if e != nil {
		return nil, e
	}
	// format_parser applies dtype.newbyteorder after calculating aligned
	// offsets. NumPy 2.5 retains the padding but not the aligned flag.
	if opt.Aligned && opt.ByteOrder != nil {
		for i, f := range s.Fields() {
			spec[i].Offset = np.Int(f.Offset)
		}
		return records.NewSchema(spec, false, s.ItemSize())
	}
	return s, nil
}
func FormatParser(formats []string, names []string, titles []any, aligned bool, order binary.ByteOrder) (*records.Schema, error) {
	types := make([]records.Type, len(formats))
	for i, s := range formats {
		t, e := parseFormat(s)
		if e != nil {
			return nil, e
		}
		types[i] = t
	}
	return schemaFromTypes(types, Options{Names: names, Titles: titles, Aligned: aligned, ByteOrder: order})
}
func explicitSchema(opt Options) (*records.Schema, error) {
	if opt.Schema != nil {
		return opt.Schema, nil
	}
	if len(opt.Formats) == 0 {
		return nil, nil
	}
	return FormatParser(opt.Formats, opt.Names, opt.Titles, opt.Aligned, opt.ByteOrder)
}

func NewRecord(schema *records.Schema, values ...any) (*records.Record, error) {
	return records.NewRecord(schema, values...)
}
func RecordFromBytes(schema *records.Schema, buffer []byte, readonly bool) (*records.Record, error) {
	a, e := records.FromBytes(schema, buffer, []int{}, readonly)
	if e != nil {
		return nil, e
	}
	return a.At()
}

// RecArray accepts an optional Go buffer, explicit shape and byte strides.
// With no buffer a fresh zero-initialized allocation is returned.
func RecArray(buffer []byte, opt Options) (*records.Array, error) {
	s, e := explicitSchema(opt)
	if e != nil {
		return nil, e
	}
	if s == nil || opt.Shape == nil || s.ItemSize() <= 0 {
		return nil, np.ErrDType
	}
	if opt.Offset < 0 {
		return nil, np.ErrValue
	}
	n, e := np.ElementCount(np.Bool, opt.Shape...)
	if e != nil {
		return nil, e
	}
	if n > int(^uint(0)>>1)/s.ItemSize() {
		return nil, np.ErrShape
	}
	order := opt.Order
	if order == "" {
		order = "C"
	}
	if order != "C" && order != "F" {
		return nil, np.ErrValue
	}
	if buffer == nil {
		if opt.Offset != 0 {
			return nil, np.ErrValue
		}
		if order == "C" && opt.Strides == nil {
			return records.Zeros(s, opt.Shape...)
		}
		buffer = make([]byte, n*s.ItemSize())
	}
	strides := opt.Strides
	if strides == nil && order == "F" {
		strides = make([]int, len(opt.Shape))
		step := s.ItemSize()
		if n > 0 {
			for i, dim := range opt.Shape {
				strides[i] = step
				step *= dim
			}
		}
	}
	a, e := np.FromRecordStridedBuffer(s, buffer, opt.Shape, strides, opt.Offset, false)
	if e != nil {
		return nil, e
	}
	return records.FromNDArray(a)
}

func inferRecordShape(a *np.Array, t records.Type) ([]int, error) {
	shape := a.Shape()
	if len(shape) < len(t.Shape) {
		return nil, np.ErrShape
	}
	start := len(shape) - len(t.Shape)
	if !slices.Equal(shape[start:], t.Shape) {
		return nil, np.ErrShape
	}
	return shape[:start], nil
}

func FromArrays(columns []*np.Array, opt Options) (*records.Array, error) {
	if len(columns) == 0 {
		return nil, np.ErrShape
	}
	s, e := explicitSchema(opt)
	if e != nil {
		return nil, e
	}
	if s == nil {
		types := make([]records.Type, len(columns))
		for i, a := range columns {
			d, e := dtypes.FromArray(a)
			if e != nil {
				return nil, e
			}
			types[i], e = d.RecordType()
			if e != nil {
				return nil, e
			}
		}
		s, e = schemaFromTypes(types, opt)
		if e != nil {
			return nil, e
		}
	}
	fields := s.Fields()
	if len(fields) != len(columns) {
		return nil, np.ErrShape
	}
	shape := opt.Shape
	if shape == nil {
		shape, e = inferRecordShape(columns[0], fields[0].Type)
		if e != nil {
			return nil, e
		}
	}
	out, e := records.Zeros(s, shape...)
	if e != nil {
		return nil, e
	}
	for i, a := range columns {
		sh, e := inferRecordShape(a, fields[i].Type)
		if e != nil || !slices.Equal(shape, sh) {
			return nil, np.ErrShape
		}
		f, e := out.Field(fields[i].Name)
		if e != nil {
			return nil, e
		}
		if e = f.Assign(a); e != nil {
			return nil, e
		}
	}
	return out, nil
}

func resolvedShape(s *records.Schema, opt Options, remaining int) ([]int, int, error) {
	if s == nil || s.ItemSize() <= 0 {
		return nil, 0, np.ErrDType
	}
	sh := opt.Shape
	if sh == nil || len(sh) == 1 && sh[0] == -1 {
		sh = []int{remaining / s.ItemSize()}
	}
	n, e := np.ElementCount(np.Bool, sh...)
	if e != nil {
		return nil, 0, e
	}
	if n > int(^uint(0)>>1)/s.ItemSize() {
		return nil, 0, np.ErrShape
	}
	return sh, n * s.ItemSize(), nil
}
func FromString(buffer []byte, opt Options) (*records.Array, error) {
	s, e := explicitSchema(opt)
	if e != nil {
		return nil, e
	}
	if opt.Offset < 0 || opt.Offset > len(buffer) {
		return nil, np.ErrValue
	}
	data := buffer[opt.Offset:]
	shape, n, e := resolvedShape(s, opt, len(data))
	if e != nil {
		return nil, e
	}
	if n > len(data) {
		return nil, np.ErrShape
	}
	return records.FromBytes(s, data[:n], shape, !opt.WritableBuffer)
}
func FromFile(r io.Reader, opt Options) (*records.Array, error) {
	s, e := explicitSchema(opt)
	if e != nil {
		return nil, e
	}
	if r == nil || opt.Offset < 0 {
		return nil, np.ErrValue
	}
	if s == nil || s.ItemSize() <= 0 {
		return nil, np.ErrDType
	}
	if opt.Offset > 0 {
		if _, e = io.CopyN(io.Discard, r, int64(opt.Offset)); e != nil {
			return nil, np.ErrShape
		}
	}
	if opt.Shape == nil || len(opt.Shape) == 1 && opt.Shape[0] == -1 {
		raw, e := io.ReadAll(r)
		if e != nil {
			return nil, e
		}
		opt.Offset = 0
		opt.WritableBuffer = true
		opt.Schema = s
		return FromString(raw, opt)
	}
	shape, n, e := resolvedShape(s, opt, 0)
	if e != nil {
		return nil, e
	}
	raw := make([]byte, n)
	if _, e = io.ReadFull(r, raw); e != nil {
		if e == io.EOF || e == io.ErrUnexpectedEOF {
			return nil, np.ErrShape
		}
		return nil, e
	}
	return records.FromBytes(s, raw, shape, false)
}

// Array dispatches on explicitly typed Go inputs instead of guessing
// whether a Python list holds records or columns.
func Array(obj any, opt Options) (*records.Array, error) {
	switch x := obj.(type) {
	case nil:
		return RecArray(nil, opt)
	case []byte:
		if opt.Strides != nil {
			return RecArray(x, opt)
		}
		return FromString(x, opt)
	case [][]any:
		return FromRecords(x, opt)
	case []*np.Array:
		return FromArrays(x, opt)
	case io.Reader:
		return FromFile(x, opt)
	case *records.Array:
		return Array(x.NDArray(), opt)
	case *np.Array:
		if x.DType() != np.RecordDType {
			return nil, np.ErrDType
		}
		s, e := explicitSchema(opt)
		if e != nil {
			return nil, e
		}
		if s != nil && !x.StructuredSchema().EqualSchema(s) {
			x, e = x.CastRecords(s)
			if e != nil {
				return nil, e
			}
		} else if !opt.NoCopy {
			x, e = x.Copy()
			if e != nil {
				return nil, e
			}
		}
		if opt.Shape != nil {
			x, e = x.Reshape(opt.Shape...)
			if e != nil {
				return nil, e
			}
		}
		return records.FromNDArray(x)
	default:
		return nil, np.ErrDType
	}
}
func FindDuplicate(values []string) []string {
	count := make(map[string]int, len(values))
	out := []string{}
	for _, s := range values {
		count[s]++
	}
	for _, s := range values {
		if count[s] > 1 {
			out = append(out, s)
			count[s] = 0
		}
	}
	return out
}
