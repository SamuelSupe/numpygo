package records

import (
	"encoding/binary"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/SamuelSupe/numpygo"
)

// Record is a storage-sharing view of one structured scalar (np.void/record).
type Record struct {
	schema   *Schema
	data     []byte
	readOnly bool
	slots    *np.RecordSlots
	start    int
	base     *np.Array
	owns     bool
}

func NewRecord(schema *Schema, values ...any) (*Record, error) {
	if schema == nil {
		return nil, np.ErrDType
	}
	r := &Record{schema: schema, data: make([]byte, schema.itemsize), owns: true}
	if schema.HasObject() {
		var e error
		r.slots, e = np.NewRecordSlots(r.data, schema.objectOffsets)
		if e != nil {
			return nil, e
		}
	}
	if len(values) != 0 {
		if e := r.SetValues(values); e != nil {
			return nil, e
		}
	}
	return r, nil
}
func (r *Record) Schema() *Schema { return r.schema }

// Bytes is unavailable for pointer-managed records.
func (r *Record) Bytes() []byte {
	if r.schema.HasObject() {
		return nil
	}
	return slices.Clone(r.data)
}
func (r *Record) Field(name string) (any, error) {
	f, e := r.schema.Field(name)
	if e != nil {
		return nil, e
	}
	n, e := f.Type.Size()
	if e != nil {
		return nil, e
	}
	return readStoredValue(f.Type, r.storage().sub(f.Offset, n), r.readOnly)
}
func (r *Record) Set(name string, value any) error {
	if r.readOnly {
		return np.ErrReadOnly
	}
	f, e := r.schema.Field(name)
	if e != nil {
		return e
	}
	n, e := f.Type.Size()
	if e != nil {
		return e
	}
	return writeStoredValue(f.Type, r.storage().sub(f.Offset, n), value)
}
func (r *Record) Values() ([]any, error) {
	v := make([]any, len(r.schema.fields))
	for i, f := range r.schema.fields {
		var e error
		v[i], e = r.Field(f.Name)
		if e != nil {
			return nil, e
		}
	}
	return v, nil
}
func (r *Record) SetValues(v []any) error {
	if r.readOnly {
		return np.ErrReadOnly
	}
	if len(v) != len(r.schema.fields) {
		return np.ErrShape
	}
	// Stage record updates so invalid values cannot leave a half-written row.
	tmp, e := NewRecord(r.schema)
	if e != nil {
		return e
	}
	copyRecordStorage(tmp, r)
	for i, f := range r.schema.fields {
		if e := tmp.Set(f.Name, v[i]); e != nil {
			return e
		}
	}
	copyRecordStorage(r, tmp)
	return nil
}

func equalField(a, b any) (bool, error) {
	switch x := a.(type) {
	case *Record:
		y, ok := b.(*Record)
		if !ok {
			return false, nil
		}
		return x.EqualRecord(y)
	case *np.Array:
		y, ok := b.(*np.Array)
		if !ok {
			return false, nil
		}
		if x.DType().IsFlexible() && y.DType().IsFlexible() && x.DType() != y.DType() {
			return false, nil
		}
		z, e := np.Equal(x, y)
		if e != nil {
			return false, e
		}
		flags, e := np.Values[bool](z)
		if e != nil {
			return false, e
		}
		for _, v := range flags {
			if !v {
				return false, nil
			}
		}
		return true, nil
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false, nil
		}
		for i, v := range x {
			same, e := equalField(v, y[i])
			if e != nil || !same {
				return same, e
			}
		}
		return true, nil
	case Bytes, Text, Void:
		return reflect.DeepEqual(a, b), nil
	default:
		z, e := np.Equal(np.ObjectScalar(a), np.ObjectScalar(b))
		if e != nil {
			return false, e
		}
		v, e := z.AtValue()
		if e != nil {
			return false, e
		}
		return v.(bool), nil
	}
}
func (s *Schema) EqualScalar(a, b []byte) (bool, error) {
	return s.EqualScalarOther(a, s, b)
}
func (s *Schema) EqualScalarOther(a []byte, other np.StructuredSchema, b []byte) (bool, error) {
	t, ok := other.(*Schema)
	if !ok || t == nil || !slices.Equal(s.Names(), t.Names()) {
		return false, np.ErrDType
	}
	if s.HasObject() || t.HasObject() {
		return false, np.ErrDType
	}
	x, y := &Record{schema: s, data: a, readOnly: true}, &Record{schema: t, data: b, readOnly: true}
	return x.EqualRecord(y)
}

func (x *Record) EqualRecord(other any) (bool, error) {
	y, ok := other.(*Record)
	if !ok || y == nil || !slices.Equal(x.schema.Names(), y.schema.Names()) {
		return false, np.ErrDType
	}
	s, t := x.schema, y.schema
	a, b := x.data, y.data
	for i, f := range s.fields {
		g := t.fields[i]
		if !slices.Equal(f.Type.Shape, g.Type.Shape) {
			return false, np.ErrDType
		}
		var u, v any
		var e error
		if f.Type.Kind == TemporalKind && g.Type.Kind == TemporalKind {
			fw, e := f.Type.Size()
			if e != nil {
				return false, e
			}
			gw, e := g.Type.Size()
			if e != nil {
				return false, e
			}
			u, e = np.FromTemporalBuffer(a[f.Offset:f.Offset+fw], f.Type.DType, f.Type.Unit.String(), f.Type.Shape, true, f.Type.byteOrder())
			if e != nil {
				return false, e
			}
			v, e = np.FromTemporalBuffer(b[g.Offset:g.Offset+gw], g.Type.DType, g.Type.Unit.String(), g.Type.Shape, true, g.Type.byteOrder())
			if e != nil {
				return false, e
			}
		} else if f.Type.Kind == NumericKind && g.Type.Kind == NumericKind {
			// Compare typed fields directly so mixed int64/uint64
			// comparison cannot round through an object float64 loop.
			fw, e := f.Type.Size()
			if e != nil {
				return false, e
			}
			gw, e := g.Type.Size()
			if e != nil {
				return false, e
			}
			u, e = np.FromBuffer(a[f.Offset:f.Offset+fw], f.Type.DType, -1, 0, f.Type.byteOrder())
			if e != nil {
				return false, e
			}
			v, e = np.FromBuffer(b[g.Offset:g.Offset+gw], g.Type.DType, -1, 0, g.Type.byteOrder())
			if e != nil {
				return false, e
			}
		} else {
			u, e = x.Field(f.Name)
			if e != nil {
				return false, e
			}
			v, e = y.Field(f.Name)
		}
		if e != nil {
			return false, e
		}
		same, e := equalField(u, v)
		if e != nil || !same {
			return same, e
		}
	}
	return true, nil
}

func readValue(t Type, b []byte, readonly bool) (any, error) {
	if t.Kind == TemporalKind {
		a, e := np.FromTemporalBuffer(b, t.DType, t.Unit.String(), t.Shape, readonly, t.byteOrder())
		if e != nil {
			return nil, e
		}
		if len(t.Shape) > 0 {
			return a, nil
		}
		return a.AtValue()
	}
	if d, ok := t.flexibleType(); ok {
		a, e := np.FromFlexibleBuffer(b, d, t.Shape, readonly)
		if e != nil {
			return nil, e
		}
		if len(t.Shape) > 0 {
			return a, nil
		}
		return a.AtValue()
	}
	if len(t.Shape) > 0 {
		n, e := np.ElementCount(np.Bool, t.Shape...)
		if e != nil {
			return nil, e
		}
		if t.Kind == NumericKind {
			a, e := np.FromBuffer(b, t.DType, n, 0, t.byteOrder())
			if e != nil {
				return nil, e
			}
			if readonly {
				if e = a.SetWritable(false); e != nil {
					return nil, e
				}
			}
			return a.Reshape(t.Shape...)
		}
		base := t.scalar()
		width, e := base.Size()
		if e != nil {
			return nil, e
		}
		vals := make([]any, n)
		for i := range vals {
			vals[i], e = readValue(base, b[i*width:(i+1)*width], readonly)
			if e != nil {
				return nil, e
			}
		}
		return vals, nil
	}
	switch t.Kind {
	case NumericKind:
		a, e := np.FromBuffer(b, t.DType, 1, 0, t.byteOrder())
		if e != nil {
			return nil, e
		}
		return a.AtValue(0)
	case RecordKind:
		return &Record{schema: t.Schema, data: b, readOnly: readonly}, nil
	}
	return nil, np.ErrDType
}
func unpackValue(v any, n int) ([]any, error) {
	if x, ok := v.([]any); ok {
		if len(x) != n {
			return nil, np.ErrShape
		}
		return x, nil
	}
	r := reflect.ValueOf(v)
	if r.IsValid() && (r.Kind() == reflect.Slice || r.Kind() == reflect.Array) {
		// Flatten nested slices for shaped subarray input.
		out := make([]any, 0, n)
		var appendValues func(reflect.Value)
		appendValues = func(r reflect.Value) {
			if r.Kind() == reflect.Slice || r.Kind() == reflect.Array {
				for i := 0; i < r.Len(); i++ {
					appendValues(r.Index(i))
				}
				return
			}
			out = append(out, r.Interface())
		}
		appendValues(r)
		if len(out) != n {
			return nil, np.ErrShape
		}
		return out, nil
	}
	out := make([]any, n)
	for i := range out {
		out[i] = v
	}
	return out, nil
}
func writeValue(t Type, b []byte, v any) error {
	if t.Kind == TemporalKind {
		a, e := np.FromTemporalBuffer(b, t.DType, t.Unit.String(), t.Shape, false, t.byteOrder())
		if e != nil {
			return e
		}
		if len(t.Shape) == 0 {
			return a.SetValue(v)
		}
		if src, ok := v.(*np.Array); ok {
			return a.Assign(nil, src)
		}
		values, e := unpackValue(v, a.Size())
		if e != nil {
			return e
		}
		tmp, e := np.FromTemporalBuffer(slices.Clone(b), t.DType, t.Unit.String(), t.Shape, false, t.byteOrder())
		if e != nil {
			return e
		}
		it, e := np.NDIndex(t.Shape...)
		if e != nil {
			return e
		}
		i := 0
		for ix := range it {
			if e = tmp.SetValue(values[i], ix...); e != nil {
				return e
			}
			i++
		}
		return a.Assign(nil, tmp)
	}
	if d, ok := t.flexibleType(); ok {
		a, e := np.FromFlexibleBuffer(b, d, t.Shape, false)
		if e != nil {
			return e
		}
		if len(t.Shape) == 0 {
			return a.SetValue(v)
		}
		if src, ok := v.(*np.Array); ok {
			return a.Assign(nil, src)
		}
		n := a.Size()
		values, e := unpackValue(v, n)
		if e != nil {
			return e
		}
		src, e := np.FromFlexible(values, d, t.Shape...)
		if e != nil {
			return e
		}
		return a.Assign(nil, src)
	}
	if len(t.Shape) > 0 {
		n, e := np.ElementCount(np.Bool, t.Shape...)
		if e != nil {
			return e
		}
		if a, ok := v.(*np.Array); ok && t.Kind == NumericKind {
			target, e := np.FromBuffer(b, t.DType, n, 0, t.byteOrder())
			if e != nil {
				return e
			}
			target, e = target.Reshape(t.Shape...)
			if e != nil {
				return e
			}
			return target.Assign(nil, a)
		}
		values, e := unpackValue(v, n)
		if e != nil {
			return e
		}
		base := t.scalar()
		width, e := base.Size()
		if e != nil {
			return e
		}
		tmp := slices.Clone(b)
		for i, x := range values {
			if e = writeValue(base, tmp[i*width:(i+1)*width], x); e != nil {
				return e
			}
		}
		copy(b, tmp)
		return nil
	}
	switch t.Kind {
	case NumericKind:
		a, e := np.FromBuffer(b, t.DType, 1, 0, t.byteOrder())
		if e != nil {
			return e
		}
		return a.SetValue(v, 0)
	case RecordKind:
		r := &Record{schema: t.Schema, data: b}
		switch x := v.(type) {
		case *Record:
			v, e := t.Schema.EncodeScalar(x)
			if e != nil {
				return e
			}
			copy(b, v)
		case []any:
			return r.SetValues(x)
		default:
			v, e := t.Schema.EncodeScalar(x)
			if e != nil {
				return e
			}
			copy(b, v)
		}
	default:
		return np.ErrDType
	}
	return nil
}

type Array struct {
	schema *Schema
	nd     *np.Array
}

func FromNDArray(a *np.Array) (*Array, error) {
	s, ok := a.StructuredSchema().(*Schema)
	if !ok {
		return nil, np.ErrDType
	}
	return &Array{s, a}, nil
}
func (a *Array) NDArray() *np.Array { return a.nd }
func wrap(a *np.Array, e error) (*Array, error) {
	if e != nil {
		return nil, e
	}
	return FromNDArray(a)
}
func FromBytes(schema *Schema, data []byte, shape []int, readonly bool) (*Array, error) {
	if schema == nil {
		return nil, np.ErrDType
	}
	return wrap(np.FromRecordBuffer(schema, data, shape, readonly))
}
func Zeros(schema *Schema, shape ...int) (*Array, error) {
	if schema == nil {
		return nil, np.ErrDType
	}
	return wrap(np.ZerosRecords(schema, shape...))
}
func FromRows(schema *Schema, rows [][]any, shape ...int) (*Array, error) {
	a, e := Zeros(schema, shape...)
	if e != nil {
		return nil, e
	}
	if a.Size() != len(rows) {
		return nil, np.ErrShape
	}
	it, e := np.NDIndex(shape...)
	if e != nil {
		return nil, e
	}
	i := 0
	for ix := range it {
		r, e := a.At(ix...)
		if e != nil {
			return nil, e
		}
		if e = r.SetValues(rows[i]); e != nil {
			return nil, e
		}
		i++
	}
	return a, nil
}
func (a *Array) Schema() *Schema { return a.schema }
func (a *Array) Shape() []int    { return a.nd.Shape() }
func (a *Array) Size() int       { return a.nd.Size() }
func (a *Array) NDim() int       { return a.nd.NDim() }
func (a *Array) Writable() bool  { return a.nd.Writable() }
func (a *Array) Strides() []int  { return a.nd.Strides() }
func (a *Array) At(indices ...int) (*Record, error) {
	v, e := a.nd.AtValue(indices...)
	if e != nil {
		return nil, e
	}
	return v.(*Record), nil
}
func (a *Array) Bytes() []byte                             { v, _ := a.nd.ToBytes("C"); return v }
func (a *Array) Slice(indices ...np.Index) (*Array, error) { return wrap(a.nd.Slice(indices...)) }
func (a *Array) Transpose(axes ...int) (*Array, error)     { return wrap(a.nd.Transpose(axes...)) }
func (a *Array) Reshape(shape ...int) (*Array, error)      { return wrap(a.nd.Reshape(shape...)) }
func (a *Array) Copy() (*Array, error)                     { return wrap(a.nd.Copy()) }
func (a *Array) BroadcastTo(shape ...int) (*Array, error)  { return wrap(a.nd.BroadcastTo(shape...)) }
func (a *Array) Subrecords(path ...string) (*Array, error) { return wrap(a.nd.Field(path...)) }

// Column is a field view. It addresses bytes through record strides so
// non-aligned or packed records can be edited without silently copying.
type Column struct {
	a     *Array
	Field Field
	path  []string
}

func (a *Array) Field(path ...string) (*Column, error) {
	if len(path) == 0 {
		return nil, np.ErrIndex
	}
	s := a.schema
	offset := 0
	var f Field
	for i, name := range path {
		var e error
		f, e = s.Field(name)
		if e != nil {
			return nil, e
		}
		offset += f.Offset
		if i < len(path)-1 {
			if f.Type.Kind != RecordKind || len(f.Type.Shape) > 0 {
				return nil, np.ErrDType
			}
			s = f.Type.Schema
		}
	}
	f.Offset = offset
	return &Column{a, f, slices.Clone(path)}, nil
}
func (c *Column) Shape() []int { return append(c.a.Shape(), c.Field.Type.Shape...) }
func (c *Column) fieldStorage(r *Record) fieldStorage {
	w, _ := c.Field.Type.Size()
	off := c.Field.Offset
	return r.storage().sub(off, w)
}
func (c *Column) At(indices ...int) (any, error) {
	r, e := c.a.At(indices...)
	if e != nil {
		return nil, e
	}
	return readStoredValue(c.Field.Type, c.fieldStorage(r), r.readOnly)
}
func (c *Column) Set(v any, indices ...int) error {
	r, e := c.a.At(indices...)
	if e != nil {
		return e
	}
	if r.readOnly {
		return np.ErrReadOnly
	}
	return writeStoredValue(c.Field.Type, c.fieldStorage(r), v)
}
func (c *Column) Values() ([]any, error) {
	it, e := np.NDIndex(c.a.Shape()...)
	if e != nil {
		return nil, e
	}
	out := make([]any, 0, c.a.Size())
	for ix := range it {
		v, e := c.At(ix...)
		if e != nil {
			return nil, e
		}
		out = append(out, v)
	}
	return out, nil
}

// NumericArray exposes a shared field ndarray, including non-aligned strides.
func (c *Column) NumericArray() (*np.Array, error) {
	if c.Field.Type.Kind != NumericKind {
		return nil, np.ErrDType
	}
	return c.NDArray()
}

// NDArray exposes the field's shared ndarray, including fixed-width
// character and opaque descriptors and nested records.
func (c *Column) NDArray() (*np.Array, error) {
	return c.a.nd.Field(c.path...)
}
func (c *Column) Assign(values *np.Array) error {
	field, e := c.NDArray()
	if e != nil {
		return e
	}
	return field.Assign(nil, values)
}

// FromRegex parses numeric/string capture groups into scalar fields of a
// structured array. Go's RE2 syntax is used rather than Python regex objects.
func FromRegex(reader io.Reader, pattern *regexp.Regexp, schema *Schema) (*Array, error) {
	if pattern == nil || schema == nil || pattern.NumSubexp() != len(schema.fields) {
		return nil, np.ErrShape
	}
	text, e := io.ReadAll(reader)
	if e != nil {
		return nil, e
	}
	matches := pattern.FindAllSubmatch(text, -1)
	rows := make([][]any, len(matches))
	for i, row := range matches {
		rows[i] = make([]any, len(schema.fields))
		for j, f := range schema.fields {
			s := string(row[j+1])
			t := f.Type
			if len(t.Shape) > 0 {
				return nil, np.ErrDType
			}
			switch t.Kind {
			case BytesKind, UnicodeKind:
				rows[i][j] = s
			case NumericKind:
				switch {
				case t.DType == np.Bool:
					rows[i][j] = s != "0" && s != "False"
				case t.DType.IsSigned():
					rows[i][j], e = strconv.ParseInt(s, 10, 64)
				case t.DType.IsUnsigned():
					rows[i][j], e = strconv.ParseUint(s, 10, 64)
				case t.DType.IsFloat():
					rows[i][j], e = strconv.ParseFloat(s, 64)
				case t.DType.IsComplex():
					rows[i][j], e = strconv.ParseComplex(strings.ReplaceAll(s, "j", "i"), 128)
				default:
					return nil, np.ErrDType
				}
				if e != nil {
					return nil, fmt.Errorf("%w: capture %d", np.ErrValue, j)
				}
			default:
				return nil, np.ErrDType
			}
		}
	}
	return FromRows(schema, rows, len(rows))
}

// NativeOrder exposes the default record byte order for callers assembling
// external binary layouts.
var NativeOrder binary.ByteOrder = binary.NativeEndian
