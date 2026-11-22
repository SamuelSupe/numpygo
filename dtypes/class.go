package dtypes

import (
	"encoding/binary"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/records"
	"maps"
	"math"
	"reflect"
	"slices"
	"strings"
)

// Subarray describes a fixed-shape element. Allocation expands the
// subarray dimensions after the requested outer array dimensions.
func Subarray(base Descriptor, shape ...int) (Descriptor, error) {
	if base.DType() == 0 {
		return Descriptor{}, np.ErrDType
	}
	if len(shape) == 0 {
		return base, nil
	}
	n, e := np.ElementCount(np.Bool, shape...)
	if e != nil {
		return Descriptor{}, e
	}
	w := base.ItemSize()
	if w != 0 && n > int(^uint(0)>>1)/w {
		return Descriptor{}, np.ErrShape
	}
	w *= n
	d, e := New("VoidDType", Options{Width: w})
	if e != nil {
		return Descriptor{}, e
	}
	d.sub = &base
	sh := slices.Clone(shape)
	d.shape = &sh
	return d, nil
}
func (d Descriptor) Base() Descriptor {
	if d.sub != nil {
		return *d.sub
	}
	return d
}
func (d Descriptor) Shape() []int {
	if d.shape == nil {
		return nil
	}
	return slices.Clone(*d.shape)
}
func (d Descriptor) NDim() int {
	if d.shape == nil {
		return 0
	}
	return len(*d.shape)
}
func (d Descriptor) Subdtype() (Descriptor, []int, bool) {
	if d.sub == nil {
		return Descriptor{}, nil, false
	}
	return *d.sub, d.Shape(), true
}
func (d Descriptor) Names() []string {
	if d.schema == nil {
		return nil
	}
	return d.schema.Names()
}

type FieldDescription struct {
	Name   string
	Title  any
	Offset int
	DType  Descriptor
}

func (d Descriptor) Field(name string) (FieldDescription, error) {
	if d.schema == nil {
		return FieldDescription{}, np.ErrDType
	}
	f, e := d.schema.Field(name)
	if e != nil {
		return FieldDescription{}, e
	}
	t, e := fieldDescriptor(f.Type)
	if e != nil {
		return FieldDescription{}, e
	}
	return FieldDescription{f.Name, f.Title, f.Offset, t}, nil
}

func fieldDescriptor(t records.Type) (Descriptor, error) {
	d, e := FromRecordType(t)
	if e != nil {
		return Descriptor{}, e
	}
	if len(t.NestedShapes) > 0 {
		for i := len(t.NestedShapes) - 1; i >= 0; i-- {
			d, e = Subarray(d, t.NestedShapes[i]...)
			if e != nil {
				return Descriptor{}, e
			}
		}
		return d, nil
	}
	if len(t.Shape) > 0 {
		return Subarray(d, t.Shape...)
	}
	return d, nil
}
func (d Descriptor) FieldAt(i int) (FieldDescription, error) {
	names := d.Names()
	if i < 0 {
		i += len(names)
	}
	if i < 0 || i >= len(names) {
		return FieldDescription{}, np.ErrIndex
	}
	return d.Field(names[i])
}

// WithNames returns a fresh descriptor with renamed fields, retaining
// original field offsets, titles, byte order and total record width.
func (d Descriptor) WithNames(names []string) (Descriptor, error) {
	if d.schema == nil || len(names) != len(d.schema.Fields()) {
		return Descriptor{}, np.ErrValue
	}
	spec := make([]records.FieldSpec, len(names))
	for i, f := range d.schema.Fields() {
		spec[i] = records.FieldSpec{Name: names[i], Title: f.Title, Type: f.Type, Offset: np.Int(f.Offset)}
	}
	s, e := records.NewSchema(spec, d.schema.Aligned(), d.ItemSize())
	if e != nil {
		return Descriptor{}, e
	}
	d.schema = s
	return d, nil
}
func (d Descriptor) IsAlignedStruct() bool { return d.schema != nil && d.schema.Aligned() }
func (d Descriptor) IsNative() bool {
	if d.sub != nil {
		return true
	}
	if d.schema != nil {
		for _, f := range d.schema.Fields() {
			t, e := FromRecordType(f.Type)
			if e != nil || len(f.Type.Shape) == 0 && !t.IsNative() {
				return false
			}
		}
		return true
	}
	b := d.Metadata().ByteOrder
	return b == '|' || d.order == normalizeOrder(binary.NativeEndian)
}
func (d Descriptor) IsBuiltin() int {
	if d.derived || d.sub != nil || d.schema != nil || d.meta != nil || !d.IsNative() || d.DType().IsTemporal() || d.DType() == np.StringDType {
		return 0
	}
	if d.DType().IsFlexible() && d.ItemSize() != 0 {
		return 0
	}
	if d.DType() == 0 {
		return 0
	}
	return 1
}

// Flags uses NumPy 2.5's descriptor flag bits, not ndarray runtime flags.
func (d Descriptor) Flags() int {
	if d.sub != nil {
		return d.sub.Flags()
	}
	switch d.DType() {
	case np.Object:
		return 63
	case np.StringDType:
		return 107
	case np.UnicodeDType:
		return 8
	case np.RecordDType:
		v := 16
		if d.schema.Aligned() {
			v |= 128
		}
		for _, f := range d.schema.Fields() {
			t, e := FromRecordType(f.Type)
			if e == nil {
				v |= t.Flags() & (1 | 2 | 8)
			}
		}
		return v
	}
	return 0
}
func (d Descriptor) UserMetadata() map[string]any {
	if d.meta == nil {
		return nil
	}
	return maps.Clone(*d.meta)
}
func (d Descriptor) WithMetadata(meta map[string]any) Descriptor {
	d.meta = nil
	if meta != nil {
		m := maps.Clone(meta)
		d.meta = &m
	}
	return d
}

// WithoutMetadata removes user metadata recursively from descriptor-only
// subarray layers without changing dtype, shape, or storage byte order.
func (d Descriptor) WithoutMetadata() Descriptor {
	d.meta = nil
	if d.sub != nil {
		b := d.sub.WithoutMetadata()
		d.sub = &b
	}
	return d
}
func (d Descriptor) Coerce() (bool, error) {
	if d.DType() != np.StringDType {
		return false, np.ErrDType
	}
	return !d.str.Strict, nil
}
func (d Descriptor) NAObject() (any, bool) {
	return d.str.NA, d.DType() == np.StringDType && d.str.HasNA
}

// ScalarTypeName describes the reference scalar class without exposing a
// Python type object.
func (d Descriptor) ScalarTypeName() string {
	switch d.def.name {
	case "BoolDType":
		return "bool"
	case "LongLongDType":
		return "longlong"
	case "ULongLongDType":
		return "ulonglong"
	case "LongDoubleDType":
		return "longdouble"
	case "CLongDoubleDType":
		return "clongdouble"
	case "BytesDType":
		return "bytes_"
	case "StrDType":
		return "str_"
	case "VoidDType":
		return "void"
	case "DateTime64DType":
		return "datetime64"
	case "TimeDelta64DType":
		return "timedelta64"
	case "ObjectDType":
		return "object_"
	case "StringDType":
		return "str"
	default:
		return d.DType().String()
	}
}

// Descr returns the dtype.descr data structure (including padding fields).
// NPY's descriptor rules reject overlapping or out-of-order fields.
func (d Descriptor) Descr() ([]any, error) {
	if d.DType() == 0 {
		return nil, np.ErrDType
	}
	if d.schema != nil {
		s, e := d.schema.NPYDescriptor()
		if e != nil {
			return nil, e
		}
		v, e := np.ParseNPYDescriptorLiteral(s)
		if e != nil {
			return nil, e
		}
		return v.([]any), nil
	}
	return []any{[]any{"", d.String()}}, nil
}

// NewByteOrder implements dtype.newbyteorder recursively, retaining the
// original record layout instead of repacking. StringDType rejects it.
func (d Descriptor) NewByteOrder(order string) (Descriptor, error) {
	if d.DType() == 0 || d.DType() == np.StringDType {
		return Descriptor{}, np.ErrDType
	}
	d.derived = true
	switch strings.ToLower(order) {
	case "s", "swap":
		order = "S"
	case "<", "little", "l":
		order = "<"
	case ">", "big", "b":
		order = ">"
	case "=", "native", "n":
		order = "="
	case "|", "ignore", "i":
		return d, nil
	default:
		return Descriptor{}, np.ErrValue
	}
	if d.sub != nil {
		b, e := d.sub.NewByteOrder(order)
		if e != nil {
			return Descriptor{}, e
		}
		d.sub = &b
		return d, nil
	}
	if d.schema != nil {
		spec := make([]records.FieldSpec, 0, len(d.schema.Fields()))
		for _, f := range d.schema.Fields() {
			b, e := fieldDescriptor(f.Type)
			if e != nil {
				return Descriptor{}, e
			}
			b, e = b.NewByteOrder(order)
			if e != nil {
				return Descriptor{}, e
			}
			t, e := b.RecordType()
			if e != nil {
				return Descriptor{}, e
			}
			spec = append(spec, records.FieldSpec{Name: f.Name, Title: f.Title, Offset: np.Int(f.Offset), Type: t})
		}
		s, e := records.NewSchema(spec, d.schema.Aligned(), d.ItemSize())
		if e != nil {
			return Descriptor{}, e
		}
		if d.schema.Alignment() == 1 && s.Alignment() != 1 {
			s = s.AsVoidDescriptor()
		}
		d.schema = s
		return d, nil
	}
	if d.Metadata().ByteOrder == '|' {
		return d, nil
	}
	var o binary.ByteOrder
	switch order {
	case "<":
		o = binary.LittleEndian
	case ">":
		o = binary.BigEndian
	case "=":
		o = binary.NativeEndian
	case "S":
		if d.order == binary.LittleEndian {
			o = binary.BigEndian
			order = ">"
		} else {
			o = binary.LittleEndian
			order = "<"
		}
	}
	d = d.WithByteOrder(o)
	d.byteMarker = order[0]
	return d, nil
}

// Equal compares exact descriptors; unlike casting-equivalence it doesn't
// interpret a requested width of zero as an unspecified string width.
func (d Descriptor) Equal(q Descriptor) bool {
	if d.DType() == 0 || d.DType() != q.DType() {
		return false
	}
	if d.sub != nil || q.sub != nil {
		return d.sub != nil && q.sub != nil && slices.Equal(*d.shape, *q.shape) && d.sub.Equal(*q.sub)
	}
	if d.schema != nil {
		return d.schema.EqualSchema(q.schema)
	}
	if d.DType() == np.StringDType {
		a, b := d.str, q.str
		if a.Strict != b.Strict || a.HasNA != b.HasNA {
			return false
		}
		if !a.HasNA {
			return true
		}
		if x, ok := a.NA.(float64); ok && math.IsNaN(x) {
			y, ok := b.NA.(float64)
			return ok && math.IsNaN(y)
		}
		return reflect.DeepEqual(a.NA, b.NA)
	}
	if d.DType().IsFlexible() && d.flex.Length != q.flex.Length {
		return false
	}
	return equivalent(d, q, false)
}
func (d Descriptor) Compare(q Descriptor, op string) (bool, error) {
	switch op {
	case "==":
		return d.Equal(q), nil
	case "!=":
		return !d.Equal(q), nil
	case "<", "<=", ">", ">=":
		a, b := d, q
		if op[0] == '>' {
			a, b = b, a
		}
		if d.Equal(q) {
			return len(op) == 2, nil
		}
		return CanCast(a, b, "safe")
	}
	return false, np.ErrValue
}
