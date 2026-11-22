// Package records implements NumPy-style fixed-width strings, opaque values
// and named structured records, including packed/non-aligned field offsets.
package records

import (
	"encoding/binary"
	"fmt"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

type Kind uint8

const (
	NumericKind Kind = iota + 1
	BytesKind
	UnicodeKind
	VoidKind
	RecordKind
	TemporalKind
	ObjectKind
)

// Bytes and Text are scalar equivalents of NumPy bytes_ and str_. Record
// fields have a declared capacity; returned scalars omit trailing NULs.
type Bytes = np.BytesScalar
type Text = np.TextScalar
type Void = np.VoidScalar

func ByteString(value []byte) Bytes   { return Bytes(string(value)) }
func UnicodeString(value string) Text { return Text(value) }
func NewVoid(data []byte) Void        { return slices.Clone(data) }

type Type struct {
	Kind         Kind
	DType        np.DType
	Length       int
	Order        binary.ByteOrder
	Schema       *Schema
	Shape        []int   // optional subarray dimensions (one field per record)
	NestedShapes [][]int // outermost first; nil for a single subarray shape
	Unit         np.DateUnit
}

func Numeric(dtype np.DType, order binary.ByteOrder) Type {
	return Type{Kind: NumericKind, DType: dtype, Order: order}
}
func Temporal(dtype np.DType, unit string, order binary.ByteOrder) (Type, error) {
	if !dtype.IsTemporal() {
		return Type{}, np.ErrDType
	}
	u, e := np.ParseDateUnit(unit)
	if e != nil {
		return Type{}, e
	}
	return Type{Kind: TemporalKind, DType: dtype, Unit: u, Order: order}, nil
}
func FixedBytes(length int) Type { return Type{Kind: BytesKind, Length: length} }
func FixedUnicode(length int, order binary.ByteOrder) Type {
	return Type{Kind: UnicodeKind, Length: length, Order: order}
}
func Opaque(length int) Type     { return Type{Kind: VoidKind, Length: length} }
func Struct(schema *Schema) Type { return Type{Kind: RecordKind, Schema: schema} }
func Object() Type               { return Type{Kind: ObjectKind, DType: np.Object} }
func Subarray(t Type, shape ...int) Type {
	t.Shape = slices.Clone(shape)
	t.NestedShapes = nil
	return t
}

// WrapSubarray retains an existing subarray as the inner descriptor.
// Shape is flattened for field views, NestedShapes retains dtype grouping.
func WrapSubarray(t Type, shape ...int) Type {
	if len(shape) == 0 {
		return t
	}
	if len(t.Shape) == 0 {
		return Subarray(t, shape...)
	}
	groups := [][]int{slices.Clone(shape)}
	if len(t.NestedShapes) > 0 {
		groups = append(groups, cloneShapes(t.NestedShapes)...)
	} else {
		groups = append(groups, slices.Clone(t.Shape))
	}
	t.Shape = append(slices.Clone(shape), t.Shape...)
	t.NestedShapes = groups
	return t
}
func cloneShapes(v [][]int) [][]int {
	if v == nil {
		return nil
	}
	out := make([][]int, len(v))
	for i, sh := range v {
		out[i] = slices.Clone(sh)
	}
	return out
}
func (t Type) IsCharacter() bool { return t.Kind == BytesKind || t.Kind == UnicodeKind }
func (t Type) IsFlexible() bool  { return t.IsCharacter() || t.Kind == VoidKind || t.Kind == RecordKind }
func (t Type) HasObject() bool {
	return t.Kind == ObjectKind || t.Kind == RecordKind && t.Schema != nil && t.Schema.HasObject()
}
func (t Type) flexibleType() (np.FlexibleDType, bool) {
	switch t.Kind {
	case BytesKind:
		return np.FixedBytes(t.Length), true
	case UnicodeKind:
		return np.FixedUnicode(t.Length, t.byteOrder()), true
	case VoidKind:
		return np.FixedVoid(t.Length), true
	}
	return np.FlexibleDType{}, false
}
func (t Type) scalar() Type { t.Shape = nil; t.NestedShapes = nil; return t }
func (t Type) byteOrder() binary.ByteOrder {
	if t.Order == nil {
		return binary.NativeEndian
	}
	return t.Order
}
func (t Type) scalarSize() (int, error) {
	switch t.Kind {
	case NumericKind:
		if n := t.DType.ItemSize(); t.DType.IsNativeNumeric() || t.DType.IsExtended() {
			return n, nil
		}
	case TemporalKind:
		if u, e := np.ParseDateUnit(t.Unit.String()); t.DType.IsTemporal() && e == nil && u == t.Unit {
			return 8, nil
		}
	case ObjectKind:
		return np.Object.ItemSize(), nil
	case BytesKind, VoidKind:
		if t.Length >= 0 {
			return t.Length, nil
		}
	case UnicodeKind:
		if t.Length >= 0 && t.Length <= int(^uint(0)>>1)/4 {
			return t.Length * 4, nil
		}
	case RecordKind:
		if t.Schema != nil {
			return t.Schema.itemsize, nil
		}
	}
	return 0, np.ErrDType
}
func (t Type) Size() (int, error) {
	width, e := t.scalarSize()
	if e != nil {
		return 0, e
	}
	n, e := np.ElementCount(np.Bool, t.Shape...)
	if e != nil {
		return 0, e
	}
	if width != 0 && n > int(^uint(0)>>1)/width {
		return 0, np.ErrShape
	}
	return width * n, nil
}
func (t Type) Alignment() int {
	switch t.Kind {
	case NumericKind:
		n := t.DType.ItemSize()
		if t.DType.IsComplex() {
			n /= 2
		}
		return max(n, 1)
	case TemporalKind:
		return 8
	case ObjectKind:
		return np.Object.ItemSize()
	case UnicodeKind:
		return 4
	case RecordKind:
		if t.Schema != nil {
			return t.Schema.alignment
		}
	}
	return 1
}
func (t Type) Descriptor() string {
	order := "<"
	if t.byteOrder() == binary.BigEndian {
		order = ">"
	}
	var base string
	switch t.Kind {
	case NumericKind:
		n := t.DType.ItemSize()
		if n == 1 {
			order = "|"
		}
		kind := "f"
		switch {
		case t.DType == np.Bool:
			kind = "b"
		case t.DType.IsSigned():
			kind = "i"
		case t.DType.IsUnsigned():
			kind = "u"
		case t.DType.IsComplex():
			kind = "c"
		}
		base = fmt.Sprintf("%s%s%d", order, kind, n)
	case TemporalKind:
		kind := "M"
		if t.DType == np.TimeDelta64 {
			kind = "m"
		}
		base = order + kind + "8"
		if t.Unit.Name != "generic" {
			base += "[" + t.Unit.String() + "]"
		}
	case BytesKind:
		base = fmt.Sprintf("|S%d", t.Length)
	case UnicodeKind:
		base = fmt.Sprintf("%sU%d", order, t.Length)
	case VoidKind:
		base = fmt.Sprintf("|V%d", t.Length)
	case RecordKind:
		base = fmt.Sprintf("|V%d", t.Schema.ItemSize())
	case ObjectKind:
		base = "|O"
	}
	return base
}

type FieldSpec struct {
	Name   string
	Type   Type
	Offset *int
	Title  any // nil, string alias, integer or bool metadata
}
type Field struct {
	Name   string
	Type   Type
	Offset int
	Title  any
}
type Schema struct {
	fields              []Field
	names               map[string]int
	itemsize, alignment int
	aligned             bool
	hasObject           bool
	objectOffsets       []int
}

func (s *Schema) EqualSchema(other np.StructuredSchema) bool {
	t, ok := other.(*Schema)
	if !ok || t == nil || s.itemsize != t.itemsize || len(s.fields) != len(t.fields) {
		return false
	}
	for i, f := range s.fields {
		g := t.fields[i]
		x, y := f.Type, g.Type
		if f.Name != g.Name || f.Title != g.Title || f.Offset != g.Offset || x.Kind != y.Kind || x.DType != y.DType || x.Length != y.Length || x.Unit != y.Unit || !slices.Equal(x.Shape, y.Shape) || !slices.EqualFunc(x.NestedShapes, y.NestedShapes, slices.Equal[[]int]) {
			return false
		}
		if x.byteOrder().Uint16([]byte{1, 0}) != y.byteOrder().Uint16([]byte{1, 0}) && (x.Kind == NumericKind || x.Kind == UnicodeKind || x.Kind == TemporalKind) && x.DType.ItemSize() != 1 {
			return false
		}
		if x.Kind == RecordKind && !x.Schema.EqualSchema(y.Schema) {
			return false
		}
	}
	return true
}

// NativeSchema implements NumPy's record result-type normalization: fields
// use native endian order and are repacked with the original alignment policy.
func (s *Schema) NativeSchema() (np.StructuredSchema, error) {
	specs := make([]FieldSpec, len(s.fields))
	for i, f := range s.fields {
		t := f.Type
		t.Order = binary.NativeEndian
		if t.Kind == RecordKind {
			v, e := t.Schema.NativeSchema()
			if e != nil {
				return nil, e
			}
			t.Schema = v.(*Schema)
		}
		specs[i] = FieldSpec{Name: f.Name, Type: t, Title: f.Title}
	}
	return NewSchema(specs, s.aligned, 0)
}

// PromoteSchema follows named-field promotion and native repacking.
// Subarray dimensions are part of the field contract and may not change.
func (s *Schema) PromoteSchema(other np.StructuredSchema) (np.StructuredSchema, error) {
	t, ok := other.(*Schema)
	if !ok || t == nil || !slices.Equal(s.Names(), t.Names()) {
		return nil, np.ErrDType
	}
	specs := make([]FieldSpec, len(s.fields))
	for i, f := range s.fields {
		if f.Title != t.fields[i].Title {
			return nil, np.ErrDType
		}
		x, y := f.Type, t.fields[i].Type
		if !slices.Equal(x.Shape, y.Shape) || !slices.EqualFunc(x.NestedShapes, y.NestedShapes, slices.Equal[[]int]) {
			return nil, np.ErrDType
		}
		v := x
		v.Order = binary.NativeEndian
		switch {
		case x.Kind == ObjectKind || y.Kind == ObjectKind:
			v = Object()
			v.Shape = slices.Clone(x.Shape)
			v.NestedShapes = cloneShapes(x.NestedShapes)
		case x.Kind == NumericKind && y.Kind == NumericKind:
			d, e := np.PromoteTypes(x.DType, y.DType)
			if e != nil {
				return nil, e
			}
			v.DType = d
		case x.Kind == TemporalKind && y.Kind == TemporalKind:
			d, e := np.PromoteTypes(x.DType, y.DType)
			if e != nil {
				return nil, e
			}
			u, e := np.PromoteDateUnits(x.Unit, y.Unit, d == np.DateTime64)
			if e != nil {
				return nil, e
			}
			v.DType, v.Unit = d, u
		case x.Kind == TemporalKind && y.Kind == NumericKind || y.Kind == TemporalKind && x.Kind == NumericKind:
			temporal, numeric := x, y
			if y.Kind == TemporalKind {
				temporal, numeric = y, x
			}
			d, e := np.PromoteTypes(temporal.DType, numeric.DType)
			if e != nil || d != np.TimeDelta64 {
				return nil, np.ErrDType
			}
			v = temporal
			v.Order = binary.NativeEndian
		case x.Kind == RecordKind && y.Kind == RecordKind:
			d, e := np.PromoteRecordSchemas(x.Schema, y.Schema)
			if e != nil {
				return nil, e
			}
			v.Schema = d.(*Schema)
		case x.IsCharacter() && y.IsCharacter():
			v.Length = max(x.Length, y.Length)
			if x.Kind == UnicodeKind || y.Kind == UnicodeKind {
				v.Kind = UnicodeKind
			} else {
				v.Kind = BytesKind
			}
		case x.IsCharacter() && y.Kind == NumericKind || y.IsCharacter() && x.Kind == NumericKind:
			chars, num := x, y
			if y.IsCharacter() {
				chars, num = y, x
			}
			n, e := np.FlexibleStringWidth(num.DType)
			if e != nil {
				return nil, e
			}
			v = chars
			v.Order = binary.NativeEndian
			v.Length = max(chars.Length, n)
		case x.Kind == VoidKind && y.Kind == VoidKind && x.Length == y.Length:
		default:
			return nil, np.ErrDType
		}
		specs[i] = FieldSpec{Name: f.Name, Type: v, Title: f.Title}
	}
	return NewSchema(specs, s.aligned || t.aligned, 0)
}
func (s *Schema) FieldLayout(name string) (np.StructuredField, error) {
	f, e := s.Field(name)
	if e != nil {
		return np.StructuredField{}, e
	}
	t := f.Type
	l := np.StructuredField{Offset: f.Offset, Shape: t.Shape, Order: t.byteOrder()}
	switch t.Kind {
	case NumericKind, ObjectKind:
		l.DType = t.DType
	case TemporalKind:
		l.DType = t.DType
		l.Unit = t.Unit
	case RecordKind:
		l.DType = np.RecordDType
		l.Schema = t.Schema
	case BytesKind, UnicodeKind, VoidKind:
		l.Flexible, _ = t.flexibleType()
		l.DType = l.Flexible.Kind
	default:
		return np.StructuredField{}, np.ErrDType
	}
	return l, nil
}
func (s *Schema) DecodeScalar(b []byte, readonly bool) any {
	return &Record{schema: s, data: b, readOnly: readonly}
}
func (s *Schema) DecodeManagedScalar(b []byte, slots *np.RecordSlots, start int, readonly bool) any {
	return &Record{schema: s, data: b, readOnly: readonly, slots: slots, start: start}
}
func (s *Schema) EncodeScalar(v any) ([]byte, error) {
	if s.HasObject() {
		return nil, np.ErrDType
	}
	r, e := s.encodeRecord(v)
	if e != nil {
		return nil, e
	}
	return r.data, nil
}
func (s *Schema) EncodeManagedScalar(v any) ([]byte, *np.RecordSlots, error) {
	r, e := s.encodeRecord(v)
	if e != nil {
		return nil, nil, e
	}
	return r.data, r.slots, nil
}
func (s *Schema) encodeRecord(v any) (*Record, error) {
	switch x := v.(type) {
	case *Record:
		if x == nil {
			return nil, np.ErrDType
		}
		if s.EqualSchema(x.schema) {
			r, e := NewRecord(s)
			if e != nil {
				return nil, e
			}
			copyRecordStorage(r, x)
			return r, nil
		}
		return s.castRecord(x)
	case []any:
		return NewRecord(s, x...)
	default:
		// Scalar-to-structured assignment replicates a value into
		// every positional field (including subarray fields).
		v := make([]any, len(s.fields))
		for i := range v {
			v[i] = x
		}
		return NewRecord(s, v...)
	}
}

func alignedOffset(offset, alignment int) (int, error) {
	r := offset % alignment
	if r == 0 {
		return offset, nil
	}
	n := alignment - r
	if offset > int(^uint(0)>>1)-n {
		return 0, np.ErrShape
	}
	return offset + n, nil
}

// NewSchema packs fields unless aligned is true. Explicit offsets support
// overlapped or padded records; itemsize=0 selects the minimum valid size.
func NewSchema(specs []FieldSpec, aligned bool, itemsize int) (*Schema, error) {
	if itemsize < 0 {
		return nil, np.ErrShape
	}
	s := &Schema{names: map[string]int{}, aligned: aligned, alignment: 1}
	end := 0
	for _, f := range specs {
		if f.Name == "" {
			return nil, np.ErrValue
		}
		if _, ok := s.names[f.Name]; ok {
			return nil, np.ErrValue
		}
		switch title := f.Title.(type) {
		case nil, int, bool:
		case string:
			if _, exists := s.names[title]; exists || title == f.Name {
				return nil, np.ErrValue
			}
			s.names[title] = len(s.fields)
		default:
			return nil, np.ErrDType
		}
		t := f.Type
		t.Shape = slices.Clone(t.Shape)
		t.NestedShapes = cloneShapes(t.NestedShapes)
		if len(t.NestedShapes) > 0 {
			var flat []int
			for _, sh := range t.NestedShapes {
				if len(sh) == 0 {
					return nil, np.ErrShape
				}
				flat = append(flat, sh...)
			}
			if len(t.NestedShapes) < 2 || !slices.Equal(flat, t.Shape) {
				return nil, np.ErrShape
			}
		}
		width, e := t.Size()
		if e != nil {
			return nil, e
		}
		align := 1
		if aligned {
			align = t.Alignment()
			s.alignment = max(s.alignment, align)
		}
		offset := end
		if f.Offset != nil {
			offset = *f.Offset
			if offset < 0 || offset%align != 0 {
				return nil, np.ErrValue
			}
		} else {
			offset, e = alignedOffset(offset, align)
			if e != nil {
				return nil, e
			}
		}
		if offset > int(^uint(0)>>1)-width {
			return nil, np.ErrShape
		}
		end = max(end, offset+width)
		s.names[f.Name] = len(s.fields)
		s.fields = append(s.fields, Field{Name: f.Name, Type: t, Offset: offset, Title: f.Title})
	}
	if itemsize == 0 {
		var e error
		itemsize, e = alignedOffset(end, s.alignment)
		if e != nil {
			return nil, e
		}
	}
	if itemsize < end || itemsize%s.alignment != 0 {
		return nil, np.ErrShape
	}
	s.itemsize = itemsize
	if e := s.collectObjectOffsets(); e != nil {
		return nil, e
	}
	return s, nil
}
func (s *Schema) ItemSize() int        { return s.itemsize }
func (s *Schema) Alignment() int       { return s.alignment }
func (s *Schema) Aligned() bool        { return s.aligned }
func (s *Schema) HasObject() bool      { return s.hasObject }
func (s *Schema) ObjectOffsets() []int { return slices.Clone(s.objectOffsets) }

func (s *Schema) collectObjectOffsets() error {
	for i, f := range s.fields {
		if !f.Type.HasObject() {
			continue
		}
		s.hasObject = true
		var offsets []int
		if f.Type.Kind == ObjectKind {
			offsets = []int{0}
		} else {
			offsets = f.Type.Schema.objectOffsets
		}
		if len(offsets) == 0 {
			continue
		}
		n, e := np.ElementCount(np.Bool, f.Type.Shape...)
		if e != nil {
			return e
		}
		w, e := f.Type.scalarSize()
		if e != nil {
			return e
		}
		for k := 0; k < n; k++ {
			for _, relative := range offsets {
				off := f.Offset + k*w + relative
				for j, g := range s.fields {
					if i == j {
						continue
					}
					gw, e := g.Type.Size()
					if e != nil {
						return e
					}
					if gw > 0 && off < g.Offset+gw && g.Offset < off+np.Object.ItemSize() {
						return np.ErrValue
					}
				}
				s.objectOffsets = append(s.objectOffsets, off)
			}
		}
	}
	return nil
}

// AsVoidDescriptor retains the verified field layout and aligned flag but
// applies the scalar void alignment. NumPy's repack_fields constructs
// dtype((void, dtype)), which has alignment 1 even for aligned fields.
func (s *Schema) AsVoidDescriptor() *Schema {
	v := *s
	v.alignment = 1
	return &v
}
func (s *Schema) Fields() []Field {
	a := slices.Clone(s.fields)
	for i := range a {
		a[i].Type.Shape = slices.Clone(a[i].Type.Shape)
		a[i].Type.NestedShapes = cloneShapes(a[i].Type.NestedShapes)
	}
	return a
}
func (s *Schema) Names() []string {
	a := make([]string, len(s.fields))
	for i, f := range s.fields {
		a[i] = f.Name
	}
	return a
}
func (s *Schema) Field(name string) (Field, error) {
	i, ok := s.names[name]
	if !ok {
		return Field{}, np.ErrIndex
	}
	f := s.fields[i]
	f.Type.Shape = slices.Clone(f.Type.Shape)
	f.Type.NestedShapes = cloneShapes(f.Type.NestedShapes)
	return f, nil
}
func (s *Schema) MaskSchema() (*Schema, error) {
	spec := make([]FieldSpec, len(s.fields))
	for i, f := range s.fields {
		t := Numeric(np.Bool, nil)
		if f.Type.Kind == RecordKind {
			m, e := f.Type.Schema.MaskSchema()
			if e != nil {
				return nil, e
			}
			t = Struct(m)
		}
		t.Shape = slices.Clone(f.Type.Shape)
		t.NestedShapes = cloneShapes(f.Type.NestedShapes)
		spec[i] = FieldSpec{Name: f.Name, Type: t, Title: f.Title}
	}
	return NewSchema(spec, false, 0)
}
