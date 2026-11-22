// Package dtypes supplies explicit NumPy-style dtype descriptors on Go
// arrays. Parametric types retain width, temporal unit, field schema and
// missing-value metadata rather than reducing them to a family enum.
package dtypes

import (
	"encoding/binary"
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/records"
	"slices"
	"strings"
)

type classDef struct {
	name string
	d    np.DType
	char byte
	num  int
}

var definitions = []classDef{
	{"BoolDType", np.Bool, '?', 0},
	{"Int8DType", np.Int8, 'b', 1}, {"UInt8DType", np.Uint8, 'B', 2},
	{"Int16DType", np.Int16, 'h', 3}, {"UInt16DType", np.Uint16, 'H', 4},
	{"Int32DType", np.Int32, 'i', 5}, {"UInt32DType", np.Uint32, 'I', 6},
	{"Int64DType", np.Int64, 'l', 7}, {"UInt64DType", np.Uint64, 'L', 8},
	{"LongLongDType", np.Int64, 'q', 9}, {"ULongLongDType", np.Uint64, 'Q', 10},
	{"Float16DType", np.Float16, 'e', 23}, {"Float32DType", np.Float32, 'f', 11}, {"Float64DType", np.Float64, 'd', 12},
	{"LongDoubleDType", np.Float128, 'g', 13}, {"Complex64DType", np.Complex64, 'F', 14},
	{"Complex128DType", np.Complex128, 'D', 15}, {"CLongDoubleDType", np.Complex256, 'G', 16},
	{"ObjectDType", np.Object, 'O', 17}, {"BytesDType", np.BytesDType, 'S', 18},
	{"StrDType", np.UnicodeDType, 'U', 19}, {"VoidDType", np.VoidDType, 'V', 20},
	{"DateTime64DType", np.DateTime64, 'M', 21}, {"TimeDelta64DType", np.TimeDelta64, 'm', 22},
	{"StringDType", np.StringDType, 'T', 2056},
}
var aliases = map[string]string{"ByteDType": "Int8DType", "UByteDType": "UInt8DType", "ShortDType": "Int16DType", "UShortDType": "UInt16DType",
	"IntDType": "Int32DType", "UIntDType": "UInt32DType", "LongDType": "Int64DType", "ULongDType": "UInt64DType"}

type Options struct {
	Width  int // bytes for S/V, code points for U
	Unit   string
	Order  binary.ByteOrder
	Schema *records.Schema
	String np.StringDescriptor
}
type Descriptor struct {
	def        classDef
	order      binary.ByteOrder
	flex       np.FlexibleDType
	unit       np.DateUnit
	schema     *records.Schema
	str        np.StringDescriptor
	sub        *Descriptor
	shape      *[]int
	meta       *map[string]any
	byteMarker byte
	derived    bool
}

func classFor(d np.DType) string {
	if d == np.RecordDType {
		return "VoidDType"
	}
	for _, v := range definitions {
		if v.d == d {
			return v.name
		}
	}
	return ""
}
func normalizeOrder(o binary.ByteOrder) binary.ByteOrder {
	if o == nil {
		o = binary.NativeEndian
	}
	if o.Uint16([]byte{1, 0}) == 1 {
		return binary.LittleEndian
	}
	return binary.BigEndian
}

// New constructs any public NumPy dtype class by its exported class name.
// The non-instantiable legacy Void/DateTime classes use explicit Go options.
func New(class string, opt Options) (Descriptor, error) {
	if v, ok := aliases[class]; ok {
		class = v
	}
	var d Descriptor
	for _, v := range definitions {
		if v.name == class {
			d.def = v
			break
		}
	}
	if d.def.d == 0 {
		return d, np.ErrDType
	}
	d.order = normalizeOrder(opt.Order)
	if opt.Width < 0 || opt.Width != 0 && !d.def.d.IsFlexible() {
		return Descriptor{}, np.ErrDType
	}
	if opt.Unit != "" && !d.def.d.IsTemporal() {
		return Descriptor{}, np.ErrDType
	}
	if opt.Schema != nil && d.def.d != np.VoidDType {
		return Descriptor{}, np.ErrDType
	}
	if (opt.String.HasNA || opt.String.Strict) && d.def.d != np.StringDType {
		return Descriptor{}, np.ErrDType
	}
	switch d.def.d {
	case np.BytesDType:
		d.flex = np.FixedBytes(opt.Width)
	case np.UnicodeDType:
		if opt.Width > int(^uint(0)>>1)/4 {
			return Descriptor{}, np.ErrShape
		}
		d.flex = np.FixedUnicode(opt.Width, d.order)
	case np.VoidDType:
		if opt.Schema != nil {
			d.schema = opt.Schema
			d.def.d = np.RecordDType
		} else {
			d.flex = np.FixedVoid(opt.Width)
		}
	case np.DateTime64, np.TimeDelta64:
		u, e := np.ParseDateUnit(opt.Unit)
		if e != nil {
			return Descriptor{}, e
		}
		d.unit = u
	case np.StringDType:
		if _, e := np.FromStringSlots(nil, nil, opt.String, 0); e != nil {
			return Descriptor{}, e
		}
		d.str = opt.String
	}
	return d, nil
}
func (d Descriptor) DType() np.DType                    { return d.def.d }
func (d Descriptor) Unit() np.DateUnit                  { return d.unit }
func (d Descriptor) Schema() *records.Schema            { return d.schema }
func (d Descriptor) StringOptions() np.StringDescriptor { return d.str }
func (d Descriptor) Order() binary.ByteOrder            { return d.order }
func (d Descriptor) ItemSize() int {
	if d.schema != nil {
		return d.schema.ItemSize()
	}
	if d.DType().IsFlexible() {
		return d.flex.ItemSize()
	}
	return d.DType().ItemSize()
}

// Metadata reports the reference Linux/64-bit NumPy dtype semantics.
// Alias-specific type numbers and chars are retained even when Go storage
// uses the same integer type.
type Metadata struct {
	Class, Name, Str         string
	Kind, Char, ByteOrder    byte
	ItemSize, Alignment, Num int
	HasObject                bool
}

func (d Descriptor) Metadata() Metadata {
	t := d.DType()
	m := Metadata{Class: d.def.name, Name: t.String(), Char: d.def.char, Num: d.def.num, ItemSize: d.ItemSize()}
	switch {
	case d.sub != nil:
		m.Kind = 'V'
		m.Name = "void"
		if m.ItemSize != 0 {
			m.Name = fmt.Sprintf("void%d", m.ItemSize*8)
		}
		m.Str = fmt.Sprintf("|V%d", m.ItemSize)
		m.Alignment = d.sub.Metadata().Alignment
		m.HasObject = d.sub.Metadata().HasObject
	case t == np.RecordDType:
		m.Kind = 'V'
		m.Name = fmt.Sprintf("void%d", m.ItemSize*8)
		m.Str = fmt.Sprintf("|V%d", m.ItemSize)
		m.Alignment = d.schema.Alignment()
		m.HasObject = d.schema.HasObject()
	case t == np.StringDType:
		m.Kind = 'T'
		m.Name = "StringDType128"
		m.Str = d.str.String()
		m.Alignment = 8
		m.HasObject = true
	case t == np.Object:
		m.Kind = 'O'
		m.Str = "|O"
		m.Alignment = 8
		m.HasObject = true
	default:
		r, e := d.RecordType()
		if e != nil {
			return m
		}
		m.Str = r.Descriptor()
		m.Kind = m.Str[1]
		m.Alignment = r.Alignment()
		if t.IsTemporal() {
			m.Name = t.String()
			if d.unit.Name != "generic" {
				m.Name += "[" + d.unit.String() + "]"
			}
		}
		if t.IsFlexible() {
			if m.ItemSize > 0 {
				m.Name = fmt.Sprintf("%s%d", t.String(), m.ItemSize*8)
			}
		}
	}
	if t == np.Bool {
		m.Kind = 'b'
	}
	if len(m.Str) > 0 {
		m.ByteOrder = m.Str[0]
		if t == np.StringDType {
			m.ByteOrder = '|'
		}
		if m.ByteOrder == '<' && binary.NativeEndian.Uint16([]byte{1, 0}) == 1 || m.ByteOrder == '>' && binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
			m.ByteOrder = '='
		}
	}
	if d.byteMarker != 0 {
		m.ByteOrder = d.byteMarker
	}
	return m
}
func (d Descriptor) String() string { return d.Metadata().Str }
func (d Descriptor) RecordType() (records.Type, error) {
	if d.sub != nil {
		t, e := d.sub.RecordType()
		if e != nil {
			return t, e
		}
		return records.WrapSubarray(t, d.Shape()...), nil
	}
	switch d.DType() {
	case np.BytesDType:
		return records.FixedBytes(d.flex.Length), nil
	case np.UnicodeDType:
		return records.FixedUnicode(d.flex.Length, d.order), nil
	case np.VoidDType:
		return records.Opaque(d.flex.Length), nil
	case np.RecordDType:
		return records.Struct(d.schema), nil
	case np.DateTime64, np.TimeDelta64:
		return records.Temporal(d.DType(), d.unit.String(), d.order)
	case np.Object:
		return records.Object(), nil
	case np.StringDType, 0:
		return records.Type{}, np.ErrDType
	default:
		return records.Numeric(d.DType(), d.order), nil
	}
}
func FromRecordType(t records.Type) (Descriptor, error) {
	opt := Options{Order: t.Order}
	var d np.DType
	switch t.Kind {
	case records.NumericKind, records.ObjectKind:
		d = t.DType
	case records.TemporalKind:
		d = t.DType
		opt.Unit = t.Unit.String()
	case records.BytesKind:
		d = np.BytesDType
		opt.Width = t.Length
	case records.UnicodeKind:
		d = np.UnicodeDType
		opt.Width = t.Length
	case records.VoidKind:
		d = np.VoidDType
		opt.Width = t.Length
	case records.RecordKind:
		d = np.RecordDType
		opt.Schema = t.Schema
	default:
		return Descriptor{}, np.ErrDType
	}
	return New(classFor(d), opt)
}
func FromArray(a *np.Array) (Descriptor, error) {
	opt := Options{}
	d := a.DType()
	if d == 0 {
		return Descriptor{}, np.ErrDType
	}
	if !a.NativeByteOrder() {
		opt.Order = binary.BigEndian
		if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
			opt.Order = binary.LittleEndian
		}
	}
	if d.IsFlexible() {
		f, _ := a.FlexibleType()
		opt.Width = f.Length
		opt.Order = f.Order
	}
	if d.IsTemporal() {
		u, _ := a.DateUnit()
		opt.Unit = u.String()
	}
	if d == np.RecordDType {
		s, ok := a.StructuredSchema().(*records.Schema)
		if !ok {
			return Descriptor{}, np.ErrDType
		}
		opt.Schema = s
	}
	if d == np.StringDType {
		opt.String, _ = a.StringDescriptor()
	}
	return New(classFor(d), opt)
}

// Parse accepts NumPy type strings, native aliases and explicit byte order.
func Parse(spec string) (Descriptor, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return Descriptor{}, np.ErrDType
	}
	opt := Options{}
	prefix := "="
	if strings.ContainsRune("<>=|", rune(spec[0])) {
		prefix, spec = spec[:1], spec[1:]
		if prefix == ">" {
			opt.Order = binary.BigEndian
		} else if prefix == "<" {
			opt.Order = binary.LittleEndian
		}
	}
	if spec == "" {
		return Descriptor{}, np.ErrDType
	}
	if strings.HasPrefix(spec, "datetime64") {
		spec = "M8" + strings.TrimPrefix(spec, "datetime64")
	}
	if strings.HasPrefix(spec, "timedelta64") {
		spec = "m8" + strings.TrimPrefix(spec, "timedelta64")
	}
	if strings.HasSuffix(spec, "DType") {
		return New(spec, opt)
	}
	if n, ok := map[string]string{"int": "Int64DType", "uint": "UInt64DType", "float": "Float64DType", "complex": "Complex128DType", "longlong": "LongLongDType", "ulonglong": "ULongLongDType", "long": "LongDType", "ulong": "ULongDType"}[spec]; ok {
		return New(n, opt)
	}
	if spec == "M" || spec == "M8" {
		spec = "M8"
	}
	if spec == "m" || spec == "m8" {
		spec = "m8"
	}
	if strings.HasPrefix(spec, "M8") || strings.HasPrefix(spec, "m8") {
		d, u, o, e := np.ParseTemporalDescriptor(prefix + spec)
		if e != nil {
			return Descriptor{}, e
		}
		return New(classFor(d), Options{Order: o, Unit: u.String()})
	}
	if strings.ContainsRune("SUV", rune(spec[0])) {
		if len(spec) == 1 {
			spec += "0"
		}
		f, e := np.ParseFlexibleDType(prefix + spec)
		if e != nil {
			return Descriptor{}, e
		}
		return New(classFor(f.Kind), Options{Order: f.Order, Width: f.Length})
	}
	if spec == "O" || spec == "O8" || spec == "object_" {
		return New("ObjectDType", opt)
	}
	if spec == "T" {
		return New("StringDType", opt)
	}
	if len(spec) == 1 {
		for _, v := range definitions {
			if v.char == spec[0] {
				return New(v.name, opt)
			}
		}
	}
	if d, e := np.ParseDType(spec); e == nil {
		return New(classFor(d), opt)
	}
	if d, o, e := np.ParseNPYScalarDescriptor(prefix + spec); e == nil {
		return New(classFor(d), Options{Order: o})
	}
	return New(spec, opt)
}

func (d Descriptor) Zeros(shape ...int) (*np.Array, error) {
	if d.sub != nil {
		// NumPy represents zero-width subarrays as zero-width void
		// elements and does not expand their trailing dimensions.
		if d.ItemSize() == 0 {
			return np.ZerosFlexible(np.FixedVoid(0), shape...)
		}
		return d.sub.Zeros(append(slices.Clone(shape), (*d.shape)...)...)
	}
	t := d.DType()
	if t == 0 {
		return nil, np.ErrDType
	}
	if d.schema != nil {
		return np.ZerosRecords(d.schema, shape...)
	}
	if t.IsFlexible() {
		return np.ZerosFlexible(d.flex, shape...)
	}
	if t == np.Object {
		return np.Zeros(t, shape...)
	}
	n, e := np.ElementCount(t, shape...)
	if e != nil {
		return nil, e
	}
	if t == np.StringDType {
		return np.FromStringSlots(make([]string, n), nil, d.str, shape...)
	}
	if t.IsTemporal() {
		return np.FromTemporalBuffer(make([]byte, n*8), t, d.unit.String(), shape, false, d.order)
	}
	if d.Metadata().ByteOrder == '=' || d.Metadata().ByteOrder == '|' {
		return np.Zeros(t, shape...)
	}
	a, e := np.FromBuffer(make([]byte, n*d.ItemSize()), t, n, 0, d.order)
	if e != nil {
		return nil, e
	}
	return a.Reshape(shape...)
}
