package np

import (
	"bytes"
	"encoding/binary"
	"math/big"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

// StructuredSchema is the descriptor contract implemented by records.Schema.
// It keeps byte layout and scalar validation in the records package while
// Array owns indexing, broadcasting, assignment and memory sharing.
type StructuredSchema interface {
	ItemSize() int
	EqualSchema(StructuredSchema) bool
	NativeSchema() (StructuredSchema, error)
	DecodeScalar([]byte, bool) any
	EncodeScalar(any) ([]byte, error)
	EqualScalar([]byte, []byte) (bool, error)
	FieldLayout(string) (StructuredField, error)
}
type StructuredField struct {
	Offset   int
	DType    DType
	Shape    []int
	Order    binary.ByteOrder
	Schema   StructuredSchema
	Flexible FlexibleDType
	Unit     DateUnit
}

func FromRecordBuffer(schema StructuredSchema, data []byte, shape []int, readonly bool) (*Array, error) {
	if schema == nil || schema.ItemSize() < 0 || schemaHasObjects(schema) {
		return nil, ErrDType
	}
	n, e := shapeSizeWidth(max(schema.ItemSize(), 1), shape)
	if e != nil {
		return nil, e
	}
	if len(data) != n*schema.ItemSize() {
		return nil, ErrShape
	}
	if data == nil {
		data = make([]byte, 0)
	}
	a := owned(&storage{dtype: RecordDType, schema: schema, buffer: data, order: binary.NativeEndian}, shape, n)
	a.readOnly = readonly
	return a, nil
}

// FromRecordStridedBuffer shares packed record storage with explicit byte
// strides and offset. Validate the entire addressable interval before
// exposing a view, including negative and non-item-aligned strides.
func FromRecordStridedBuffer(schema StructuredSchema, data []byte, shape, strides []int, offset int, readonly bool) (*Array, error) {
	if schema == nil || schema.ItemSize() < 0 || schemaHasObjects(schema) {
		return nil, ErrDType
	}
	width := schema.ItemSize()
	n, e := shapeSizeWidth(max(width, 1), shape)
	if e != nil {
		return nil, e
	}
	if offset < 0 || offset > len(data) {
		return nil, ErrShape
	}
	if strides == nil {
		strides = contiguousStrides(shape, n)
		for i := range strides {
			strides[i] *= width
		}
	}
	if e = boundedLayout(len(data), offset, width, n, shape, strides); e != nil {
		return nil, e
	}
	if n == 0 {
		offset = 0
	}
	if data == nil {
		data = make([]byte, 0)
	}
	return &Array{data: &storage{dtype: RecordDType, schema: schema, buffer: data, order: binary.NativeEndian, byteIndexed: true},
		shape: slices.Clone(shape), strides: slices.Clone(strides), offset: offset, size: n, readOnly: readonly}, nil
}
func ZerosRecords(schema StructuredSchema, shape ...int) (*Array, error) {
	if schema == nil || schema.ItemSize() < 0 {
		return nil, ErrDType
	}
	n, e := shapeSizeWidth(max(schema.ItemSize(), 1), shape)
	if e != nil {
		return nil, e
	}
	data := make([]byte, n*schema.ItemSize())
	if schemaHasObjects(schema) {
		slots, e := newRecordSlots(data, schema.(managedRecordSchema).ObjectOffsets(), n, schema.ItemSize())
		if e != nil {
			return nil, e
		}
		return owned(&storage{dtype: RecordDType, schema: schema, buffer: data, slots: slots, order: binary.NativeEndian}, shape, n), nil
	}
	return FromRecordBuffer(schema, data, shape, false)
}
func (a *Array) StructuredSchema() StructuredSchema {
	if a.DType() != RecordDType {
		return nil
	}
	return a.data.schema
}
func (a *Array) recordBytes(off int) []byte {
	start := off * a.strideUnit()
	return a.data.buffer[start : start+a.ItemSize()]
}

// Field returns a shared numerical, temporal, flexible or nested-structure ndarray,
// including packed byte strides and trailing subarray dimensions.
func (a *Array) Field(path ...string) (*Array, error) {
	if len(path) == 0 {
		return nil, ErrIndex
	}
	if a.DType() != RecordDType {
		return nil, ErrDType
	}
	f, e := a.data.schema.FieldLayout(path[0])
	if e != nil {
		return nil, e
	}
	width := f.DType.ItemSize()
	if f.DType == RecordDType {
		if f.Schema == nil {
			return nil, ErrDType
		}
		width = f.Schema.ItemSize()
	}
	if f.DType.IsFlexible() {
		if f.Flexible.Kind != f.DType {
			return nil, ErrDType
		}
		width, e = f.Flexible.width()
		if e != nil {
			return nil, e
		}
	}
	if (width <= 0 && !f.DType.IsFlexible()) || f.Offset < 0 || f.DType == StringDType || f.DType == Object && a.data.slots == nil {
		return nil, ErrDType
	}
	if f.DType.IsTemporal() {
		if u, e := ParseDateUnit(f.Unit.String()); e != nil || u != f.Unit {
			return nil, ErrDType
		}
	}
	count, e := shapeSizeWidth(max(width, 1), f.Shape)
	if e != nil {
		return nil, e
	}
	if f.Offset > a.ItemSize()-width*count {
		return nil, ErrShape
	}
	shape := append(a.Shape(), f.Shape...)
	strides := append(a.Strides(), contiguousStrides(f.Shape, count)...)
	for i := a.NDim(); i < len(strides); i++ {
		strides[i] *= width
	}
	size, e := shapeSizeWidth(max(width, 1), shape)
	if e != nil {
		return nil, e
	}
	off := a.offset*a.strideUnit() + f.Offset
	if size == 0 {
		off = 0
	}
	order := f.Order
	if order == nil {
		order = binary.NativeEndian
	}
	v := &Array{data: &storage{dtype: f.DType, schema: f.Schema, flex: f.Flexible, unit: f.Unit, buffer: a.data.buffer, order: order, byteIndexed: true}, shape: shape, strides: strides, offset: off, size: size, readOnly: a.readOnly, base: a.viewBase()}
	if f.DType == Object || f.DType == RecordDType && schemaHasObjects(f.Schema) {
		v.data.slots = a.data.slots
		if v.data.slots == nil {
			return nil, ErrDType
		}
		if f.DType == Object && !validObjectSlots(v) {
			return nil, ErrShape
		}
	}
	if len(path) > 1 {
		return v.Field(path[1:]...)
	}
	return v, nil
}

// FromObjects copies the reference slots, but does not deep-copy any Go
// object in them, matching ndarray object-array copy semantics.
func FromObjects(values []any, shape ...int) (*Array, error) {
	n, e := shapeSize(Object, shape)
	if e != nil {
		return nil, e
	}
	if len(values) != n {
		return nil, ErrShape
	}
	return owned(&storage{dtype: Object, raw: slices.Clone(values)}, shape, n), nil
}
func ObjectScalar(v any) *Array { a, _ := FromObjects([]any{v}); return a }
func (a *Array) ObjectValues() ([]any, error) {
	if a.DType() != Object {
		return nil, ErrDType
	}
	out := make([]any, a.Size())
	a.walk(func(i, off int) { out[i] = a.rawAt(off) })
	return out, nil
}
func objectTruth(v any) bool {
	if v == nil {
		return false
	}
	if d := dtypeOf(v); d != 0 {
		return elementFrom(v).truth(d)
	}
	r := reflect.ValueOf(v)
	switch r.Kind() {
	case reflect.Array, reflect.Slice, reflect.Map, reflect.String:
		return r.Len() != 0
	case reflect.Pointer, reflect.Interface, reflect.Func, reflect.Chan:
		return !r.IsNil()
	}
	return true
}
func recordTruth(v any) bool {
	if r, ok := v.(managedRecordValue); ok {
		for _, rel := range r.schema.ObjectOffsets() {
			v, _ := r.slots.Get(r.offset + rel)
			if objectTruth(v) {
				return true
			}
		}
		v = r.bytes
	}
	b, _ := v.([]byte)
	for _, x := range b {
		if x != 0 {
			return true
		}
	}
	return false
}

func extendedValue(e element, d DType) x87.Complex {
	switch d {
	case Float128:
		v, _ := e.extra.(x87.Float)
		return x87.Complex{Real: v}
	case Complex256:
		v, _ := e.extra.(x87.Complex)
		return v
	case Bool, Uint8, Uint16, Uint32, Uint64:
		return x87.Complex{Real: x87.FromUint64(e.i)}
	case Int8, Int16, Int32, Int64:
		return x87.Complex{Real: x87.FromInt64(int64(e.i))}
	default:
		return x87.FromComplex128(e.z)
	}
}
func extendedElement(z x87.Complex, d DType) element {
	if d == Float128 {
		return element{extra: z.Real}
	}
	return element{extra: z}
}
func castExtended(e element, from, to DType) element {
	z := extendedValue(e, from)
	if to.IsExtended() {
		return extendedElement(z, to)
	}
	if to == Bool {
		return boolElement(!z.Real.IsZero() || !z.Imag.IsZero())
	}
	if to.IsInteger() && z.Real.IsFinite() {
		r := z.Real.Rat()
		q := new(big.Int).Quo(r.Num(), r.Denom())
		if q.IsInt64() {
			return roundElement(element{i: uint64(q.Int64())}, to)
		}
		if q.IsUint64() {
			return roundElement(element{i: q.Uint64()}, to)
		}
	}
	return castElement(element{z: z.Complex128()}, Complex128, to)
}
func scalarFromElement(e element, d DType) any {
	if d.IsFlexible() {
		return flexScalar(e)
	}
	switch d {
	case Bool:
		return e.i != 0
	case Int8:
		return int8(e.i)
	case Int16:
		return int16(e.i)
	case Int32:
		return int32(e.i)
	case Int64:
		return int64(e.i)
	case Uint8:
		return uint8(e.i)
	case Uint16:
		return uint16(e.i)
	case Uint32:
		return uint32(e.i)
	case Uint64:
		return e.i
	case Float16:
		return HalfFromFloat64(real(e.z))
	case Float32:
		return float32(real(e.z))
	case Float64:
		return real(e.z)
	case Complex64:
		return complex64(e.z)
	case Complex128:
		return e.z
	default:
		return e.extra
	}
}
func (a *Array) zerosLike(shape []int) (*Array, error) {
	d := a.DType()
	if d.IsFlexible() {
		return ZerosFlexible(a.data.flex, shape...)
	}
	if d == StringDType {
		out, e := Zeros(d, shape...)
		if e == nil {
			out.setStringDescriptor(a.stringDescriptor())
		}
		return out, e
	}
	if d == RecordDType {
		return ZerosRecords(a.data.schema, shape...)
	}
	if d.IsTemporal() {
		n, e := shapeSize(d, shape)
		if e != nil {
			return nil, e
		}
		s := makeStorage(d, n)
		s.unit = a.data.unit
		return owned(s, shape, n), nil
	}
	return Zeros(d, shape...)
}
func (a *Array) sameDescriptor(b *Array) bool {
	if a.DType() != b.DType() {
		return false
	}
	if a.DType().IsTemporal() {
		return a.data.unit == b.data.unit
	}
	if a.DType() == RecordDType {
		return a.data.schema.EqualSchema(b.data.schema)
	}
	if a.DType().IsFlexible() {
		return a.data.flex.Equal(b.data.flex)
	}
	if a.DType() == StringDType {
		return sameStringDescriptor(a.stringDescriptor(), b.stringDescriptor())
	}
	return true
}
func (a *Array) convertValue(v any) (element, error) {
	d := a.DType()
	switch d {
	case StringDType:
		return a.convertString(v)
	case BytesDType, UnicodeDType, VoidDType:
		x, e := encodeFlexible(v, a.data.flex)
		return element{extra: x}, e
	case Object:
		return element{extra: v}, nil
	case RecordDType:
		if schemaHasObjects(a.data.schema) {
			s := a.data.schema.(managedRecordSchema)
			b, slots, e := s.EncodeManagedScalar(v)
			if e != nil {
				return element{}, e
			}
			if len(b) != a.ItemSize() {
				return element{}, ErrShape
			}
			return element{extra: managedRecordValue{b, slots, 0, s}}, nil
		}
		b, e := a.data.schema.EncodeScalar(v)
		if e != nil {
			return element{}, e
		}
		if len(b) != a.ItemSize() {
			return element{}, ErrShape
		}
		return element{extra: b}, nil
	case DateTime64, TimeDelta64:
		return convertTemporal(v, d, a.data.unit)
	}
	if k := dtypeOf(v); k.IsFlexible() {
		if d == Bool {
			return boolElement(elementFrom(v).truth(k)), nil
		}
		switch x := v.(type) {
		case BytesScalar:
			v = string(x)
		case TextScalar:
			v = string(x)
		default:
			return element{}, ErrDType
		}
	}
	if d == Bool && dtypeOf(v) == 0 {
		return boolElement(objectTruth(v)), nil
	}
	from := dtypeOf(v)
	if from == 0 {
		if s, ok := v.(string); ok {
			s = strings.TrimSpace(s)
			if d.IsExtended() {
				if d == Float128 {
					f, e := x87.Parse(s)
					return element{extra: f}, e
				}
				z, e := parseNumber(s, Complex256)
				if e != nil {
					return element{}, ErrValue
				}
				return castElement(elementFrom(z), Complex128, Complex256), nil
			}
			if d.IsSigned() {
				x, e := strconv.ParseInt(s, 10, 64)
				if e != nil {
					return element{}, ErrValue
				}
				v = x
			}
			if d.IsUnsigned() {
				x, e := strconv.ParseUint(s, 10, 64)
				if e != nil {
					return element{}, ErrValue
				}
				v = x
			}
			if d.IsFloat() {
				x, e := strconv.ParseFloat(s, 64)
				if e != nil {
					return element{}, ErrValue
				}
				v = x
			}
			if d.IsComplex() {
				x, e := parseNumber(s, d)
				if e != nil {
					return element{}, ErrValue
				}
				v = x
			}
			from = dtypeOf(v)
		}
	}
	if from == 0 || from.IsTemporal() {
		return element{}, ErrDType
	}
	return castElement(elementFrom(v), from, d), nil
}
func allocationFor(arrays []*Array, d DType, shape []int) (*Array, error) {
	if d == StringDType {
		r, e := Zeros(d, shape...)
		if e == nil {
			var desc StringDescriptor
			desc, e = mergeStringDescriptors(arrays...)
			if e == nil {
				r.setStringDescriptor(desc)
			}
		}
		return r, e
	}
	if d.IsFlexible() {
		f, e := flexibleDescriptorFor(arrays, d)
		if e != nil {
			return nil, e
		}
		return ZerosFlexible(f, shape...)
	}
	if d == RecordDType {
		s, e := arrays[0].data.schema.NativeSchema()
		if e != nil {
			return nil, e
		}
		for _, a := range arrays[1:] {
			if a.DType() != RecordDType {
				return nil, ErrDType
			}
			s, e = PromoteRecordSchemas(s, a.data.schema)
			if e != nil {
				return nil, e
			}
		}
		return ZerosRecords(s, shape...)
	}
	if d.IsTemporal() {
		u := DateUnit{Name: "generic", Step: 1}
		for _, a := range arrays {
			if a.DType() != d {
				if d == TimeDelta64 {
					if p, e := PromoteTypes(d, a.DType()); e == nil && p == d {
						continue
					}
				}
				return nil, ErrDType
			}
			var e error
			u, e = commonDateUnit(u, a.data.unit, d == DateTime64)
			if e != nil {
				return nil, e
			}
		}
		n, e := shapeSize(d, shape)
		if e != nil {
			return nil, e
		}
		s := makeStorage(d, n)
		s.unit = u
		return owned(s, shape, n), nil
	}
	return Zeros(d, shape...)
}

// PromoteRecordSchemas resolves both field descriptors and the resulting
// native byte layout; RecordDType alone cannot encode that information.
func PromoteRecordSchemas(a, b StructuredSchema) (StructuredSchema, error) {
	if a == nil || b == nil {
		return nil, ErrDType
	}
	if p, ok := a.(interface {
		PromoteSchema(StructuredSchema) (StructuredSchema, error)
	}); ok {
		return p.PromoteSchema(b)
	}
	if a.EqualSchema(b) {
		return a.NativeSchema()
	}
	return nil, ErrDType
}

// CastRecords materializes an array with an explicit target schema. Record
// values convert positionally; ordinary scalars are replicated across fields.
// It does not reinterpret raw bytes.
func (a *Array) CastRecords(schema StructuredSchema) (*Array, error) {
	if a.DType() == 0 || schema == nil {
		return nil, ErrDType
	}
	out, e := ZerosRecords(schema, a.shape...)
	if e != nil {
		return nil, e
	}
	a.walk(func(i, off int) {
		if e != nil {
			return
		}
		v, err := out.convertValue(a.rawAt(off))
		if err != nil {
			e = err
			return
		}
		out.setElement(i, v)
	})
	if e != nil {
		return nil, e
	}
	return out, nil
}
func (a *Array) astypeSpecial(d DType) (*Array, error) {
	if d.ItemSize() == 0 {
		return nil, ErrDType
	}
	if a.DType() == d {
		out, e := a.zerosLike(a.shape)
		if e != nil {
			return nil, e
		}
		a.walk(func(i, off int) { out.setElement(i, a.elementAt(off)) })
		return out, nil
	}
	if d == StringDType {
		return a.castStrings(false)
	}
	if d.IsFlexible() {
		f, e := flexibleDescriptorFor([]*Array{a}, d)
		if e != nil {
			return nil, e
		}
		return a.CastFlexible(f)
	}
	if d.IsTemporal() || d == RecordDType {
		return nil, ErrDType
	}
	if a.DType() == RecordDType && d != Object {
		return nil, ErrDType
	}
	if a.DType().IsTemporal() && d != Object && !d.IsInteger() && !d.IsFloat() && !d.IsComplex() && d != Bool {
		return nil, ErrDType
	}
	out, e := Zeros(d, a.shape...)
	if e != nil {
		return nil, e
	}
	a.walk(func(i, off int) {
		if e != nil {
			return
		}
		v := a.rawAt(off)
		if a.DType() == StringDType {
			x := a.elementAt(off)
			if d == Bool {
				out.setElement(i, boolElement(x.truth(StringDType)))
				return
			}
			if _, missing := x.extra.(stringNull); missing && d != Object {
				e = ErrValue
				return
			}
		}
		if a.DType().IsFlexible() && d == Bool {
			out.setElement(i, boolElement(a.elementAt(off).truth(a.DType())))
			return
		}
		if a.DType().IsTemporal() && d != Object {
			if d == Bool {
				out.setElement(i, boolElement(temporalTicks(a.elementAt(off)) != 0))
				return
			}
			v = temporalTicks(a.elementAt(off))
		}
		z, err := out.convertValue(v)
		if err != nil {
			e = err
			return
		}
		out.setElement(i, z)
	})
	if e != nil {
		return nil, e
	}
	return out, nil
}
func (a *Array) castLike(source *Array) (*Array, error) {
	if a.DType() == StringDType {
		return source.castStringDescriptor(a.stringDescriptor())
	}
	if a.DType().IsFlexible() {
		return source.CastFlexible(a.data.flex)
	}
	if a.DType().IsTemporal() {
		return source.CastTemporal(a.DType(), a.data.unit.String())
	}
	if a.DType() == RecordDType {
		if a.sameDescriptor(source) {
			return source.Copy()
		}
		return source.CastRecords(a.data.schema)
	}
	return source.Astype(a.DType())
}
func recordEqual(x, y element) bool {
	if a, ok := x.extra.(managedRecordValue); ok {
		b, ok := y.extra.(managedRecordValue)
		if !ok {
			return false
		}
		r := a.schema.DecodeManagedScalar(a.bytes, a.slots, a.offset, true)
		s := b.schema.DecodeManagedScalar(b.bytes, b.slots, b.offset, true)
		if c, ok := r.(interface{ EqualRecord(any) (bool, error) }); ok {
			v, e := c.EqualRecord(s)
			return e == nil && v
		}
		return false
	}
	a, _ := x.extra.([]byte)
	b, _ := y.extra.([]byte)
	return bytes.Equal(a, b)
}
