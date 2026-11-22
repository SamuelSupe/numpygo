package np

import (
	"fmt"
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

var dtypeNames = [...]string{"invalid", "float64", "complex128", "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "complex64", "float128", "complex256", "datetime64", "timedelta64", "object", "record", "bytes", "str", "void", "StringDType()"}

func (d DType) IsSigned() bool        { return d >= Int8 && d <= Int64 }
func (d DType) IsUnsigned() bool      { return d >= Uint8 && d <= Uint64 }
func (d DType) IsInteger() bool       { return d.IsSigned() || d.IsUnsigned() }
func (d DType) IsFloat() bool         { return d == Float16 || d == Float32 || d == Float64 || d == Float128 }
func (d DType) IsComplex() bool       { return d == Complex64 || d == Complex128 || d == Complex256 }
func (d DType) IsExtended() bool      { return d == Float128 || d == Complex256 }
func (d DType) IsTemporal() bool      { return d == DateTime64 || d == TimeDelta64 }
func (d DType) IsSpecial() bool       { return d >= Float128 }
func (d DType) IsNativeNumeric() bool { return d >= Float64 && d <= Complex64 }
func (d DType) IsCharacter() bool     { return d == BytesDType || d == UnicodeDType }
func (d DType) IsFlexible() bool      { return d == BytesDType || d == UnicodeDType || d == VoidDType }

type ExtendedFloat = x87.Float
type ExtendedComplex = x87.Complex
type ExtendedFloatInfo = x87.FloatInfo

func ParseDType(name string) (DType, error) {
	switch name {
	case "T", "StringDType":
		return StringDType, nil
	case "longdouble":
		return Float128, nil
	case "clongdouble":
		return Complex256, nil
	case "object_":
		return Object, nil
	case "bytes_":
		return BytesDType, nil
	case "str_":
		return UnicodeDType, nil
	}
	for i, n := range dtypeNames {
		if i != 0 && n == name {
			return DType(i), nil
		}
	}
	return 0, fmt.Errorf("%w: %s", ErrDType, name)
}

// Half stores the IEEE-754 binary16 bit pattern, keeping NaNs and signed zero.
type Half uint16

func HalfFromFloat64(x float64) Half {
	sign := Half(0)
	if math.Signbit(x) {
		sign = 0x8000
	}
	x = math.Abs(x)
	if math.IsNaN(x) {
		return sign | 0x7e00
	}
	if math.IsInf(x, 0) {
		return sign | 0x7c00
	}
	if x < math.Ldexp(1, -14) {
		return sign | Half(math.RoundToEven(math.Ldexp(x, 24)))
	}
	e := math.Ilogb(x)
	if e > 15 {
		return sign | 0x7c00
	}
	fraction := int(math.RoundToEven(math.Ldexp(x, 10-e)))
	if fraction == 2048 {
		e++
		fraction = 1024
	}
	if e > 15 {
		return sign | 0x7c00
	}
	return sign | Half((e+15)<<10) | Half(fraction-1024)
}

func (h Half) Float64() float64 {
	e, frac := int(h>>10&31), int(h&1023)
	var x float64
	switch e {
	case 0:
		x = math.Ldexp(float64(frac), -24)
	case 31:
		if frac != 0 {
			x = math.NaN()
		} else {
			x = math.Inf(1)
		}
	default:
		x = math.Ldexp(float64(1024+frac), e-25)
	}
	if h&0x8000 != 0 {
		return math.Copysign(x, -1)
	}
	return x
}

// Number is the set of directly representable numerical Go element types.
type Number interface {
	bool | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | Half | float32 | float64 | complex64 | complex128 | x87.Float | x87.Complex
}

func dtypeOf(value any) DType {
	switch value.(type) {
	case bool:
		return Bool
	case int8:
		return Int8
	case int16:
		return Int16
	case int32:
		return Int32
	case int, int64:
		return Int64
	case uint8:
		return Uint8
	case uint16:
		return Uint16
	case uint32:
		return Uint32
	case uint, uint64:
		return Uint64
	case Half:
		return Float16
	case float32:
		return Float32
	case float64:
		return Float64
	case complex64:
		return Complex64
	case complex128:
		return Complex128
	case x87.Float:
		return Float128
	case x87.Complex:
		return Complex256
	case DateTime:
		return DateTime64
	case TimeDelta:
		return TimeDelta64
	case BytesScalar:
		return BytesDType
	case TextScalar:
		return UnicodeDType
	case VoidScalar:
		return VoidDType
	}
	return 0
}

func makeStorage(dtype DType, n int) *storage {
	s := &storage{dtype: dtype}
	switch dtype {
	case Float64:
		s.real = make([]float64, n)
	case Complex128:
		s.cplx = make([]complex128, n)
	case Bool:
		s.raw = make([]bool, n)
	case Int8:
		s.raw = make([]int8, n)
	case Int16:
		s.raw = make([]int16, n)
	case Int32:
		s.raw = make([]int32, n)
	case Int64:
		s.raw = make([]int64, n)
	case Uint8:
		s.raw = make([]uint8, n)
	case Uint16:
		s.raw = make([]uint16, n)
	case Uint32:
		s.raw = make([]uint32, n)
	case Uint64:
		s.raw = make([]uint64, n)
	case Float16:
		s.raw = make([]Half, n)
	case Float32:
		s.raw = make([]float32, n)
	case Complex64:
		s.raw = make([]complex64, n)
	case Float128:
		s.raw = make([]x87.Float, n)
	case Complex256:
		s.raw = make([]x87.Complex, n)
	case DateTime64, TimeDelta64:
		s.raw = make([]int64, n)
	case Object:
		v := make([]any, n)
		for i := range v {
			v[i] = int64(0)
		}
		s.raw = v
	case StringDType:
		v := make([]any, n)
		for i := range v {
			v[i] = ""
		}
		s.raw = v
	}
	return s
}

// WrapSlice shares the supplied typed slice as an array in C order.
func WrapSlice[T Number](values []T, shape ...int) (*Array, error) {
	var sample T
	dtype := dtypeOf(sample)
	size, err := shapeSize(dtype, shape)
	if err != nil {
		return nil, err
	}
	if len(values) != size {
		return nil, ErrShape
	}
	s := &storage{dtype: dtype}
	switch values := any(values).(type) {
	case []float64:
		s.real = values
	case []complex128:
		s.cplx = values
	default:
		s.raw = values
	}
	return owned(s, shape, size), nil
}

// FromSlice copies a typed Go slice into a numerical array.
func FromSlice[T Number](values []T, shape ...int) (*Array, error) {
	return WrapSlice(slices.Clone(values), shape...)
}

func ScalarOf[T Number](value T) *Array {
	a, _ := WrapSlice([]T{value})
	return a
}

func FromInt64(values []int64, shape ...int) (*Array, error)     { return FromSlice(values, shape...) }
func FromUint64(values []uint64, shape ...int) (*Array, error)   { return FromSlice(values, shape...) }
func FromBool(values []bool, shape ...int) (*Array, error)       { return FromSlice(values, shape...) }
func FromFloat32(values []float32, shape ...int) (*Array, error) { return FromSlice(values, shape...) }
func FromComplex64(values []complex64, shape ...int) (*Array, error) {
	return FromSlice(values, shape...)
}

// element keeps integer precision independent of the floating-point path.
type element struct {
	i     uint64
	z     complex128
	extra any
}

func (e element) number(dtype DType) complex128 {
	if dtype.IsExtended() {
		return extendedValue(e, dtype).Complex128()
	}
	if dtype.IsSigned() {
		return complex(float64(int64(e.i)), 0)
	}
	if dtype.IsUnsigned() || dtype == Bool {
		return complex(float64(e.i), 0)
	}
	return e.z
}

func (e element) truth(dtype DType) bool {
	if dtype.IsExtended() {
		v := extendedValue(e, dtype)
		return !v.Real.IsZero() || !v.Imag.IsZero()
	}
	if dtype.IsTemporal() {
		return temporalTicks(e) != 0
	}
	if dtype == Object {
		return objectTruth(e.extra)
	}
	if dtype == StringDType {
		if _, missing := e.extra.(stringNull); missing {
			return stringNaN(e)
		}
		return flexKey(e) != ""
	}
	if dtype == RecordDType {
		return recordTruth(e.extra)
	}
	if dtype.IsFlexible() {
		return flexTruth(e, dtype)
	}
	if dtype.IsInteger() || dtype == Bool {
		return e.i != 0
	}
	return e.z != 0
}

func elementFrom(value any) element {
	switch x := value.(type) {
	case bool:
		if x {
			return element{i: 1}
		}
	case int8:
		return element{i: uint64(int64(x))}
	case int16:
		return element{i: uint64(int64(x))}
	case int32:
		return element{i: uint64(int64(x))}
	case int64:
		return element{i: uint64(x)}
	case int:
		return element{i: uint64(x)}
	case uint8:
		return element{i: uint64(x)}
	case uint16:
		return element{i: uint64(x)}
	case uint32:
		return element{i: uint64(x)}
	case uint64:
		return element{i: x}
	case uint:
		return element{i: uint64(x)}
	case Half:
		return element{z: complex(x.Float64(), 0)}
	case float32:
		return element{z: complex(float64(x), 0)}
	case float64:
		return element{z: complex(x, 0)}
	case complex64:
		return element{z: complex128(x)}
	case complex128:
		return element{z: x}
	case x87.Float, x87.Complex, DateTime, TimeDelta, BytesScalar, TextScalar, VoidScalar:
		return element{extra: x}
	}
	return element{}
}

func (a *Array) rawAt(offset int) any {
	if r := a.mappingRegion(); r != nil && r.closed.Load() {
		panic(ErrClosedMapping)
	}
	if a.DType() == Object && a.data.slots != nil {
		v, _ := a.data.slots.Get(offset * a.strideUnit())
		return v
	}
	if a.DType() == RecordDType && schemaHasObjects(a.data.schema) {
		v := a.managedRecord(offset)
		return v.schema.DecodeManagedScalar(v.bytes, v.slots, v.offset, a.readOnly)
	}
	if a.data.buffer != nil {
		return a.bufferAt(offset)
	}
	if a.DType().IsTemporal() {
		return temporalValue(a.DType(), a.data.raw.([]int64)[offset], a.data.unit)
	}
	if p := a.data.part; p != nil {
		if p.dtype == Complex256 {
			z := p.raw.([]x87.Complex)[offset/2]
			if offset%2 == 0 {
				return z.Real
			}
			return z.Imag
		}
		if p.dtype == Complex128 {
			z := p.cplx[offset/2]
			if offset%2 == 0 {
				return real(z)
			}
			return imag(z)
		}
		z := p.raw.([]complex64)[offset/2]
		if offset%2 == 0 {
			return real(z)
		}
		return imag(z)
	}
	switch a.DType() {
	case StringDType:
		v := a.data.raw.([]any)[offset]
		if n, ok := v.(stringNull); ok {
			return n.value
		}
		return v
	case Float64:
		return a.data.real[offset]
	case Complex128:
		return a.data.cplx[offset]
	}
	switch v := a.data.raw.(type) {
	case []bool:
		return v[offset]
	case []int8:
		return v[offset]
	case []int16:
		return v[offset]
	case []int32:
		return v[offset]
	case []int64:
		return v[offset]
	case []uint8:
		return v[offset]
	case []uint16:
		return v[offset]
	case []uint32:
		return v[offset]
	case []uint64:
		return v[offset]
	case []Half:
		return v[offset]
	case []float32:
		return v[offset]
	case []complex64:
		return v[offset]
	case []x87.Float:
		return v[offset]
	case []x87.Complex:
		return v[offset]
	case []any:
		return v[offset]
	case []string:
		return v[offset]
	}
	return nil
}

func (a *Array) elementAt(offset int) element {
	if r := a.mappingRegion(); r != nil && r.closed.Load() {
		panic(ErrClosedMapping)
	}
	d := a.data.dtype
	if d == StringDType {
		return element{extra: a.data.raw.([]any)[offset]}
	}
	if d == Object {
		return element{extra: a.rawAt(offset)}
	}
	if d == RecordDType {
		if schemaHasObjects(a.data.schema) {
			return element{extra: a.managedRecord(offset)}
		}
		return element{extra: a.recordBytes(offset)}
	}
	if d.IsFlexible() {
		return element{extra: flexElement{a.flexBytes(offset), a.data.flex}}
	}
	if a.data.buffer != nil && d.IsNativeNumeric() {
		return a.bufferNumericElementAt(offset, d)
	}
	if a.data.part != nil || a.data.buffer != nil {
		return elementFrom(a.rawAt(offset))
	}
	// Numerical kernels do not need to materialize Go interface values.
	// AtValue/rawAt keep their exact typed-scalar contract separately.
	switch d {
	case Float64:
		return element{z: complex(a.data.real[offset], 0)}
	case Complex128:
		return element{z: a.data.cplx[offset]}
	case Bool:
		return boolElement(a.data.raw.([]bool)[offset])
	case Int8:
		return element{i: uint64(int64(a.data.raw.([]int8)[offset]))}
	case Int16:
		return element{i: uint64(int64(a.data.raw.([]int16)[offset]))}
	case Int32:
		return element{i: uint64(int64(a.data.raw.([]int32)[offset]))}
	case Int64:
		return element{i: uint64(a.data.raw.([]int64)[offset])}
	case Uint8:
		return element{i: uint64(a.data.raw.([]uint8)[offset])}
	case Uint16:
		return element{i: uint64(a.data.raw.([]uint16)[offset])}
	case Uint32:
		return element{i: uint64(a.data.raw.([]uint32)[offset])}
	case Uint64:
		return element{i: a.data.raw.([]uint64)[offset]}
	case Float16:
		return element{z: complex(a.data.raw.([]Half)[offset].Float64(), 0)}
	case Float32:
		return element{z: complex(float64(a.data.raw.([]float32)[offset]), 0)}
	case Complex64:
		return element{z: complex128(a.data.raw.([]complex64)[offset])}
	default:
		return elementFrom(a.rawAt(offset))
	}
}

func (a *Array) setElement(offset int, e element) {
	if r := a.mappingRegion(); r != nil && r.closed.Load() {
		panic(ErrClosedMapping)
	}
	d := a.data.dtype
	if d == Object && a.data.slots != nil {
		_ = a.data.slots.Set(offset*a.strideUnit(), e.extra)
		return
	}
	if d == RecordDType && schemaHasObjects(a.data.schema) {
		a.storeManagedRecord(offset, e)
		return
	}
	if a.data.buffer != nil {
		a.bufferSet(offset, e)
		return
	}
	if p := a.data.part; p != nil {
		if p.dtype == Complex256 {
			v := p.raw.([]x87.Complex)
			z := v[offset/2]
			r := extendedValue(e, Float128).Real
			if offset%2 == 0 {
				z.Real = r
			} else {
				z.Imag = r
			}
			v[offset/2] = z
			return
		}
		if p.dtype == Complex128 {
			z := p.cplx[offset/2]
			if offset%2 == 0 {
				p.cplx[offset/2] = complex(real(e.z), imag(z))
			} else {
				p.cplx[offset/2] = complex(real(z), real(e.z))
			}
		} else {
			v := p.raw.([]complex64)
			z := v[offset/2]
			r := float32(real(e.z))
			if offset%2 == 0 {
				v[offset/2] = complex(r, imag(z))
			} else {
				v[offset/2] = complex(real(z), r)
			}
		}
		return
	}
	switch d {
	case Float64:
		a.data.real[offset] = real(e.z)
	case Complex128:
		a.data.cplx[offset] = e.z
	case Bool:
		a.data.raw.([]bool)[offset] = e.i != 0
	case Int8:
		a.data.raw.([]int8)[offset] = int8(e.i)
	case Int16:
		a.data.raw.([]int16)[offset] = int16(e.i)
	case Int32:
		a.data.raw.([]int32)[offset] = int32(e.i)
	case Int64:
		a.data.raw.([]int64)[offset] = int64(e.i)
	case Uint8:
		a.data.raw.([]uint8)[offset] = uint8(e.i)
	case Uint16:
		a.data.raw.([]uint16)[offset] = uint16(e.i)
	case Uint32:
		a.data.raw.([]uint32)[offset] = uint32(e.i)
	case Uint64:
		a.data.raw.([]uint64)[offset] = e.i
	case Float16:
		a.data.raw.([]Half)[offset] = HalfFromFloat64(real(e.z))
	case Float32:
		a.data.raw.([]float32)[offset] = float32(real(e.z))
	case Complex64:
		a.data.raw.([]complex64)[offset] = complex64(e.z)
	case Float128:
		a.data.raw.([]x87.Float)[offset] = extendedValue(e, Float128).Real
	case Complex256:
		a.data.raw.([]x87.Complex)[offset] = extendedValue(e, Complex256)
	case DateTime64, TimeDelta64:
		a.data.raw.([]int64)[offset] = temporalTicks(e)
	case Object:
		a.data.raw.([]any)[offset] = e.extra
	case StringDType:
		a.data.raw.([]any)[offset] = e.extra
	}
}

func castElement(e element, from, to DType) element {
	if from == to {
		return e
	}
	if to == Object {
		if from == StringDType {
			if n, ok := e.extra.(stringNull); ok {
				return element{extra: n.value}
			}
		}
		return element{extra: scalarFromElement(e, from)}
	}
	if to == Bool && from.IsFlexible() {
		return boolElement(e.truth(from))
	}
	if from.IsExtended() || to.IsExtended() {
		return castExtended(e, from, to)
	}
	if to == Bool {
		if e.truth(from) {
			return element{i: 1}
		}
		return element{}
	}
	if to.IsInteger() {
		if from.IsInteger() || from == Bool {
			return roundElement(element{i: e.i}, to)
		}
		x := math.Trunc(real(e.z))
		// NumPy's fixed-width Linux casts use the signed conversion
		// indefinite value on invalid conversions. Narrow signed and
		// unsigned casts truncate that intermediate int32.
		switch to {
		case Int8, Int16, Int32, Uint8, Uint16:
			if math.IsNaN(x) || x < -0x1p31 || x >= 0x1p31 {
				return roundElement(element{i: 1 << 31}, to)
			}
			return roundElement(element{i: uint64(int64(x))}, to)
		case Int64:
			if math.IsNaN(x) || x < -0x1p63 || x >= 0x1p63 {
				return element{i: 1 << 63}
			}
			return element{i: uint64(int64(x))}
		case Uint64:
			if x >= 0x1p64 {
				return element{}
			}
			if math.IsNaN(x) || x < -0x1p63 {
				return element{i: 1 << 63}
			}
			if x < 0 {
				return element{i: uint64(int64(x))}
			}
			return element{i: uint64(x)}
		case Uint32:
			// Keep scalar conversion semantics: NumPy SIMD uint32
			// invalid-cast sentinels may vary with array layout.
			if math.IsNaN(x) || x < -0x1p63 || x >= 0x1p63 {
				return element{}
			}
			return roundElement(element{i: uint64(int64(x))}, to)
		}
		return element{}
	}
	z := e.number(from)
	// Convert exact integers directly to float32 rather than rounding twice
	// through float64 at a halfway value.
	if to == Float32 || to == Complex64 {
		if from.IsSigned() {
			z = complex(float64(float32(int64(e.i))), 0)
		} else if from.IsUnsigned() {
			z = complex(float64(float32(e.i)), 0)
		}
	}
	if !to.IsComplex() {
		z = complex(real(z), 0)
	}
	return roundElement(element{z: z}, to)
}

// roundElement is the value after storing and reloading it in dtype d.
// Kernels with a low-precision accumulator must use it between operations.
func roundElement(e element, d DType) element {
	switch {
	case d == Bool:
		return boolElement(e.i != 0)
	case d.IsSigned():
		shift := 64 - 8*d.ItemSize()
		return element{i: uint64(int64(e.i<<shift) >> shift)}
	case d.IsUnsigned():
		shift := 64 - 8*d.ItemSize()
		return element{i: e.i << shift >> shift}
	case d == Float16:
		return element{z: complex(HalfFromFloat64(real(e.z)).Float64(), 0)}
	case d == Float32:
		return element{z: complex(float64(float32(real(e.z))), 0)}
	case d == Complex64:
		return element{z: complex128(complex64(e.z))}
	default:
		return e
	}
}

// AtValue gives the exact typed Go scalar, including integers larger than 2^53.
func (a *Array) AtValue(indices ...int) (any, error) {
	off, err := a.elementOffset(indices)
	if err != nil {
		return nil, err
	}
	v := a.rawAt(off)
	if a.DType() == RecordDType {
		if r, ok := v.(interface{ BindBase(*Array) }); ok {
			a.viewBase()
			r.BindBase(a)
		}
	}
	return v, nil
}

// Values returns a detached typed C-order copy. The requested Go type must
// match the dtype. Use Astype for numerical conversions.
func Values[T Number](a *Array) ([]T, error) {
	var sample T
	if a.DType() != dtypeOf(sample) {
		return nil, ErrDType
	}
	v := make([]T, a.size)
	if a.data.part == nil && a.data.buffer == nil {
		if source, ok := storageSlice(a.data).([]T); ok {
			if a.IsCContiguous() {
				copy(v, source[a.offset:a.offset+a.size])
			} else if a.NDim() == 2 {
				copyMatrix(v, source, a.shape[0], a.shape[1], 0, a.offset,
					[2]int{a.shape[1], 1}, [2]int{a.strides[0], a.strides[1]})
			} else {
				a.walk(func(i, off int) { v[i] = source[off] })
			}
			return v, nil
		}
	}
	a.walk(func(i, off int) { v[i] = a.rawAt(off).(T) })
	return v, nil
}

// SetValue accepts an explicitly typed Go scalar and casts it to the array's dtype.
func (a *Array) SetValue(value any, indices ...int) error {
	off, err := a.elementOffset(indices)
	if err != nil {
		return err
	}
	if !a.Writable() {
		return ErrReadOnly
	}
	if a.DType().IsSpecial() {
		e, err := a.convertValue(value)
		if err != nil {
			return err
		}
		a.setElement(off, e)
		return nil
	}
	dtype := dtypeOf(value)
	if dtype.IsFlexible() {
		e, err := a.convertValue(value)
		if err != nil {
			return err
		}
		a.setElement(off, e)
		return nil
	}
	if dtype == 0 {
		return ErrDType
	}
	e := elementFrom(value)
	if dtype.IsTemporal() {
		return ErrDType
	}
	if dtype.IsComplex() && !a.DType().IsComplex() && a.DType() != Bool && imag(e.number(dtype)) != 0 {
		return ErrDType
	}
	a.setElement(off, castElement(e, dtype, a.DType()))
	return nil
}

// Astype creates an independent array of the requested dtype, casting each value.
func (a *Array) Astype(dtype DType) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.DType().IsSpecial() || dtype.IsSpecial() {
		return a.astypeSpecial(dtype)
	}
	r, err := Zeros(dtype, a.shape...)
	if err != nil {
		return nil, err
	}
	if dtype == a.DType() && a.data.part == nil && a.data.buffer == nil {
		copyNative(r, a)
		return r, nil
	}
	if nativeFloatingCast(r, a) {
		return r, nil
	}
	a.walk(func(i, off int) { r.setElement(i, castElement(a.elementAt(off), a.DType(), dtype)) })
	return r, nil
}

// PromoteTypes matches NumPy's strong-array dtype promotion for built-in
// fixed-width numerical types. Python weak scalar rules are separate.
func PromoteTypes(a, b DType) (DType, error) {
	if a.ItemSize() == 0 || b.ItemSize() == 0 {
		return 0, ErrDType
	}
	if a == b {
		return a, nil
	}
	if a == Object || b == Object {
		return Object, nil
	}
	if a == StringDType || b == StringDType {
		if a == UnicodeDType || b == UnicodeDType {
			return StringDType, nil
		}
		return 0, ErrDType
	}
	if a.IsFlexible() || b.IsFlexible() {
		if a == VoidDType || b == VoidDType {
			return 0, ErrDType
		}
		for _, d := range []DType{a, b} {
			if !d.IsCharacter() {
				if _, e := FlexibleStringWidth(d); e != nil {
					return 0, e
				}
			}
		}
		if a == UnicodeDType || b == UnicodeDType {
			return UnicodeDType, nil
		}
		return BytesDType, nil
	}
	if a.IsTemporal() || b.IsTemporal() {
		if a.IsTemporal() && b.IsTemporal() {
			return DateTime64, nil
		}
		if b.IsTemporal() {
			a, b = b, a
		}
		if a == TimeDelta64 && (b == Bool || b.IsSigned() || b.IsUnsigned() && b.ItemSize() < 8) {
			return TimeDelta64, nil
		}
		return 0, ErrDType
	}
	if a == RecordDType || b == RecordDType {
		return 0, ErrDType
	}
	if a == Bool {
		return b, nil
	}
	if b == Bool {
		return a, nil
	}
	if a.IsInteger() && b.IsInteger() {
		as, bs := a.ItemSize(), b.ItemSize()
		if a.IsSigned() == b.IsSigned() {
			if as >= bs {
				return a, nil
			}
			return b, nil
		}
		s, u := a, b
		if !s.IsSigned() {
			s, u = b, a
		}
		for _, d := range []DType{Int8, Int16, Int32, Int64} {
			if d.ItemSize() >= s.ItemSize() && d.ItemSize() > u.ItemSize() {
				return d, nil
			}
		}
		return Float64, nil
	}
	precision := func(d DType) int {
		if d.IsComplex() {
			return d.ItemSize() / 2
		}
		if d.IsFloat() {
			return d.ItemSize()
		}
		if d.ItemSize() == 1 {
			return 2
		}
		if d.ItemSize() == 2 {
			return 4
		}
		return 8
	}
	p := max(precision(a), precision(b))
	if a.IsComplex() || b.IsComplex() {
		if p > 8 {
			return Complex256, nil
		}
		if p <= 4 {
			return Complex64, nil
		}
		return Complex128, nil
	}
	if p <= 2 {
		return Float16, nil
	}
	if p <= 4 {
		return Float32, nil
	}
	if p > 8 {
		return Float128, nil
	}
	return Float64, nil
}
