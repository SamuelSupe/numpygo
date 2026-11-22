// Package np implements NumPy-style CPU arrays for native Go applications.
package np

import (
	"encoding/binary"
	"errors"
	"fmt"
	"slices"
	"sync/atomic"

	numkernel "github.com/SamuelSupe/numpygo/internal/numeric"
)

// DType describes the representation of each element in an Array.
type DType uint8

const (
	Float64 DType = iota + 1
	Complex128
	Bool
	Int8
	Int16
	Int32
	Int64
	Uint8
	Uint16
	Uint32
	Uint64
	Float16
	Float32
	Complex64
	Float128
	Complex256
	DateTime64
	TimeDelta64
	Object
	RecordDType
	BytesDType
	UnicodeDType
	VoidDType
	StringDType
)

const LongDouble = Float128
const CLongDouble = Complex256

func (d DType) String() string {
	if int(d) < len(dtypeNames) {
		return dtypeNames[d]
	}
	return "invalid"
}

// ItemSize returns the number of bytes in one element, or zero for an invalid dtype.
func (d DType) ItemSize() int {
	switch d {
	case Bool, Int8, Uint8:
		return 1
	case Int16, Uint16, Float16:
		return 2
	case Int32, Uint32, Float32:
		return 4
	case Float64, Int64, Uint64, Complex64:
		return 8
	case Complex128:
		return 16
	case StringDType:
		// NumPy's 64-bit variable-length metadata slot, not its payload.
		return 16
	case Float128:
		return 16
	case Complex256:
		return 32
	case DateTime64, TimeDelta64:
		return 8
	case Object:
		return 8
	case RecordDType, BytesDType, UnicodeDType, VoidDType:
		// Descriptor-sized dtypes use Array.ItemSize for the actual width.
		return 1
	default:
		return 0
	}
}

var (
	ErrShape    = errors.New("invalid array shape")
	ErrAxis     = errors.New("invalid axis")
	ErrIndex    = errors.New("invalid index")
	ErrDType    = errors.New("incompatible dtype")
	ErrReadOnly = errors.New("array is read-only")
	ErrValue    = errors.New("invalid numerical value")
)

const maxInt = int(^uint(0) >> 1)
const maxDims = 64

type storage struct {
	dtype        DType
	real         []float64
	cplx         []complex128
	raw          any
	part         *storage // virtual real/imaginary view of complex storage
	buffer       []byte
	order        binary.ByteOrder
	mapping      *mappedRegion
	shared       atomic.Bool
	unit         DateUnit
	schema       StructuredSchema
	slots        *RecordSlots
	flex         FlexibleDType
	strictString bool
	stringNA     *stringNull
	byteIndexed  bool
}

// Array is a strided, mutable N-dimensional array. Views retain the storage of
// the original array. Concurrent writes to shared elements require caller synchronization.
// Construct arrays with the functions in this package; the zero value is invalid.
type Array struct {
	data     *storage
	shape    []int
	strides  []int // in elements; the public Strides method reports bytes
	offset   int
	size     int
	readOnly bool
	owner    bool
	base     *Array
}

// shapeSize limits both element counts and byte offsets to the native int range.
// Even an empty array must have representable dimensions and strides.
func shapeSize(dtype DType, shape []int) (int, error) {
	return shapeSizeWidth(dtype.ItemSize(), shape)
}
func shapeSizeWidth(itemSize int, shape []int) (int, error) {
	if itemSize == 0 {
		return 0, ErrDType
	}
	if len(shape) > maxDims {
		return 0, fmt.Errorf("%w: maximum rank is %d", ErrShape, maxDims)
	}
	product, empty := 1, false
	for _, dim := range shape {
		if dim < 0 {
			return 0, fmt.Errorf("%w: negative dimension %d", ErrShape, dim)
		}
		if dim == 0 {
			empty = true
			continue
		}
		if product > maxInt/itemSize/dim {
			return 0, fmt.Errorf("%w: array is too large", ErrShape)
		}
		product *= dim
	}
	if empty {
		return 0, nil
	}
	return product, nil
}

// ElementCount validates a shape and returns its size, without allocating data.
func ElementCount(dtype DType, shape ...int) (int, error) { return shapeSize(dtype, shape) }

func contiguousStrides(shape []int, size int) []int {
	strides := make([]int, len(shape))
	if size == 0 {
		return strides
	}
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func owned(data *storage, shape []int, size int) *Array {
	return &Array{data: data, shape: slices.Clone(shape), strides: contiguousStrides(shape, size), size: size, owner: true}
}

// Zeros creates a writable, C-contiguous array. An empty shape creates a scalar.
func Zeros(dtype DType, shape ...int) (*Array, error) {
	if dtype == RecordDType || dtype.IsTemporal() || dtype.IsFlexible() {
		return nil, ErrDType
	}
	size, err := shapeSize(dtype, shape)
	if err != nil {
		return nil, err
	}
	data := makeStorage(dtype, size)
	return owned(data, shape, size), nil
}

// WrapFloat64 creates an array sharing the supplied slice in C order. Its
// length must exactly equal the product of shape (one for a scalar).
func WrapFloat64(values []float64, shape ...int) (*Array, error) {
	size, err := shapeSize(Float64, shape)
	if err != nil {
		return nil, err
	}
	if size != len(values) {
		return nil, fmt.Errorf("%w: got %d values for size %d", ErrShape, len(values), size)
	}
	return owned(&storage{dtype: Float64, real: values}, shape, size), nil
}

// WrapComplex128 is the complex-valued equivalent of WrapFloat64.
func WrapComplex128(values []complex128, shape ...int) (*Array, error) {
	size, err := shapeSize(Complex128, shape)
	if err != nil {
		return nil, err
	}
	if size != len(values) {
		return nil, fmt.Errorf("%w: got %d values for size %d", ErrShape, len(values), size)
	}
	return owned(&storage{dtype: Complex128, cplx: values}, shape, size), nil
}

// FromFloat64 creates an independent copy of the supplied data in C order.
func FromFloat64(values []float64, shape ...int) (*Array, error) {
	a, err := WrapFloat64(values, shape...)
	if err != nil {
		return nil, err
	}
	a.data.real = slices.Clone(values)
	return a, nil
}

// FromComplex128 creates an independent complex array in C order.
func FromComplex128(values []complex128, shape ...int) (*Array, error) {
	a, err := WrapComplex128(values, shape...)
	if err != nil {
		return nil, err
	}
	a.data.cplx = slices.Clone(values)
	return a, nil
}

func Scalar(value float64) *Array {
	return owned(&storage{dtype: Float64, real: []float64{value}}, nil, 1)
}

func ComplexScalar(value complex128) *Array {
	return owned(&storage{dtype: Complex128, cplx: []complex128{value}}, nil, 1)
}

func (a *Array) DType() DType {
	if a == nil || a.data == nil {
		return 0
	}
	if r := a.mappingRegion(); r != nil && r.closed.Load() {
		return 0
	}
	return a.data.dtype
}

func (a *Array) Shape() []int { return slices.Clone(a.shape) }
func (a *Array) NDim() int    { return len(a.shape) }
func (a *Array) Size() int    { return a.size }

// ItemSize is the array's actual element width, including variable descriptors.
func (a *Array) ItemSize() int {
	if a.DType() == RecordDType && a.data.schema != nil {
		return a.data.schema.ItemSize()
	}
	if a.DType().IsFlexible() {
		return a.data.flex.ItemSize()
	}
	return a.DType().ItemSize()
}
func (a *Array) NBytes() int { return a.size * a.ItemSize() }
func (a *Array) shapeSize(shape []int) (int, error) {
	if a.DType().IsFlexible() || a.DType() == RecordDType {
		return shapeSizeWidth(max(a.ItemSize(), 1), shape)
	}
	return shapeSizeWidth(a.ItemSize(), shape)
}
func (a *Array) strideUnit() int {
	if a.data.byteIndexed {
		return 1
	}
	return a.ItemSize()
}
func (a *Array) contiguousStep() int {
	if a.strideUnit() == 0 {
		return 1
	}
	return a.ItemSize() / a.strideUnit()
}

// Base is the original allocation's array for an array-derived view. Owning
// arrays and externally supplied Go buffers have no array base.
func (a *Array) Base() *Array { return a.base }
func (a *Array) viewBase() *Array {
	root := a
	if a.base != nil {
		root = a.base
	}
	root.data.shared.Store(true)
	return root
}
func (a *Array) Writable() bool {
	return a.DType() != 0 && !a.readOnly
}

// SetWritable changes the write flag on this array. Only independently
// allocated headers may re-enable writes; derived read-only views remain protected.
func (a *Array) SetWritable(writable bool) error {
	if a.DType() == 0 {
		return ErrDType
	}
	if writable && a.readOnly && !a.owner {
		return ErrReadOnly
	}
	if writable && a.mappingRegion() != nil && a.mappingRegion().mode == "r" {
		return ErrReadOnly
	}
	a.readOnly = !writable
	return nil
}

// Strides returns the signed byte stride for each dimension.
func (a *Array) Strides() []int {
	strides := slices.Clone(a.strides)
	for i := range strides {
		strides[i] *= a.strideUnit()
	}
	return strides
}

func (a *Array) IsCContiguous() bool {
	if a.DType() == 0 {
		return false
	}
	if a.size == 0 {
		return true
	}
	stride := a.contiguousStep()
	for i := len(a.shape) - 1; i >= 0; i-- {
		if a.shape[i] > 1 && a.strides[i] != stride {
			return false
		}
		stride *= a.shape[i]
	}
	return true
}

func (a *Array) IsFContiguous() bool {
	if a.DType() == 0 {
		return false
	}
	if a.size == 0 {
		return true
	}
	stride := a.contiguousStep()
	for i, dim := range a.shape {
		if dim > 1 && a.strides[i] != stride {
			return false
		}
		stride *= dim
	}
	return true
}

func normalizeIndex(index, dim int) (int, error) {
	if index < 0 {
		index += dim
	}
	if index < 0 || index >= dim {
		return 0, ErrIndex
	}
	return index, nil
}

func (a *Array) elementOffset(indices []int) (int, error) {
	if r := a.mappingRegion(); r != nil && r.closed.Load() {
		return 0, ErrClosedMapping
	}
	if a.DType() == 0 {
		return 0, ErrDType
	}
	if len(indices) != len(a.shape) {
		return 0, fmt.Errorf("%w: expected %d indices, got %d", ErrIndex, len(a.shape), len(indices))
	}
	offset := a.offset
	for axis, index := range indices {
		i, err := normalizeIndex(index, a.shape[axis])
		if err != nil {
			return 0, fmt.Errorf("%w: index %d on axis %d of length %d", err, index, axis, a.shape[axis])
		}
		offset += i * a.strides[axis]
	}
	return offset, nil
}

func (a *Array) complexAt(offset int) complex128 {
	return a.elementAt(offset).number(a.DType())
}

// At reads a single element, promoting a real value to complex128 with zero imaginary part.
func (a *Array) At(indices ...int) (complex128, error) {
	if a.DType().IsTemporal() || a.DType() == Object || a.DType() == StringDType || a.DType() == RecordDType || a.DType().IsFlexible() {
		return 0, ErrDType
	}
	offset, err := a.elementOffset(indices)
	if err != nil {
		return 0, err
	}
	return a.complexAt(offset), nil
}

// Set writes a single element through the array's view. A complex value with
// a nonzero imaginary component cannot be assigned to a real array.
func (a *Array) Set(value complex128, indices ...int) error {
	if a.DType().IsSpecial() {
		return a.SetValue(value, indices...)
	}
	offset, err := a.elementOffset(indices)
	if err != nil {
		return err
	}
	if !a.Writable() {
		return ErrReadOnly
	}
	if !a.DType().IsComplex() && imag(value) != 0 {
		return ErrDType
	}
	a.setElement(offset, castElement(element{z: value}, Complex128, a.DType()))
	return nil
}

// walk iterates offsets in logical C order, without allocating an index per element.
func (a *Array) walk(visit func(flat, offset int)) {
	if a.size == 0 {
		return
	}
	if a.IsCContiguous() {
		for i := 0; i < a.size; i++ {
			visit(i, a.offset+i*a.contiguousStep())
		}
		return
	}
	index := make([]int, len(a.shape))
	offset := a.offset
	for i := 0; i < a.size; i++ {
		visit(i, offset)
		for axis := len(index) - 1; axis >= 0; axis-- {
			if index[axis]+1 < a.shape[axis] {
				index[axis]++
				offset += a.strides[axis]
				break
			}
			offset -= index[axis] * a.strides[axis]
			index[axis] = 0
		}
	}
}

// Float64s returns a detached copy in logical C order, requiring a Float64 dtype.
func (a *Array) Float64s() ([]float64, error) {
	return Values[float64](a)
}

// Complex128s returns a detached C-order copy, promoting real input if needed.
func (a *Array) Complex128s() ([]complex128, error) {
	if a.DType() == 0 || a.DType().IsTemporal() || a.DType() == Object || a.DType() == StringDType || a.DType() == RecordDType || a.DType().IsFlexible() {
		return nil, ErrDType
	}
	if a.DType() == Complex128 && a.data.part == nil {
		return Values[complex128](a)
	}
	values := make([]complex128, a.size)
	if a.data.part == nil && a.data.buffer == nil {
		switch a.DType() {
		case Float64:
			convertNative(values, a.data.real, a, func(v float64) complex128 { return complex(v, 0) })
			return values, nil
		case Float32:
			convertNative(values, a.data.raw.([]float32), a, func(v float32) complex128 { return complex(float64(v), 0) })
			return values, nil
		case Complex64:
			x := a.data.raw.([]complex64)
			if !convertNativeLanes(values, x, a, numkernel.WidenComplex) {
				convertNative(values, x, a, func(v complex64) complex128 { return complex128(v) })
			}
			return values, nil
		}
	}
	a.walk(func(i, offset int) { values[i] = a.complexAt(offset) })
	return values, nil
}

// Copy creates a writable, C-contiguous array independent of the original storage.
func (a *Array) Copy() (*Array, error) {
	if a.DType().IsSpecial() {
		return a.Astype(a.DType())
	}
	if a.DType() != 0 && a.data.buffer != nil {
		raw, e := a.ToBytes("C")
		if e != nil {
			return nil, e
		}
		out, e := FromBuffer(raw, a.DType(), -1, 0, a.data.order)
		if e != nil {
			return nil, e
		}
		return owned(out.data, a.shape, a.size), nil
	}
	return a.Astype(a.DType())
}

// NormalizeAxis resolves a negative axis against the rank of an array.
func NormalizeAxis(axis, ndim int) (int, error) {
	original := axis
	if axis < 0 {
		axis += ndim
	}
	if axis < 0 || axis >= ndim {
		return 0, fmt.Errorf("%w: %d for rank %d", ErrAxis, original, ndim)
	}
	return axis, nil
}

// Int is a convenience for specifying an optional integer argument, including zero.
func Int(value int) *int { return &value }
