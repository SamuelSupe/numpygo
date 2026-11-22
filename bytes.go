package np

import (
	"encoding/binary"
	"github.com/SamuelSupe/numpygo/internal/x87"
	"io"
	"math"
	"reflect"
	"slices"
	"unsafe"
)

// FromBuffer creates a writable array sharing a byte slice, with explicit
// endian interpretation. count=-1 consumes all remaining complete elements.
// The caller must not concurrently mutate the buffer.
func FromBuffer(buffer []byte, dtype DType, count, offset int, order binary.ByteOrder) (*Array, error) {
	if dtype == Object || dtype == StringDType || dtype == RecordDType || dtype.IsTemporal() || dtype.IsFlexible() {
		return nil, ErrDType
	}
	width := dtype.ItemSize()
	if width == 0 {
		return nil, ErrDType
	}
	if order == nil {
		order = binary.NativeEndian
	}
	if dtype.IsExtended() && order.Uint16([]byte{1, 0}) != 1 {
		return nil, ErrDType
	}
	if offset < 0 || offset > len(buffer) || count < -1 {
		return nil, ErrValue
	}
	b := buffer[offset:]
	if b == nil {
		b = make([]byte, 0)
	}
	if count == -1 {
		if len(b)%width != 0 {
			return nil, ErrValue
		}
		count = len(b) / width
	}
	if count > len(b)/width {
		return nil, ErrValue
	}
	if _, err := shapeSize(dtype, []int{count}); err != nil {
		return nil, err
	}
	return owned(&storage{dtype: dtype, buffer: b[:count*width], order: order}, []int{count}, count), nil
}

func (a *Array) bufferAt(offset int) any {
	d := a.DType()
	b := a.data.buffer[offset*a.strideUnit():]
	o := a.data.order
	switch d {
	case Bool:
		return b[0] != 0
	case Int8:
		return int8(b[0])
	case Uint8:
		return uint8(b[0])
	case Int16:
		return int16(o.Uint16(b))
	case Uint16:
		return o.Uint16(b)
	case Int32:
		return int32(o.Uint32(b))
	case Uint32:
		return o.Uint32(b)
	case Int64:
		return int64(o.Uint64(b))
	case Uint64:
		return o.Uint64(b)
	case Float16:
		return Half(o.Uint16(b))
	case Float32:
		return math.Float32frombits(o.Uint32(b))
	case Float64:
		return math.Float64frombits(o.Uint64(b))
	case Complex64:
		return complex(math.Float32frombits(o.Uint32(b)), math.Float32frombits(o.Uint32(b[4:])))
	case Complex128:
		return complex(math.Float64frombits(o.Uint64(b)), math.Float64frombits(o.Uint64(b[8:])))
	case Float128:
		v, e := x87.FromRaw(b[:16])
		if e != nil {
			return x87.NaN()
		}
		return v
	case Complex256:
		r, e := x87.FromRaw(b[:16])
		if e != nil {
			r = x87.NaN()
		}
		i, e := x87.FromRaw(b[16:32])
		if e != nil {
			i = x87.NaN()
		}
		return x87.Complex{Real: r, Imag: i}
	case DateTime64, TimeDelta64:
		return temporalValue(d, int64(o.Uint64(b)), a.data.unit)
	case RecordDType:
		return a.data.schema.DecodeScalar(b[:a.ItemSize()], a.readOnly)
	case BytesDType, UnicodeDType, VoidDType:
		return decodeFlexible(b[:a.ItemSize()], a.data.flex)
	}
	return nil
}

// Decode native numerical fields directly into the kernel representation
// without boxing a Go scalar. Byte order and byte-indexed packed layouts
// remain identical to bufferAt.
func (a *Array) bufferNumericElementAt(offset int, d DType) element {
	if !a.data.byteIndexed {
		offset *= d.ItemSize()
	}
	b := a.data.buffer[offset:]
	o := a.data.order
	switch d {
	case Bool:
		return boolElement(b[0] != 0)
	case Int8:
		return element{i: uint64(int64(int8(b[0])))}
	case Uint8:
		return element{i: uint64(b[0])}
	case Int16:
		return element{i: uint64(int64(int16(o.Uint16(b))))}
	case Uint16:
		return element{i: uint64(o.Uint16(b))}
	case Int32:
		return element{i: uint64(int64(int32(o.Uint32(b))))}
	case Uint32:
		return element{i: uint64(o.Uint32(b))}
	case Int64, Uint64:
		return element{i: o.Uint64(b)}
	case Float16:
		return element{z: complex(Half(o.Uint16(b)).Float64(), 0)}
	case Float32:
		return element{z: complex(float64(math.Float32frombits(o.Uint32(b))), 0)}
	case Float64:
		return element{z: complex(math.Float64frombits(o.Uint64(b)), 0)}
	case Complex64:
		return element{z: complex(float64(math.Float32frombits(o.Uint32(b))), float64(math.Float32frombits(o.Uint32(b[4:]))))}
	case Complex128:
		return element{z: complex(math.Float64frombits(o.Uint64(b)), math.Float64frombits(o.Uint64(b[8:])))}
	}
	return element{}
}
func (a *Array) bufferSet(offset int, e element) {
	d := a.DType()
	b := a.data.buffer[offset*a.strideUnit():]
	o := a.data.order
	switch d {
	case Bool:
		if e.i == 0 {
			b[0] = 0
		} else {
			b[0] = 1
		}
	case Int8, Uint8:
		b[0] = byte(e.i)
	case Int16, Uint16:
		o.PutUint16(b, uint16(e.i))
	case Int32, Uint32:
		o.PutUint32(b, uint32(e.i))
	case Int64, Uint64:
		o.PutUint64(b, e.i)
	case Float16:
		o.PutUint16(b, uint16(HalfFromFloat64(real(e.z))))
	case Float32:
		o.PutUint32(b, math.Float32bits(float32(real(e.z))))
	case Float64:
		o.PutUint64(b, math.Float64bits(real(e.z)))
	case Complex64:
		o.PutUint32(b, math.Float32bits(float32(real(e.z))))
		o.PutUint32(b[4:], math.Float32bits(float32(imag(e.z))))
	case Complex128:
		o.PutUint64(b, math.Float64bits(real(e.z)))
		o.PutUint64(b[8:], math.Float64bits(imag(e.z)))
	case Float128:
		v := extendedValue(e, d).Real.Bytes16()
		copy(b, v[:])
	case Complex256:
		v := extendedValue(e, d)
		r, i := v.Real.Bytes16(), v.Imag.Bytes16()
		copy(b, r[:])
		copy(b[16:], i[:])
	case DateTime64, TimeDelta64:
		o.PutUint64(b, uint64(temporalTicks(e)))
	case RecordDType:
		v, _ := e.extra.([]byte)
		copy(b[:a.ItemSize()], v)
	case BytesDType, UnicodeDType, VoidDType:
		if v, ok := e.extra.(flexElement); ok {
			copy(b[:a.ItemSize()], v.raw)
		} else {
			clear(b[:a.ItemSize()])
		}
	}
}

// Native numeric Go slices contain no pointers. Their complete byte span can
// be viewed as bytes without a copy; the byte slice retains the allocation.
func rawStorageBytes(s *storage) []byte {
	if s.part != nil {
		s = s.part
	}
	if s.buffer != nil {
		return s.buffer
	}
	v := reflect.ValueOf(storageSlice(s))
	return unsafe.Slice((*byte)(v.UnsafePointer()), v.Len()*s.dtype.ItemSize())
}
func storageOrder(s *storage) binary.ByteOrder {
	if s.part != nil {
		s = s.part
	}
	if s.buffer != nil {
		return s.order
	}
	return binary.NativeEndian
}

// NativeByteOrder reports whether numerical storage uses the host byte order.
// One-byte dtypes have no endian distinction.
func (a *Array) NativeByteOrder() bool {
	d := a.DType()
	if d.ItemSize() == 0 {
		return false
	}
	if d == Object || d == StringDType || d == RecordDType || d == BytesDType || d == VoidDType {
		return true
	}
	if d.ItemSize() == 1 && d != UnicodeDType {
		return true
	}
	b := [2]byte{1, 0}
	return storageOrder(a.data).Uint16(b[:]) == binary.NativeEndian.Uint16(b[:])
}

// ToBytes serializes the logical array in C, F or A order, preserving actual
// byte representations (including endian order, signed zero and NaN bits).
func (a *Array) ToBytes(order string) ([]byte, error) {
	if r := a.mappingRegion(); r != nil && r.closed.Load() {
		return nil, ErrClosedMapping
	}
	if a.DType() == 0 || a.rawReferenceStorage() {
		return nil, ErrDType
	}
	if order == "" {
		order = "C"
	}
	if order == "A" {
		order = "C"
		if a.IsFContiguous() {
			order = "F"
		}
	}
	if order == "F" {
		var e error
		a, e = a.Transpose()
		if e != nil {
			return nil, e
		}
	} else if order != "C" {
		return nil, ErrValue
	}
	width := a.ItemSize()
	out := make([]byte, a.NBytes())
	if a.DType().IsExtended() && a.data.buffer == nil {
		a.walk(func(i, off int) {
			v := extendedValue(a.elementAt(off), a.DType())
			r := v.Real.Bytes16()
			copy(out[i*width:], r[:])
			if a.DType() == Complex256 {
				im := v.Imag.Bytes16()
				copy(out[i*width+16:], im[:])
			}
		})
		return out, nil
	}
	raw := rawStorageBytes(a.data)
	a.walk(func(i, off int) { start := off * a.strideUnit(); copy(out[i*width:], raw[start:start+width]) })
	return out, nil
}

// DataBytes exposes a mutable byte span for C-contiguous writable arrays.
// It retains the same backing memory, unlike ToBytes.
func (a *Array) DataBytes() ([]byte, error) {
	if a.DType() == 0 || a.rawReferenceStorage() {
		return nil, ErrDType
	}
	if !a.Writable() {
		return nil, ErrReadOnly
	}
	// An escaped byte slice could outlive an explicitly closed mapping.
	if a.mappingRegion() != nil {
		return nil, ErrValue
	}
	if !a.IsCContiguous() {
		return nil, ErrShape
	}
	start := a.offset * a.strideUnit()
	b := rawStorageBytes(a.data)
	a.data.shared.Store(true)
	return b[start : start+a.NBytes()], nil
}

// ByteSwap reverses the bytes of each scalar component without changing its
// dtype interpretation. Complex arrays swap the two components independently.
func (a *Array) ByteSwap(inplace bool) (*Array, error) {
	if a.rawReferenceStorage() {
		return nil, ErrDType
	}
	if a.DType().IsFlexible() {
		out := a
		if !inplace {
			var e error
			out, e = a.Copy()
			if e != nil {
				return nil, e
			}
		} else if !a.Writable() {
			return nil, ErrReadOnly
		}
		if out.DType() == UnicodeDType {
			out.walk(func(_, off int) {
				v := out.flexBytes(off)
				for i := 0; i < len(v); i += 4 {
					slices.Reverse(v[i : i+4])
				}
			})
		}
		return out, nil
	}
	if a.DType() == 0 || a.DType().IsSpecial() {
		return nil, ErrDType
	}
	out := a
	if !inplace {
		b, e := a.ToBytes("C")
		if e != nil {
			return nil, e
		}
		out, e = FromBuffer(b, a.DType(), a.size, 0, storageOrder(a.data))
		if e != nil {
			return nil, e
		}
		out, e = out.Reshape(a.shape...)
		if e != nil {
			return nil, e
		}
	} else if !a.Writable() {
		return nil, ErrReadOnly
	}
	raw := rawStorageBytes(out.data)
	width := out.DType().ItemSize()
	unit := width
	if out.DType().IsComplex() {
		unit /= 2
	}
	if unit <= 1 {
		return out, nil
	}
	out.walk(func(_, off int) {
		beg := off * out.strideUnit()
		for start := beg; start < beg+width; start += unit {
			slices.Reverse(raw[start : start+unit])
		}
	})
	return out, nil
}

// View reinterprets the same storage using a native-order numeric dtype.
// Changing itemsize requires a contiguous last dimension.
func (a *Array) View(dtype DType) (*Array, error) {
	if a.DType() == 0 || dtype.ItemSize() == 0 {
		return nil, ErrDType
	}
	if a.rawReferenceStorage() || dtype == StringDType || dtype == Object || dtype == RecordDType || dtype.IsTemporal() || dtype.IsFlexible() {
		return nil, ErrDType
	}
	old, newWidth := a.ItemSize(), dtype.ItemSize()
	shape := slices.Clone(a.shape)
	strides := a.Strides()
	offset := a.offset * a.strideUnit()
	if old != newWidth {
		if a.NDim() == 0 {
			return nil, ErrShape
		}
		ax := len(shape) - 1
		if shape[ax] > 1 && strides[ax] != old || shape[ax]*old%newWidth != 0 {
			return nil, ErrShape
		}
		shape[ax] = shape[ax] * old / newWidth
		strides[ax] = newWidth
	}
	// Slice the allocation at a byte remainder to retain addressability for
	// a dtype of greater alignment than the source.
	remainder := offset % newWidth
	raw := rawStorageBytes(a.data)
	if remainder > len(raw) {
		return nil, ErrShape
	}
	raw = raw[remainder:]
	for i, s := range strides {
		if s%newWidth != 0 && shape[i] > 1 {
			return nil, ErrShape
		}
		strides[i] = s / newWidth
	}
	n, e := shapeSize(dtype, shape)
	if e != nil {
		return nil, e
	}
	return &Array{data: &storage{dtype: dtype, buffer: raw, order: binary.NativeEndian}, shape: shape, strides: strides, offset: offset / newWidth, size: n, readOnly: a.readOnly, base: a.viewBase()}, nil
}

// GetField returns a view of one fixed-width field inside each numeric item.
func (a *Array) GetField(dtype DType, byteOffset int) (*Array, error) {
	old, width := a.ItemSize(), dtype.ItemSize()
	if old == 0 || width == 0 {
		return nil, ErrDType
	}
	if a.rawReferenceStorage() || dtype == StringDType || dtype == Object || dtype == RecordDType || dtype.IsTemporal() || dtype.IsFlexible() {
		return nil, ErrDType
	}
	if byteOffset < 0 || byteOffset > old-width {
		return nil, ErrValue
	}
	offset := a.offset*a.strideUnit() + byteOffset
	raw := rawStorageBytes(a.data)
	strides := a.Strides()
	if a.size == 0 {
		offset = 0
	}
	return &Array{data: &storage{dtype: dtype, buffer: raw, order: binary.NativeEndian, byteIndexed: true}, shape: slices.Clone(a.shape), strides: strides, offset: offset, size: a.size, readOnly: a.readOnly, base: a.viewBase()}, nil
}
func (a *Array) SetField(values *Array, dtype DType, offset int) error {
	field, e := a.GetField(dtype, offset)
	if e != nil {
		return e
	}
	return field.Assign(nil, values)
}

// ToFile writes a raw C-order binary array without dtype or shape metadata.
func (a *Array) ToFile(w io.Writer) error {
	b, e := a.ToBytes("C")
	if e != nil {
		return e
	}
	n, e := w.Write(b)
	if e == nil && n != len(b) {
		return io.ErrShortWrite
	}
	return e
}
func FromFile(r io.Reader, dtype DType, count int, byteOffset int64, order binary.ByteOrder) (*Array, error) {
	width := dtype.ItemSize()
	if width == 0 {
		return nil, ErrDType
	}
	if byteOffset < 0 || count < -1 {
		return nil, ErrValue
	}
	if byteOffset != 0 {
		if _, e := io.CopyN(io.Discard, r, byteOffset); e != nil {
			return nil, e
		}
	}
	if count >= 0 {
		if count > maxInt/width {
			return nil, ErrShape
		}
		r = io.LimitReader(r, int64(count*width))
	}
	b, e := io.ReadAll(r)
	if e != nil {
		return nil, e
	}
	b = b[:len(b)/width*width]
	return FromBuffer(b, dtype, -1, 0, order)
}
