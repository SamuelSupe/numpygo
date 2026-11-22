package np

import (
	"math"
	"reflect"
	"slices"
	"sort"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

func Real(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if !a.DType().IsComplex() {
		return a, nil
	}
	return componentView(a, 0), nil
}
func Imag(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.DType().IsComplex() {
		return componentView(a, 1), nil
	}
	out, err := a.zerosLike(a.shape)
	if err != nil {
		return nil, err
	}
	out.readOnly = true
	return out, nil
}
func componentView(a *Array, component int) *Array {
	d := Float64
	if a.DType() == Complex64 {
		d = Float32
	}
	if a.DType() == Complex256 {
		d = Float128
	}
	strides := slices.Clone(a.strides)
	offset := 0
	if a.data.byteIndexed {
		offset = a.offset + component*d.ItemSize()
	} else {
		for i := range strides {
			strides[i] *= 2
		}
		offset = a.offset*2 + component
	}
	if a.size == 0 {
		offset = 0
	}
	data := &storage{dtype: d, part: a.data}
	if a.data.buffer != nil {
		data = &storage{dtype: d, buffer: a.data.buffer, order: a.data.order, byteIndexed: a.data.byteIndexed}
	}
	return &Array{data: data, shape: slices.Clone(a.shape), strides: strides, offset: offset, size: a.size, readOnly: a.readOnly, base: a.viewBase()}
}
func RealIfClose(a *Array, tol float64) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if !a.DType().IsComplex() {
		return a, nil
	}
	if a.DType() == Complex256 {
		threshold := x87.FromFloat64(tol)
		if tol > 1 {
			threshold = threshold.Mul(x87.Info().Eps)
		}
		closeEnough := true
		a.walk(func(_, off int) {
			v := extendedValue(a.elementAt(off), Complex256).Imag.Abs()
			c, ok := v.Compare(threshold)
			if !ok || c >= 0 {
				closeEnough = false
			}
		})
		if closeEnough {
			return Real(a)
		}
		return a, nil
	}
	threshold := tol
	if tol > 1 {
		eps := 2.220446049250313e-16
		if a.DType() == Complex64 {
			eps = 1.1920928955078125e-7
		}
		threshold *= eps
	}
	closeEnough := true
	a.walk(func(_, off int) {
		if !(math.Abs(imag(a.elementAt(off).z)) < threshold) {
			closeEnough = false
		}
	})
	if closeEnough {
		return Real(a)
	}
	return a, nil
}
func AsContiguousArray(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 0 {
		return a.Reshape(1)
	}
	if a.IsCContiguous() {
		return a, nil
	}
	return a.Copy()
}
func AsFortranArray(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 0 {
		return a.Reshape(1)
	}
	if a.IsFContiguous() {
		return a, nil
	}
	r, err := a.Transpose()
	if err != nil {
		return nil, err
	}
	r, err = r.Copy()
	if err != nil {
		return nil, err
	}
	return r.Transpose()
}
func Atleast1D(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 0 {
		return a.Reshape(1)
	}
	return a, nil
}
func Atleast2D(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	switch a.NDim() {
	case 0:
		return a.Reshape(1, 1)
	case 1:
		return a.Slice(NewAxis, Slice{})
	default:
		return a, nil
	}
}
func Atleast3D(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	switch a.NDim() {
	case 0:
		return a.Reshape(1, 1, 1)
	case 1:
		return a.Slice(NewAxis, Slice{}, NewAxis)
	case 2:
		return a.Slice(Slice{}, Slice{}, NewAxis)
	default:
		return a, nil
	}
}
func IsFortran(a *Array) bool { return a.DType() != 0 && a.IsFContiguous() && !a.IsCContiguous() }

func ArrayEqual(a, b *Array, equalNaN bool) (bool, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return false, ErrDType
	}
	if !slices.Equal(a.shape, b.shape) {
		return false, nil
	}
	return arrayEquiv(a, b, equalNaN)
}
func ArrayEquiv(a, b *Array) (bool, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return false, ErrDType
	}
	return arrayEquiv(a, b, false)
}
func arrayEquiv(a, b *Array, equalNaN bool) (bool, error) {
	if _, err := BroadcastShapes(a.shape, b.shape); err != nil {
		return false, nil
	}
	eq, err := Equal(a, b)
	if err != nil {
		return false, err
	}
	if equalNaN {
		x, err := IsNaN(a)
		if err != nil {
			return false, err
		}
		y, err := IsNaN(b)
		if err != nil {
			return false, err
		}
		both, err := LogicalAnd(x, y)
		if err != nil {
			return false, err
		}
		eq, err = LogicalOr(eq, both)
		if err != nil {
			return false, err
		}
	}
	result, err := All(eq)
	if err != nil {
		return false, err
	}
	v, err := result.AtValue()
	if err != nil {
		return false, err
	}
	return v.(bool), nil
}

// Fill writes one scalar to all logical elements of an array or view.
func (a *Array) Fill(value any) error {
	if !a.Writable() {
		return ErrReadOnly
	}
	if a.DType().IsSpecial() {
		e, err := a.convertValue(value)
		if err != nil {
			return err
		}
		a.walk(func(_, off int) { a.setElement(off, e) })
		return nil
	}
	d := dtypeOf(value)
	if d.IsFlexible() {
		e, err := a.convertValue(value)
		if err != nil {
			return err
		}
		a.walk(func(_, off int) { a.setElement(off, e) })
		return nil
	}
	if d == 0 || d.IsTemporal() {
		return ErrDType
	}
	e := elementFrom(value)
	if d.IsComplex() && !a.DType().IsComplex() && a.DType() != Bool && imag(e.number(d)) != 0 {
		return ErrDType
	}
	e = castElement(e, d, a.DType())
	a.walk(func(_, off int) { a.setElement(off, e) })
	return nil
}

// CopyTo broadcasts the source and optional mask over dst, with RHS snapshot semantics.
func CopyTo(dst, src, where *Array) error {
	if !dst.Writable() {
		return ErrReadOnly
	}
	if src.DType() == 0 {
		return ErrDType
	}
	x, err := src.BroadcastTo(dst.shape...)
	if err != nil {
		return err
	}
	x, err = dst.castLike(x)
	if err != nil {
		return err
	}
	if where == nil {
		return dst.Assign(nil, x)
	}
	if where.DType() == 0 {
		return ErrDType
	}
	w, err := where.BroadcastTo(dst.shape...)
	if err != nil {
		return err
	}
	w, err = w.Astype(Bool)
	if err != nil {
		return err
	}
	dst.walk(func(i, off int) {
		if w.elementAt(i).i != 0 {
			dst.setElement(off, x.elementAt(i))
		}
	})
	return nil
}

func memoryStart(a *Array) uintptr {
	s := a.data
	if s.part != nil {
		s = s.part
	}
	if s.buffer != nil {
		return reflect.ValueOf(s.buffer).Pointer()
	}
	return reflect.ValueOf(storageSlice(s)).Pointer()
}

// DataIdentity is an opaque address token for the first logical element.
// It is for alias diagnostics only, not a foreign pointer or a lifetime guarantee.
func (a *Array) DataIdentity() uintptr {
	if a.DType() == 0 || a.Size() == 0 {
		return 0
	}
	return memoryStart(a) + uintptr(a.offset)*a.memoryUnit()
}
func (a *Array) memoryWidth() uintptr {
	if a.data.slots != nil {
		return uintptr(a.ItemSize())
	}
	if a.DType() == Object {
		return reflect.TypeFor[any]().Size()
	}
	if a.DType() == StringDType {
		return reflect.TypeFor[any]().Size()
	}
	return uintptr(a.ItemSize())
}
func (a *Array) memoryUnit() uintptr {
	if a.data.byteIndexed {
		return 1
	}
	return a.memoryWidth()
}
func memoryBounds(a *Array) (uintptr, uintptr) {
	low, high := a.offset, a.offset
	for i, d := range a.shape {
		v := (d - 1) * a.strides[i]
		if v < 0 {
			low += v
		} else {
			high += v
		}
	}
	size := a.memoryUnit()
	base := memoryStart(a)
	return base + uintptr(low)*size, base + uintptr(high)*size + a.memoryWidth()
}
func MayShareMemory(a, b *Array) bool {
	if a.DType() == 0 || b.DType() == 0 || a.size == 0 || b.size == 0 {
		return false
	}
	al, ah := memoryBounds(a)
	bl, bh := memoryBounds(b)
	return al < bh && bl < ah
}
func SharesMemory(a, b *Array) bool {
	if !MayShareMemory(a, b) {
		return false
	}
	ac := a.IsCContiguous() || a.IsFContiguous()
	bc := b.IsCContiguous() || b.IsFContiguous()
	if ac && bc {
		return true
	}
	if ac || bc {
		if bc {
			a, b = b, a
		}
		al, ah := memoryBounds(a)
		base := memoryStart(b)
		size := b.memoryUnit()
		found := false
		b.walk(func(_, off int) {
			p := base + uintptr(off)*size
			if p < ah && p+b.memoryWidth() > al {
				found = true
			}
		})
		return found
	}
	if a.size > b.size {
		a, b = b, a
	}
	addresses := make([]uintptr, a.size)
	base := memoryStart(a)
	as := a.memoryWidth()
	a.walk(func(i, off int) { addresses[i] = base + uintptr(off)*a.memoryUnit() })
	sort.Slice(addresses, func(i, j int) bool { return addresses[i] < addresses[j] })
	base = memoryStart(b)
	bs := b.memoryWidth()
	found := false
	b.walk(func(_, off int) {
		if found {
			return
		}
		p := base + uintptr(off)*b.memoryUnit()
		j := sort.Search(len(addresses), func(j int) bool { return addresses[j]+as > p })
		if j < len(addresses) && addresses[j] < p+bs {
			found = true
		}
	})
	return found
}

func BroadcastArrays(arrays ...*Array) ([]*Array, error) {
	shapes := make([][]int, len(arrays))
	for i, a := range arrays {
		if a.DType() == 0 {
			return nil, ErrDType
		}
		shapes[i] = a.shape
	}
	shape, err := BroadcastShapes(shapes...)
	if err != nil {
		return nil, err
	}
	out := make([]*Array, len(arrays))
	for i, a := range arrays {
		out[i], err = a.BroadcastTo(shape...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
