package np

import (
	"fmt"
	"slices"
)

// Index describes a NumPy-style index: a basic index or an integer/boolean Array.
type Index interface{ basicIndex() }

// Slice selects a range using Python's exclusive stop and optional bounds.
// A nil Step means +1. With a negative step, nil Stop means before the first
// element, which differs from an explicit stop of -1.
type Slice struct {
	Start, Stop, Step *int
}

func (Slice) basicIndex() {}

// AtIndex selects an integer index and removes that dimension from the view.
type AtIndex int

func (AtIndex) basicIndex() {}

type specialIndex uint8

func (specialIndex) basicIndex() {}

const (
	NewAxis  specialIndex = iota + 1 // insert a size-one axis without consuming an input axis
	Ellipsis                         // expand to the remaining input dimensions
)

func Range(start, stop int) Slice { return Slice{Start: Int(start), Stop: Int(stop)} }

func Strided(start, stop, step int) Slice {
	return Slice{Start: Int(start), Stop: Int(stop), Step: Int(step)}
}

func Reverse() Slice { return Slice{Step: Int(-1)} }

func (s Slice) normalize(dim int) (start, length, step int, err error) {
	step = 1
	if s.Step != nil {
		step = *s.Step
	}
	if step == 0 {
		return 0, 0, 0, fmt.Errorf("%w: slice step is zero", ErrIndex)
	}
	lower, upper := 0, dim
	start, stop := lower, upper
	if step < 0 {
		lower, upper = -1, dim-1
		start, stop = upper, lower
	}
	if s.Start != nil {
		start = *s.Start
		if start < 0 {
			start += dim
		}
		start = min(max(start, lower), upper)
	}
	if s.Stop != nil {
		stop = *s.Stop
		if stop < 0 {
			stop += dim
		}
		stop = min(max(stop, lower), upper)
	}
	if step > 0 && start < stop {
		length = 1 + (stop-start-1)/step
	} else if step < 0 && start > stop {
		// Dividing by the negative step avoids negating the minimum int.
		length = 1 + (stop-start+1)/step
	}
	if length == 0 {
		return 0, 0, 1, nil
	}
	return start, length, step, nil
}

func checkedStride(stride, factor, itemSize int) (int, bool) {
	if stride == 0 || factor == 0 {
		return 0, true
	}
	limit := maxInt / max(itemSize, 1)
	if factor == -maxInt-1 || stride == -maxInt-1 {
		return 0, false
	}
	if max(stride, -stride) > limit/max(factor, -factor) {
		return 0, false
	}
	return stride * factor, true
}

// Slice returns a storage-sharing view. Unspecified trailing axes are kept.
// Integer indexes drop an axis; use At when a scalar value is needed instead.
func (a *Array) Slice(indices ...Index) (*Array, error) {
	for _, index := range indices {
		if _, ok := index.(*Array); ok {
			return a.Index(indices...)
		}
	}
	if a.DType() == 0 {
		return nil, ErrDType
	}
	consumed, ellipses, inserted := 0, 0, 0
	for _, index := range indices {
		switch x := index.(type) {
		case Slice, AtIndex:
			consumed++
		case specialIndex:
			if x == Ellipsis {
				ellipses++
			} else if x == NewAxis {
				inserted++
			} else {
				return nil, ErrIndex
			}
		default:
			return nil, ErrIndex
		}
	}
	if consumed > a.NDim() || ellipses > 1 {
		return nil, ErrIndex
	}
	remaining := a.NDim() - consumed
	shape := make([]int, 0, a.NDim()+inserted)
	strides := make([]int, 0, cap(shape))
	offset, axis := a.offset, 0
	appendAxes := func(n int) {
		shape = append(shape, a.shape[axis:axis+n]...)
		strides = append(strides, a.strides[axis:axis+n]...)
		axis += n
	}
	for _, index := range indices {
		switch x := index.(type) {
		case Slice:
			start, length, step, err := x.normalize(a.shape[axis])
			if err != nil {
				return nil, err
			}
			stride, ok := checkedStride(a.strides[axis], step, a.strideUnit())
			if !ok {
				return nil, fmt.Errorf("%w: slice stride overflow", ErrShape)
			}
			offset += start * a.strides[axis]
			shape = append(shape, length)
			strides = append(strides, stride)
			axis++
		case AtIndex:
			i, err := normalizeIndex(int(x), a.shape[axis])
			if err != nil {
				return nil, err
			}
			offset += i * a.strides[axis]
			axis++
		case specialIndex:
			if x == Ellipsis {
				appendAxes(remaining)
			} else {
				shape = append(shape, 1)
				strides = append(strides, 0)
			}
		}
	}
	appendAxes(a.NDim() - axis)
	size, err := a.shapeSize(shape)
	if err != nil {
		return nil, err
	}
	if size == 0 {
		offset = 0 // there are no addressable elements in an empty view
	}
	return &Array{data: a.data, shape: shape, strides: strides, offset: offset, size: size, readOnly: a.readOnly, base: a.viewBase()}, nil
}

// Transpose permutes axes without copying elements. With no arguments it reverses all axes.
func (a *Array) Transpose(axes ...int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if len(axes) == 0 {
		axes = make([]int, a.NDim())
		for i := range axes {
			axes[i] = len(axes) - i - 1
		}
	}
	if len(axes) != a.NDim() {
		return nil, ErrAxis
	}
	shape, strides := make([]int, len(axes)), make([]int, len(axes))
	seen := make([]bool, len(axes))
	for i, axis := range axes {
		axis, err := NormalizeAxis(axis, a.NDim())
		if err != nil || seen[axis] {
			return nil, ErrAxis
		}
		seen[axis] = true
		shape[i], strides[i] = a.shape[axis], a.strides[axis]
	}
	return &Array{data: a.data, shape: shape, strides: strides, offset: a.offset, size: a.size, readOnly: a.readOnly, base: a.viewBase()}, nil
}

// Reshape changes shape in logical C order. It shares storage when the existing
// layout permits the reshape, otherwise it copies. One dimension may be -1.
func (a *Array) Reshape(shape ...int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	shape = slices.Clone(shape)
	unknown := -1
	for i, dim := range shape {
		if dim == -1 {
			if unknown != -1 {
				return nil, ErrShape
			}
			unknown = i
			shape[i] = 1
		}
	}
	size, err := a.shapeSize(shape)
	if err != nil {
		return nil, err
	}
	if unknown != -1 {
		if size == 0 || a.size%size != 0 {
			return nil, ErrShape
		}
		shape[unknown] = a.size / size
		size, err = a.shapeSize(shape)
		if err != nil {
			return nil, err
		}
	}
	if size != a.size {
		return nil, fmt.Errorf("%w: cannot reshape size %d to %v", ErrShape, a.size, shape)
	}
	if slices.Equal(a.shape, shape) {
		return a.Slice()
	}
	if strides, ok := a.reshapeStrides(shape); ok {
		return &Array{data: a.data, shape: shape, strides: strides, offset: a.offset, size: size, readOnly: a.readOnly, base: a.viewBase()}, nil
	}
	result, err := a.Copy()
	if err != nil {
		return nil, err
	}
	result.shape = shape
	result.strides = contiguousStrides(shape, size)
	return result, nil
}

// reshapeStrides coalesces contiguous runs of existing axes in logical C order.
// It also handles uniformly stepped and reversed arrays without copying.
func (a *Array) reshapeStrides(shape []int) ([]int, bool) {
	strides := make([]int, len(shape))
	if a.size == 0 || a.IsCContiguous() {
		stride := a.contiguousStep()
		for i := len(shape) - 1; i >= 0; i-- {
			strides[i] = stride
			stride *= max(shape[i], 1)
		}
		return strides, true
	}
	oldShape, oldStrides := make([]int, 0, a.NDim()), make([]int, 0, a.NDim())
	for i, dim := range a.shape {
		if dim != 1 {
			oldShape = append(oldShape, dim)
			oldStrides = append(oldStrides, a.strides[i])
		}
	}
	oi, ni := 0, 0
	for oi < len(oldShape) && ni < len(shape) {
		oj, nj := oi+1, ni+1
		op, np := oldShape[oi], shape[ni]
		for op != np {
			if op < np {
				if oj == len(oldShape) {
					return nil, false
				}
				op *= oldShape[oj]
				oj++
			} else {
				if nj == len(shape) {
					return nil, false
				}
				np *= shape[nj]
				nj++
			}
		}
		for i := oi; i < oj-1; i++ {
			stride, ok := checkedStride(oldStrides[i+1], oldShape[i+1], a.strideUnit())
			if !ok || oldStrides[i] != stride {
				return nil, false
			}
		}
		strides[nj-1] = oldStrides[oj-1]
		for i := nj - 1; i > ni; i-- {
			stride, ok := checkedStride(strides[i], shape[i], a.strideUnit())
			if !ok {
				return nil, false
			}
			strides[i-1] = stride
		}
		oi, ni = oj, nj
	}
	lastStride := a.contiguousStep()
	if ni > 0 {
		lastStride = strides[ni-1]
	}
	for i := ni; i < len(shape); i++ {
		strides[i] = lastStride
	}
	return strides, true
}

// BroadcastShapes computes the common shape by comparing dimensions from the right.
func BroadcastShapes(shapes ...[]int) ([]int, error) {
	rank := 0
	for _, shape := range shapes {
		rank = max(rank, len(shape))
	}
	if rank > maxDims {
		return nil, ErrShape
	}
	result := make([]int, rank)
	for i := range result {
		result[i] = 1
	}
	for _, shape := range shapes {
		for i, dim := range shape {
			axis := rank - len(shape) + i
			if dim < 0 || (dim != 1 && result[axis] != 1 && dim != result[axis]) {
				return nil, fmt.Errorf("%w: cannot broadcast %v with %v", ErrShape, result, shape)
			}
			if result[axis] == 1 {
				result[axis] = dim
			}
		}
	}
	return result, nil
}

// BroadcastTo returns a read-only storage-sharing view using zero strides for
// replicated elements. The target may add leading axes but cannot drop axes.
func (a *Array) BroadcastTo(shape ...int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	size, err := a.shapeSize(shape)
	if err != nil {
		return nil, err
	}
	combined, err := BroadcastShapes(a.shape, shape)
	if err != nil || !slices.Equal(combined, shape) {
		return nil, ErrShape
	}
	strides := make([]int, len(shape))
	for i, dim := range a.shape {
		if dim != 1 {
			strides[len(shape)-a.NDim()+i] = a.strides[i]
		}
	}
	offset := a.offset
	if size == 0 {
		offset = 0
	}
	return &Array{data: a.data, shape: slices.Clone(shape), strides: strides, offset: offset, size: size, readOnly: true, base: a.viewBase()}, nil
}
