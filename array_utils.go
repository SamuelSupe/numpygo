package np

import "fmt"

// NormalizeAxisIndex resolves an axis against the rank. The supplied name
// (if nonempty) labels the invalid-axis error.
func NormalizeAxisIndex(axis, ndim int, name string) (int, error) {
	v, e := NormalizeAxis(axis, ndim)
	if e != nil && name != "" {
		return 0, fmt.Errorf("%s: %w", name, e)
	}
	return v, e
}

// NormalizeAxisTuple preserves the supplied axis order; duplicates are
// rejected unless allowDuplicate is set. Nil denotes an empty axis tuple.
func NormalizeAxisTuple(axes []int, ndim int, allowDuplicate bool) ([]int, error) {
	if ndim < 0 {
		return nil, ErrAxis
	}
	out := make([]int, len(axes))
	seen := map[int]bool{}
	for i, a := range axes {
		x, e := NormalizeAxis(a, ndim)
		if e != nil {
			return nil, e
		}
		if !allowDuplicate && seen[x] {
			return nil, ErrValue
		}
		seen[x] = true
		out[i] = x
	}
	return out, nil
}

// ByteBounds reports opaque address tokens for the first and one-past-last
// addressed bytes, including gaps, negative and zero strides. Empty arrays
// return (0,0), as they have no addressable element in Go. Object/string slots
// report the actual Go GC slot span, not NumPy's pointer ABI.
func ByteBounds(a *Array) (low, high uintptr, err error) {
	if a.DType() == 0 {
		return 0, 0, ErrDType
	}
	if a.Size() == 0 {
		return 0, 0, nil
	}
	low, high = memoryBounds(a)
	return low, high, nil
}
