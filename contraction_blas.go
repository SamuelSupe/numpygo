package np

import (
	"slices"

	kernel "github.com/SamuelSupe/numpygo/internal/numeric"
)

func tensorGemm(a, b *Array, d DType, axesA, axesB []int, usedA, usedB []bool) (*Array, error) {
	if !d.IsNativeNumeric() {
		return nil, ErrDType
	}
	var left, right, reduce, pa, pb []int
	for i, dim := range a.shape {
		if !usedA[i] {
			left = append(left, dim)
			pa = append(pa, i)
		}
	}
	for _, i := range axesA {
		reduce = append(reduce, a.shape[i])
		pa = append(pa, i)
	}
	pb = append(pb, axesB...)
	for i, dim := range b.shape {
		if !usedB[i] {
			right = append(right, dim)
			pb = append(pb, i)
		}
	}
	shape := append(slices.Clone(left), right...)
	size, e := shapeSize(d, shape)
	if e != nil {
		return nil, e
	}
	if size == 0 {
		return Zeros(d, shape...)
	}
	k, e := shapeSize(d, reduce)
	if e != nil {
		return nil, e
	}
	if k == 0 {
		return Zeros(d, shape...)
	}
	m, e := shapeSize(d, left)
	if e != nil {
		return nil, e
	}
	n, e := shapeSize(d, right)
	if e != nil {
		return nil, e
	}
	a, e = a.Transpose(pa...)
	if e != nil {
		return nil, e
	}
	b, e = b.Transpose(pb...)
	if e != nil {
		return nil, e
	}
	if a.DType() != d {
		a, e = a.Astype(d)
		if e != nil {
			return nil, e
		}
	}
	if b.DType() != d {
		b, e = b.Astype(d)
		if e != nil {
			return nil, e
		}
	}
	switch d {
	case Float64:
		return gemmProduct[float64](a, b, shape, size, m, k, n)
	case Float32:
		return gemmProduct[float32](a, b, shape, size, m, k, n)
	case Complex64:
		return gemmProduct[complex64](a, b, shape, size, m, k, n)
	case Complex128:
		return gemmProduct[complex128](a, b, shape, size, m, k, n)
	}
	return nil, ErrDType
}
func gemmProduct[T kernel.Floating](a, b *Array, shape []int, size, m, k, n int) (*Array, error) {
	av, e := gemmInput[T](a)
	if e != nil {
		return nil, e
	}
	bv, e := gemmInput[T](b)
	if e != nil {
		return nil, e
	}
	out := make([]T, size)
	kernel.Gemm(m, k, n, av, bv, out)
	return WrapSlice(out, shape...)
}

// BLAS only reads its input operands. Pack a view only when the storage
// is not already a contiguous native slice; Values itself still copies.
func gemmInput[T kernel.Floating](a *Array) ([]T, error) {
	if a.IsCContiguous() && a.data.part == nil && a.data.buffer == nil {
		if s, ok := storageSlice(a.data).([]T); ok {
			return s[a.offset : a.offset+a.size], nil
		}
	}
	return Values[T](a)
}

// Only ordinary two-operand dot contractions are lowered to BLAS. Diagonals,
// shared retained axes, broadcast contraction dimensions and >2 operands
// still use contract. The unplanned Einsum preserves its original order.
func contractPlanned(arrays []*Array, labels [][]int, output []int) (*Array, error) {
	if len(arrays) != 2 || len(labels) != 2 {
		return contract(arrays, labels, output)
	}
	a, b := arrays[0], arrays[1]
	al, bl := labels[0], labels[1]
	if len(al) != a.NDim() || len(bl) != b.NDim() || len(toLabelSet(al)) != len(al) || len(toLabelSet(bl)) != len(bl) {
		return contract(arrays, labels, output)
	}
	d, e := PromoteTypes(a.DType(), b.DType())
	if e != nil || !(d == Float32 || d == Float64 || d.IsComplex()) {
		return contract(arrays, labels, output)
	}
	var aa, bb, canonical []int
	for i, l := range al {
		j := slices.Index(bl, l)
		kept := slices.Contains(output, l)
		if kept {
			if j >= 0 {
				return contract(arrays, labels, output)
			}
			canonical = append(canonical, l)
		} else {
			if j < 0 || a.shape[i] != b.shape[j] {
				return contract(arrays, labels, output)
			}
			aa = append(aa, i)
			bb = append(bb, j)
		}
	}
	for _, l := range bl {
		if slices.Contains(output, l) {
			canonical = append(canonical, l)
		} else if !slices.Contains(al, l) {
			return contract(arrays, labels, output)
		}
	}
	if len(aa) == 0 || len(canonical) != len(output) {
		return contract(arrays, labels, output)
	}
	permutation := make([]int, len(output))
	seen := make([]bool, len(output))
	for i, l := range output {
		j := slices.Index(canonical, l)
		if j < 0 || seen[j] {
			return contract(arrays, labels, output)
		}
		permutation[i] = j
		seen[j] = true
	}
	v, e := Tensordot(a, b, aa, bb)
	if e != nil {
		return nil, e
	}
	if slices.Equal(canonical, output) {
		return v, nil
	}
	v, e = v.Transpose(permutation...)
	if e != nil {
		return nil, e
	}
	return v.Copy()
}
