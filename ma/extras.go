package ma

import (
	"encoding/binary"
	"iter"
	"slices"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/poly"
)

func IsMaskedArray(v any) bool { a, ok := v.(*Array); return ok && valid(a) }
func CommonFillValue(a, b *Array) (any, error) {
	if !valid(a) || !valid(b) {
		return nil, np.ErrDType
	}
	x, e := np.Full(a.DType(), a.FillValue())
	if e != nil {
		return nil, e
	}
	y, e := np.Full(b.DType(), b.FillValue())
	if e != nil {
		return nil, e
	}
	eq, e := np.ArrayEqual(x, y, false)
	if e != nil {
		return nil, e
	}
	if !eq {
		return nil, nil
	}
	return a.FillValue(), nil
}
func FlattenMask(mask *np.Array) (*np.Array, error) {
	b, e := MakeMask(mask)
	if e != nil {
		return nil, e
	}
	return np.Ravel(b)
}
func Empty(dtype np.DType, shape ...int) (*Array, error) {
	d, e := np.Empty(dtype, shape...)
	if e != nil {
		return nil, e
	}
	return New(d, nil)
}
func EmptyLike(a *Array, dtype np.DType, order string) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	data, e := np.EmptyLike(a.data, dtype, order)
	if e != nil {
		return nil, e
	}
	mask, e := a.mask.Copy()
	if e != nil {
		return nil, e
	}
	return a.with(data, mask), nil
}
func FromBuffer(b []byte, dtype np.DType, count, offset int, order binary.ByteOrder) (*Array, error) {
	d, e := np.FromBuffer(b, dtype, count, offset, order)
	if e != nil {
		return nil, e
	}
	return New(d, nil)
}
func FromFunction(shape []int, d np.DType, fn func(...*np.Array) (*np.Array, error)) (*Array, error) {
	a, e := np.FromFunction(shape, d, fn)
	if e != nil {
		return nil, e
	}
	return New(a, nil)
}
func Indices(shape []int) (*Array, error) {
	d, e := np.DenseIndices(shape)
	if e != nil {
		return nil, e
	}
	return New(d, nil)
}
func CompressND(a *Array, axes []int) (*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	if axes == nil {
		axes = make([]int, a.NDim())
		for i := range axes {
			axes[i] = i
		}
	}
	normalized := make([]int, len(axes))
	seen := make(map[int]bool, len(axes))
	for i, axis := range axes {
		ax, e := np.NormalizeAxis(axis, a.NDim())
		if e != nil || seen[ax] {
			return nil, np.ErrAxis
		}
		seen[ax] = true
		normalized[i] = ax
	}
	if !IsMasked(a) {
		return a.data, nil
	}
	all, e := np.All(a.mask)
	if e != nil {
		return nil, e
	}
	fully, _ := all.AtValue()
	if fully.(bool) {
		return np.Zeros(np.Float64, 0)
	}
	data := a.data
	for _, ax := range normalized {
		other := make([]int, 0, a.NDim()-1)
		for i := 0; i < a.NDim(); i++ {
			if i != ax {
				other = append(other, i)
			}
		}
		bad, e := np.Any(a.mask, np.SumOptions{Axes: other})
		if e != nil {
			return nil, e
		}
		good, e := np.LogicalNot(bad)
		if e != nil {
			return nil, e
		}
		data, e = np.Compress(good, data, &ax)
		if e != nil {
			return nil, e
		}
	}
	return data, nil
}
func CompressRowCols(a *Array, axis *int) (*np.Array, error) {
	if !valid(a) || a.NDim() != 2 {
		return nil, np.ErrShape
	}
	if axis == nil {
		return CompressND(a, nil)
	}
	return CompressND(a, []int{*axis})
}
func MaskRowCols(a *Array, axis *int) (*Array, error) {
	if !valid(a) || a.NDim() != 2 {
		return nil, np.ErrShape
	}
	if axis != nil {
		ax, e := np.NormalizeAxis(*axis, 2)
		if e != nil {
			return nil, e
		}
		return maskAxis(a, ax)
	}
	rows, e := np.Any(a.mask, np.SumOptions{Axes: []int{1}, KeepDims: true})
	if e != nil {
		return nil, e
	}
	cols, e := np.Any(a.mask, np.SumOptions{Axes: []int{0}, KeepDims: true})
	if e != nil {
		return nil, e
	}
	both, e := np.LogicalOr(rows, cols)
	if e != nil {
		return nil, e
	}
	return MaskedWhere(both, a)
}

func FlatNotMaskedEdges(a *Array) (*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	// nomask (including a zero-sized array) uses the two endpoint indices.
	if !IsMasked(a) {
		return np.FromSlice([]int64{0, int64(a.Size() - 1)}, 2)
	}
	mask, e := np.LogicalNot(a.mask)
	if e != nil {
		return nil, e
	}
	idx, e := np.FlatNonzero(mask)
	if e != nil {
		return nil, e
	}
	if idx.Size() == 0 {
		return nil, nil
	}
	f, _ := idx.AtValue(0)
	l, _ := idx.AtValue(-1)
	return np.FromSlice([]int64{f.(int64), l.(int64)}, 2)
}
func FlatNotMaskedContiguous(a *Array) ([][2]int, error) { return ClumpUnmasked(a) }
func NotMaskedContiguous(a *Array, axis *int) ([][][2]int, error) {
	if !valid(a) || a.NDim() > 2 {
		return nil, np.ErrShape
	}
	if axis == nil || a.NDim() == 1 {
		v, e := FlatNotMaskedContiguous(a)
		return [][][2]int{v}, e
	}
	ax, e := np.NormalizeAxis(*axis, 2)
	if e != nil {
		return nil, e
	}
	out := make([][][2]int, a.Shape()[1-ax])
	for i := range out {
		index := []np.Index{np.Slice{}, np.Slice{}}
		index[1-ax] = np.AtIndex(i)
		view, e := a.Index(index...)
		if e != nil {
			return nil, e
		}
		out[i], e = FlatNotMaskedContiguous(view)
		if e != nil {
			return nil, e
		}
	}
	return out, nil
}

// NotMaskedEdges returns coordinate arrays for the first and last valid
// positions of each lane along axis. Lanes with no valid values are omitted.
func NotMaskedEdges(a *Array, axis int) (first, last []*np.Array, err error) {
	if !valid(a) {
		return nil, nil, np.ErrDType
	}
	ax, e := np.NormalizeAxis(axis, a.NDim())
	if e != nil {
		return nil, nil, e
	}
	shape := a.Shape()
	outer := append(slices.Clone(shape[:ax]), shape[ax+1:]...)
	it, e := np.NDIndex(outer...)
	if e != nil {
		return nil, nil, e
	}
	f, l := make([][]int64, a.NDim()), make([][]int64, a.NDim())
	for idx := range it {
		index := append(slices.Clone(idx[:ax]), 0)
		index = append(index, idx[ax:]...)
		begin, end := -1, -1
		for k := 0; k < shape[ax]; k++ {
			index[ax] = k
			masked, _ := a.mask.AtValue(index...)
			if !masked.(bool) {
				if begin < 0 {
					begin = k
				}
				end = k
			}
		}
		if begin < 0 {
			continue
		}
		index[ax] = begin
		for j, v := range index {
			f[j] = append(f[j], int64(v))
		}
		index[ax] = end
		for j, v := range index {
			l[j] = append(l[j], int64(v))
		}
	}
	first, last = make([]*np.Array, a.NDim()), make([]*np.Array, a.NDim())
	for i := range first {
		first[i], e = np.FromSlice(f[i], len(f[i]))
		if e != nil {
			return nil, nil, e
		}
		last[i], e = np.FromSlice(l[i], len(l[i]))
		if e != nil {
			return nil, nil, e
		}
	}
	return first, last, nil
}

type MaskedValue struct {
	Value  any
	Masked bool
}

func NDEnumerate(a *Array, compressed bool) (iter.Seq2[[]int, MaskedValue], error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	it, e := np.NDEnumerate(a.data)
	if e != nil {
		return nil, e
	}
	return func(yield func([]int, MaskedValue) bool) {
		for index, value := range it {
			m, _ := a.mask.AtValue(index...)
			masked := m.(bool)
			if masked && compressed {
				continue
			}
			if masked {
				value = nil
			}
			if !yield(index, MaskedValue{value, masked}) {
				return
			}
		}
	}, nil
}

func Put(a *Array, indices *np.Array, values *Array, mode string) error {
	if !valid(a) || !valid(values) {
		return np.ErrDType
	}
	if !a.data.Writable() || !a.mask.Writable() {
		return np.ErrReadOnly
	}
	data, e := a.data.Copy()
	if e != nil {
		return e
	}
	mask, e := a.mask.Copy()
	if e != nil {
		return e
	}
	if e = np.Put(data, indices, values.data, mode); e != nil {
		return e
	}
	if e = np.Put(mask, indices, values.mask, mode); e != nil {
		return e
	}
	if a.hard {
		data, e = np.Where(a.mask, a.data, data)
		if e != nil {
			return e
		}
		mask, e = np.LogicalOr(a.mask, mask)
		if e != nil {
			return e
		}
	}
	if e = a.data.Assign(nil, data); e != nil {
		return e
	}
	if e = a.mask.Assign(nil, mask); e != nil {
		return e
	}
	a.explicitMask = IsMasked(a)
	return nil
}
func Putmask(a *Array, mask *np.Array, values *Array) error {
	if !valid(a) || !valid(values) {
		return np.ErrDType
	}
	if !a.data.Writable() || !a.mask.Writable() {
		return np.ErrReadOnly
	}
	m, e := a.mask.Copy()
	if e != nil {
		return e
	}
	if e = np.CopyTo(m, values.mask, mask); e != nil {
		return e
	}
	if a.hard {
		m, e = np.LogicalOr(m, a.mask)
		if e != nil {
			return e
		}
	}
	if e = np.CopyTo(a.data, values.data, mask); e != nil {
		return e
	}
	if e = a.mask.Assign(nil, m); e != nil {
		return e
	}
	a.explicitMask = a.explicitMask || values.explicitMask
	return nil
}

func Vander(a *Array, n int) (*np.Array, error) {
	if !valid(a) || a.NDim() != 1 {
		return nil, np.ErrShape
	}
	v, e := np.Vander(a.data, n, false)
	if e != nil {
		return nil, e
	}
	m, e := np.ExpandDims(a.mask, 1)
	if e != nil {
		return nil, e
	}
	return np.Where(m, np.ScalarOf(int64(0)), v)
}
func Polyfit(x, y, weights *Array, degree int) (poly.FitResult, error) {
	var zero poly.FitResult
	if !valid(x) || !valid(y) || x.NDim() != 1 || y.NDim() < 1 || y.NDim() > 2 || y.Shape()[0] != x.Size() {
		return zero, np.ErrShape
	}
	m := y.mask
	var e error
	if y.NDim() == 2 {
		m, e = np.Any(m, np.SumOptions{Axes: []int{1}})
		if e != nil {
			return zero, e
		}
	}
	m, e = np.LogicalOr(m, x.mask)
	if e != nil {
		return zero, e
	}
	if weights != nil {
		if !valid(weights) || weights.NDim() != 1 || weights.Size() != x.Size() {
			return zero, np.ErrShape
		}
		m, e = np.LogicalOr(m, weights.mask)
		if e != nil {
			return zero, e
		}
	}
	good, e := np.LogicalNot(m)
	if e != nil {
		return zero, e
	}
	xx, e := x.data.Index(good)
	if e != nil {
		return zero, e
	}
	yy, e := y.data.Index(good)
	if e != nil {
		return zero, e
	}
	opt := poly.FitOptions{}
	if weights != nil {
		opt.Weights, e = weights.data.Index(good)
		if e != nil {
			return zero, e
		}
	}
	return poly.LegacyFit(xx, yy, degree, opt)
}
