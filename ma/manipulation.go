package ma

import (
	"github.com/SamuelSupe/numpygo"
)

func transform(a *Array, f func(*np.Array) (*np.Array, error)) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	data, err := f(a.data)
	if err != nil {
		return nil, err
	}
	mask, err := f(a.mask)
	if err != nil {
		return nil, err
	}
	return a.with(data, mask), nil
}
func Reshape(a *Array, shape ...int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return x.Reshape(shape...) })
}
func Transpose(a *Array, axes ...int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return x.Transpose(axes...) })
}
func Ravel(a *Array) (*Array, error)     { return transform(a, np.Ravel) }
func Atleast1D(a *Array) (*Array, error) { return transform(a, np.Atleast1D) }
func Atleast2D(a *Array) (*Array, error) { return transform(a, np.Atleast2D) }
func Atleast3D(a *Array) (*Array, error) { return transform(a, np.Atleast3D) }
func ExpandDims(a *Array, axis int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return np.ExpandDims(x, axis) })
}
func Squeeze(a *Array, axes ...int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return np.Squeeze(x, axes...) })
}
func SwapAxes(a *Array, x, y int) (*Array, error) {
	return transform(a, func(v *np.Array) (*np.Array, error) { return np.SwapAxes(v, x, y) })
}
func Resize(a *Array, shape ...int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return np.Resize(x, shape...) })
}
func Repeat(a *Array, repeats int, axis *int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return np.Repeat(x, repeats, axis) })
}
func Take(a *Array, indices *np.Array, axis *int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return np.Take(x, indices, axis) })
}
func Compress(condition *np.Array, a *Array, axis *int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return np.Compress(condition, x, axis) })
}
func Diag(a *Array, k int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return np.Diag(x, k) })
}
func Diagflat(a *Array, k int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return np.Diagflat(x, k) })
}
func Diagonal(a *Array, offset, axis1, axis2 int) (*Array, error) {
	return transform(a, func(x *np.Array) (*np.Array, error) { return np.Diagonal(x, offset, axis1, axis2) })
}
func Trace(a *Array, offset, axis1, axis2 int, dtype np.DType) (*Array, error) {
	d, err := Diagonal(a, offset, axis1, axis2)
	if err != nil {
		return nil, err
	}
	// numpy.ma.trace fills the diagonal with zero and defaults to float64,
	// rather than propagating a fully masked diagonal as ma.sum does.
	if dtype == 0 {
		dtype = np.Float64
	}
	data, err := d.Filled(int64(0))
	if err != nil {
		return nil, err
	}
	data, err = data.Astype(dtype)
	if err != nil {
		return nil, err
	}
	out, err := np.Sum(data, np.SumOptions{Axes: []int{-1}, DType: dtype})
	if err != nil {
		return nil, err
	}
	return New(out, nil)
}
func concat(arrays []*Array, f func([]*np.Array) (*np.Array, error)) (*Array, error) {
	if len(arrays) == 0 {
		return nil, np.ErrShape
	}
	data, mask := make([]*np.Array, len(arrays)), make([]*np.Array, len(arrays))
	explicit := false
	for i, a := range arrays {
		if !valid(a) {
			return nil, np.ErrDType
		}
		data[i], mask[i] = a.data, a.mask
		explicit = explicit || a.explicitMask || IsMasked(a)
	}
	d, err := f(data)
	if err != nil {
		return nil, err
	}
	m, err := f(mask)
	if err != nil {
		return nil, err
	}
	out, err := New(d, m)
	if err != nil {
		return nil, err
	}
	out.explicitMask = explicit
	return out, nil
}
func Concatenate(arrays []*Array, axis int) (*Array, error) {
	return concat(arrays, func(v []*np.Array) (*np.Array, error) { return np.Concatenate(v, axis) })
}
func Stack(arrays []*Array, axis int) (*Array, error) {
	return concat(arrays, func(v []*np.Array) (*np.Array, error) { return np.Stack(v, axis) })
}
func Append(a, b *Array, axis *int) (*Array, error) {
	return concat([]*Array{a, b}, func(v []*np.Array) (*np.Array, error) { return np.Append(v[0], v[1], axis) })
}
func HStack(arrays ...*Array) (*Array, error) {
	return concat(arrays, func(v []*np.Array) (*np.Array, error) { return np.HStack(v...) })
}
func VStack(arrays ...*Array) (*Array, error) {
	return concat(arrays, func(v []*np.Array) (*np.Array, error) { return np.VStack(v...) })
}
func DStack(arrays ...*Array) (*Array, error) {
	return concat(arrays, func(v []*np.Array) (*np.Array, error) { return np.DStack(v...) })
}
func ColumnStack(arrays ...*Array) (*Array, error) {
	return concat(arrays, func(v []*np.Array) (*np.Array, error) { return np.ColumnStack(v...) })
}
func Split(a *Array, sections, axis int) ([]*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	d, err := np.Split(a.data, sections, axis)
	if err != nil {
		return nil, err
	}
	m, err := np.Split(a.mask, sections, axis)
	if err != nil {
		return nil, err
	}
	out := make([]*Array, len(d))
	for i := range out {
		out[i] = a.with(d[i], m[i])
	}
	return out, nil
}
func HSplit(a *Array, sections int) ([]*Array, error) {
	if !valid(a) || a.NDim() < 1 {
		return nil, np.ErrShape
	}
	axis := 0
	if a.NDim() > 1 {
		axis = 1
	}
	return Split(a, sections, axis)
}
func VSplit(a *Array, sections int) ([]*Array, error) {
	if !valid(a) || a.NDim() < 2 {
		return nil, np.ErrShape
	}
	return Split(a, sections, 0)
}
func Diff(a *Array, n, axis int) (*Array, error) {
	if !valid(a) || n < 0 {
		return nil, np.ErrValue
	}
	ax, err := np.NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	data, err := np.Diff(a.data, n, ax)
	if err != nil {
		return nil, err
	}
	mask := a.mask
	for i := 0; i < n; i++ {
		length := mask.Shape()[ax]
		if length == 0 {
			break
		}
		left, right := make([]np.Index, ax+1), make([]np.Index, ax+1)
		for j := 0; j < ax; j++ {
			left[j], right[j] = np.Slice{}, np.Slice{}
		}
		left[ax], right[ax] = np.Range(0, length-1), np.Range(1, length)
		x, err := mask.Slice(left...)
		if err != nil {
			return nil, err
		}
		y, err := mask.Slice(right...)
		if err != nil {
			return nil, err
		}
		mask, err = np.LogicalOr(x, y)
		if err != nil {
			return nil, err
		}
	}
	return a.with(data, mask), nil
}
func Ediff1D(a, begin, end *Array) (*Array, error) {
	x, err := Ravel(a)
	if err != nil {
		return nil, err
	}
	d, err := Diff(x, 1, 0)
	if err != nil {
		return nil, err
	}
	arrays := []*Array{d}
	if begin != nil {
		x, err := Ravel(begin)
		if err != nil {
			return nil, err
		}
		arrays = append([]*Array{x}, arrays...)
	}
	if end != nil {
		x, err := Ravel(end)
		if err != nil {
			return nil, err
		}
		arrays = append(arrays, x)
	}
	return Concatenate(arrays, 0)
}
func Where(condition, x, y *Array) (*Array, error) {
	if !valid(condition) || !valid(x) || !valid(y) {
		return nil, np.ErrDType
	}
	data, err := np.Where(condition.data, x.data, y.data)
	if err != nil {
		return nil, err
	}
	m, err := np.Where(condition.data, x.mask, y.mask)
	if err != nil {
		return nil, err
	}
	m, err = np.LogicalOr(m, condition.mask)
	if err != nil {
		return nil, err
	}
	return New(data, m)
}
func Choose(indices *Array, choices []*Array) (*Array, error) {
	if !valid(indices) || len(choices) == 0 {
		return nil, np.ErrShape
	}
	d, m := make([]*np.Array, len(choices)), make([]*np.Array, len(choices))
	for i, a := range choices {
		if !valid(a) {
			return nil, np.ErrDType
		}
		d[i], m[i] = a.data, a.mask
	}
	safe, err := indices.Filled(int64(0))
	if err != nil {
		return nil, err
	}
	data, err := np.Choose(safe, d)
	if err != nil {
		return nil, err
	}
	mask, err := np.Choose(safe, m)
	if err != nil {
		return nil, err
	}
	mask, err = np.LogicalOr(mask, indices.mask)
	if err != nil {
		return nil, err
	}
	return New(data, mask)
}

func Argsort(a *Array, axis *int, endWith bool) (*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	sentinel := MinimumFillValue(a.DType())
	if !endWith {
		sentinel = MaximumFillValue(a.DType())
	}
	data, err := a.Filled(sentinel)
	if err != nil {
		return nil, err
	}
	return np.Argsort(data, axis)
}
func Sort(a *Array, axis *int, endWith bool) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	if axis == nil {
		var err error
		a, err = Ravel(a)
		if err != nil {
			return nil, err
		}
		axis = np.Int(0)
	}
	idx, err := Argsort(a, axis, endWith)
	if err != nil {
		return nil, err
	}
	data, err := np.TakeAlongAxis(a.data, idx, axis)
	if err != nil {
		return nil, err
	}
	mask, err := np.TakeAlongAxis(a.mask, idx, axis)
	if err != nil {
		return nil, err
	}
	return a.with(data, mask), nil
}
func Nonzero(a *Array) ([]*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	v, err := a.Filled(int64(0))
	if err != nil {
		return nil, err
	}
	return np.Nonzero(v)
}
func logicalContract(a, b *Array, op func(*np.Array, *np.Array) (*np.Array, error)) (*Array, error) {
	if !valid(a) || !valid(b) {
		return nil, np.ErrDType
	}
	x, err := a.Filled(int64(0))
	if err != nil {
		return nil, err
	}
	y, err := b.Filled(int64(0))
	if err != nil {
		return nil, err
	}
	data, err := op(x, y)
	if err != nil {
		return nil, err
	}
	am, err := np.LogicalNot(a.mask)
	if err != nil {
		return nil, err
	}
	bm, err := np.LogicalNot(b.mask)
	if err != nil {
		return nil, err
	}
	am, err = am.Astype(np.Int64)
	if err != nil {
		return nil, err
	}
	bm, err = bm.Astype(np.Int64)
	if err != nil {
		return nil, err
	}
	counts, err := op(am, bm)
	if err != nil {
		return nil, err
	}
	mask, err := np.Equal(counts, np.ScalarOf(int64(0)))
	if err != nil {
		return nil, err
	}
	return New(data, mask)
}
func Dot(a, b *Array) (*Array, error)   { return logicalContract(a, b, np.Dot) }
func Inner(a, b *Array) (*Array, error) { return logicalContract(a, b, np.Inner) }
func Outer(a, b *Array) (*Array, error) { return logicalContract(a, b, np.Outer) }

func ClumpMasked(a *Array) ([][2]int, error)   { return clumps(a, true) }
func ClumpUnmasked(a *Array) ([][2]int, error) { return clumps(a, false) }
func clumps(a *Array, want bool) ([][2]int, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	values, err := np.Values[bool](a.mask)
	if err != nil {
		return nil, err
	}
	var result [][2]int
	for i := 0; i < len(values); {
		if values[i] != want {
			i++
			continue
		}
		start := i
		i++
		for i < len(values) && values[i] == want {
			i++
		}
		result = append(result, [2]int{start, i})
	}
	return result, nil
}

func CompressRows(a *Array) (*np.Array, error) { return compressAxis(a, 0) }
func CompressCols(a *Array) (*np.Array, error) { return compressAxis(a, 1) }
func compressAxis(a *Array, axis int) (*np.Array, error) {
	if !valid(a) || a.NDim() != 2 {
		return nil, np.ErrShape
	}
	return CompressND(a, []int{axis})
}
func MaskRows(a *Array) (*Array, error) { return maskAxis(a, 0) }
func MaskCols(a *Array) (*Array, error) { return maskAxis(a, 1) }
func maskAxis(a *Array, axis int) (*Array, error) {
	if !valid(a) || a.NDim() != 2 {
		return nil, np.ErrShape
	}
	bad, err := np.Any(a.mask, np.SumOptions{Axes: []int{1 - axis}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	return MaskedWhere(bad, a)
}

func Zeros(dtype np.DType, shape ...int) (*Array, error) {
	data, err := np.Zeros(dtype, shape...)
	if err != nil {
		return nil, err
	}
	return New(data, nil)
}
func Ones(dtype np.DType, shape ...int) (*Array, error) {
	data, err := np.Ones(dtype, shape...)
	if err != nil {
		return nil, err
	}
	return New(data, nil)
}
func Identity(n int, dtype np.DType) (*Array, error) {
	data, err := np.Identity(n, dtype)
	if err != nil {
		return nil, err
	}
	return New(data, nil)
}
func Arange(start, stop, step float64, dtype np.DType) (*Array, error) {
	data, err := np.Arange(start, stop, step, dtype)
	if err != nil {
		return nil, err
	}
	return New(data, nil)
}
func ZerosLike(a *Array) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	data, err := np.Zeros(a.DType(), a.Shape()...)
	if err != nil {
		return nil, err
	}
	mask, err := a.mask.Copy()
	if err != nil {
		return nil, err
	}
	return a.with(data, mask), nil
}
func OnesLike(a *Array) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	data, err := np.Ones(a.DType(), a.Shape()...)
	if err != nil {
		return nil, err
	}
	mask, err := a.mask.Copy()
	if err != nil {
		return nil, err
	}
	return a.with(data, mask), nil
}
