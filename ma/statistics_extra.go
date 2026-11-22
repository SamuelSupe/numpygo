package ma

import (
	"slices"

	"github.com/SamuelSupe/numpygo"
)

func convolve(a, b *Array, mode np.SignalMode, propagate bool, op func(*np.Array, *np.Array, np.SignalMode) (*np.Array, error)) (*Array, error) {
	if !valid(a) || !valid(b) {
		return nil, np.ErrDType
	}
	if propagate {
		p, e := np.Ones(np.Bool, a.Shape()...)
		if e != nil {
			return nil, e
		}
		q, e := np.Ones(np.Bool, b.Shape()...)
		if e != nil {
			return nil, e
		}
		m1, e := op(a.mask, q, mode)
		if e != nil {
			return nil, e
		}
		m2, e := op(p, b.mask, mode)
		if e != nil {
			return nil, e
		}
		m, e := np.LogicalOr(m1, m2)
		if e != nil {
			return nil, e
		}
		d, e := op(a.data, b.data, mode)
		if e != nil {
			return nil, e
		}
		return New(d, m)
	}
	p, e := np.LogicalNot(a.mask)
	if e != nil {
		return nil, e
	}
	q, e := np.LogicalNot(b.mask)
	if e != nil {
		return nil, e
	}
	m, e := op(p, q, mode)
	if e != nil {
		return nil, e
	}
	m, e = np.LogicalNot(m)
	if e != nil {
		return nil, e
	}
	x, e := a.Filled(int64(0))
	if e != nil {
		return nil, e
	}
	y, e := b.Filled(int64(0))
	if e != nil {
		return nil, e
	}
	d, e := op(x, y, mode)
	if e != nil {
		return nil, e
	}
	return New(d, m)
}
func Convolve(a, b *Array, mode np.SignalMode, propagate bool) (*Array, error) {
	return convolve(a, b, mode, propagate, np.Convolve)
}
func Correlate(a, b *Array, mode np.SignalMode, propagate bool) (*Array, error) {
	return convolve(a, b, mode, propagate, np.Correlate)
}

// Cov centers each variable using all its valid observations, then computes
// the pairwise product sums and per-pair observation count minus ddof.
func Cov(x, y *Array, rowvar, allowMasked bool, ddof int) (*Array, error) {
	if !valid(x) || x.NDim() > 2 || x.DType().IsComplex() {
		return nil, np.ErrShape
	}
	if !allowMasked && IsMasked(x) {
		return nil, np.ErrValue
	}
	x, e := Atleast2D(x)
	if e != nil {
		return nil, e
	}
	xData, e := x.data.Astype(np.Float64)
	if e != nil {
		return nil, e
	}
	x = x.with(xData, x.mask)
	if x.Shape()[0] == 1 {
		rowvar = true
	}
	if y != nil {
		if !valid(y) || y.NDim() > 2 || y.DType().IsComplex() {
			return nil, np.ErrShape
		}
		if !allowMasked && IsMasked(y) {
			return nil, np.ErrValue
		}
		y, e = Atleast2D(y)
		if e != nil {
			return nil, e
		}
		yData, e := y.data.Astype(np.Float64)
		if e != nil {
			return nil, e
		}
		y = y.with(yData, y.mask)
		if slices.Equal(x.Shape(), y.Shape()) {
			m, e := np.LogicalOr(x.mask, y.mask)
			if e != nil {
				return nil, e
			}
			x = x.with(x.data, m)
			y = y.with(y.data, m)
		}
		axis := 0
		if !rowvar {
			axis = 1
		}
		x, e = Concatenate([]*Array{x, y}, axis)
		if e != nil {
			return nil, e
		}
	}
	if !rowvar {
		x, e = Transpose(x)
		if e != nil {
			return nil, e
		}
	}
	mu, e := Mean(x, np.SumOptions{Axes: []int{1}, KeepDims: true})
	if e != nil {
		return nil, e
	}
	centered, e := Subtract(x, mu)
	if e != nil {
		return nil, e
	}
	data, e := centered.Filled(0.)
	if e != nil {
		return nil, e
	}
	good, e := np.LogicalNot(x.mask)
	if e != nil {
		return nil, e
	}
	good, e = good.Astype(np.Float64)
	if e != nil {
		return nil, e
	}
	goodT, e := good.Transpose()
	if e != nil {
		return nil, e
	}
	count, e := np.Dot(good, goodT)
	if e != nil {
		return nil, e
	}
	fact, e := np.Subtract(count, np.Scalar(float64(ddof)))
	if e != nil {
		return nil, e
	}
	mask, e := np.LessEqual(fact, np.Scalar(0))
	if e != nil {
		return nil, e
	}
	t, e := data.Transpose()
	if e != nil {
		return nil, e
	}
	d, e := np.Dot(data, t)
	if e != nil {
		return nil, e
	}
	d, e = np.Divide(d, fact)
	if e != nil {
		return nil, e
	}
	a, e := New(d, mask)
	if e != nil {
		return nil, e
	}
	return Squeeze(a)
}
func Corrcoef(x, y *Array, rowvar, allowMasked bool) (*Array, error) {
	c, e := Cov(x, y, rowvar, allowMasked, 1)
	if e != nil {
		return nil, e
	}
	if c.NDim() < 2 {
		return MaskedAll(np.Float64)
	}
	d, e := Diagonal(c, 0, 0, 1)
	if e != nil {
		return nil, e
	}
	d, e = Sqrt(d)
	if e != nil {
		return nil, e
	}
	den, e := Outer(d, d)
	if e != nil {
		return nil, e
	}
	return Divide(c, den)
}

func ApplyAlongAxis(a *Array, axis int, fn func(*Array) (*Array, error)) (*Array, error) {
	if !valid(a) || fn == nil {
		return nil, np.ErrValue
	}
	ax, e := np.NormalizeAxis(axis, a.NDim())
	if e != nil {
		return nil, e
	}
	shape := a.Shape()
	outer := append(slices.Clone(shape[:ax]), shape[ax+1:]...)
	it, e := np.NDIndex(outer...)
	if e != nil {
		return nil, e
	}
	var results []*Array
	var dtype np.DType
	var resultShape []int
	var targets [][]int
	for idx := range it {
		index := make([]np.Index, a.NDim())
		for i := range index {
			if i < ax {
				index[i] = np.AtIndex(idx[i])
			} else if i == ax {
				index[i] = np.Slice{}
			} else {
				index[i] = np.AtIndex(idx[i-1])
			}
		}
		x, e := a.Index(index...)
		if e != nil {
			return nil, e
		}
		v, e := fn(x)
		if e != nil {
			return nil, e
		}
		if !valid(v) {
			return nil, np.ErrDType
		}
		if len(results) == 0 {
			resultShape = v.Shape()
			dtype = v.DType()
		} else {
			if !slices.Equal(resultShape, v.Shape()) {
				return nil, np.ErrShape
			}
			dtype, e = np.PromoteTypes(dtype, v.DType())
			if e != nil {
				return nil, e
			}
		}
		results = append(results, v)
		targets = append(targets, idx)
	}
	if len(results) == 0 {
		return nil, np.ErrValue
	}
	s := append(slices.Clone(shape[:ax]), resultShape...)
	s = append(s, shape[ax+1:]...)
	output, e := Zeros(dtype, s...)
	if e != nil {
		return nil, e
	}
	resultRank := len(resultShape)
	for j, idx := range targets {
		target := make([]np.Index, output.NDim())
		for i := range target {
			if i < ax {
				target[i] = np.AtIndex(idx[i])
			} else if i < ax+resultRank {
				target[i] = np.Slice{}
			} else {
				target[i] = np.AtIndex(idx[i-resultRank])
			}
		}
		if e = output.Assign(target, results[j]); e != nil {
			return nil, e
		}
	}
	return output, nil
}
func ApplyOverAxes(a *Array, axes []int, fn func(*Array, int) (*Array, error)) (*Array, error) {
	if !valid(a) || fn == nil {
		return nil, np.ErrValue
	}
	rank := a.NDim()
	for _, axis := range axes {
		ax, e := np.NormalizeAxis(axis, rank)
		if e != nil {
			return nil, e
		}
		v, e := fn(a, ax)
		if e != nil {
			return nil, e
		}
		if !valid(v) {
			return nil, np.ErrDType
		}
		if v.NDim() == rank-1 {
			v, e = ExpandDims(v, ax)
			if e != nil {
				return nil, e
			}
		} else if v.NDim() != rank {
			return nil, np.ErrShape
		}
		a = v
	}
	return a, nil
}
