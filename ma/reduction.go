package ma

import (
	"slices"

	"github.com/SamuelSupe/numpygo"
)

func Count(a *Array, options ...np.SumOptions) (*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	validMask, err := np.LogicalNot(a.mask)
	if err != nil {
		return nil, err
	}
	return np.CountNonzero(validMask, options...)
}
func CountMasked(a *Array, options ...np.SumOptions) (*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	return np.CountNonzero(a.mask, options...)
}
func reductionMask(a *Array, count *np.Array) (*np.Array, error) {
	if !a.explicitMask && !IsMasked(a) {
		return np.Zeros(np.Bool, count.Shape()...)
	}
	return np.Equal(count, np.ScalarOf(int64(0)))
}
func reduce(a *Array, fill any, fn func(*np.Array, ...np.SumOptions) (*np.Array, error), options []np.SumOptions) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	data, err := a.Filled(fill)
	if err != nil {
		return nil, err
	}
	result, err := fn(data, options...)
	if err != nil {
		return nil, err
	}
	count, err := Count(a, options...)
	if err != nil {
		return nil, err
	}
	mask, err := reductionMask(a, count)
	if err != nil {
		return nil, err
	}
	if result.NDim() == 0 {
		m, _ := mask.AtValue()
		if m.(bool) {
			return MaskedAll(np.Float64)
		}
	}
	out, err := New(result, mask)
	if err != nil {
		return nil, err
	}
	out.explicitMask = a.explicitMask || IsMasked(a)
	return out, nil
}
func Sum(a *Array, options ...np.SumOptions) (*Array, error) {
	return reduce(a, int64(0), np.Sum, options)
}
func Prod(a *Array, options ...np.SumOptions) (*Array, error) {
	return reduce(a, int64(1), np.Prod, options)
}
func Min(a *Array, options ...np.SumOptions) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	return reduce(a, MinimumFillValue(a.DType()), np.Min, options)
}
func Max(a *Array, options ...np.SumOptions) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	return reduce(a, MaximumFillValue(a.DType()), np.Max, options)
}
func All(a *Array, options ...np.SumOptions) (*Array, error) { return reduce(a, true, np.All, options) }
func Any(a *Array, options ...np.SumOptions) (*Array, error) {
	return reduce(a, false, np.Any, options)
}
func PTP(a *Array, options ...np.SumOptions) (*Array, error) {
	hi, err := Max(a, options...)
	if err != nil {
		return nil, err
	}
	lo, err := Min(a, options...)
	if err != nil {
		return nil, err
	}
	return Subtract(hi, lo)
}

func sumOption(options []np.SumOptions) (np.SumOptions, error) {
	if len(options) > 1 {
		return np.SumOptions{}, np.ErrAxis
	}
	if len(options) == 0 {
		return np.SumOptions{}, nil
	}
	return options[0], nil
}
func meanType(a *Array) np.DType {
	d := a.DType()
	if d.IsInteger() || d == np.Bool {
		return np.Float64
	}
	if a.explicitMask || IsMasked(a) {
		if d == np.Float32 {
			return np.Float64
		}
		if d == np.Complex64 {
			return np.Complex128
		}
	}
	return d
}
func Mean(a *Array, options ...np.SumOptions) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	opt, err := sumOption(options)
	if err != nil {
		return nil, err
	}
	if !a.explicitMask && !IsMasked(a) {
		data, err := np.Mean(a.data, options...)
		if err != nil {
			return nil, err
		}
		return New(data, nil)
	}
	d := meanType(a)
	if opt.DType != 0 {
		d = opt.DType
	}
	optSum := opt
	// Float16 means use a float32 accumulator, like NumPy.
	if a.DType() == np.Float16 && opt.DType == 0 {
		optSum.DType = np.Float32
	}
	numerator, err := Sum(a, optSum)
	if err != nil {
		return nil, err
	}
	count, err := Count(a, opt)
	if err != nil {
		return nil, err
	}
	num, err := numerator.data.Astype(d)
	if err != nil {
		return nil, err
	}
	den, err := count.Astype(d)
	if err != nil {
		return nil, err
	}
	data, err := np.Divide(num, den)
	if err != nil {
		return nil, err
	}
	data, err = data.Astype(d)
	if err != nil {
		return nil, err
	}
	return New(data, numerator.mask)
}
func Var(a *Array, options ...np.VarOptions) (*Array, error) { return variance(a, false, options) }
func Std(a *Array, options ...np.VarOptions) (*Array, error) { return variance(a, true, options) }
func variance(a *Array, sqrt bool, options []np.VarOptions) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	if len(options) > 1 {
		return nil, np.ErrAxis
	}
	var opt np.VarOptions
	if len(options) == 1 {
		opt = options[0]
	}
	if !a.explicitMask && !IsMasked(a) {
		var data *np.Array
		var err error
		if sqrt {
			data, err = np.Std(a.data, options...)
		} else {
			data, err = np.Var(a.data, options...)
		}
		if err != nil {
			return nil, err
		}
		return New(data, nil)
	}
	d := meanType(a)
	if d == np.Complex64 {
		d = np.Float32
	} else if d == np.Complex128 {
		d = np.Float64
	}
	if opt.DType != 0 {
		d = opt.DType
	}
	if opt.DType == 0 && d == np.Float16 {
		d = np.Float64
	}
	m, err := Mean(a, np.SumOptions{Axes: opt.Axes, KeepDims: true, DType: opt.DType})
	if err != nil {
		return nil, err
	}
	diff, err := np.Subtract(a.data, m.data)
	if err != nil {
		return nil, err
	}
	diff, err = np.Absolute(diff)
	if err != nil {
		return nil, err
	}
	squares, err := np.Multiply(diff, diff)
	if err != nil {
		return nil, err
	}
	arr, err := New(squares, a.mask)
	if err != nil {
		return nil, err
	}
	reduceOpt := np.SumOptions{Axes: opt.Axes, KeepDims: opt.KeepDims, DType: opt.DType}
	sum, err := Sum(arr, reduceOpt)
	if err != nil {
		return nil, err
	}
	count, err := Count(a, reduceOpt)
	if err != nil {
		return nil, err
	}
	n, err := count.Astype(np.Float64)
	if err != nil {
		return nil, err
	}
	den, err := np.Subtract(n, np.Scalar(opt.DDOF))
	if err != nil {
		return nil, err
	}
	bad, err := np.LessEqual(den, np.Scalar(0))
	if err != nil {
		return nil, err
	}
	mask, err := np.LogicalOr(sum.mask, bad)
	if err != nil {
		return nil, err
	}
	data, err := np.Divide(sum.data, den)
	if err != nil {
		return nil, err
	}
	if sqrt {
		data, err = np.Sqrt(data)
		if err != nil {
			return nil, err
		}
	}
	data, err = data.Astype(d)
	if err != nil {
		return nil, err
	}
	return New(data, mask)
}
func ArgMin(a *Array, options ...np.ArgOptions) (*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	v, err := a.Filled(MinimumFillValue(a.DType()))
	if err != nil {
		return nil, err
	}
	return np.ArgMin(v, options...)
}
func ArgMax(a *Array, options ...np.ArgOptions) (*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	v, err := a.Filled(MaximumFillValue(a.DType()))
	if err != nil {
		return nil, err
	}
	return np.ArgMax(v, options...)
}
func Cumsum(a *Array, axis *int, dtype np.DType) (*Array, error) {
	return cumulative(a, axis, dtype, false)
}
func Cumprod(a *Array, axis *int, dtype np.DType) (*Array, error) {
	return cumulative(a, axis, dtype, true)
}
func cumulative(a *Array, axis *int, dtype np.DType, prod bool) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	identity := int64(0)
	if prod {
		identity = 1
	}
	data, err := a.Filled(identity)
	if err != nil {
		return nil, err
	}
	if prod {
		data, err = np.Cumprod(data, axis, dtype)
	} else {
		data, err = np.Cumsum(data, axis, dtype)
	}
	if err != nil {
		return nil, err
	}
	mask, err := a.mask.Reshape(data.Shape()...)
	if err != nil {
		return nil, err
	}
	mask, err = mask.Copy()
	if err != nil {
		return nil, err
	}
	return a.with(data, mask), nil
}

// Median groups valid elements by their retained coordinates. It retains
// unmasked NaNs; only explicitly masked elements are removed.
func Median(a *Array, options ...np.SumOptions) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	opt, err := sumOption(options)
	if err != nil {
		return nil, err
	}
	if !IsMasked(a) {
		data, err := np.Median(a.data, opt)
		if err != nil {
			return nil, err
		}
		return New(data, nil)
	}
	shape := a.Shape()
	reduceAxis := make([]bool, len(shape))
	if opt.Axes == nil {
		for i := range reduceAxis {
			reduceAxis[i] = true
		}
	} else {
		for _, axis := range opt.Axes {
			ax, err := np.NormalizeAxis(axis, len(shape))
			if err != nil || reduceAxis[ax] {
				return nil, np.ErrAxis
			}
			reduceAxis[ax] = true
		}
	}
	var outShape []int
	for i, d := range shape {
		if !reduceAxis[i] {
			outShape = append(outShape, d)
		} else if opt.KeepDims {
			outShape = append(outShape, 1)
		}
	}
	n, err := np.ElementCount(np.Int64, outShape...)
	if err != nil {
		return nil, err
	}
	groups := make([][]int64, n)
	mask, _ := np.Values[bool](a.mask)
	for flat, m := range mask {
		if m {
			continue
		}
		index, stride := 0, 1
		v := flat
		for ax := len(shape) - 1; ax >= 0; ax-- {
			j := v % shape[ax]
			v /= shape[ax]
			if !reduceAxis[ax] {
				index += j * stride
				stride *= shape[ax]
			}
		}
		groups[index] = append(groups[index], int64(flat))
	}
	d := a.DType()
	if d.IsInteger() || d == np.Bool {
		d = np.Float64
	}
	out, err := np.Zeros(d, outShape...)
	if err != nil {
		return nil, err
	}
	outMask, err := np.Zeros(np.Bool, outShape...)
	if err != nil {
		return nil, err
	}
	flat, err := np.Ravel(a.data)
	if err != nil {
		return nil, err
	}
	flatOut, err := np.Ravel(out)
	if err != nil {
		return nil, err
	}
	flatMask, err := np.Ravel(outMask)
	if err != nil {
		return nil, err
	}
	for i, ix := range groups {
		if len(ix) == 0 {
			if err := flatMask.SetValue(true, i); err != nil {
				return nil, err
			}
			continue
		}
		indices, err := np.WrapSlice(ix, len(ix))
		if err != nil {
			return nil, err
		}
		values, err := np.Take(flat, indices, nil)
		if err != nil {
			return nil, err
		}
		median, err := np.Median(values)
		if err != nil {
			return nil, err
		}
		v, err := median.AtValue()
		if err != nil {
			return nil, err
		}
		if err = flatOut.SetValue(v, i); err != nil {
			return nil, err
		}
	}
	return New(out, outMask)
}

func Average(a, weights *Array, options ...np.SumOptions) (*Array, *Array, error) {
	if !valid(a) {
		return nil, nil, np.ErrDType
	}
	opt, err := sumOption(options)
	if err != nil {
		return nil, nil, err
	}
	if weights == nil {
		mean, err := Mean(a, opt)
		if err != nil {
			return nil, nil, err
		}
		count, err := Count(a, opt)
		if err != nil {
			return nil, nil, err
		}
		count, err = count.Astype(mean.DType())
		if err != nil {
			return nil, nil, err
		}
		total, err := New(count, mean.mask)
		return mean, total, err
	}
	if !valid(weights) {
		return nil, nil, np.ErrDType
	}
	if !slices.Equal(a.Shape(), weights.Shape()) {
		if len(opt.Axes) != 1 || weights.NDim() != 1 {
			return nil, nil, np.ErrShape
		}
		ax, err := np.NormalizeAxis(opt.Axes[0], a.NDim())
		if err != nil || weights.Size() != a.Shape()[ax] {
			return nil, nil, np.ErrShape
		}
		sh := make([]int, a.NDim())
		for i := range sh {
			sh[i] = 1
		}
		sh[ax] = weights.Size()
		weights, err = Reshape(weights, sh...)
		if err != nil {
			return nil, nil, err
		}
	}
	wdata, err := weights.data.BroadcastTo(a.Shape()...)
	if err != nil {
		return nil, nil, err
	}
	wmask, err := weights.mask.BroadcastTo(a.Shape()...)
	if err != nil {
		return nil, nil, err
	}
	mask, err := np.LogicalOr(a.mask, wmask)
	if err != nil {
		return nil, nil, err
	}
	w, err := New(wdata, mask)
	if err != nil {
		return nil, nil, err
	}
	x, err := New(a.data, mask)
	if err != nil {
		return nil, nil, err
	}
	d, err := np.PromoteTypes(a.DType(), weights.DType())
	if err != nil {
		return nil, nil, err
	}
	if a.DType().IsInteger() || a.DType() == np.Bool {
		d, err = np.PromoteTypes(d, np.Float64)
		if err != nil {
			return nil, nil, err
		}
	}
	if opt.DType != 0 {
		d = opt.DType
	}
	x.data, err = x.data.Astype(d)
	if err != nil {
		return nil, nil, err
	}
	w.data, err = w.data.Astype(d)
	if err != nil {
		return nil, nil, err
	}
	prod, err := Multiply(x, w)
	if err != nil {
		return nil, nil, err
	}
	num, err := Sum(prod, opt)
	if err != nil {
		return nil, nil, err
	}
	den, err := Sum(w, opt)
	if err != nil {
		return nil, nil, err
	}
	mean, err := Divide(num, den)
	return mean, den, err
}

func AllClose(a, b *Array, rtol, atol float64, maskedEqual bool) (bool, error) {
	if !valid(a) || !valid(b) {
		return false, np.ErrDType
	}
	close, err := np.IsClose(a.data, b.data, rtol, atol, false)
	if err != nil {
		return false, err
	}
	either, err := np.LogicalOr(a.mask, b.mask)
	if err != nil {
		return false, err
	}
	if maskedEqual {
		close, err = np.LogicalOr(close, either)
	} else {
		not, err2 := np.LogicalNot(either)
		if err2 != nil {
			return false, err2
		}
		close, err = np.LogicalAnd(close, not)
	}
	if err != nil {
		return false, err
	}
	all, err := np.All(close)
	if err != nil {
		return false, err
	}
	v, err := all.AtValue()
	if err != nil {
		return false, err
	}
	return v.(bool), nil
}
func AllEqual(a, b *Array, maskedEqual bool) (bool, error) {
	if !valid(a) || !valid(b) {
		return false, np.ErrDType
	}
	eq, err := np.Equal(a.data, b.data)
	if err != nil {
		return false, err
	}
	mask, err := np.LogicalOr(a.mask, b.mask)
	if err != nil {
		return false, err
	}
	if maskedEqual {
		eq, err = np.LogicalOr(eq, mask)
	} else {
		not, err2 := np.LogicalNot(mask)
		if err2 != nil {
			return false, err2
		}
		eq, err = np.LogicalAnd(eq, not)
	}
	if err != nil {
		return false, err
	}
	r, err := np.All(eq)
	if err != nil {
		return false, err
	}
	v, err := r.AtValue()
	if err != nil {
		return false, err
	}
	return v.(bool), nil
}

func Anomalies(a *Array, options ...np.SumOptions) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	opt, err := sumOption(options)
	if err != nil {
		return nil, err
	}
	opt.KeepDims = true
	mean, err := Mean(a, opt)
	if err != nil {
		return nil, err
	}
	return Subtract(a, mean)
}
