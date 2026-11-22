package np

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"
)

func meanDType(d DType) DType {
	if d.IsInteger() || d == Bool {
		return Float64
	}
	return d
}

func statistics(a *Array, options []SumOptions, dtype DType, f func(reductionLayout, int) (element, error)) (*Array, error) {
	r, _, err := reduction(a, options)
	if err != nil {
		return nil, err
	}
	out, err := Zeros(dtype, r.shape...)
	if err != nil {
		return nil, err
	}
	for i := 0; i < out.size; i++ {
		z, err := f(r, r.base(a, i))
		if err != nil {
			return nil, err
		}
		out.setElement(i, z)
	}
	return out, nil
}

func optionDType(options []SumOptions, defaultType DType) DType {
	if len(options) == 1 && options[0].DType != 0 {
		return options[0].DType
	}
	return defaultType
}

func Mean(a *Array, o ...SumOptions) (*Array, error)    { return mean(a, false, o) }
func NanMean(a *Array, o ...SumOptions) (*Array, error) { return mean(a, true, o) }
func mean(a *Array, skip bool, o []SumOptions) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialReduce(a, o, "mean", skip)
	}
	d := optionDType(o, meanDType(a.DType()))
	skip = skip && (a.DType().IsFloat() || a.DType().IsComplex())
	if skip && (d.IsInteger() || d == Bool) {
		return nil, ErrDType
	}
	acc := d
	if !skip && d == Float16 && (len(o) == 0 || o[0].DType == 0) {
		acc = Float32
	}
	r, _, err := reduction(a, o)
	if err != nil {
		return nil, err
	}
	out, err := Zeros(d, r.shape...)
	if err != nil {
		return nil, err
	}
	if !skip && acc == d && nativeRealReduction(a, out, r, true) {
		return out, nil
	}
	for i := 0; i < out.size; i++ {
		z, _ := groupMean(a, r, r.base(a, i), skip, acc)
		out.setElement(i, castElement(z, acc, d))
	}
	return out, nil
}
func groupMean(a *Array, r reductionLayout, base int, skip bool, d DType) (element, int) {
	n := r.count
	from := a.DType()
	z := sumGroup(r.count, func(j int) element {
		e := a.elementAt(r.offset(base, j))
		if skip && nanElement(e, from) {
			n--
			return element{}
		}
		return castElement(e, from, d)
	}, d, r.pairwise)
	return divideReduction(z, d, float64(n)), n
}

func divideReduction(z element, d DType, den float64) element {
	if d.IsExtended() {
		v, _ := extendedArithmetic(z, castElement(element{z: complex(den, 0)}, Float64, d), d, divOp)
		return v
	}
	v := z.number(d)
	if d.IsComplex() {
		return roundElement(element{z: complexDivide(v, complex(den, 0))}, d)
	}
	return castElement(element{z: complex(real(v)/den, 0)}, Float64, d)
}

// VarOptions supports NumPy's axis, dtype, keepdims and degrees-of-freedom correction.
type VarOptions struct {
	Axes             []int
	KeepDims         bool
	DType            DType
	DDOF             float64
	Correction       *float64
	Where, Mean, Out *Array
}

// varianceLayout keeps the supplied mean in keepdims shape, independent of
// whether the caller requests a squeezed result.
type varianceLayout struct {
	r, maskLayout reductionLayout
	mask, center  *Array
}

func prepareVariance(a *Array, opt VarOptions) (varianceLayout, error) {
	var p varianceLayout
	var e error
	p.r, _, e = reductionAny(a, []SumOptions{{Axes: opt.Axes, KeepDims: opt.KeepDims}})
	if e != nil {
		return p, e
	}
	if opt.Where != nil {
		if opt.Where.DType() != Bool {
			return p, ErrDType
		}
		p.mask, e = opt.Where.BroadcastTo(a.shape...)
		if e != nil {
			return p, e
		}
		p.maskLayout = p.r.companion(p.mask, opt.KeepDims)
	}
	if opt.Mean != nil {
		k, _, e := reductionAny(a, []SumOptions{{Axes: opt.Axes, KeepDims: true}})
		if e != nil {
			return p, e
		}
		p.center, e = opt.Mean.BroadcastTo(k.shape...)
		if e != nil {
			return p, e
		}
	}
	return p, nil
}
func (p varianceLayout) maskAt(i int) func(int) bool {
	if p.mask == nil {
		return func(int) bool { return true }
	}
	base := p.maskLayout.base(p.mask, i)
	return func(j int) bool { return p.mask.elementAt(p.maskLayout.offset(base, j)).i != 0 }
}
func (p varianceLayout) mean(a *Array, i int, skip bool, d DType) (element, int) {
	r := p.r
	base := r.base(a, i)
	if p.mask == nil && p.center == nil {
		return groupMean(a, r, base, skip, d)
	}
	keep := p.maskAt(i)
	n := r.count
	z := sumGroup(r.count, func(j int) element {
		v := a.elementAt(r.offset(base, j))
		if !keep(j) || skip && nanElement(v, a.DType()) {
			n--
			return element{}
		}
		return castElement(v, a.DType(), d)
	}, d, r.pairwise)
	if p.center != nil {
		v := p.center.elementAt(p.center.offset + flatOffset(i, p.center.shape, p.center.strides))
		return castElement(v, p.center.DType(), d), n
	}
	return divideReduction(z, d, float64(n)), n
}
func reductionOutput(out, result *Array) (*Array, error) {
	if out == nil {
		return result, nil
	}
	if !slices.Equal(out.shape, result.shape) {
		return nil, ErrShape
	}
	if e := CopyTo(out, result, nil); e != nil {
		return nil, e
	}
	return out, nil
}

func realDType(d DType) DType {
	if d == Complex256 {
		return Float128
	}
	if d == Complex64 {
		return Float32
	}
	if d == Complex128 {
		return Float64
	}
	return d
}
func Var(a *Array, o ...VarOptions) (*Array, error)    { return variance(a, false, false, o) }
func Std(a *Array, o ...VarOptions) (*Array, error)    { return variance(a, false, true, o) }
func NanVar(a *Array, o ...VarOptions) (*Array, error) { return variance(a, true, false, o) }
func NanStd(a *Array, o ...VarOptions) (*Array, error) { return variance(a, true, true, o) }
func variance(a *Array, skip, sqrt bool, options []VarOptions) (*Array, error) {
	if len(options) > 1 {
		return nil, ErrAxis
	}
	var opt VarOptions
	if len(options) == 1 {
		opt = options[0]
	}
	if opt.Correction != nil {
		if opt.DDOF != 0 {
			return nil, ErrValue
		}
		opt.DDOF = *opt.Correction
	}
	if a.DType().IsExtended() || opt.DType.IsExtended() {
		return extendedVariance(a, skip, sqrt, opt)
	}
	acc := meanDType(a.DType())
	d := realDType(acc)
	if opt.DType != 0 {
		d, acc = opt.DType, opt.DType
	}
	from := a.DType()
	skip = skip && (from.IsFloat() || from.IsComplex())
	if skip && (acc.IsInteger() || acc == Bool) {
		return nil, ErrDType
	}
	diffType, err := PromoteTypes(from, acc)
	if err != nil {
		return nil, err
	}
	if diffType == Bool {
		return nil, ErrDType
	}
	// nanvar subtracts in a copy of the input, while var uses a promoted
	// temporary. Each subtraction, square and sum rounds in that dtype.
	computeType := diffType
	if skip {
		diffType = from
	}
	magnitude := from.IsComplex() || from == Bool && diffType.IsComplex()
	squareType := diffType
	if magnitude {
		squareType = realDType(diffType)
	}
	p, err := prepareVariance(a, opt)
	if err != nil {
		return nil, err
	}
	r := p.r
	out, err := Zeros(d, r.shape...)
	if err != nil {
		return nil, err
	}
	if !skip && p.mask == nil && p.center == nil && acc == from && nativeRealVariance(a, out, r, opt.DDOF, sqrt) {
		return reductionOutput(opt.Out, out)
	}
	for i := 0; i < out.size; i++ {
		base := r.base(a, i)
		keep := p.maskAt(i)
		mean, n := p.mean(a, i, skip, acc)
		mu := castElement(mean, acc, computeType)
		ss := sumGroup(r.count, func(j int) element {
			e := a.elementAt(r.offset(base, j))
			if !keep(j) || skip && nanElement(e, from) {
				return element{}
			}
			x := castElement(e, from, computeType)
			var diff element
			if computeType.IsInteger() {
				diff.i = x.i - mu.i
			} else {
				diff.z = x.z - mu.z
			}
			diff = castElement(roundElement(diff, computeType), computeType, diffType)
			var sq element
			if magnitude {
				re := roundElement(element{z: complex(real(diff.z)*real(diff.z), 0)}, squareType)
				im := roundElement(element{z: complex(imag(diff.z)*imag(diff.z), 0)}, squareType)
				sq = roundElement(addScalar(re, im, squareType), squareType)
			} else {
				sq = roundElement(multiplyScalar(diff, diff, diffType), diffType)
			}
			return castElement(sq, squareType, d)
		}, d, r.pairwise)
		den := math.Max(float64(n)-opt.DDOF, 0)
		z := divideReduction(ss, d, den)
		if skip && den == 0 {
			z = castElement(element{z: complex(math.NaN(), math.NaN())}, Complex128, d)
		}
		if sqrt {
			if d.IsComplex() {
				z = roundElement(element{z: cmplx.Sqrt(z.z)}, d)
			} else {
				if (d.IsInteger() || d == Bool) && len(r.shape) > 0 {
					return nil, ErrDType
				}
				z = castElement(element{z: complex(math.Sqrt(real(z.number(d))), 0)}, Float64, d)
			}
		}
		out.setElement(i, z)
	}
	return reductionOutput(opt.Out, out)
}

func Prod(a *Array, o ...SumOptions) (*Array, error)    { return product(a, false, o) }
func NanProd(a *Array, o ...SumOptions) (*Array, error) { return product(a, true, o) }
func NanSum(a *Array, o ...SumOptions) (*Array, error)  { return skipSum(a, o) }
func product(a *Array, skip bool, o []SumOptions) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialReduce(a, o, "prod", skip)
	}
	d := optionDType(o, sumDType(a.DType()))
	return statistics(a, o, d, func(r reductionLayout, base int) (element, error) {
		acc := d
		if d == Float16 {
			acc = Float32
		}
		z := castElement(element{i: 1}, Int64, acc)
		for j := 0; j < r.count; j++ {
			e := a.elementAt(r.offset(base, j))
			if skip && nanElement(e, a.DType()) {
				continue
			}
			v := castElement(castElement(e, a.DType(), d), d, acc)
			z = roundElement(multiplyScalar(z, v, acc), acc)
		}
		return castElement(z, acc, d), nil
	})
}
func skipSum(a *Array, o []SumOptions) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialReduce(a, o, "sum", true)
	}
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if !a.DType().IsFloat() && !a.DType().IsComplex() {
		return Sum(a, o...)
	}
	b, err := a.Copy()
	if err != nil {
		return nil, err
	}
	b.walk(func(_, off int) {
		if nanElement(b.elementAt(off), b.DType()) {
			b.setElement(off, element{})
		}
	})
	return Sum(b, o...)
}
func Min(a *Array, o ...SumOptions) (*Array, error)    { return reduceExtrema(a, false, false, o) }
func Max(a *Array, o ...SumOptions) (*Array, error)    { return reduceExtrema(a, true, false, o) }
func NanMin(a *Array, o ...SumOptions) (*Array, error) { return reduceExtrema(a, false, true, o) }
func NanMax(a *Array, o ...SumOptions) (*Array, error) { return reduceExtrema(a, true, true, o) }
func reduceExtrema(a *Array, maximum, skip bool, o []SumOptions) (*Array, error) {
	if a.DType().IsSpecial() {
		kind := "min"
		if maximum {
			kind = "max"
		}
		return specialReduce(a, o, kind, skip)
	}
	d := a.DType()
	return statistics(a, o, d, func(r reductionLayout, base int) (element, error) {
		if r.count == 0 {
			return element{}, fmt.Errorf("%w: zero-size reduction without identity", ErrValue)
		}
		z := a.elementAt(base)
		for j := 1; j < r.count; j++ {
			z = selectExtrema(z, a.elementAt(r.offset(base, j)), d, maximum, skip)
		}
		return z, nil
	})
}

func All(a *Array, o ...SumOptions) (*Array, error) { return truthReduction(a, true, o) }
func Any(a *Array, o ...SumOptions) (*Array, error) { return truthReduction(a, false, o) }
func truthReduction(a *Array, all bool, o []SumOptions) (*Array, error) {
	if a.DType().IsSpecial() {
		kind := "any"
		if all {
			kind = "all"
		}
		return specialReduce(a, o, kind, false)
	}
	return statistics(a, o, Bool, func(r reductionLayout, base int) (element, error) {
		for j := 0; j < r.count; j++ {
			if a.elementAt(r.offset(base, j)).truth(a.DType()) != all {
				return boolElement(!all), nil
			}
		}
		return boolElement(all), nil
	})
}
func CountNonzero(a *Array, o ...SumOptions) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialReduce(a, o, "count", false)
	}
	return statistics(a, o, Int64, func(r reductionLayout, base int) (element, error) {
		var n uint64
		for j := 0; j < r.count; j++ {
			if a.elementAt(r.offset(base, j)).truth(a.DType()) {
				n++
			}
		}
		return element{i: n}, nil
	})
}

// ArgOptions uses nil Axis for C-order flattening.
type ArgOptions struct {
	Axis     *int
	KeepDims bool
}

func ArgMin(a *Array, o ...ArgOptions) (*Array, error)    { return argExtrema(a, false, false, o) }
func ArgMax(a *Array, o ...ArgOptions) (*Array, error)    { return argExtrema(a, true, false, o) }
func NanArgMin(a *Array, o ...ArgOptions) (*Array, error) { return argExtrema(a, false, true, o) }
func NanArgMax(a *Array, o ...ArgOptions) (*Array, error) { return argExtrema(a, true, true, o) }
func argExtrema(a *Array, maximum, skip bool, options []ArgOptions) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if len(options) > 1 {
		return nil, ErrAxis
	}
	var opt ArgOptions
	if len(options) == 1 {
		opt = options[0]
	}
	orig := a.shape
	if opt.Axis == nil {
		var err error
		a, err = a.Reshape(-1)
		if err != nil {
			return nil, err
		}
		opt.Axis = Int(0)
	}
	o := []SumOptions{{Axes: []int{*opt.Axis}, KeepDims: opt.KeepDims}}
	if a.DType() == VoidDType {
		return nil, ErrDType
	}
	calc := func(r reductionLayout, base int) (element, error) {
		best := -1
		var value element
		for j := 0; j < r.count; j++ {
			e := a.elementAt(r.offset(base, j))
			if a.DType() == StringDType {
				if r.count > 1 && stringBadOrder(e) {
					return element{}, ErrValue
				}
				if skip && stringNaN(e) {
					continue
				}
				if best < 0 || maximum && (sortLess(value, e, StringDType) || stringNaN(e)) || !maximum && sortLess(e, value, StringDType) {
					best = j
					value = e
				}
				continue
			}
			if nanElement(e, a.DType()) {
				if skip {
					continue
				}
				return element{i: uint64(j)}, nil
			}
			if best < 0 || maximum && lessElement(value, e, a.DType()) || !maximum && lessElement(e, value, a.DType()) {
				best = j
				value = e
			}
		}
		if best < 0 {
			return element{}, ErrValue
		}
		return element{i: uint64(best)}, nil
	}
	var out *Array
	var err error
	if a.DType().IsCharacter() || a.DType() == StringDType {
		r, _, e := reductionAny(a, o)
		if e != nil {
			return nil, e
		}
		out, e = Zeros(Int64, r.shape...)
		if e != nil {
			return nil, e
		}
		for i := 0; i < out.size; i++ {
			x, e := calc(r, r.base(a, i))
			if e != nil {
				return nil, e
			}
			out.setElement(i, x)
		}
	} else {
		out, err = statistics(a, o, Int64, calc)
	}
	if err != nil {
		return nil, err
	}
	if (len(options) == 0 || options[0].Axis == nil) && opt.KeepDims {
		shape := make([]int, len(orig))
		for i := range shape {
			shape[i] = 1
		}
		return out.Reshape(shape...)
	}
	return out, nil
}

func quantile(a *Array, q float64, skip, median bool, o []SumOptions) (*Array, error) {
	if !median {
		if len(o) > 1 {
			return nil, ErrAxis
		}
		var opts QuantileOptions
		if len(o) == 1 {
			opts.Axes = o[0].Axes
			opts.KeepDims = o[0].KeepDims
		}
		return weakQuantile(a, q, opts, skip)
	}
	if median && a.DType().IsComplex() {
		return complexMedian(a, skip, o)
	}
	if a.DType() == 0 || a.DType().IsComplex() {
		return nil, ErrDType
	}
	if math.IsNaN(q) || q < 0 || q > 1 {
		return nil, ErrValue
	}
	d := Float64
	if a.DType().IsFloat() {
		d = a.DType()
	}
	return statistics(a, o, d, func(r reductionLayout, base int) (element, error) {
		values := make([]float64, 0, r.count)
		for j := 0; j < r.count; j++ {
			x := real(a.elementAt(r.offset(base, j)).number(a.DType()))
			if math.IsNaN(x) {
				if skip {
					continue
				}
				return element{z: complex(x, 0)}, nil
			}
			values = append(values, x)
		}
		if len(values) == 0 {
			return element{z: complex(math.NaN(), 0)}, nil
		}
		sort.Float64s(values)
		p := q * float64(len(values)-1)
		i := int(p)
		f := p - float64(i)
		x := values[i]
		if i+1 < len(values) && f != 0 {
			if median {
				x = (x + values[i+1]) / 2
			} else {
				x += (values[i+1] - x) * f
			}
		}
		return element{z: complex(x, 0)}, nil
	})
}
func complexMedian(a *Array, skip bool, o []SumOptions) (*Array, error) {
	return statistics(a, o, a.DType(), func(r reductionLayout, base int) (element, error) {
		values := make([]element, 0, r.count)
		for j := 0; j < r.count; j++ {
			e := a.elementAt(r.offset(base, j))
			if nanElement(e, a.DType()) {
				if skip {
					continue
				}
				return e, nil
			}
			values = append(values, e)
		}
		if len(values) == 0 {
			return element{z: complex(math.NaN(), math.NaN())}, nil
		}
		sort.SliceStable(values, func(i, j int) bool { return sortLess(values[i], values[j], a.DType()) })
		n := len(values)
		z := values[n/2].z
		if n%2 == 0 {
			z = (values[n/2-1].z + z) / 2
		}
		return element{z: z}, nil
	})
}
func Median(a *Array, o ...SumOptions) (*Array, error)    { return quantile(a, .5, false, true, o) }
func NanMedian(a *Array, o ...SumOptions) (*Array, error) { return quantile(a, .5, true, true, o) }
func Quantile(a *Array, q float64, o ...SumOptions) (*Array, error) {
	return quantile(a, q, false, false, o)
}
func NanQuantile(a *Array, q float64, o ...SumOptions) (*Array, error) {
	return quantile(a, q, true, false, o)
}
func Percentile(a *Array, q float64, o ...SumOptions) (*Array, error) {
	return Quantile(a, q/100, o...)
}
func NanPercentile(a *Array, q float64, o ...SumOptions) (*Array, error) {
	return NanQuantile(a, q/100, o...)
}

func PTP(a *Array, o ...SumOptions) (*Array, error) {
	hi, err := Max(a, o...)
	if err != nil {
		return nil, err
	}
	lo, err := Min(a, o...)
	if err != nil {
		return nil, err
	}
	return Subtract(hi, lo)
}

func Cumsum(a *Array, axis *int, dtype DType) (*Array, error) {
	return cumulative(a, axis, dtype, false, false)
}
func Cumprod(a *Array, axis *int, dtype DType) (*Array, error) {
	return cumulative(a, axis, dtype, true, false)
}
func NanCumsum(a *Array, axis *int, dtype DType) (*Array, error) {
	return cumulative(a, axis, dtype, false, true)
}
func NanCumprod(a *Array, axis *int, dtype DType) (*Array, error) {
	return cumulative(a, axis, dtype, true, true)
}
func cumulative(a *Array, axis *int, dtype DType, product, skip bool) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if axis == nil {
		var err error
		a, err = a.Reshape(-1)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	ax, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, err
	}
	if dtype == 0 {
		dtype = sumDType(a.DType())
	}
	if a.DType().IsSpecial() || dtype.IsSpecial() {
		return specialCumulative(a, ax, dtype, product, skip)
	}
	out, err := Zeros(dtype, a.shape...)
	if err != nil {
		return nil, err
	}
	r, _, err := reduction(a, []SumOptions{{Axes: []int{ax}}})
	if err != nil {
		return nil, err
	}
	if a.size == 0 {
		return out, nil
	}
	step := out.strides[ax]
	for i := 0; i < a.size/a.shape[ax]; i++ {
		base := r.base(a, i)
		obase := flatOffset(i, r.shape, append(append([]int(nil), out.strides[:ax]...), out.strides[ax+1:]...))
		z := element{}
		if product {
			z = castElement(element{i: 1}, Int64, dtype)
		}
		for j := 0; j < r.count; j++ {
			e := a.elementAt(base + j*a.strides[ax])
			if !skip || !nanElement(e, a.DType()) {
				v := castElement(e, a.DType(), dtype)
				if dtype.IsInteger() || dtype == Bool {
					if product {
						z.i *= v.i
					} else if dtype == Bool {
						z.i |= v.i
					} else {
						z.i += v.i
					}
				} else {
					if product {
						if dtype.IsComplex() {
							z.z *= v.z
						} else {
							z.z = complex(real(z.z)*real(v.z), 0)
						}
					} else {
						z.z += v.z
					}
					switch dtype {
					case Float16:
						z.z = complex(HalfFromFloat64(real(z.z)).Float64(), 0)
					case Float32:
						z.z = complex(float64(float32(real(z.z))), 0)
					case Complex64:
						z.z = complex128(complex64(z.z))
					}
				}
			}
			out.setElement(obase+j*step, z)
		}
	}
	return out, nil
}

// Sort and Argsort use a stable ordering within each axis. Nil axis flattens.
func Sort(a *Array, axis *int) (*Array, error)    { return sortAxis(a, axis, false, nil, true) }
func Argsort(a *Array, axis *int) (*Array, error) { return sortAxis(a, axis, true, nil, true) }

type SortOptions struct {
	Order  []string // structured field priority
	Kind   string
	Stable *bool
}

func sortOptions(opt SortOptions) (bool, error) {
	if opt.Kind != "" && opt.Stable != nil {
		return false, ErrValue
	}
	switch opt.Kind {
	case "", "stable", "mergesort":
		return opt.Stable == nil || *opt.Stable, nil
	case "quicksort", "heapsort":
		return false, nil
	default:
		return false, ErrValue
	}
}

// Explicit unstable algorithms do not promise any order among equal keys.
func SortWithOptions(a *Array, axis *int, opt SortOptions) (*Array, error) {
	s, e := sortOptions(opt)
	if e != nil {
		return nil, e
	}
	return sortAxis(a, axis, false, opt.Order, s)
}
func ArgsortWithOptions(a *Array, axis *int, opt SortOptions) (*Array, error) {
	s, e := sortOptions(opt)
	if e != nil {
		return nil, e
	}
	return sortAxis(a, axis, true, opt.Order, s)
}
func sortAxis(a *Array, axis *int, indices bool, fields []string, stable bool) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if axis == nil {
		var err error
		a, err = a.Reshape(-1)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	ax, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, err
	}
	cmp, err := arrayOrder(a, fields)
	if err != nil {
		return nil, err
	}
	d := a.DType()
	if indices {
		d = Int64
	}
	var out *Array
	if indices {
		out, err = Zeros(d, a.shape...)
	} else {
		out, err = a.zerosLike(a.shape)
	}
	if err != nil {
		return nil, err
	}
	if a.size == 0 {
		return out, nil
	}
	r, _, _ := reductionAny(a, []SumOptions{{Axes: []int{ax}}})
	os := append(append([]int(nil), out.strides[:ax]...), out.strides[ax+1:]...)
	for i := 0; i < a.size/a.shape[ax]; i++ {
		base := r.base(a, i)
		obase := flatOffset(i, r.shape, os)
		order := make([]int, r.count)
		for j := range order {
			if a.DType() == StringDType && r.count > 1 && stringBadOrder(a.elementAt(base+j*a.strides[ax])) {
				return nil, ErrValue
			}
			order[j] = j
		}
		less := func(i, j int) bool {
			return cmp.less(base+order[i]*a.strides[ax], base+order[j]*a.strides[ax])
		}
		if stable {
			sort.SliceStable(order, less)
		} else {
			sort.Slice(order, less)
		}
		for j, k := range order {
			v := element{i: uint64(k)}
			if !indices {
				v = a.elementAt(base + k*a.strides[ax])
			}
			out.setElement(obase+j*out.strides[ax], v)
		}
	}
	return out, nil
}

// Cov computes unweighted covariance with variables in rows and observations in columns.
func Cov(a *Array, ddof float64) (*Array, error) {
	if !a.DType().IsNativeNumeric() {
		return nil, ErrDType
	}
	if a.NDim() != 1 && a.NDim() != 2 {
		return nil, ErrShape
	}
	rows, n := 1, a.shape[0]
	if a.NDim() == 2 {
		rows, n = a.shape[0], a.shape[1]
	}
	d := Float64
	if a.DType().IsComplex() {
		d = Complex128
	}
	shape := []int{rows, rows}
	if rows == 1 {
		shape = nil
	}
	out, err := Zeros(d, shape...)
	if err != nil {
		return nil, err
	}
	v, _ := a.Complex128s()
	means := make([]complex128, rows)
	for row := 0; row < rows; row++ {
		for _, z := range v[row*n : (row+1)*n] {
			means[row] += z
		}
		means[row] /= complex(float64(n), 0)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < rows; j++ {
			var z complex128
			for k := 0; k < n; k++ {
				z += (v[i*n+k] - means[i]) * cmplx.Conj(v[j*n+k]-means[j])
			}
			z /= complex(math.Max(float64(n)-ddof, 0), 0)
			out.setElement(i*rows+j, castElement(element{z: z}, Complex128, d))
		}
	}
	return out, nil
}

func Corrcoef(a *Array) (*Array, error) {
	c, err := Cov(a, 1)
	if err != nil {
		return nil, err
	}
	if c.NDim() == 0 {
		z := c.elementAt(0).number(c.DType())
		if c.DType().IsComplex() {
			return ComplexScalar(z / z), nil
		}
		return Scalar(real(z / z)), nil
	}
	n := c.shape[0]
	diagonal := make([]float64, n)
	for i := range diagonal {
		diagonal[i] = math.Sqrt(real(c.elementAt(i*n + i).number(c.DType())))
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			z := c.elementAt(i*n+j).number(c.DType()) / complex(diagonal[i]*diagonal[j], 0)
			z = complex(math.Max(-1, math.Min(1, real(z))), math.Max(-1, math.Min(1, imag(z))))
			c.setElement(i*n+j, castElement(element{z: z}, Complex128, c.DType()))
		}
	}
	return c, nil
}
