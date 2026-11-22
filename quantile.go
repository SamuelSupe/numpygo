package np

import (
	"math"
	"slices"
	"sort"
)

// QuantileOptions supports NumPy's 13 sample quantile methods. Weights are
// accepted only for inverted_cdf; their shape must match a or the reduced axes.
// The shape of q is prepended to the reduced output shape.
type QuantileOptions struct {
	Axes         []int
	KeepDims     bool
	Method       string
	Weights, Out *Array
}

func QuantileArray(a, q *Array, opt QuantileOptions) (*Array, error) {
	return quantileArray(a, q, opt, false)
}
func NanQuantileArray(a, q *Array, opt QuantileOptions) (*Array, error) {
	return quantileArray(a, q, opt, true)
}
func PercentileArray(a, q *Array, opt QuantileOptions) (*Array, error) {
	return percentileArray(a, q, opt, false)
}
func NanPercentileArray(a, q *Array, opt QuantileOptions) (*Array, error) {
	return percentileArray(a, q, opt, true)
}
func percentileArray(a, q *Array, opt QuantileOptions, skip bool) (*Array, error) {
	if q.DType() == 0 || q.DType().IsComplex() {
		return nil, ErrDType
	}
	d := q.DType()
	if !d.IsFloat() {
		d = Float64
	}
	q, e := q.Astype(d)
	if e != nil {
		return nil, e
	}
	v, e := Full(d, 100.)
	if e != nil {
		return nil, e
	}
	q, e = Divide(q, v)
	if e != nil {
		return nil, e
	}
	return quantileArray(a, q, opt, skip)
}
func weakQuantile(a *Array, q float64, opt QuantileOptions, skip bool) (*Array, error) {
	d := Float64
	if a.DType().IsFloat() {
		d = a.DType()
	}
	// Validate prior to converting a Python-style weak scalar to low precision.
	if math.IsNaN(q) || q < 0 || q > 1 {
		return nil, ErrValue
	}
	v, e := Full(d, q)
	if e != nil {
		return nil, e
	}
	return quantileArray(a, v, opt, skip)
}
func QuantileWithOptions(a *Array, q float64, opt QuantileOptions) (*Array, error) {
	return weakQuantile(a, q, opt, false)
}
func NanQuantileWithOptions(a *Array, q float64, opt QuantileOptions) (*Array, error) {
	return weakQuantile(a, q, opt, true)
}

func quantileMethod(method string) (string, bool, error) {
	if method == "" {
		method = "linear"
	}
	switch method {
	case "inverted_cdf", "closest_observation", "lower", "higher", "nearest":
		return method, true, nil
	case "averaged_inverted_cdf", "interpolated_inverted_cdf", "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased", "midpoint":
		return method, false, nil
	default:
		return "", false, ErrValue
	}
}

type quantileSample struct {
	value  element
	weight float64
}

func quantileWeights(a, w *Array, axes []int) (*Array, error) {
	if w.DType() == 0 || w.DType().IsComplex() {
		return nil, ErrDType
	}
	if slices.Equal(w.shape, a.shape) {
		return w, nil
	}
	if axes == nil {
		return nil, ErrDType
	}
	if len(axes) != w.NDim() {
		return nil, ErrShape
	}
	shape := make([]int, a.NDim())
	for i := range shape {
		shape[i] = 1
	}
	strides := make([]int, a.NDim())
	for i, axis := range axes {
		ax, e := NormalizeAxis(axis, a.NDim())
		if e != nil {
			return nil, e
		}
		if shape[ax] != 1 || w.shape[i] != a.shape[ax] {
			return nil, ErrShape
		}
		shape[ax], strides[ax] = w.shape[i], w.strides[i]
	}
	v := &Array{data: w.data, shape: shape, strides: strides, size: w.size, offset: w.offset, readOnly: true, base: w.viewBase()}
	return v.BroadcastTo(a.shape...)
}

func quantileArray(a, q *Array, opt QuantileOptions, skip bool) (*Array, error) {
	ad, qd := a.DType(), q.DType()
	if ad == 0 || qd == 0 || ad.IsComplex() || qd.IsComplex() {
		return nil, ErrDType
	}
	method, discrete, e := quantileMethod(opt.Method)
	if e != nil {
		return nil, e
	}
	if opt.Weights != nil && method != "inverted_cdf" {
		return nil, ErrValue
	}
	if method == "linear" && (qd.IsInteger() || qd == Bool) {
		discrete = true
	}
	if ad == Bool && !discrete {
		return nil, ErrDType
	}
	qs := make([]float64, q.size)
	q.walk(func(i, off int) { qs[i] = real(q.elementAt(off).number(qd)) })
	for _, x := range qs {
		if math.IsNaN(x) || x < 0 || x > 1 {
			return nil, ErrValue
		}
	}
	r, _, e := reduction(a, []SumOptions{{Axes: opt.Axes, KeepDims: opt.KeepDims}})
	if e != nil {
		return nil, e
	}
	// NumPy's nanquantile delegates a globally empty input to nanmean;
	// the quantile dimensions are not prepended in that special case.
	if skip && a.size == 0 {
		v, e := NanMean(a, SumOptions{Axes: opt.Axes, KeepDims: opt.KeepDims})
		if e != nil {
			return nil, e
		}
		if opt.Out == nil {
			return v, nil
		}
		if !slices.Equal(opt.Out.shape, v.shape) {
			return nil, ErrShape
		}
		if e = CopyTo(opt.Out, v, nil); e != nil {
			return nil, e
		}
		return opt.Out, nil
	}
	d := ad
	if !qd.IsFloat() {
		qd = Float64
	}
	if !discrete {
		d, e = PromoteTypes(ad, qd)
		if e != nil {
			return nil, e
		}
	}
	calcD := d
	groups, e := shapeSize(d, r.shape)
	if e != nil {
		return nil, e
	}
	// The nanquantile lane routine returns the input dtype for an all-NaN
	// first lane. The outer apply-along-axis allocation follows that lane.
	if skip && ad.IsFloat() && groups > 0 && r.count > 0 {
		allNaN := true
		for j := 0; j < r.count; j++ {
			if !nanElement(a.elementAt(r.offset(r.base(a, 0), j)), ad) {
				allNaN = false
				break
			}
		}
		if allNaN {
			d = ad
		}
	} else if !skip && ad.IsFloat() && q.NDim() == 0 && len(r.shape) == 0 && opt.Out == nil {
		for j := 0; j < r.count; j++ {
			if nanElement(a.elementAt(r.offset(r.base(a, 0), j)), ad) {
				d = ad
				break
			}
		}
	}
	shape := append(slices.Clone(q.shape), r.shape...)
	out, e := Zeros(d, shape...)
	if e != nil {
		return nil, e
	}
	if opt.Out != nil {
		if !opt.Out.Writable() {
			return nil, ErrReadOnly
		}
		if !slices.Equal(opt.Out.shape, shape) {
			return nil, ErrShape
		}
		ok, e := CanCast(d, opt.Out.DType(), "same_kind")
		if e != nil || !ok {
			return nil, ErrDType
		}
	}
	var weights *Array
	var wr reductionLayout
	if opt.Weights != nil {
		weights, e = quantileWeights(a, opt.Weights, opt.Axes)
		if e != nil {
			return nil, e
		}
		bad := false
		weights.walk(func(_, off int) {
			if real(weights.elementAt(off).number(weights.DType())) < 0 {
				bad = true
			}
		})
		if bad {
			return nil, ErrValue
		}
		wr = r.companion(weights, opt.KeepDims)
	}
	if r.count == 0 && !skip && groups > 0 {
		return nil, ErrIndex
	}
	samples := make([]quantileSample, 0, r.count)
	for g := 0; g < groups; g++ {
		samples = samples[:0]
		base := r.base(a, g)
		wbase := 0
		if weights != nil {
			wbase = wr.base(weights, g)
		}
		hasNaN := false
		for j := 0; j < r.count; j++ {
			v := a.elementAt(r.offset(base, j))
			if nanElement(v, ad) {
				if skip {
					continue
				}
				hasNaN = true
			}
			item := quantileSample{value: v}
			if weights != nil {
				item.weight = real(weights.elementAt(wr.offset(wbase, j)).number(weights.DType()))
			}
			samples = append(samples, item)
		}
		sort.SliceStable(samples, func(i, j int) bool { return sortLess(samples[i].value, samples[j].value, ad) })
		var cdf []float64
		if weights != nil && len(samples) > 0 {
			cdf = make([]float64, len(samples))
			total := 0.
			for i, s := range samples {
				total += s.weight
				cdf[i] = total
			}
			if total <= 0 || math.IsInf(total, 0) || math.IsNaN(total) {
				return nil, ErrValue
			}
			for i, c := range cdf {
				c = c / total
				if qd.IsFloat() {
					c = quantileRound(c, qd)
				}
				if c == 0 {
					c = -1
				}
				cdf[i] = c
			}
		}
		for k, x := range qs {
			var v element
			switch {
			case hasNaN || len(samples) == 0:
				v = castElement(element{z: complex(math.NaN(), 0)}, Float64, calcD)
			case weights != nil:
				j := sort.SearchFloat64s(cdf, x)
				v = samples[min(j, len(samples)-1)].value
			default:
				v = sampleQuantile(samples, x, method, discrete, ad, qd, calcD)
			}
			out.setElement(k*groups+g, castElement(v, calcD, d))
		}
	}
	if opt.Out != nil {
		if e = CopyTo(opt.Out, out, nil); e != nil {
			return nil, e
		}
		return opt.Out, nil
	}
	return out, nil
}

func quantileRound(v float64, d DType) float64 {
	return real(roundElement(element{z: complex(v, 0)}, d).z)
}
func sampleQuantile(s []quantileSample, q float64, method string, discrete bool, ad, qd, d DType) element {
	n := float64(len(s))
	// Virtual indices are computed in the precision of q. Their integer
	// selections retain the original dtype, including exact uint64 values.
	r := func(v float64) float64 { return quantileRound(v, qd) }
	p := r((n - 1) * q)
	switch method {
	case "inverted_cdf":
		p = math.Ceil(r(r(n*q) - 1))
	case "closest_observation":
		p = r(r(r(n*q)-1) - .5)
		i := math.Floor(p)
		if p == i && int64(i)%2 != 0 {
			p = i
		} else {
			p = i + 1
		}
	case "lower":
		p = math.Floor(p)
	case "higher":
		p = math.Ceil(p)
	case "nearest":
		p = math.RoundToEven(p)
	case "averaged_inverted_cdf":
		p = r(r(n*q) - 1)
	case "interpolated_inverted_cdf":
		p = r(r(n*q) - 1)
	case "hazen":
		p = r(r(r(n*q)+.5) - 1)
	case "weibull":
		p = r(r(r(n*q)+q) - 1)
	case "median_unbiased":
		p = r(r(r(n*q)+r(1./3+r(q*(1./3)))) - 1)
	case "normal_unbiased":
		p = r(r(r(n*q)+r(3./8+r(q*.25))) - 1)
	case "midpoint":
		p = r(.5 * r(math.Floor(p)+math.Ceil(p)))
	}
	clamp := func(i float64) int { return int(max(0., min(n-1, i))) }
	if discrete {
		return s[clamp(p)].value
	}
	// Follow numpy's two-sided lerp to reduce cancellation. Subtraction
	// occurs in the sample dtype before promotion by the interpolation weight.
	prev, next := math.Floor(p), math.Floor(p)+1
	if p < 0 {
		prev, next = 0, 0
	}
	if p >= n-1 {
		prev, next = n-1, n-1
	}
	gamma := p - math.Floor(p)
	if method == "averaged_inverted_cdf" {
		if gamma == 0 {
			gamma = .5
		} else {
			gamma = 1
		}
	}
	if method == "midpoint" && gamma != 0 {
		gamma = .5
	}
	x, y := s[clamp(prev)].value, s[clamp(next)].value
	var diff element
	if ad.IsInteger() {
		diff = roundElement(element{i: y.i - x.i}, ad)
	} else {
		diff = roundElement(element{z: y.z - x.z}, ad)
	}
	a, b := real(castElement(x, ad, d).z), real(castElement(y, ad, d).z)
	dv := real(castElement(diff, ad, d).z)
	gamma = r(gamma)
	if gamma >= .5 {
		return roundElement(element{z: complex(b-quantileRound(dv*r(1-gamma), d), 0)}, d)
	}
	return roundElement(element{z: complex(a+quantileRound(dv*gamma, d), 0)}, d)
}
