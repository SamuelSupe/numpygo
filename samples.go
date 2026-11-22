package np

import (
	"math"
	"slices"
	"sort"
)

// Average computes an unweighted or weighted mean. Weights may match a's shape
// or be a one-dimensional array along the sole reduction axis.
func Average(a, weights *Array, options ...SumOptions) (*Array, error) {
	avg, _, err := AverageWithWeights(a, weights, options...)
	return avg, err
}
func AverageWithWeights(a, weights *Array, options ...SumOptions) (*Array, *Array, error) {
	r, opt, err := reduction(a, options)
	if err != nil {
		return nil, nil, err
	}
	if weights == nil {
		avg, err := Mean(a, options...)
		if err != nil {
			return nil, nil, err
		}
		sum, err := Full(avg.DType(), float64(r.count), avg.shape...)
		return avg, sum, err
	}
	if weights.DType() == 0 {
		return nil, nil, ErrDType
	}
	if !slices.Equal(a.shape, weights.shape) {
		if opt.Axes == nil || len(opt.Axes) != weights.NDim() {
			return nil, nil, ErrShape
		}
		sh := make([]int, a.NDim())
		strides := make([]int, a.NDim())
		for i := range sh {
			sh[i] = 1
		}
		for i, axis := range opt.Axes {
			ax, e := NormalizeAxis(axis, a.NDim())
			if e != nil || weights.shape[i] != a.shape[ax] {
				return nil, nil, ErrShape
			}
			sh[ax], strides[ax] = weights.shape[i], weights.strides[i]
		}
		weights = &Array{data: weights.data, shape: sh, strides: strides, offset: weights.offset, size: weights.size, readOnly: true, base: weights.viewBase()}
	}
	weights, err = weights.BroadcastTo(a.shape...)
	if err != nil {
		return nil, nil, err
	}
	d, err := PromoteTypes(a.DType(), weights.DType())
	if err != nil {
		return nil, nil, err
	}
	if a.DType().IsInteger() || a.DType() == Bool {
		d, err = PromoteTypes(d, Float64)
		if err != nil {
			return nil, nil, err
		}
	}
	if d.IsInteger() || d == Bool {
		d = Float64
	}
	if opt.DType != 0 {
		d = opt.DType
	}
	a, err = a.Astype(d)
	if err != nil {
		return nil, nil, err
	}
	weights, err = weights.Astype(d)
	if err != nil {
		return nil, nil, err
	}
	opt.DType = d
	den, err := Sum(weights, opt)
	if err != nil {
		return nil, nil, err
	}
	zero := false
	den.walk(func(_, off int) {
		if !den.elementAt(off).truth(d) {
			zero = true
		}
	})
	if zero {
		return nil, nil, ErrValue
	}
	num, err := Multiply(a, weights)
	if err != nil {
		return nil, nil, err
	}
	num, err = Sum(num, opt)
	if err != nil {
		return nil, nil, err
	}
	out, err := Divide(num, den)
	return out, den, err
}

// Bincount counts nonnegative integer values, optionally accumulating real weights.
func Bincount(a, weights *Array, minlength int) (*Array, error) {
	if !a.DType().IsInteger() && a.DType() != Bool {
		return nil, ErrDType
	}
	if a.NDim() != 1 || minlength < 0 {
		return nil, ErrShape
	}
	d := Int64
	if weights != nil {
		if weights.DType() == 0 || weights.DType().IsComplex() {
			return nil, ErrDType
		}
		if weights.NDim() != 1 || weights.size != a.size {
			return nil, ErrShape
		}
		d = Float64
	}
	indices := make([]int, a.size)
	n := minlength
	for i := range indices {
		v := a.elementAt(a.offset + i*a.strides[0])
		if a.DType().IsSigned() && int64(v.i) < 0 || v.i >= uint64(maxInt/8) {
			return nil, ErrValue
		}
		indices[i] = int(v.i)
		n = max(n, indices[i]+1)
	}
	out, err := Zeros(d, n)
	if err != nil {
		return nil, err
	}
	for i, j := range indices {
		if weights == nil {
			out.setElement(j, element{i: out.elementAt(j).i + 1})
		} else {
			v := real(weights.elementAt(weights.offset + i*weights.strides[0]).number(weights.DType()))
			out.data.real[j] += v
		}
	}
	return out, nil
}

type HistogramOptions struct {
	Count   int
	Edges   []float64
	Range   *[2]float64
	Weights *Array
	Density bool
}

func Histogram(a *Array, opt HistogramOptions) (hist, edges *Array, err error) {
	if !a.DType().IsNativeNumeric() || a.DType().IsComplex() {
		return nil, nil, ErrDType
	}
	if opt.Weights != nil && (!slices.Equal(a.shape, opt.Weights.shape) || opt.Weights.DType() == 0) {
		return nil, nil, ErrShape
	}
	data, _ := a.Complex128s()
	bins := slices.Clone(opt.Edges)
	if opt.Edges == nil {
		n := opt.Count
		if n == 0 {
			n = 10
		}
		if n < 1 || n == maxInt {
			return nil, nil, ErrValue
		}
		low, high := 0., 1.
		if opt.Range != nil {
			low, high = opt.Range[0], opt.Range[1]
		} else if len(data) > 0 {
			low, high = real(data[0]), real(data[0])
			for _, z := range data[1:] {
				low = math.Min(low, real(z))
				high = math.Max(high, real(z))
			}
		}
		if math.IsNaN(low) || math.IsNaN(high) || math.IsInf(low, 0) || math.IsInf(high, 0) || low > high {
			return nil, nil, ErrValue
		}
		if low == high {
			low -= .5
			high += .5
		}
		if _, err := shapeSize(Float64, []int{n + 1}); err != nil {
			return nil, nil, err
		}
		bins = make([]float64, n+1)
		for i := range bins {
			bins[i] = low + (high-low)*float64(i)/float64(n)
		}
		bins[n] = high
	}
	if len(bins) < 2 {
		return nil, nil, ErrValue
	}
	for i := 1; i < len(bins); i++ {
		if bins[i] < bins[i-1] || math.IsNaN(bins[i]) || math.IsNaN(bins[i-1]) {
			return nil, nil, ErrValue
		}
	}
	d := Int64
	var weights *Array
	if opt.Weights != nil {
		d = opt.Weights.DType()
		weights, err = opt.Weights.Copy()
		if err != nil {
			return nil, nil, err
		}
	}
	hist, err = Zeros(d, len(bins)-1)
	if err != nil {
		return nil, nil, err
	}
	for i, z := range data {
		v := real(z)
		if math.IsNaN(v) || v < bins[0] || v > bins[len(bins)-1] {
			continue
		}
		j := sort.Search(len(bins), func(k int) bool { return bins[k] > v }) - 1
		if j == hist.size {
			j--
		}
		if weights == nil {
			hist.setElement(j, element{i: hist.elementAt(j).i + 1})
		} else {
			hist.setElement(j, addScalar(hist.elementAt(j), weights.elementAt(i), d))
		}
	}
	edges, err = WrapFloat64(bins, len(bins))
	if err != nil {
		return nil, nil, err
	}
	if opt.Edges == nil && (a.DType() == Float16 || a.DType() == Float32) {
		edges, err = edges.Astype(a.DType())
		if err != nil {
			return nil, nil, err
		}
	}
	if opt.Density {
		vals, _ := hist.Complex128s()
		var total complex128
		for _, v := range vals {
			total += v
		}
		for i := range vals {
			vals[i] /= total * complex(bins[i+1]-bins[i], 0)
		}
		hist, err = WrapComplex128(vals, len(vals))
		if err != nil {
			return nil, nil, err
		}
		if !d.IsComplex() {
			hist, err = hist.Astype(Float64)
		}
	}
	return hist, edges, err
}

// Gradient approximates a derivative with uniform scalar spacing along one axis.
func Gradient(a *Array, axis int, spacing float64, edgeOrder int) (*Array, error) {
	return gradientAxis(a, axis, Scalar(spacing), edgeOrder)
}

// GradientAxes accepts scalar spacing or one-dimensional coordinates for
// each requested axis. Nil axes means all axes; nil spacings means unit spacing.
func GradientAxes(a *Array, axes []int, spacings []*Array, edgeOrder int) ([]*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if axes == nil {
		axes = make([]int, a.NDim())
		for i := range axes {
			axes[i] = i
		}
	}
	if len(spacings) != 0 && len(spacings) != 1 && len(spacings) != len(axes) {
		return nil, ErrShape
	}
	result := make([]*Array, len(axes))
	seen := make(map[int]bool, len(axes))
	for i, axis := range axes {
		ax, e := NormalizeAxis(axis, a.NDim())
		if e != nil {
			return nil, e
		}
		if seen[ax] {
			return nil, ErrAxis
		}
		seen[ax] = true
		s := Scalar(1.)
		if len(spacings) == 1 {
			s = spacings[0]
		} else if len(spacings) > 1 {
			s = spacings[i]
		}
		if s.DType() == 0 {
			return nil, ErrDType
		}
		if len(spacings) == 1 && len(axes) > 1 && s.NDim() != 0 {
			return nil, ErrShape
		}
		result[i], e = gradientAxis(a, ax, s, edgeOrder)
		if e != nil {
			return nil, e
		}
	}
	return result, nil
}

func gradientAxis(a *Array, axis int, spacing *Array, edgeOrder int) (*Array, error) {
	if a.DType() == 0 || spacing.DType() == 0 ||
		!a.DType().IsNativeNumeric() && !a.DType().IsExtended() && !a.DType().IsTemporal() ||
		!spacing.DType().IsInteger() && !spacing.DType().IsFloat() {
		return nil, ErrDType
	}
	ax, err := NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	n := a.shape[ax]
	if edgeOrder != 1 && edgeOrder != 2 || n < edgeOrder+1 {
		return nil, ErrValue
	}
	d := meanDType(a.DType())
	temporal := d.IsTemporal()
	if temporal {
		d = TimeDelta64
	}
	work := Complex128
	if a.DType().IsExtended() || spacing.DType().IsExtended() || temporal {
		work = Float128
		if a.DType().IsComplex() {
			work = Complex256
		}
	}
	var out *Array
	if temporal {
		n, e := shapeSize(TimeDelta64, a.shape)
		if e != nil {
			return nil, e
		}
		store := makeStorage(TimeDelta64, n)
		store.unit = a.data.unit
		out = owned(store, a.shape, n)
	} else {
		out, err = Zeros(d, a.shape...)
	}
	if err != nil {
		return nil, err
	}
	if a.size == 0 {
		return out, nil
	}
	r, _, err := reductionAny(a, []SumOptions{{Axes: []int{ax}}})
	if err != nil {
		return nil, err
	}
	if a.DType() == Float64 && a.data.buffer == nil && a.data.part == nil &&
		spacing.NDim() == 0 && !spacing.DType().IsExtended() {
		h := real(spacing.elementAt(spacing.offset).number(spacing.DType()))
		gradientUniformFloat64(a, out, r, ax, n, h, edgeOrder)
		return out, nil
	}
	constant := func(v float64) element { return castElement(element{z: complex(v, 0)}, Float64, work) }
	add := func(a, b element) element { v, _ := arithmeticScalar(a, b, work, addOp); return v }
	sub := func(a, b element) element { v, _ := arithmeticScalar(a, b, work, subOp); return v }
	mul := func(a, b element) element { v, _ := arithmeticScalar(a, b, work, mulOp); return v }
	div := func(a, b element) element { v, _ := arithmeticScalar(a, b, work, divOp); return v }
	weighted := func(c, v element) element {
		z := mul(c, v)
		if temporal {
			// NumPy multiplies each timedelta term independently, rounding
			// toward zero in its declared tick unit before summation.
			t := int64(castElement(z, work, Int64).i)
			if t == NaT {
				return constant(math.NaN())
			}
			return castElement(element{i: uint64(t)}, Int64, work)
		}
		return z
	}
	two := constant(2)
	dx := make([]element, n-1)
	if spacing.NDim() == 0 {
		v := castElement(spacing.elementAt(spacing.offset), spacing.DType(), work)
		for i := range dx {
			dx[i] = v
		}
	} else {
		if spacing.NDim() != 1 || spacing.size != n {
			return nil, ErrShape
		}
		for i := range dx {
			x, y := spacing.elementAt(spacing.offset+i*spacing.strides[0]), spacing.elementAt(spacing.offset+(i+1)*spacing.strides[0])
			dx[i] = sub(castElement(y, spacing.DType(), work), castElement(x, spacing.DType(), work))
		}
	}
	os := append(append([]int(nil), out.strides[:ax]...), out.strides[ax+1:]...)
	for i := 0; i < a.size/n; i++ {
		base, ob := r.base(a, i), flatOffset(i, r.shape, os)
		at := func(j int) element {
			v := a.elementAt(base + j*a.strides[ax])
			if temporal {
				t := temporalTicks(v)
				if t == NaT {
					return constant(math.NaN())
				}
				return castElement(element{i: uint64(t)}, Int64, work)
			}
			return castElement(v, a.DType(), work)
		}
		for j := 0; j < n; j++ {
			var z element
			switch {
			case j == 0:
				if edgeOrder == 1 {
					z = div(sub(at(1), at(0)), dx[0])
				} else {
					h, k := dx[0], dx[1]
					c0 := div(sub(element{}, add(mul(two, h), k)), mul(h, add(h, k)))
					c1 := div(add(h, k), mul(h, k))
					c2 := div(sub(element{}, h), mul(k, add(h, k)))
					z = add(add(weighted(c0, at(0)), weighted(c1, at(1))), weighted(c2, at(2)))
				}
			case j == n-1:
				if edgeOrder == 1 {
					z = div(sub(at(j), at(j-1)), dx[j-1])
				} else {
					h, k := dx[j-2], dx[j-1]
					c0 := div(k, mul(h, add(h, k)))
					c1 := div(sub(element{}, add(h, k)), mul(h, k))
					c2 := div(add(mul(two, k), h), mul(k, add(h, k)))
					z = add(add(weighted(c0, at(j-2)), weighted(c1, at(j-1))), weighted(c2, at(j)))
				}
			default:
				h, k := dx[j-1], dx[j]
				if equalElement(h, k, work) {
					z = div(sub(at(j+1), at(j-1)), mul(two, h))
				} else {
					c0 := div(sub(element{}, k), mul(h, add(h, k)))
					c1 := div(sub(k, h), mul(h, k))
					c2 := div(h, mul(k, add(h, k)))
					z = add(add(weighted(c0, at(j-1)), weighted(c1, at(j))), weighted(c2, at(j+1)))
				}
			}
			if temporal {
				t := int64(castElement(z, work, Int64).i)
				out.setElement(ob+j*out.strides[ax], temporalElement(TimeDelta64, t, a.data.unit))
			} else {
				out.setElement(ob+j*out.strides[ax], castElement(z, work, d))
			}
		}
	}
	return out, nil
}

// Use the uniform spacing formula directly: it avoids allocating n-1
// step sizes and does not square h (which may underflow or overflow).
func gradientUniformFloat64(a, out *Array, r reductionLayout, ax, n int, h float64, edgeOrder int) {
	src, dst := a.data.real, out.data.real
	s, ds := a.strides[ax], out.strides[ax]
	os := append(append([]int(nil), out.strides[:ax]...), out.strides[ax+1:]...)
	twoH := 2 * h
	for i := 0; i < a.size/n; i++ {
		base, ob := r.base(a, i), flatOffset(i, r.shape, os)
		at := func(j int) float64 { return src[base+j*s] }
		if edgeOrder == 1 {
			dst[ob] = (at(1) - at(0)) / h
			dst[ob+(n-1)*ds] = (at(n-1) - at(n-2)) / h
		} else {
			c0, c1, c2 := -1.5/h, 2/h, -0.5/h
			dst[ob] = c0*at(0) + c1*at(1) + c2*at(2)
			dst[ob+(n-1)*ds] = -c2*at(n-3) - c1*at(n-2) - c0*at(n-1)
		}
		for j := 1; j < n-1; j++ {
			dst[ob+j*ds] = (at(j+1) - at(j-1)) / twoH
		}
	}
}

// Trapezoid integrates along axis with scalar spacing dx or coordinate array x.
func Trapezoid(a, x *Array, dx float64, axis int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	ax, err := NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	n := a.shape[ax]
	left, right := make([]Index, ax+1), make([]Index, ax+1)
	for i := 0; i < ax; i++ {
		left[i], right[i] = Slice{}, Slice{}
	}
	left[ax], right[ax] = Range(0, max(n-1, 0)), Range(1, n)
	l, err := a.Slice(left...)
	if err != nil {
		return nil, err
	}
	r, err := a.Slice(right...)
	if err != nil {
		return nil, err
	}
	y, err := Add(l, r)
	if err != nil {
		return nil, err
	}
	var spacing *Array
	if x == nil {
		d := meanDType(y.DType())
		spacing, err = Scalar(dx).Astype(d)
	} else {
		if x.NDim() == 1 {
			spacing, err = Diff(x, 1, 0)
			if err == nil {
				sh := make([]int, a.NDim())
				for i := range sh {
					sh[i] = 1
				}
				sh[ax] = spacing.size
				spacing, err = spacing.Reshape(sh...)
			}
		} else {
			spacing, err = Diff(x, 1, ax)
		}
	}
	if err != nil {
		return nil, err
	}
	y, err = Multiply(spacing, y)
	if err != nil {
		return nil, err
	}
	two, err := Scalar(2).Astype(meanDType(y.DType()))
	if err != nil {
		return nil, err
	}
	y, err = Divide(y, two)
	if err != nil {
		return nil, err
	}
	return Sum(y, SumOptions{Axes: []int{ax}})
}

// Interp linearly interpolates a sorted one-dimensional sample grid.
func Interp(x, xp, fp *Array, left, right *complex128) (*Array, error) {
	if !x.DType().IsNativeNumeric() || !xp.DType().IsNativeNumeric() || !fp.DType().IsNativeNumeric() || x.DType().IsComplex() || xp.DType().IsComplex() {
		return nil, ErrDType
	}
	if xp.NDim() != 1 || fp.NDim() != 1 || xp.size != fp.size || xp.size == 0 {
		return nil, ErrShape
	}
	points, _ := xp.Complex128s()
	values, _ := fp.Complex128s()
	for _, v := range points {
		if math.IsNaN(real(v)) {
			return nil, ErrValue
		}
	}
	for i := 1; i < len(points); i++ {
		if real(points[i]) < real(points[i-1]) {
			return nil, ErrValue
		}
	}
	lo, hi := values[0], values[len(values)-1]
	if left != nil {
		lo = *left
	}
	if right != nil {
		hi = *right
	}
	out, err := unary(x, Complex128, func(e element, d DType) (element, error) {
		v := real(e.number(d))
		if math.IsNaN(v) {
			return element{z: complex(v, 0)}, nil
		}
		if v < real(points[0]) {
			return element{z: lo}, nil
		}
		if v > real(points[len(points)-1]) {
			return element{z: hi}, nil
		}
		i := sort.Search(len(points), func(i int) bool { return real(points[i]) > v }) - 1
		if i == len(points)-1 || v == real(points[i]) {
			return element{z: values[i]}, nil
		}
		t := (v - real(points[i])) / (real(points[i+1]) - real(points[i]))
		return element{z: values[i] + complex(t, 0)*(values[i+1]-values[i])}, nil
	})
	if err != nil || fp.DType().IsComplex() {
		return out, err
	}
	return out.Astype(Float64)
}

func InterpPeriod(x, xp, fp *Array, period float64) (*Array, error) {
	period = math.Abs(period)
	if period == 0 || math.IsNaN(period) || math.IsInf(period, 0) {
		return nil, ErrValue
	}
	if xp.NDim() != 1 || fp.NDim() != 1 || xp.size != fp.size || xp.size == 0 {
		return nil, ErrShape
	}
	if x.DType().IsComplex() || xp.DType().IsComplex() {
		return nil, ErrDType
	}
	xv, e := x.Astype(Float64)
	if e != nil {
		return nil, e
	}
	p, e := xp.Astype(Float64)
	if e != nil {
		return nil, e
	}
	mod := func(v float64) float64 {
		r := math.Mod(v, period)
		if r < 0 {
			r += period
		}
		return r
	}
	for i := 0; i < xv.size; i++ {
		xv.setElement(i, element{z: complex(mod(real(xv.elementAt(i).z)), 0)})
	}
	for i := 0; i < p.size; i++ {
		p.setElement(i, element{z: complex(mod(real(p.elementAt(i).z)), 0)})
	}
	order, e := Argsort(p, Int(0))
	if e != nil {
		return nil, e
	}
	p, e = Take(p, order, Int(0))
	if e != nil {
		return nil, e
	}
	f, e := Take(fp, order, Int(0))
	if e != nil {
		return nil, e
	}
	left, right := real(p.elementAt(p.size-1).z)-period, real(p.elementAt(0).z)+period
	l, _ := WrapFloat64([]float64{left}, 1)
	r, _ := WrapFloat64([]float64{right}, 1)
	points, e := Concatenate([]*Array{l, p, r}, 0)
	if e != nil {
		return nil, e
	}
	first, e := f.Slice(Range(0, 1))
	if e != nil {
		return nil, e
	}
	last, e := f.Slice(Range(f.size-1, f.size))
	if e != nil {
		return nil, e
	}
	values, e := Concatenate([]*Array{last, f, first}, 0)
	if e != nil {
		return nil, e
	}
	return Interp(xv, points, values, nil, nil)
}
