package np

import (
	"math"
	"slices"
	"sort"
)

// HistogramBinEdges returns bin boundaries for a fixed count/explicit edges,
// or one of NumPy's named unweighted bin estimators.
func HistogramBinEdges(a *Array, opt HistogramOptions, method string) (*Array, error) {
	if method == "" {
		_, e, err := Histogram(a, opt)
		return e, err
	}
	if !a.DType().IsNativeNumeric() || a.DType().IsComplex() {
		return nil, ErrDType
	}
	if opt.Weights != nil {
		return nil, ErrDType
	}
	if opt.Edges != nil {
		return nil, ErrValue
	}
	switch method {
	case "sqrt", "sturges", "rice", "scott", "stone", "doane", "fd", "auto":
	default:
		return nil, ErrValue
	}
	values, _ := a.Complex128s()
	lo, hi := 0., 1.
	if len(values) > 0 {
		lo, hi = real(values[0]), real(values[0])
		for _, z := range values {
			lo = math.Min(lo, real(z))
			hi = math.Max(hi, real(z))
		}
	}
	if opt.Range != nil {
		lo, hi = opt.Range[0], opt.Range[1]
	}
	if lo > hi || math.IsNaN(lo) || math.IsNaN(hi) || math.IsInf(lo, 0) || math.IsInf(hi, 0) {
		return nil, ErrValue
	}
	if lo == hi {
		lo -= .5
		hi += .5
	}
	x := make([]float64, 0, len(values))
	for _, z := range values {
		v := real(z)
		if v >= lo && v <= hi {
			x = append(x, v)
		}
	}
	n := float64(len(x))
	width := 0.
	if len(x) > 0 {
		slices.Sort(x)
		ptp := x[len(x)-1] - x[0]
		mean := 0.
		for _, v := range x {
			mean += v
		}
		mean /= n
		std := 0.
		for _, v := range x {
			d := v - mean
			std += d * d
		}
		std = math.Sqrt(std / n)
		quantile := func(q float64) float64 {
			p := q * (n - 1)
			i := int(p)
			if i+1 == len(x) {
				return x[i]
			}
			return x[i] + (p-float64(i))*(x[i+1]-x[i])
		}
		fd := 2 * (quantile(.75) - quantile(.25)) * math.Pow(n, -1./3)
		sturges := ptp / (math.Log2(n) + 1)
		sqrt := ptp / math.Sqrt(n)
		switch method {
		case "sqrt":
			width = sqrt
		case "sturges":
			width = sturges
		case "rice":
			width = ptp / (2 * math.Cbrt(n))
		case "scott":
			width = math.Cbrt(24*math.Sqrt(math.Pi)/n) * std
		case "fd":
			width = fd
		case "auto":
			width = math.Min(math.Max(fd, sqrt/2), sturges)
		case "doane":
			if n > 2 && std > 0 {
				g1 := 0.
				for _, v := range x {
					p := (v - mean) / std
					g1 += p * p * p
				}
				g1 /= n
				sg := math.Sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
				width = ptp / (1 + math.Log2(n) + math.Log2(1+math.Abs(g1)/sg))
			}
		case "stone":
			if n > 1 && ptp > 0 {
				best := math.Inf(1)
				bestCount := 1
				for count := 1; count <= max(100, int(math.Sqrt(n))); count++ {
					bins := make([]int, count)
					for _, v := range x {
						j := min(count-1, int((v-lo)/(hi-lo)*float64(count)))
						bins[j]++
					}
					sum := 0.
					for _, v := range bins {
						p := float64(v) / n
						sum += p * p
					}
					cost := (2 - (n+1)*sum) / (ptp / float64(count))
					if cost < best {
						best = cost
						bestCount = count
					}
				}
				width = ptp / float64(bestCount)
			}
		}
	}
	count := 1
	if width > 0 {
		if a.DType().IsInteger() && width < 1 {
			width = 1
		}
		f := math.Ceil((hi - lo) / width)
		if f > float64(maxInt/8-1) {
			return nil, ErrShape
		}
		count = max(1, int(f))
	}
	opt.Count = count
	opt.Range = &[2]float64{lo, hi}
	_, edges, e := Histogram(a, opt)
	return edges, e
}

// HistogramDD bins rows of a two-dimensional (samples, dimensions) array.
// Per-dimension counts, edges and ranges are specified by bins; weights have
// one value per sample. As in NumPy, the histogram has float64 dtype.
func HistogramDD(sample *Array, bins []HistogramOptions, weights *Array, density bool) (*Array, []*Array, error) {
	if sample.DType() == 0 || sample.DType().IsComplex() {
		return nil, nil, ErrDType
	}
	if sample.NDim() != 2 || sample.shape[1] == 0 {
		return nil, nil, ErrShape
	}
	n, dim := sample.shape[0], sample.shape[1]
	if len(bins) == 0 {
		bins = make([]HistogramOptions, dim)
	} else if len(bins) != dim {
		return nil, nil, ErrShape
	}
	if weights != nil && (weights.DType() == 0 || weights.DType().IsComplex() || weights.NDim() != 1 || weights.size != n) {
		return nil, nil, ErrShape
	}
	edges := make([]*Array, dim)
	values := make([][]float64, dim)
	shape := make([]int, dim)
	for axis, opt := range bins {
		if opt.Weights != nil || opt.Density {
			return nil, nil, ErrValue
		}
		column, e := sample.Slice(Slice{}, AtIndex(axis))
		if e != nil {
			return nil, nil, e
		}
		edges[axis], e = HistogramBinEdges(column, opt, "")
		if e != nil {
			return nil, nil, e
		}
		f, e := edges[axis].Astype(Float64)
		if e != nil {
			return nil, nil, e
		}
		values[axis], _ = f.Float64s()
		shape[axis] = len(values[axis]) - 1
	}
	hist, e := Zeros(Float64, shape...)
	if e != nil {
		return nil, nil, e
	}
	for row := 0; row < n; row++ {
		flat := 0
		valid := true
		for axis, v := range values {
			z := real(sample.elementAt(sample.offset + row*sample.strides[0] + axis*sample.strides[1]).number(sample.DType()))
			if math.IsNaN(z) || z < v[0] || z > v[len(v)-1] {
				valid = false
				break
			}
			idx := sort.Search(len(v), func(j int) bool { return v[j] > z }) - 1
			idx = min(idx, shape[axis]-1)
			flat = flat*shape[axis] + idx
		}
		if valid {
			w := 1.
			if weights != nil {
				w = real(weights.elementAt(weights.offset + row*weights.strides[0]).number(weights.DType()))
			}
			hist.data.real[flat] += w
		}
	}
	if density {
		total := 0.
		for _, v := range hist.data.real {
			total += v
		}
		for flat, v := range hist.data.real {
			width := 1.
			q := flat
			for axis := dim - 1; axis >= 0; axis-- {
				j := q % shape[axis]
				q /= shape[axis]
				width *= values[axis][j+1] - values[axis][j]
			}
			hist.data.real[flat] = v / total / width
		}
	}
	return hist, edges, nil
}
func Histogram2D(x, y *Array, bins []HistogramOptions, weights *Array, density bool) (hist, xedges, yedges *Array, err error) {
	if x.DType() == 0 || y.DType() == 0 || x.NDim() != 1 || y.NDim() != 1 || x.size != y.size {
		return nil, nil, nil, ErrShape
	}
	sample, e := Stack([]*Array{x, y}, 1)
	if e != nil {
		return nil, nil, nil, e
	}
	h, edges, e := HistogramDD(sample, bins, weights, density)
	if e != nil {
		return nil, nil, nil, e
	}
	return h, edges[0], edges[1], nil
}
