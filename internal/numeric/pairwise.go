package numeric

// Pairwise64/32 retain NumPy's eight independent accumulators, 128-value
// recursion split and final addition tree. SIMD only updates independent
// lanes; no horizontal reduction changes the floating-point order.
func Pairwise64(x []float64) float64 { return pairwise(x) }
func Pairwise32(x []float32) float32 { return pairwise(x) }

func pairwise[T float32 | float64](x []T) T {
	n := len(x)
	if n < 8 {
		var s T
		for _, v := range x {
			s += v
		}
		return s
	}
	if n > 128 {
		m := (n / 2) &^ 7
		return pairwise(x[:m]) + pairwise(x[m:])
	}
	var p [8]T
	full := n &^ 7
	fast := false
	if full >= 32 {
		switch dst := any(&p).(type) {
		case *[8]float64:
			fast = partial64SIMD(dst, any(x[:full]).([]float64))
		case *[8]float32:
			fast = partial32SIMD(dst, any(x[:full]).([]float32))
		}
	}
	if !fast {
		copy(p[:], x[:8])
		for i := 8; i < full; i += 8 {
			for j := range p {
				p[j] += x[i+j]
			}
		}
	}
	s := ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))
	for _, v := range x[full:] {
		s += v
	}
	return s
}
