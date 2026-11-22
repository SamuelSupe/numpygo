package fft

import (
	"math"
	"math/bits"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/numeric"
)

// A call-local fused radix-4/2 plan reuses bit-reversed offsets and stage twiddles
// across all lanes. Real transforms pack even/odd samples into a half-size
// complex FFT and recover the Hermitian spectrum without a second transform.
type pow2Plan struct {
	rev   []int
	roots []complex128
	work  []complex128
}

func newPow2Plan(n int, inverse bool, realTwiddles []complex128) *pow2Plan {
	p := &pow2Plan{rev: make([]int, n), roots: make([]complex128, n), work: make([]complex128, n)}
	shift := bits.UintSize - bits.Len(uint(n-1))
	for i := range p.rev {
		p.rev[i] = int(bits.Reverse(uint(i)) >> shift)
	}
	fillPow2Roots(p.roots, p.work, inverse, realTwiddles)
	return p
}

// Mixed-radix plans borrow their power-of-two roots and setup scratch
// from the same arena used by the outer stages.
func fillPow2Roots(roots, work []complex128, inverse bool, realTwiddles []complex128) {
	n := len(roots)
	sign := -1.
	if inverse {
		sign = 1
	}
	half := n / 2
	if realTwiddles != nil {
		for j := 0; j < half; j++ {
			z := realTwiddles[2*j]
			if inverse {
				z = complex(real(z), -imag(z))
			}
			roots[half+j] = z
		}
	} else {
		for j := 1; j < half/2; j++ {
			s, c := math.Sincos(sign * math.Pi * float64(j) / float64(half))
			roots[half+j] = complex(c, s)
			roots[n-j] = complex(-c, s)
		}
	}
	roots[half] = 1
	if half >= 2 {
		roots[half+half/2] = complex(0, sign)
	}
	// Earlier radix-2 stages use exact subsets of the final stage.
	// Avoid evaluating the same trigonometric values twice.
	for h := half / 2; h >= 1; h /= 2 {
		for j := 0; j < h; j++ {
			roots[h+j] = roots[half+j*(half/h)]
		}
	}
	// A fused radix-4 stage uses three twiddle streams of h entries.
	// Reuse the latter half of the second radix-2 stage for its third
	// stream. Build it in the already-allocated scratch buffer first.
	for h := 4; h <= n/4; h *= 4 {
		for j := 0; j < h; j++ {
			k := 3 * j
			if k < 2*h {
				work[j] = roots[2*h+k]
			} else {
				work[j] = -roots[k]
			}
		}
		copy(roots[3*h:4*h], work[:h])
	}
}
func (p *pow2Plan) apply() {
	x := p.work
	if len(x) == 2 {
		x[0], x[1] = x[0]+x[1], x[0]-x[1]
		return
	}
	// Fuse the first two stages using exact twiddles 1 and ±i.
	sign := imag(p.roots[3])
	for i := 0; i < len(x); i += 4 {
		a, b := x[i]+x[i+1], x[i]-x[i+1]
		c, d := x[i+2]+x[i+3], x[i+2]-x[i+3]
		e := complex(-sign*imag(d), sign*real(d))
		x[i], x[i+1], x[i+2], x[i+3] = a+c, b+e, a-c, b-e
	}
	half := 4
	for ; half <= len(x)/4; half *= 4 {
		for base := 0; base < len(x); base += 4 * half {
			numeric.Radix4(x[base:base+4*half], p.roots[half:2*half], p.roots[2*half:3*half], p.roots[3*half:4*half], sign)
		}
	}
	for ; half < len(x); half *= 2 {
		roots := p.roots[half : 2*half]
		for base := 0; base < len(x); base += 2 * half {
			l, r := x[base:base+half], x[base+half:base+2*half]
			if half >= 32 {
				numeric.Butterfly(l, r, roots)
			} else {
				for j, w := range roots {
					u, v := l[j], r[j]*w
					l[j], r[j] = u+v, u-v
				}
			}
		}
	}
}
func realRoots(n int) []complex128 {
	roots := make([]complex128, n/2)
	roots[0] = 1
	for i := 1; i < n/4; i++ {
		s, c := math.Sincos(-2 * math.Pi * float64(i) / float64(n))
		roots[i] = complex(c, s)
		roots[n/2-i] = complex(-c, s)
	}
	roots[n/4] = complex(0, -1)
	return roots
}

func transformPow2(a *np.Array, kind transformKind, n int, shape []int, inLen, outLen, inner, outer, size int, dtype np.DType, scale float64, direct *np.Array) (*np.Array, error) {
	if n < 64 || n&(n-1) != 0 {
		return nil, nil
	}
	// The plan pays off when reused across lanes, or for a sufficiently
	// long single transform. Small single transforms retain FFTPACK.
	if outer*inner == 1 && n < 4096 {
		return nil, nil
	}
	// The packed real formula adds unnormalized complex values; keep the
	// previous FFTPACK path for nonfinite and extreme input magnitudes.
	limit := math.MaxFloat64 / (8 * float64(n))
	switch kind {
	case forwardReal:
		in, e := fftRealInput(a)
		if e != nil {
			return nil, e
		}
		for _, x := range in {
			if !(math.Abs(x) <= limit) {
				return nil, nil
			}
		}
		return pow2RealForward(in, n, shape, inLen, outLen, inner, outer, size, dtype, scale, direct)
	default:
		in, owned, e := fftComplexInput(a)
		if e != nil {
			return nil, e
		}
		for _, z := range in {
			if !(math.Abs(real(z)) <= limit && math.Abs(imag(z)) <= limit) {
				return nil, nil
			}
		}
		if kind == inverseReal {
			return pow2RealInverse(in, n, shape, inLen, inner, outer, size, dtype, scale, direct)
		}
		p := newPow2Plan(n, kind == inverseComplex, nil)
		var out []complex128
		if (direct == nil || dtype == np.Complex64) && owned && n == inLen {
			// Complex128s returned a detached copy. Each lane is fully
			// gathered before its result is written, so equal-length
			// transforms can reuse that storage without touching the input.
			out = in
		} else {
			out = fftBuffer[complex128](direct, size)
		}
		for batch := 0; batch < outer; batch++ {
			for lane := 0; lane < inner; lane++ {
				used := min(inLen, n)
				if used < n {
					clear(p.work)
				}
				base := batch*inLen*inner + lane
				for j := 0; j < used; j++ {
					p.work[p.rev[j]] = in[base+j*inner]
				}
				p.apply()
				base = batch*n*inner + lane
				if inner == 1 && scale == 1 {
					copy(out[base:base+n], p.work)
				} else {
					for j, z := range p.work {
						out[base+j*inner] = scaled(z, scale)
					}
				}
			}
		}
		return complexOutput(out, dtype, shape, direct)
	}
}
func pow2RealForward(in []float64, n int, shape []int, inLen, outLen, inner, outer, size int, dtype np.DType, scale float64, direct *np.Array) (*np.Array, error) {
	out := fftBuffer[complex128](direct, size)
	pow2RealForwardTo(in, out, n, inLen, outLen, inner, outer, scale)
	return complexOutput(out, dtype, shape, direct)
}
func pow2RealForwardTo[R float32 | float64, C complex64 | complex128](in []R, out []C, n, inLen, outLen, inner, outer int, scale float64) {
	m := n / 2
	roots := realRoots(n)
	p := newPow2Plan(m, false, roots)
	packedRealForwardTo(in, out, n, inLen, outLen, inner, outer, scale, roots, p.rev, p.work, p.apply)
}
func packedRealForwardTo[R float32 | float64, C complex64 | complex128](in []R, out []C, n, inLen, outLen, inner, outer int, scale float64, roots []complex128, rev []int, work []complex128, apply func()) {
	m := n / 2
	var line []R
	if inner != 1 || inLen < n {
		line = make([]R, n)
	}
	for batch := 0; batch < outer; batch++ {
		for lane := 0; lane < inner; lane++ {
			seq := line
			if inner == 1 && inLen >= n {
				seq = in[batch*inLen : batch*inLen+n]
			} else {
				gather(line, in, inLen, inner, batch, lane)
			}
			// Bit reversal is self-inverse; mixed-radix forward
			// plans supply the corresponding contiguous gather map.
			for j, r := range rev {
				work[j] = complex(float64(seq[2*r]), float64(seq[2*r+1]))
			}
			apply()
			base := batch*outLen*inner + lane
			z := work[0]
			out[base] = C(complex((real(z)+imag(z))*scale, 0))
			out[base+m*inner] = C(complex((real(z)-imag(z))*scale, 0))
			for j := 1; j <= (m-1)/2; j++ {
				a, b := work[j], work[m-j]
				b = complex(real(b), -imag(b))
				e, d := (a+b)*.5, (a-b)*.5
				t := complex(imag(d), -real(d)) * roots[j]
				out[base+j*inner] = C(scaled(e+t, scale))
				v := e - t
				out[base+(m-j)*inner] = C(scaled(complex(real(v), -imag(v)), scale))
			}
			if m%2 == 0 {
				z = work[m/2]
				out[base+(m/2)*inner] = C(scaled(complex(real(z), -imag(z)), scale))
			}
		}
	}
}
func pow2RealInverse(in []complex128, n int, shape []int, inLen, inner, outer, size int, dtype np.DType, scale float64, direct *np.Array) (*np.Array, error) {
	out := fftBuffer[float64](direct, size)
	pow2RealInverseTo(in, out, n, inLen, inner, outer, scale)
	return realOutput(out, dtype, shape, direct)
}
func pow2RealInverseTo[C complex64 | complex128, R float32 | float64](in []C, out []R, n, inLen, inner, outer int, scale float64) {
	m := n / 2
	roots := realRoots(n)
	p := newPow2Plan(m, true, roots)
	packedRealInverseTo(in, out, n, inLen, inner, outer, scale, roots, p.rev, p.work, p.apply)
}
func packedRealInverseTo[C complex64 | complex128, R float32 | float64](in []C, out []R, n, inLen, inner, outer int, scale float64, roots []complex128, rev []int, work []complex128, apply func()) {
	m := n / 2
	var line []C
	if inner != 1 || inLen < m+1 {
		line = make([]C, m+1)
	}
	for batch := 0; batch < outer; batch++ {
		for lane := 0; lane < inner; lane++ {
			seq := line
			if inner == 1 && inLen >= m+1 {
				seq = in[batch*inLen : batch*inLen+m+1]
			} else {
				gather(line, in, inLen, inner, batch, lane)
			}
			a, b := real(complex128(seq[0])), real(complex128(seq[m]))
			work[0] = complex(a+b, a-b)
			for j := 1; j <= (m-1)/2; j++ {
				x, y := complex128(seq[j]), complex128(seq[m-j])
				y = complex(real(y), -imag(y))
				e, d := x+y, x-y
				w := roots[j]
				o := d * complex(real(w), -imag(w))
				t := complex(-imag(o), real(o))
				work[rev[j]] = e + t
				work[rev[m-j]] = complex(real(e-t), -imag(e-t))
			}
			if m%2 == 0 {
				z := complex128(seq[m/2])
				work[rev[m/2]] = complex(2*real(z), -2*imag(z))
			}
			apply()
			base := batch*n*inner + lane
			for j, z := range work {
				out[base+2*j*inner] = R(real(z) * scale)
				out[base+(2*j+1)*inner] = R(imag(z) * scale)
			}
		}
	}
}
