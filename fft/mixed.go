package fft

import (
	"math"
	"math/bits"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/numeric"
)

// Mixed-radix plans reuse the existing power-of-two FFT for the largest
// power-of-two divisor, then combine adjacent subtransforms with radix-3
// and radix-5 stages. The digit-reversed gather and twiddles are shared
// across the entire batch.
type mixedStage struct {
	radix, width int
	roots        []complex128
}
type mixedPlan struct {
	rev       []int
	work      []complex128
	realRoots []complex128
	pow2      pow2Plan
	base      int
	stages    []mixedStage
	sign      float64
}

func smooth235(n int) bool {
	if n < 1 {
		return false
	}
	for _, f := range []int{2, 3, 5} {
		for n%f == 0 {
			n /= f
		}
	}
	return n == 1
}

func mixedRadices(n int, factors *[bits.UintSize]int) (base, count int) {
	base = 1 << bits.TrailingZeros(uint(n))
	left := n / base
	for _, r := range []int{3, 5} {
		for left%r == 0 {
			factors[count] = r
			count++
			left /= r
		}
	}
	return
}

// Enumerate mixed digits once, rather than divide each input index by
// every radix. `order` selects a contiguous-output gather map; packed
// inverse real transforms use the opposite scatter map.
func fillMixedPermutation(dst []int, factors []int, width, base, src, step, off int, order bool) {
	if len(factors) == 0 {
		shift := bits.UintSize - bits.Len(uint(base-1))
		for j := 0; j < base; j++ {
			a, b := src+j*step, off+int(bits.Reverse(uint(j))>>shift)
			if order {
				dst[b] = a
			} else {
				dst[a] = b
			}
		}
		return
	}
	r := factors[len(factors)-1]
	w := width / r
	for j := 0; j < r; j++ {
		fillMixedPermutation(dst, factors[:len(factors)-1], w, base, src+j*step, step*r, off+j*w, order)
	}
}

// Only the first twiddle stream requires sincos; higher powers reuse
// it. The stage phases are evaluated independently, not accumulated.
func fillMixedRoots(roots []complex128, radix, width, lanes int, sign float64) {
	step := sign * 2 * math.Pi / float64(radix*width)
	for j := 0; j < lanes; j++ {
		s, c := math.Sincos(step * float64(j))
		z := complex(c, s)
		roots[j] = z
		if radix >= 3 {
			z2 := z * z
			roots[lanes+j] = z2
			if radix == 5 {
				roots[2*lanes+j] = z2 * z
				roots[3*lanes+j] = z2 * z2
			}
		}
	}
}

func newMixedPlan(n int, inverse, order, realPacked bool) *mixedPlan {
	var factors [bits.UintSize]int
	base, count := mixedRadices(n, &factors)
	total := 2 * n
	if realPacked {
		total += n
	}
	arena := make([]complex128, total)
	p := &mixedPlan{rev: make([]int, n), work: arena[:n:n], base: base, stages: make([]mixedStage, count), sign: -1}
	if inverse {
		p.sign = 1
	}
	if base > 1 {
		p.pow2 = pow2Plan{roots: arena[n : n+base : n+base], work: p.work[:base]}
		fillPow2Roots(p.pow2.roots, p.pow2.work, inverse, nil)
	}
	width := base
	for i, radix := range factors[:count] {
		next := width * radix
		s := mixedStage{radix: radix, width: width, roots: arena[n+width : n+next : n+next]}
		fillMixedRoots(s.roots, radix, width, width, p.sign)
		p.stages[i] = s
		width = next
	}
	fillMixedPermutation(p.rev, factors[:count], n, base, 0, 1, 0, order)
	if realPacked {
		p.realRoots = arena[2*n:]
		fillMixedRealRoots(p.realRoots)
	}
	return p
}

func (p *mixedPlan) apply() {
	if p.base > 1 {
		for i := 0; i < len(p.work); i += p.base {
			p.pow2.work = p.work[i : i+p.base]
			p.pow2.apply()
		}
	}
	for _, s := range p.stages {
		w := s.width
		for off := 0; off < len(p.work); off += s.radix * w {
			x := p.work[off : off+s.radix*w]
			if s.radix == 3 {
				numeric.Radix3(x, s.roots[:w], s.roots[w:], p.sign)
			} else {
				numeric.Radix5(x, s.roots, p.sign)
			}
		}
	}
}

func fillMixedRealRoots(r []complex128) {
	m := len(r)
	n := 2 * m
	r[0] = 1
	for j := 1; j <= (m-1)/2; j++ {
		s, c := math.Sincos(-2 * math.Pi * float64(j) / float64(n))
		r[j] = complex(c, s)
		r[m-j] = complex(-c, s)
	}
	if m%2 == 0 {
		r[m/2] = complex(0, -1)
	}
}

// Lengths with factors other than 2/3/5 retain FFTPACK/Bluestein.
// Real even transforms halve the work length (including when its half
// is odd); odd real transforms keep packed Hermitian stages.
func transformMixed(a *np.Array, kind transformKind, n int, shape []int, inLen, outLen, inner, outer int, dtype np.DType, scale float64, direct *np.Array) (*np.Array, error) {
	if n < 64 || n&(n-1) == 0 || !smooth235(n) || inner*outer == 1 && n < 1024 {
		return nil, nil
	}
	if kind == inverseReal && !a.DType().IsComplex() {
		return nil, nil
	}
	switch a.DType() {
	case np.Float32:
		if dtype != np.Complex64 {
			return nil, nil
		}
		v, e := singleInput[float32](a)
		if e != nil {
			return nil, e
		}
		return mixedTyped(v, []complex64(nil), kind, n, shape, inLen, outLen, inner, outer, dtype, scale, direct)
	case np.Complex64:
		if kind == inverseReal && dtype != np.Float32 || kind != inverseReal && dtype != np.Complex64 {
			return nil, nil
		}
		v, e := singleInput[complex64](a)
		if e != nil {
			return nil, e
		}
		return mixedTyped([]float32(nil), v, kind, n, shape, inLen, outLen, inner, outer, dtype, scale, direct)
	case np.Float64:
		if dtype != np.Complex128 {
			return nil, nil
		}
		v, e := fftRealInput(a)
		if e != nil {
			return nil, e
		}
		return mixedTyped(v, []complex128(nil), kind, n, shape, inLen, outLen, inner, outer, dtype, scale, direct)
	case np.Complex128:
		if kind == inverseReal && dtype != np.Float64 || kind != inverseReal && dtype != np.Complex128 {
			return nil, nil
		}
		v, _, e := fftComplexInput(a)
		if e != nil {
			return nil, e
		}
		return mixedTyped([]float64(nil), v, kind, n, shape, inLen, outLen, inner, outer, dtype, scale, direct)
	}
	return nil, nil
}

func mixedTyped[R float32 | float64, C complex64 | complex128](in []R, ci []C, kind transformKind, n int, shape []int, inLen, outLen, inner, outer int, dtype np.DType, scale float64, direct *np.Array) (*np.Array, error) {
	limit := math.MaxFloat64 / (16 * float64(n))
	oddSafe := true
	hartleyCheck := kind == inverseReal && n%2 != 0
	// A Hartley representation combines the two spectrum components.
	// For large, imbalanced values, retain the full complex kernel
	// if that conversion would discard significant absolute detail.
	hartleyTolerance := 1e-13
	if dtype == np.Float32 {
		hartleyTolerance = 1e-7
	}
	for _, x := range in {
		if !(math.Abs(float64(x)) <= limit) {
			return nil, nil
		}
	}
	if hartleyCheck {
		for _, x := range ci {
			z := complex128(x)
			if !(math.Abs(real(z)) <= limit && math.Abs(imag(z)) <= limit) {
				return nil, nil
			}
			if !oddSafe {
				continue
			}
			r, i := real(z), imag(z)
			if r != 0 && i != 0 && (math.Abs(r) > 1024 || math.Abs(i) > 1024) {
				s, d := r+i, r-i
				if math.Abs((s+d)*.5-r) > hartleyTolerance || math.Abs((s-d)*.5-i) > hartleyTolerance {
					oddSafe = false
				}
			}
		}
	} else {
		for _, x := range ci {
			z := complex128(x)
			if !(math.Abs(real(z)) <= limit && math.Abs(imag(z)) <= limit) {
				return nil, nil
			}
		}
	}
	if direct == nil {
		var e error
		direct, e = np.Zeros(dtype, shape...)
		if e != nil {
			return nil, e
		}
	}
	if kind == forwardReal || kind == inverseReal {
		if n%2 != 0 {
			if kind == forwardReal {
				dst, _ := np.NativeSliceForKernel[C](direct, numeric.KernelAccess{}, true)
				oddRealForwardTo(in, dst, n, inLen, outLen, inner, outer, scale)
			} else {
				dst, _ := np.NativeSliceForKernel[R](direct, numeric.KernelAccess{}, true)
				if oddSafe {
					oddRealInverseTo(ci, dst, n, inLen, inner, outer, scale)
				} else {
					mixedOddInverseWide(ci, dst, n, inLen, inner, outer, scale)
				}
			}
			return direct, nil
		}
		p := newMixedPlan(n/2, kind == inverseReal, kind == forwardReal, true)
		roots := p.realRoots
		if kind == forwardReal {
			dst, _ := np.NativeSliceForKernel[C](direct, numeric.KernelAccess{}, true)
			packedRealForwardTo(in, dst, n, inLen, outLen, inner, outer, scale, roots, p.rev, p.work, p.apply)
		} else {
			dst, _ := np.NativeSliceForKernel[R](direct, numeric.KernelAccess{}, true)
			packedRealInverseTo(ci, dst, n, inLen, inner, outer, scale, roots, p.rev, p.work, p.apply)
		}
		return direct, nil
	}
	dst, _ := np.NativeSliceForKernel[C](direct, numeric.KernelAccess{}, true)
	p := newMixedPlan(n, kind == inverseComplex, true, false)
	used := min(n, inLen)
	for b := 0; b < outer; b++ {
		for k := 0; k < inner; k++ {
			base := b*inLen*inner + k
			if used == n && inner == 1 {
				if in != nil {
					row := in[base : base+n]
					for j, q := range p.rev {
						p.work[j] = complex(float64(row[q]), 0)
					}
				} else {
					row := ci[base : base+n]
					for j, q := range p.rev {
						p.work[j] = complex128(row[q])
					}
				}
			} else if in != nil {
				for j, q := range p.rev {
					if q < used {
						p.work[j] = complex(float64(in[base+q*inner]), 0)
					} else {
						p.work[j] = 0
					}
				}
			} else {
				for j, q := range p.rev {
					if q < used {
						p.work[j] = complex128(ci[base+q*inner])
					} else {
						p.work[j] = 0
					}
				}
			}
			p.apply()
			base = b*outLen*inner + k
			line := p.work[:outLen]
			if inner == 1 && scale == 1 {
				switch v := any(dst[base : base+outLen]).(type) {
				case []complex128:
					copy(v, line)
				case []complex64:
					numeric.NarrowComplex(v, line)
				}
			} else {
				for j, z := range line {
					dst[base+j*inner] = C(scaled(z, scale))
				}
			}
		}
	}
	return direct, nil
}

func mixedOddInverseWide[C complex64 | complex128, R float32 | float64](in []C, out []R, n, inLen, inner, outer int, scale float64) {
	p := newMixedPlan(n, true, true, false)
	for b := 0; b < outer; b++ {
		for k := 0; k < inner; k++ {
			base := b*inLen*inner + k
			for j, q := range p.rev {
				f := q
				if f > n/2 {
					f = n - f
				}
				z := complex128(0)
				if f < inLen {
					z = complex128(in[base+f*inner])
					if q == 0 {
						z = complex(real(z), 0)
					} else if q > n/2 {
						z = complex(real(z), -imag(z))
					}
				}
				p.work[j] = z
			}
			p.apply()
			base = b*n*inner + k
			for j, z := range p.work {
				out[base+j*inner] = R(real(z) * scale)
			}
		}
	}
}
