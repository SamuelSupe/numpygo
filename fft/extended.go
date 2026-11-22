package fft

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/x87"
)

// Extended FFTs keep the x87 significand and exponent throughout. Radix-2
// butterflies handle power-of-two sizes; Bluestein uses a radix-2 convolution
// for other sizes, including large primes, with O(n log n) arithmetic.
type extendedFFTPlan struct {
	n, m                       int
	inverse                    bool
	roots, chirp, kernel, work []x87.Complex
}

func newExtendedFFTPlan(n int, inverse bool) (*extendedFFTPlan, error) {
	if _, e := np.ElementCount(np.Complex256, n); e != nil {
		return nil, e
	}
	p := &extendedFFTPlan{n: n, m: n, inverse: inverse}
	if n&(n-1) != 0 {
		m := 1
		for m < 2*n-1 {
			if m > int(^uint(0)>>1)/2 {
				return nil, np.ErrShape
			}
			m *= 2
		}
		if _, e := np.ElementCount(np.Complex256, m); e != nil {
			return nil, e
		}
		p.m = m
	}
	p.roots = make([]x87.Complex, p.m/2)
	for k := range p.roots {
		s, c := x87.SinCosPiRatio(-2*int64(k), int64(p.m))
		p.roots[k] = x87.Complex{Real: c, Imag: s}
	}
	if p.m != n {
		p.chirp = make([]x87.Complex, n)
		p.kernel = make([]x87.Complex, p.m)
		p.work = make([]x87.Complex, p.m)
		sign := int64(-1)
		if inverse {
			sign = 1
		}
		square := int64(0)
		for k := 0; k < n; k++ {
			// k² modulo 2n avoids overflow and preserves exact quadrants.
			s, c := x87.SinCosPiRatio(sign*square, int64(n))
			p.chirp[k] = x87.Complex{Real: c, Imag: s}
			v := p.chirp[k].Conj()
			p.kernel[k] = v
			if k != 0 {
				p.kernel[p.m-k] = v
			}
			square = (square + 2*int64(k) + 1) % (2 * int64(n))
		}
		p.radix2(p.kernel, false)
	}
	return p, nil
}
func (p *extendedFFTPlan) radix2(v []x87.Complex, inverse bool) {
	n := len(v)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			v[i], v[j] = v[j], v[i]
		}
	}
	for width := 2; width <= n; width *= 2 {
		half, step := width/2, n/width
		for start := 0; start < n; start += width {
			for j := 0; j < half; j++ {
				root := p.roots[j*step]
				if inverse {
					root = root.Conj()
				}
				u, w := v[start+j], v[start+j+half].Mul(root)
				v[start+j], v[start+j+half] = u.Add(w), u.Sub(w)
			}
		}
		if width == n {
			break
		}
	}
}
func (p *extendedFFTPlan) transform(line []x87.Complex) {
	if p.m == p.n {
		p.radix2(line, p.inverse)
		return
	}
	clear(p.work)
	for i, z := range line {
		p.work[i] = z.Mul(p.chirp[i])
	}
	p.radix2(p.work, false)
	for i, z := range p.work {
		p.work[i] = z.Mul(p.kernel[i])
	}
	p.radix2(p.work, true)
	// The convolution FFT length is a power of two; exact binary scaling
	// avoids unnecessary rounding when undoing the inverse transform.
	ex := 0
	for n := p.m; n > 1; n >>= 1 {
		ex--
	}
	for i := range line {
		z := p.work[i].Mul(p.chirp[i])
		line[i] = x87.Complex{Real: z.Real.Ldexp(ex), Imag: z.Imag.Ldexp(ex)}
	}
}
func extendedNormalization(norm Norm, n int, inverse bool) x87.Float {
	one := x87.FromInt64(1)
	count := x87.FromInt64(int64(n))
	switch norm {
	case Ortho:
		return one.Div(count.Sqrt())
	case Forward:
		if !inverse {
			return one.Div(count)
		}
	case "", Backward:
		if inverse {
			return one.Div(count)
		}
	}
	return one
}
func transformExtended(a *np.Array, kind transformKind, norm Norm, n int, shape []int, inLen, outLen, inner, outer int, dtype np.DType) (*np.Array, error) {
	converted, e := a.Astype(np.Complex256)
	if e != nil {
		return nil, e
	}
	input, e := np.Values[np.ExtendedComplex](converted)
	if e != nil {
		return nil, e
	}
	inverse := kind == inverseComplex || kind == inverseReal
	plan, e := newExtendedFFTPlan(n, inverse)
	if e != nil {
		return nil, e
	}
	factor := extendedNormalization(norm, n, inverse)
	line := make([]x87.Complex, n)
	size, e := np.ElementCount(dtype, shape...)
	if e != nil {
		return nil, e
	}
	output := make([]x87.Complex, size)
	for batch := 0; batch < outer; batch++ {
		for lane := 0; lane < inner; lane++ {
			clear(line)
			limit := min(inLen, n)
			if kind == inverseReal {
				limit = min(inLen, n/2+1)
			}
			for i := 0; i < limit; i++ {
				line[i] = input[(batch*inLen+i)*inner+lane]
			}
			if kind == inverseReal {
				line[0].Imag = x87.Zero(false)
				if n%2 == 0 {
					line[n/2].Imag = x87.Zero(false)
				}
				for i := 1; i <= (n-1)/2; i++ {
					line[n-i] = line[i].Conj()
				}
			}
			plan.transform(line)
			if kind == forwardReal {
				line[0].Imag = x87.Zero(false)
				if n%2 == 0 {
					line[n/2].Imag = x87.Zero(false)
				}
			}
			for i := 0; i < outLen; i++ {
				z := line[i]
				output[(batch*outLen+i)*inner+lane] = x87.Complex{Real: z.Real.Mul(factor), Imag: z.Imag.Mul(factor)}
			}
		}
	}
	if kind == inverseReal {
		realOutput := make([]np.ExtendedFloat, len(output))
		for i, z := range output {
			realOutput[i] = z.Real
		}
		v, e := np.WrapSlice(realOutput, shape...)
		if e != nil {
			return nil, e
		}
		if dtype == np.Float128 {
			return v, nil
		}
		return v.Astype(dtype)
	}
	v, e := np.WrapSlice(output, shape...)
	if e != nil {
		return nil, e
	}
	if dtype == np.Complex256 {
		return v, nil
	}
	return v.Astype(dtype)
}

// FFTFreqExtended is the typed longdouble-spacing variant of FFTFreq.
func FFTFreqExtended(n int, d np.ExtendedFloat) (*np.Array, error)  { return extendedFreq(n, d, false) }
func RFFTFreqExtended(n int, d np.ExtendedFloat) (*np.Array, error) { return extendedFreq(n, d, true) }
func extendedFreq(n int, d x87.Float, realOnly bool) (*np.Array, error) {
	if n <= 0 {
		return nil, np.ErrValue
	}
	size := n
	if realOnly {
		size = n/2 + 1
	}
	if _, e := np.ElementCount(np.Float128, size); e != nil {
		return nil, e
	}
	factor := x87.FromInt64(1).Div(x87.FromInt64(int64(n)).Mul(d))
	out := make([]x87.Float, size)
	for i := range out {
		k := i
		if !realOnly && i >= n/2+n%2 {
			k = i - n
		}
		out[i] = x87.FromInt64(int64(k)).Mul(factor)
	}
	return np.WrapSlice(out, size)
}
