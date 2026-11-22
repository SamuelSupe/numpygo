package fft

import (
	"math"

	"github.com/SamuelSupe/numpygo"

	"gonum.org/v1/gonum/dsp/fourier"
)

// FFTPACK's generic radix step is quadratic in a prime factor. Bluestein
// reduces such a transform to a power-of-two convolution. The plan is local
// to the call and reused across lanes, not retained in a global cache.
func largeFFTFactor(n int) bool {
	for f := 2; f <= n/f; f++ {
		if f >= 127 {
			return n >= 127
		}
		for n%f == 0 {
			n /= f
		}
	}
	return n >= 127
}

type chirpFFT struct {
	plan                *fourier.CmplxFFT
	radix               *pow2Plan
	chirp, kernel, work []complex128
}

func newChirpFFT(n int, inverse bool, lanes int) *chirpFFT {
	m := 1
	for m < 2*n-1 {
		m *= 2
	}
	if _, e := np.ElementCount(np.Complex128, m); e != nil {
		return nil
	}
	p := &chirpFFT{chirp: make([]complex128, n), kernel: make([]complex128, m), work: make([]complex128, m)}
	// Reuse the fused radix-4/2 kernel for repeated or large convolutions;
	// small single-lane plans retain FFTPACK's lower setup cost.
	if m >= 8192 || m >= 2048 && lanes > 1 {
		p.radix = newPow2Plan(m, false, nil)
	} else {
		p.plan = fourier.NewCmplxFFT(m)
	}
	sign := -1.
	if inverse {
		sign = 1
	}
	phase := 0
	for j := 0; j < n; j++ {
		// j*j modulo 2*n avoids losing phase bits for long transforms.
		s, c := math.Sincos(sign * math.Pi * float64(phase) / float64(n))
		p.chirp[j] = complex(c, s)
		k := complex(c, -s)
		p.kernel[j] = k
		if j != 0 {
			p.kernel[m-j] = k
		}
		phase = (phase + 2*j + 1) % (2 * n)
	}
	p.transform(p.kernel, false)
	return p
}

func (p *chirpFFT) transform(data []complex128, inverse bool) {
	if p.radix == nil {
		if inverse {
			p.plan.Sequence(data, data)
		} else {
			p.plan.Coefficients(data, data)
		}
		return
	}
	q := p.radix
	if inverse {
		for i, z := range data {
			q.work[q.rev[i]] = complex(real(z), -imag(z))
		}
	} else {
		for i, z := range data {
			q.work[q.rev[i]] = z
		}
	}
	q.apply()
	if inverse {
		for i, z := range q.work {
			data[i] = complex(real(z), -imag(z))
		}
	} else {
		copy(data, q.work)
	}
}
func (p *chirpFFT) apply(line []complex128) {
	if q := p.radix; q != nil {
		// Compose chirp multiplication with the bit-reversed writes.
		// Ping-pong the already allocated buffers between the forward
		// transform and conjugate inverse, avoiding two full copies.
		clear(q.work)
		for j, z := range line {
			q.work[q.rev[j]] = z * p.chirp[j]
		}
		q.apply()
		for j, z := range q.work {
			v := z * p.kernel[j]
			p.work[q.rev[j]] = complex(real(v), -imag(v))
		}
		q.work, p.work = p.work, q.work
		q.apply()
		scale := 1 / float64(len(q.work))
		for j, z := range p.chirp {
			v := q.work[j]
			line[j] = scaled(complex(real(v), -imag(v))*z, scale)
		}
		return
	}
	for j, z := range line {
		p.work[j] = z * p.chirp[j]
	}
	clear(p.work[len(line):])
	p.transform(p.work, false)
	for j, z := range p.kernel {
		p.work[j] *= z
	}
	p.transform(p.work, true)
	scale := 1 / float64(len(p.work))
	for j, z := range p.chirp {
		line[j] = scaled(p.work[j]*z, scale)
	}
}

// Only use the convolution for finite, safely scaled data. Explicit FFTPACK
// arithmetic remains the fallback for exceptional and extreme magnitudes.
func transformChirp(a *np.Array, kind transformKind, n int, shape []int, inLen, outLen, inner, outer, size int, dtype np.DType, scale float64, direct *np.Array) (*np.Array, error) {
	if !largeFFTFactor(n) {
		return nil, nil
	}
	limit := math.MaxFloat64 / (16 * float64(n) * float64(n))
	var input []complex128
	var reals []float64
	var owned bool
	var e error
	// Real inputs need only a single complex lane for Bluestein, not
	// a promoted copy of the entire real ndarray.
	if kind == forwardReal {
		reals, e = fftRealInput(a)
		if e != nil {
			return nil, e
		}
		for _, x := range reals {
			if !(math.Abs(x) <= limit) {
				return nil, nil
			}
		}
	} else {
		input, owned, e = fftComplexInput(a)
		if e != nil {
			return nil, e
		}
		for _, z := range input {
			// The convolution has multiple unnormalized intermediate transforms.
			if !(math.Abs(real(z)) <= limit && math.Abs(imag(z)) <= limit) {
				return nil, nil
			}
		}
	}
	p := newChirpFFT(n, kind == inverseComplex || kind == inverseReal, inner*outer)
	if p == nil {
		return nil, nil
	}
	line := make([]complex128, n)
	if kind == inverseReal {
		output := fftBuffer[float64](direct, size)
		for batch := 0; batch < outer; batch++ {
			for lane := 0; lane < inner; lane++ {
				gather(line[:n/2+1], input, inLen, inner, batch, lane)
				line[0] = complex(real(line[0]), 0)
				for j := 1; j <= n/2; j++ {
					if 2*j == n {
						line[j] = complex(real(line[j]), 0)
					} else {
						line[n-j] = complex(real(line[j]), -imag(line[j]))
					}
				}
				p.apply(line)
				for j, z := range line {
					output[(batch*n+j)*inner+lane] = real(z) * scale
				}
			}
		}
		return realOutput(output, dtype, shape, direct)
	}
	var output []complex128
	if (direct == nil || dtype == np.Complex64) && kind != forwardReal && owned && n == inLen {
		output = input
	} else {
		output = fftBuffer[complex128](direct, size)
	}
	for batch := 0; batch < outer; batch++ {
		for lane := 0; lane < inner; lane++ {
			if kind == forwardReal {
				used := min(inLen, n)
				base := batch*inLen*inner + lane
				for j := 0; j < used; j++ {
					line[j] = complex(reals[base+j*inner], 0)
				}
				clear(line[used:])
			} else {
				gather(line, input, inLen, inner, batch, lane)
			}
			p.apply(line)
			if kind == forwardReal {
				line[0] = complex(real(line[0]), 0)
				if n%2 == 0 {
					line[n/2] = complex(real(line[n/2]), 0)
				}
			}
			for j, z := range line[:outLen] {
				output[(batch*outLen+j)*inner+lane] = scaled(z, scale)
			}
		}
	}
	return complexOutput(output, dtype, shape, direct)
}
