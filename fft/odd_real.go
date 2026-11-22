package fft

import (
	"math"
	"math/bits"

	"github.com/SamuelSupe/numpygo/internal/numeric"
)

// Keep one representative of each conjugate pair, in stage order.
// After the first real radix r0 (m0=(r0+1)/2), each later stage's
// frequency q at position i is (i/m0)*r0 + i%m0. DC descendants
// may include both conjugates, but the regular layout permits every
// subsequent merge to run in place without a packing/transposition.
type oddRealPlan struct {
	order  []int
	bins   []int
	first  int
	a      []complex128
	stages []mixedStage
}

func newOddRealPlan(n int, inverse bool) *oddRealPlan {
	var factors [bits.UintSize]int
	_, count := mixedRadices(n, &factors)
	first := factors[0]
	m0 := (first + 1) / 2
	buffer := n / first * m0
	arena := make([]complex128, 2*buffer-m0)
	indices := make([]int, n+(n+1)/2)
	p := &oddRealPlan{order: indices[:n:n], bins: indices[n:], first: first, a: arena[:buffer:buffer], stages: make([]mixedStage, count-1)}
	// Positive bins use a signed slot index (+1 avoids negative zero).
	for q := range p.bins {
		block, j := q/first, q%first
		if j < m0 {
			p.bins[q] = 1 + block*m0 + j
		} else {
			t := n - q
			p.bins[q] = -(1 + t/first*m0 + t%first)
		}
	}
	roots := arena[buffer:]
	width := first
	for i, r := range factors[1:count] {
		lanes := width / first * m0
		size := (r - 1) * lanes
		p.stages[i] = mixedStage{radix: r, width: width, roots: roots[:size:size]}
		step := -2 * math.Pi / float64(r*width)
		for j := 0; j < lanes; j++ {
			q := (j/m0)*first + j%m0
			s, c := math.Sincos(step * float64(q))
			z := complex(c, s)
			roots[j] = z
			z2 := z * z
			roots[lanes+j] = z2
			if r == 5 {
				roots[2*lanes+j] = z2 * z
				roots[3*lanes+j] = z2 * z2
			}
		}
		roots = roots[size:]
		width *= r
	}
	fillMixedPermutation(p.order, factors[:count], n, 1, 0, 1, 0, true)
	if inverse {
		// A real Hartley spectrum has Re(X)-Im(X) at k and
		// Re(X)+Im(X) at n-k. Encode the reflected indices.
		for i, j := range p.order {
			if j > n/2 {
				p.order[i] = j - n
			}
		}
	}
	return p
}

func firstOddReal(dst []complex128, x *[5]float64, radix int) {
	a, b, c := x[0], x[1], x[2]
	if radix == 3 {
		s, d := b+c, b-c
		dst[0] = complex(a+s, 0)
		dst[1] = complex(a-.5*s, -.86602540378443864676*d)
		return
	}
	d, e := x[3], x[4]
	p, q, r, t := b+e, c+d, b-e, c-d
	const c1 = .3090169943749474241
	const c2 = -.8090169943749474241
	const s1 = .9510565162951535721
	const s2 = .5877852522924731292
	dst[0] = complex(a+p+q, 0)
	dst[1] = complex(a+c1*p+c2*q, -(s1*r + s2*t))
	dst[2] = complex(a+c2*p+c1*q, -(s2*r - s1*t))
}

func oddRealGather[R float32 | float64, C complex64 | complex128](p *oddRealPlan, in []R, ci []C, inverse bool, base, step, inLen int) {
	r := p.first
	m := (r + 1) / 2
	// Packed, unpadded lanes need no per-sample length check. Fuse
	// the radix-3 gather and first butterfly explicitly.
	if step == 1 && r == 3 {
		if !inverse && inLen >= len(p.order) {
			row := in[base : base+len(p.order)]
			for i, j := 0, 0; i < len(p.order); i, j = i+3, j+2 {
				x := float64(row[p.order[i]])
				y, z := float64(row[p.order[i+1]]), float64(row[p.order[i+2]])
				s, d := y+z, y-z
				p.a[j] = complex(x+s, 0)
				p.a[j+1] = complex(x-.5*s, -.86602540378443864676*d)
			}
			return
		}
		if inverse && inLen >= len(p.bins) {
			row := ci[base : base+len(p.bins)]
			for i, j := 0, 0; i < len(p.order); i, j = i+3, j+2 {
				x := hartleyAt(row, p.order[i])
				y, z := hartleyAt(row, p.order[i+1]), hartleyAt(row, p.order[i+2])
				s, d := y+z, y-z
				p.a[j] = complex(x+s, 0)
				p.a[j+1] = complex(x-.5*s, -.86602540378443864676*d)
			}
			return
		}
	}
	if inverse {
		for i := 0; i < len(p.order)/r; i++ {
			var x [5]float64
			for k, j := range p.order[i*r : (i+1)*r] {
				q := j
				if q < 0 {
					q = -q
				}
				if q >= inLen {
					continue
				}
				z := complex128(ci[base+q*step])
				switch {
				case j < 0:
					x[k] = real(z) + imag(z)
				case j > 0:
					x[k] = real(z) - imag(z)
				default:
					x[k] = real(z)
				}
			}
			firstOddReal(p.a[i*m:(i+1)*m], &x, r)
		}
		return
	}
	for i := 0; i < len(p.order)/r; i++ {
		var x [5]float64
		for k, j := range p.order[i*r : (i+1)*r] {
			if j < inLen {
				x[k] = float64(in[base+j*step])
			}
		}
		firstOddReal(p.a[i*m:(i+1)*m], &x, r)
	}
}

func hartleyAt[C complex64 | complex128](row []C, j int) float64 {
	if j < 0 {
		z := complex128(row[-j])
		return real(z) + imag(z)
	}
	z := complex128(row[j])
	if j == 0 {
		return real(z)
	}
	return real(z) - imag(z)
}

func (p *oddRealPlan) apply() []complex128 {
	m0 := (p.first + 1) / 2
	for _, s := range p.stages {
		r := s.radix
		m := s.width / p.first * m0
		for off := 0; off < len(p.a); off += r * m {
			x := p.a[off : off+r*m]
			if r == 3 {
				numeric.Radix3(x, s.roots[:m], s.roots[m:], -1)
			} else {
				numeric.Radix5(x, s.roots, -1)
			}
		}
	}
	return p.a
}

func oddRealForwardTo[R float32 | float64, C complex64 | complex128](in []R, out []C, n, inLen, outLen, inner, outer int, scale float64) {
	p := newOddRealPlan(n, false)
	for b := 0; b < outer; b++ {
		for k := 0; k < inner; k++ {
			oddRealGather(p, in, []C(nil), false, b*inLen*inner+k, inner, inLen)
			line := p.apply()
			base := b*outLen*inner + k
			out[base] = C(complex(real(line[0])*scale, 0))
			for j, slot := range p.bins[1:] {
				z := oddBin(line, slot)
				out[base+(j+1)*inner] = C(scaled(z, scale))
			}
		}
	}
}

func oddRealInverseTo[C complex64 | complex128, R float32 | float64](in []C, out []R, n, inLen, inner, outer int, scale float64) {
	p := newOddRealPlan(n, true)
	for b := 0; b < outer; b++ {
		for k := 0; k < inner; k++ {
			oddRealGather(p, []R(nil), in, true, b*inLen*inner+k, inner, inLen)
			line := p.apply()
			base := b*n*inner + k
			// The discrete Hartley transform is self-inverse up
			// to n. Reconstruct it from the positive FFT bins.
			out[base] = R(real(line[0]) * scale)
			for j, slot := range p.bins[1:] {
				q := j + 1
				z := oddBin(line, slot)
				out[base+q*inner] = R((real(z) - imag(z)) * scale)
				out[base+(n-q)*inner] = R((real(z) + imag(z)) * scale)
			}
		}
	}
}

func oddBin(line []complex128, slot int) complex128 {
	if slot < 0 {
		z := line[-slot-1]
		return complex(real(z), -imag(z))
	}
	return line[slot-1]
}
