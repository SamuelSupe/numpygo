package linalg

import (
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/blas/cblas128"
)

// complexEigen computes a scaled complex Schur form by Hessenberg reduction
// and implicit shifted QR. Unlike a real embedding, it retains algebraic
// multiplicities and supports defective matrices with dependent eigenvectors.
func complexEigen(a []complex128, n int, vectors bool) ([]complex128, []complex128, error) {
	h := make([]complex128, len(a))
	var q []complex128
	if vectors {
		q = identity(n)
	}
	scale := 0.
	for _, z := range a {
		scale = max(scale, math.Abs(real(z)), math.Abs(imag(z)))
	}
	ev := make([]complex128, n)
	if scale == 0 {
		return ev, q, nil
	}
	for i, z := range a {
		h[i] = complex(real(z)/scale, imag(z)/scale)
	}
	hessenberg(h, q, n)
	if vectors {
		// QR repeatedly updates adjacent columns of Q. Transpose its
		// working storage once so these rotations are contiguous.
		for i := 0; i < n; i++ {
			for j := 0; j < i; j++ {
				q[i*n+j], q[j*n+i] = q[j*n+i], q[i*n+j]
			}
		}
	}
	const eps = 2.220446049250313e-16
	sweeps := 0
	for hi := n - 1; hi > 0; {
		lo := hi
		for lo > 0 {
			sub := cmplx.Abs(h[lo*n+lo-1])
			bound := cmplx.Abs(h[(lo-1)*n+lo-1]) + cmplx.Abs(h[lo*n+lo])
			if sub <= max(1e-300, eps*bound) {
				h[lo*n+lo-1] = 0
				break
			}
			lo--
		}
		if lo == hi {
			hi--
			sweeps = 0
			continue
		}
		if sweeps >= 1000*(hi-lo+1) {
			return nil, nil, ErrConvergence
		}
		mu := schurShift(h, n, hi)
		if sweeps > 0 && sweeps%24 == 0 {
			s := cmplx.Abs(h[hi*n+hi-1])
			mu = h[hi*n+hi] + complex(.75*s, .25*s)
		}
		for k := lo; k < hi; k++ {
			x, y := h[k*n+k]-mu, h[(k+1)*n+k]
			if k > lo {
				x, y = h[k*n+k-1], h[(k+1)*n+k-1]
			}
			c, s := complexRotation(x, y)
			start := k
			if k > lo {
				start = k - 1
			}
			end := n
			if !vectors {
				end = hi + 1
			}
			xrow, yrow := h[k*n+start:k*n+end], h[(k+1)*n+start:(k+1)*n+end]
			cs := cmplx.Conj(s)
			for j, u := range xrow {
				v := yrow[j]
				xrow[j] = scaleComplex(u, c) + s*v
				yrow[j] = -cs*u + scaleComplex(v, c)
			}
			if k > lo {
				h[(k+1)*n+k-1] = 0
			}
			first := 0
			if !vectors {
				first = lo
			}
			// In Hessenberg form columns k and k+1 are zero below
			// row k+2, including the single temporary QR bulge.
			rotateRight(h, n, k, c, s, first, min(n, k+3))
			if vectors {
				rotateQ(q, n, k, c, s)
			}
		}
		sweeps++
	}
	for j := range ev {
		lambda := h[j*n+j]
		ev[j] = scaleComplex(lambda, scale)
	}
	if !vectors {
		return ev, nil, nil
	}
	result := make([]complex128, n*n)
	x := make([]complex128, n)
	acc := make([]complex128, n)
	for j := 0; j < n; j++ {
		lambda := h[j*n+j]
		clear(x)
		x[j] = 1
		// Back-substitute in the upper triangular Schur form. A
		// repeated eigenvalue may have fewer independent eigenvectors;
		// a tiny denominator perturbation produces its limiting vector.
		smin := max(1e-300, eps*cmplx.Abs(lambda))
		for i := j - 1; i >= 0; i-- {
			var sum complex128
			for k := i + 1; k <= j; k++ {
				sum += h[i*n+k] * x[k]
			}
			den := h[i*n+i] - lambda
			absden := cmplx.Abs(den)
			if absden < smin {
				den = complex(smin, 0)
				absden = smin
			}
			if mag := cmplx.Abs(sum); mag > absden*1e100 {
				f := absden * 1e100 / mag
				for k := i + 1; k <= j; k++ {
					x[k] *= complex(f, 0)
				}
				sum *= complex(f, 0)
			}
			x[i] = -sum / den
		}
		norm := vectorNorm(x[:j+1])
		if norm == 0 || math.IsNaN(norm) || math.IsInf(norm, 0) {
			return nil, nil, ErrConvergence
		}
		normalizeVector(x[:j+1], norm)
		clear(acc)
		for k := 0; k <= j; k++ {
			col := q[k*n : (k+1)*n]
			if n >= 32 {
				cblas128.Axpy(x[k], contiguousVector(col), contiguousVector(acc))
			} else {
				for i, v := range col {
					acc[i] += v * x[k]
				}
			}
		}
		for i, v := range acc {
			result[i*n+j] = v
		}
	}
	return ev, result, nil
}

func hessenberg(h, q []complex128, n int) {
	buf, conj, work := make([]complex128, n), make([]complex128, n), make([]complex128, n)
	for k := 0; k < n-2; k++ {
		v := buf[:n-k-1]
		for i := range v {
			v[i] = h[(k+1+i)*n+k]
		}
		norm := vectorNorm(v)
		if norm == 0 {
			continue
		}
		phase := complex(1., 0)
		if v[0] != 0 {
			phase = v[0] / complex(cmplx.Abs(v[0]), 0)
		}
		normalizeVector(v, norm)
		v[0] += phase
		normalizeVector(v, vectorNorm(v))
		applyHouseholderLeft(h, n, k+1, k, v, 2, work, true)
		for i := k + 2; i < n; i++ {
			h[i*n+k] = 0
		}
		cv := conj[:len(v)]
		for i, z := range v {
			cv[i] = cmplx.Conj(z)
		}
		for _, a := range [][]complex128{h, q} {
			if a == nil {
				continue
			}
			for i := 0; i < n; i++ {
				row := a[i*n+k+1 : (i+1)*n]
				if len(v) >= 32 {
					dot := cblas128.Dotu(contiguousVector(row), contiguousVector(v))
					cblas128.Axpy(-2*dot, contiguousVector(cv), contiguousVector(row))
					continue
				}
				var dot complex128
				for j, z := range v {
					dot += row[j] * z
				}
				for j, z := range cv {
					row[j] -= 2 * dot * z
				}
			}
		}
	}
}
func complexRotation(a, b complex128) (float64, complex128) {
	if b == 0 {
		return 1, 0
	}
	aa, bb := cmplx.Abs(a), cmplx.Abs(b)
	norm := math.Hypot(aa, bb)
	phase := complex(1., 0)
	if aa != 0 {
		phase = a / complex(aa, 0)
	}
	return aa / norm, phase * (cmplx.Conj(b) / complex(norm, 0))
}
func scaleComplex(z complex128, c float64) complex128 { return complex(real(z)*c, imag(z)*c) }
func rotateRight(a []complex128, n, k int, c float64, s complex128, first, end int) {
	cs := cmplx.Conj(s)
	for i := first; i < end; i++ {
		x, y := a[i*n+k], a[i*n+k+1]
		a[i*n+k] = scaleComplex(x, c) + cs*y
		a[i*n+k+1] = -s*x + scaleComplex(y, c)
	}
}
func rotateQ(q []complex128, n, k int, c float64, s complex128) {
	l, r := q[k*n:(k+1)*n], q[(k+1)*n:(k+2)*n]
	cs := cmplx.Conj(s)
	for i, x := range l {
		y := r[i]
		l[i] = scaleComplex(x, c) + cs*y
		r[i] = -s*x + scaleComplex(y, c)
	}
}
func schurShift(h []complex128, n, hi int) complex128 {
	a, b := h[(hi-1)*n+hi-1], h[(hi-1)*n+hi]
	c, d := h[hi*n+hi-1], h[hi*n+hi]
	half := (a - d) / 2
	root := cmplx.Sqrt(half*half + b*c)
	u, v := half+root, half-root
	den := u
	if cmplx.Abs(v) > cmplx.Abs(u) {
		den = v
	}
	if den == 0 {
		return d
	}
	return d - b*c/den
}
