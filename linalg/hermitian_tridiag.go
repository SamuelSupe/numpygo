package linalg

import (
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/cblas128"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/gonum"
)

// realReflector overwrites x with v such that Hᴴ*x_original = beta*e₁,
// H = I - tau*v*vᴴ. The input has already been scaled to avoid overflow.
func realReflector(x []complex128) (beta float64, tau complex128) {
	alpha := x[0]
	tail := vectorNorm(x[1:])
	if tail == 0 && imag(alpha) == 0 {
		x[0] = 1
		return real(alpha), 0
	}
	beta = -math.Copysign(math.Hypot(cmplx.Abs(alpha), tail), real(alpha))
	ar := complex(real(alpha)/beta, imag(alpha)/beta)
	tau = 1 - ar
	for i, z := range x[1:] {
		x[i+1] = complex(real(z)/beta, imag(z)/beta) / (ar - 1)
	}
	x[0] = 1
	return beta, tau
}

// complexHermitianEigen reduces directly to a real symmetric tridiagonal
// matrix. Only the upper triangle is updated; the retired lower columns
// hold reflectors for the optional eigenvector backtransform.
func complexHermitianEigen(a []complex128, n int, vectors bool) ([]float64, []complex128, error) {
	d := make([]float64, n)
	e := make([]float64, n-1)
	scale := 0.
	for _, z := range a {
		scale = max(scale, math.Abs(real(z)), math.Abs(imag(z)))
	}
	if scale == 0 {
		if vectors {
			return d, identity(n), nil
		}
		return d, nil, nil
	}
	// Avoid normalizing an otherwise safe matrix to unit magnitude:
	// that can erase a tiny, disconnected eigenvalue in a graded
	// Hermitian matrix. Scale only outside the safe arithmetic range.
	limit := math.MaxFloat64 / (64 * float64(n) * float64(n))
	switch {
	case scale > limit:
		scale /= limit
	case scale < 1e-280:
		// Lift subnormal matrix entries before reflector arithmetic.
	default:
		scale = 1
	}
	if scale != 1 {
		for i, z := range a {
			a[i] = complex(real(z)/scale, imag(z)/scale)
		}
	}
	buf, work := make([]complex128, n), make([]complex128, n)
	cv, cw := make([]complex128, n), make([]complex128, n)
	var tau []complex128
	if vectors {
		tau = make([]complex128, n-1)
	}
	bi := cblas128.Implementation()
	for k := 0; k < n-1; k++ {
		d[k] = real(a[k*n+k])
		start := k + 1
		v, w := buf[:n-start], work[:n-start]
		for i := range v {
			v[i] = cmplx.Conj(a[k*n+start+i])
		}
		beta, t := realReflector(v)
		e[k] = beta
		if t == 0 {
			continue
		}
		trailing := a[start*n+start:]
		// Hᴴ A H = A - w*vᴴ - v*wᴴ, with
		// w = tau*A*v - (|tau|²/2)*(vᴴ*A*v)*v.
		if len(v) >= 32 {
			hermitianMatVec(trailing, n, t, v, w)
		} else {
			bi.Zhemv(blas.Upper, len(v), t, trailing, n, v, 1, 0, w, 1)
		}
		correction := -0.5 * real(t*cblas128.Dotc(contiguousVector(w), contiguousVector(v)))
		cblas128.Axpy(complex(correction, 0), contiguousVector(v), contiguousVector(w))
		if len(v) >= 32 {
			hermitianRank2(trailing, n, v, w, cv[:len(v)], cw[:len(v)])
		} else {
			bi.Zher2(blas.Upper, len(v), -1, w, 1, v, 1, trailing, n)
		}
		if vectors {
			tau[k] = t
			for i, z := range v {
				a[(start+i)*n+k] = z
			}
		}
	}
	d[n-1] = real(a[n*n-1])
	impl := gonum.Implementation{}
	if !vectors {
		safe := true
		for _, v := range [][]float64{d, e} {
			for _, z := range v {
				x := math.Abs(z)
				if x != 0 && !(x >= 1e-100 && x <= 1e100) {
					safe = false
					break
				}
			}
		}
		// Gonum v0.17 Dsterf passes an incorrect leading dimension
		// to Dlascl when it needs to rescale an extreme subblock.
		// Dsteqr's values-only mode uses the correct column stride.
		ok := false
		if safe {
			ok = impl.Dsterf(n, d, e)
		} else {
			ok = impl.Dsteqr(lapack.EVCompNone, n, d, e, nil, 1, nil)
		}
		if !ok {
			return nil, nil, ErrConvergence
		}
		for i := range d {
			d[i] *= scale
		}
		return d, nil, nil
	}
	re := make([]float64, n*n)
	if !impl.Dsteqr(lapack.EVTridiag, n, d, e, re, n, make([]float64, max(1, 2*n-2))) {
		return nil, nil, ErrConvergence
	}
	out := make([]complex128, n*n)
	for i, z := range re {
		out[i] = complex(z, 0)
	}
	for k := n - 2; k >= 0; k-- {
		if tau[k] == 0 {
			continue
		}
		v := buf[:n-k-1]
		for i := range v {
			v[i] = a[(k+1+i)*n+k]
		}
		applyHouseholderLeft(out, n, k+1, 0, v, tau[k], work, true)
	}
	for i := range d {
		d[i] *= scale
	}
	return d, out, nil
}

// Accumulate conjugate(A*v) so the upper triangle can be traversed
// with contiguous Dotu/Axpy kernels and without copying the matrix.
func hermitianMatVec(a []complex128, stride int, t complex128, v, w []complex128) {
	clear(w)
	for i, z := range v {
		row := a[i*stride+i : i*stride+len(v)]
		sum := scaleComplex(z, real(row[0]))
		tail := row[1:]
		if len(tail) >= 16 {
			sum += cblas128.Dotu(contiguousVector(tail), contiguousVector(v[i+1:]))
			cblas128.Axpy(cmplx.Conj(z), contiguousVector(tail), contiguousVector(w[i+1:]))
		} else {
			for j, x := range tail {
				sum += x * v[i+1+j]
				w[i+1+j] += cmplx.Conj(z) * x
			}
		}
		w[i] += cmplx.Conj(sum)
	}
	for i, z := range w {
		w[i] = t * cmplx.Conj(z)
	}
}

func hermitianRank2(a []complex128, stride int, v, w, cv, cw []complex128) {
	for i, z := range v {
		cv[i], cw[i] = cmplx.Conj(z), cmplx.Conj(w[i])
	}
	for i, z := range v {
		row := a[i*stride+i : i*stride+len(v)]
		if len(row) >= 16 {
			cblas128.Axpy(-w[i], contiguousVector(cv[i:]), contiguousVector(row))
			cblas128.Axpy(-z, contiguousVector(cw[i:]), contiguousVector(row))
		} else {
			for j := range row {
				row[j] -= w[i]*cv[i+j] + z*cw[i+j]
			}
		}
		row[0] = complex(real(row[0]), 0)
	}
}
