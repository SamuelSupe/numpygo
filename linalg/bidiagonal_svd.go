package linalg

import (
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/cblas128"
	"gonum.org/v1/gonum/lapack/gonum"
)

func complexTallSVD(a []complex128, m, n int, full, vectors bool) ([]complex128, []float64, []complex128, error) {
	if u, s, vh, ok := bidiagonalSVD(a, m, n, full, vectors); ok {
		return u, s, vh, nil
	}
	return jacobiSVD(a, m, n, full, vectors)
}

// bidiagonalSVD reduces a tall complex matrix to a real bidiagonal matrix.
// A severely graded spectrum or a failed bidiagonal solve falls back to
// one-sided Jacobi, preserving its relative accuracy for tiny singular values.
func bidiagonalSVD(input []complex128, m, n int, full, vectors bool) (u []complex128, s []float64, vh []complex128, ok bool) {
	if n < 16 {
		return nil, nil, nil, false
	}
	scale := 0.
	for _, z := range input {
		if z-z != 0 {
			return nil, nil, nil, false
		}
		scale = max(scale, math.Abs(real(z)), math.Abs(imag(z)))
	}
	if scale == 0 {
		return nil, nil, nil, false
	}
	a := make([]complex128, len(input))
	for j, z := range input {
		a[j] = complex(real(z)/scale, imag(z)/scale)
	}
	d, e := make([]float64, n), make([]float64, n-1)
	buf, work, conj := make([]complex128, m), make([]complex128, max(n, m)), make([]complex128, n)
	var left, right []complex128
	if vectors {
		left, right = make([]complex128, n), make([]complex128, n-1)
	}
	for k := 0; k < n; k++ {
		v := buf[:m-k]
		for i := range v {
			v[i] = a[(k+i)*n+k]
		}
		beta, t := realReflector(v)
		d[k] = beta
		if t != 0 {
			applyHouseholderLeft(a, n, k, k+1, v, cmplx.Conj(t), work, true)
		}
		if vectors {
			left[k] = t
			for i, z := range v {
				a[(k+i)*n+k] = z
			}
		}
		if k+1 == n {
			break
		}
		v = buf[:n-k-1]
		for i, z := range a[k*n+k+1 : (k+1)*n] {
			v[i] = cmplx.Conj(z)
		}
		beta, t = realReflector(v)
		e[k] = beta
		if t != 0 {
			cv := conj[:len(v)]
			for i, z := range v {
				cv[i] = cmplx.Conj(z)
			}
			for i := k + 1; i < m; i++ {
				row := a[i*n+k+1 : (i+1)*n]
				dot := cblas128.Dotu(contiguousVector(row), contiguousVector(v))
				cblas128.Axpy(-t*dot, contiguousVector(cv), contiguousVector(row))
			}
		}
		if vectors {
			right[k] = t
			copy(a[k*n+k+1:(k+1)*n], v)
		}
	}
	impl := gonum.Implementation{}
	for _, z := range d {
		if math.Abs(z) < 1e-8 {
			return nil, nil, nil, false
		}
	}
	scratch := make([]float64, 4*n)
	var ur, vr []float64
	dim, ld := 0, 1
	if vectors {
		dim, ld = n, n
		ur, vr = make([]float64, n*n), make([]float64, n*n)
		for i := 0; i < n; i++ {
			ur[i*n+i], vr[i*n+i] = 1, 1
		}
	}
	if !impl.Dbdsqr(blas.Upper, n, dim, dim, 0, d, e, vr, ld, ur, ld, nil, 1, scratch) {
		return nil, nil, nil, false
	}
	// Bidiagonal reduction is backward stable, but may lose relative
	// accuracy on strongly ill-conditioned inputs. Keep Jacobi there.
	if d[n-1] < d[0]*1e-8 {
		return nil, nil, nil, false
	}
	for i := range d {
		d[i] *= scale
	}
	if !vectors {
		return nil, d, nil, true
	}
	cols := n
	if full {
		cols = m
	}
	u = make([]complex128, m*cols)
	for i := 0; i < n; i++ {
		for j, z := range ur[i*n : (i+1)*n] {
			u[i*cols+j] = complex(z, 0)
		}
	}
	for i := n; i < cols; i++ {
		u[i*cols+i] = 1
	}
	for k := n - 1; k >= 0; k-- {
		if left[k] == 0 {
			continue
		}
		v := buf[:m-k]
		for i := range v {
			v[i] = a[(k+i)*n+k]
		}
		applyHouseholderLeft(u, cols, k, 0, v, left[k], work, true)
	}
	// V = G₀...Gₖ * transpose(VT_b); then return its adjoint.
	v := make([]complex128, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v[i*n+j] = complex(vr[j*n+i], 0)
		}
	}
	for k := n - 2; k >= 0; k-- {
		if right[k] == 0 {
			continue
		}
		r := a[k*n+k+1 : (k+1)*n]
		applyHouseholderLeft(v, n, k+1, 0, r, right[k], work, true)
	}
	return u, d, adjoint(v, n, n), true
}
