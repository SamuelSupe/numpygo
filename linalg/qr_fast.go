package linalg

import (
	"github.com/SamuelSupe/numpygo"

	"gonum.org/v1/gonum/lapack/lapack64"
)

func qrReal(av []float64, d np.DType, qs, rs []int, m, n, k, qcols, count, qsz, rsz int, ronly bool) (*np.Array, *np.Array, error) {
	rv := make([]float64, rsz)
	var qv []float64
	if !ronly {
		qv = make([]float64, qsz)
	}
	if k > 0 && count > 0 {
		tau := make([]float64, k)
		work := make([]float64, 1)
		lapack64.Geqrf(matrix(m, n, av[:m*n]), tau, work, -1)
		lwork := max(int(work[0]), n)
		if !ronly {
			lapack64.Orgqr(matrix(m, qcols, qv[:m*qcols]), tau, work, -1)
			lwork = max(lwork, int(work[0]), qcols)
		}
		work = make([]float64, lwork)
		for b := 0; b < count; b++ {
			a := av[b*m*n : (b+1)*m*n]
			lapack64.Geqrf(matrix(m, n, a), tau, work, len(work))
			r := rv[b*qcols*n : (b+1)*qcols*n]
			for i := 0; i < k; i++ {
				copy(r[i*n+i:(i+1)*n], a[i*n+i:(i+1)*n])
			}
			if !ronly {
				q := qv[b*m*qcols : (b+1)*m*qcols]
				for i := 0; i < m; i++ {
					copy(q[i*qcols:i*qcols+k], a[i*n:i*n+k])
				}
				lapack64.Orgqr(matrix(m, qcols, q), tau, work, len(work))
			}
		}
	} else if !ronly && qcols > 0 {
		for b := 0; b < count; b++ {
			for i := 0; i < min(m, qcols); i++ {
				qv[b*m*qcols+i*qcols+i] = 1
			}
		}
	}
	wrap := func(v []float64, shape []int) (*np.Array, error) {
		a, e := np.WrapFloat64(v, shape...)
		if e != nil || d == np.Float64 {
			return a, e
		}
		return a.Astype(d)
	}
	r, e := wrap(rv, rs)
	if e != nil || ronly {
		return nil, r, e
	}
	q, e := wrap(qv, qs)
	return q, r, e
}

// Q is built from the same packed reflectors as QRRaw, in reverse order,
// directly in M×K storage for reduced QR. This avoids constructing and
// then truncating a full M×M identity for tall input matrices.
func formComplexQ(packed, q []complex128, m, n, k, qcols int, tau, v, work []complex128) {
	for i := 0; i < min(m, qcols); i++ {
		q[i*qcols+i] = 1
	}
	for j := k - 1; j >= 0; j-- {
		if tau[j] == 0 {
			continue
		}
		u := v[:m-j]
		u[0] = 1
		for i := j + 1; i < m; i++ {
			u[i-j] = packed[i*n+j]
		}
		applyHouseholderLeft(q, qcols, j, j, u, tau[j], work, finiteMatrix(u) && tau[j]-tau[j] == 0)
	}
}
