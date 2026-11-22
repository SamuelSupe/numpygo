package linalg

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"

	"github.com/SamuelSupe/numpygo"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/blas/cblas128"
	"gonum.org/v1/gonum/lapack/lapack64"
)

var ErrNotPositiveDefinite = errors.New("matrix is not positive definite")

func squareShape(a *np.Array) (np.DType, []int, int, int, error) {
	d, err := linearDType(a.DType())
	if err != nil {
		return 0, nil, 0, 0, err
	}
	s := a.Shape()
	if len(s) < 2 || s[len(s)-2] != s[len(s)-1] {
		return 0, nil, 0, 0, np.ErrShape
	}
	n := s[len(s)-1]
	batch := s[:len(s)-2]
	count, err := np.ElementCount(d, batch...)
	return d, batch, n, count, err
}

// Inv returns the inverse of each square matrix.
func Inv(a *np.Array) (*np.Array, error) {
	d, _, n, _, err := squareShape(a)
	if err != nil {
		return nil, err
	}
	if _, err = np.ElementCount(d, n, n); err != nil {
		return nil, err
	}
	i, err := asComplexArray(identity(n), d, []int{n, n})
	if err != nil {
		return nil, err
	}
	return Solve(a, i)
}

// Slogdet returns determinant sign/complex phase and log absolute magnitude.
func Slogdet(a *np.Array) (*np.Array, *np.Array, error) {
	d, batch, n, count, err := squareShape(a)
	if err != nil {
		return nil, nil, err
	}
	if !d.IsComplex() {
		av, e := realLAPACKInput(a)
		if e != nil {
			return nil, nil, e
		}
		if av != nil {
			signs, logs := make([]float64, count), make([]float64, count)
			pivots := make([]int, n)
			for q := range signs {
				signs[q] = 1
				if n == 0 {
					continue
				}
				lu := av[q*n*n : (q+1)*n*n]
				if !lapack64.Getrf(matrix(n, n, lu), pivots) {
					signs[q], logs[q] = 0, math.Inf(-1)
					continue
				}
				for i, p := range pivots {
					if p != i {
						signs[q] = -signs[q]
					}
					v := lu[i*n+i]
					if v < 0 {
						signs[q] = -signs[q]
					}
					logs[q] += math.Log(math.Abs(v))
				}
			}
			return realSlogdetResult(signs, logs, d, batch)
		}
	}
	av, _ := a.Complex128s()
	signs := make([]complex128, count)
	logs := make([]float64, count)
	for q := 0; q < count; q++ {
		lu := slices.Clone(av[q*n*n : (q+1)*n*n])
		_, parity, ok := factorComplex(lu, n)
		if !ok {
			logs[q] = math.Inf(-1)
			continue
		}
		sign := complex(float64(parity), 0)
		for i := 0; i < n; i++ {
			z := lu[i*n+i]
			mag := cmplx.Abs(z)
			logs[q] += math.Log(mag)
			sign *= z / complex(mag, 0)
		}
		signs[q] = sign
	}
	s, err := asComplexArray(signs, d, batch)
	if err != nil {
		return nil, nil, err
	}
	l, err := np.WrapFloat64(logs, batch...)
	if err != nil {
		return nil, nil, err
	}
	if d == np.Float32 || d == np.Complex64 {
		l, err = l.Astype(np.Float32)
	}
	return s, l, err
}

// LAPACK mutates its input. Retain the existing explicit IEEE path for
// nonfinite matrices, where zero-skipping BLAS kernels can change results.
func realLAPACKInput(a *np.Array) ([]float64, error) {
	if a.DType() != np.Float64 {
		var e error
		a, e = a.Astype(np.Float64)
		if e != nil {
			return nil, e
		}
	}
	v, e := a.Float64s()
	if e != nil {
		return nil, e
	}
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, nil
		}
	}
	return v, nil
}
func realSlogdetResult(signs, logs []float64, d np.DType, shape []int) (*np.Array, *np.Array, error) {
	s, e := np.WrapFloat64(signs, shape...)
	if e != nil {
		return nil, nil, e
	}
	l, e := np.WrapFloat64(logs, shape...)
	if e != nil {
		return nil, nil, e
	}
	if d != np.Float64 {
		s, e = s.Astype(d)
		if e != nil {
			return nil, nil, e
		}
		l, e = l.Astype(d)
	}
	return s, l, e
}
func Det(a *np.Array) (*np.Array, error) {
	s, l, err := Slogdet(a)
	if err != nil {
		return nil, err
	}
	sign, _ := s.Complex128s()
	logs, _ := l.Complex128s()
	for i := range sign {
		m := math.Exp(real(logs[i]))
		sign[i] = complex(real(sign[i])*m, imag(sign[i])*m)
	}
	return asComplexArray(sign, s.DType(), s.Shape())
}

// Cholesky computes lower (default) or upper triangular Hermitian factors.
func Cholesky(a *np.Array, upper ...bool) (*np.Array, error) {
	if len(upper) > 1 {
		return nil, np.ErrValue
	}
	up := len(upper) == 1 && upper[0]
	d, _, n, count, err := squareShape(a)
	if err != nil {
		return nil, err
	}
	if n == 0 || count == 0 {
		return np.Zeros(d, a.Shape()...)
	}
	if !d.IsComplex() {
		av, e := realLAPACKInput(a)
		if e != nil {
			return nil, e
		}
		if av != nil {
			uplo := blas.Lower
			if up {
				uplo = blas.Upper
			}
			for q := 0; q < count; q++ {
				x := av[q*n*n : (q+1)*n*n]
				if _, ok := lapack64.Potrf(blas64.Symmetric{N: n, Stride: n, Uplo: uplo, Data: x}); !ok {
					return nil, ErrNotPositiveDefinite
				}
				for i := 0; i < n; i++ {
					if up {
						clear(x[i*n : i*n+i])
					} else {
						clear(x[i*n+i+1 : (i+1)*n])
					}
				}
			}
			result, e := np.WrapFloat64(av, a.Shape()...)
			if e != nil || d == np.Float64 {
				return result, e
			}
			return result.Astype(d)
		}
	}
	av, _ := a.Complex128s()
	out := make([]complex128, len(av))
	for q := 0; q < count; q++ {
		x := av[q*n*n : (q+1)*n*n]
		l := out[q*n*n : (q+1)*n*n]
		fast := n >= 64
		if fast {
			for _, v := range x {
				if !(math.Abs(real(v)) <= math.MaxFloat64/(8*float64(n)) && math.Abs(imag(v)) <= math.MaxFloat64/(8*float64(n))) {
					fast = false
					break
				}
			}
		}
		for i := 0; i < n; i++ {
			row := l[i*n : (i+1)*n]
			for j := 0; j <= i; j++ {
				z := x[i*n+j]
				if up {
					z = cmplx.Conj(x[j*n+i])
				}
				if fast && j >= 32 {
					z -= cblas128.Dotc(cblas128.Vector{N: j, Inc: 1, Data: l[j*n : j*n+j]}, cblas128.Vector{N: j, Inc: 1, Data: row[:j]})
				} else {
					for k, v := range l[j*n : j*n+j] {
						z -= row[k] * cmplx.Conj(v)
					}
				}
				if i == j {
					if !(real(z) > 0) {
						return nil, ErrNotPositiveDefinite
					}
					row[i] = complex(math.Sqrt(real(z)), 0)
				} else {
					row[j] = z / l[j*n+j]
				}
			}
		}
		if up {
			copy(l, adjoint(l, n, n))
		}
	}
	return asComplexArray(out, d, a.Shape())
}

type QROptions struct {
	Complete bool
	ROnly    bool
}

// QR computes a Householder QR factorization over the final two axes.
func QR(a *np.Array, options ...QROptions) (*np.Array, *np.Array, error) {
	d, err := linearDType(a.DType())
	if err != nil {
		return nil, nil, err
	}
	if a.NDim() < 2 || len(options) > 1 {
		return nil, nil, np.ErrShape
	}
	var opt QROptions
	if len(options) == 1 {
		opt = options[0]
	}
	shape := a.Shape()
	batch := shape[:len(shape)-2]
	m, n := shape[len(shape)-2], shape[len(shape)-1]
	k := min(m, n)
	qcols := k
	if opt.Complete && !opt.ROnly {
		qcols = m
	}
	qs := append(slices.Clone(batch), m, qcols)
	rs := append(slices.Clone(batch), qcols, n)
	rsz, err := np.ElementCount(d, rs...)
	if err != nil {
		return nil, nil, err
	}
	qsz := 0
	if !opt.ROnly {
		qsz, err = np.ElementCount(d, qs...)
		if err != nil {
			return nil, nil, err
		}
	}
	count, err := np.ElementCount(d, batch...)
	if err != nil {
		return nil, nil, err
	}
	if !d.IsComplex() {
		av, e := realLAPACKInput(a)
		if e != nil {
			return nil, nil, e
		}
		if av != nil {
			return qrReal(av, d, qs, rs, m, n, k, qcols, count, qsz, rsz, opt.ROnly)
		}
	}
	rv := make([]complex128, rsz)
	var qv []complex128
	if !opt.ROnly {
		qv = make([]complex128, qsz)
	}
	av, _ := a.Complex128s()
	tau, v, work := make([]complex128, k), make([]complex128, m), make([]complex128, qcols)
	for bi := 0; bi < count; bi++ {
		r := av[bi*m*n : (bi+1)*m*n]
		if finiteMatrix(r) {
			packedComplexQR(r, m, n, tau)
			dest := rv[bi*qcols*n : (bi+1)*qcols*n]
			for i := 0; i < k; i++ {
				copy(dest[i*n+i:(i+1)*n], r[i*n+i:(i+1)*n])
			}
			if !opt.ROnly {
				formComplexQ(r, qv[bi*m*qcols:(bi+1)*m*qcols], m, n, k, qcols, tau, v, work)
			}
			continue
		}
		// Preserve the original explicit propagation path for nonfinite
		// inputs; the optimized packed representation is for finite data.
		var q []complex128
		if !opt.ROnly {
			q = identity(m)
		}
		for j := 0; j < k; j++ {
			norm := normColumn(r[j*n:], m-j, n, j)
			if norm == 0 {
				continue
			}
			x0 := r[j*n+j]
			phase := complex(1., 0.)
			if x0 != 0 {
				phase = x0 / complex(cmplx.Abs(x0), 0)
			}
			v := make([]complex128, m-j)
			for i := j; i < m; i++ {
				v[i-j] = r[i*n+j] / complex(norm, 0)
			}
			v[0] += phase
			var vn float64
			for _, z := range v {
				vn = math.Hypot(vn, cmplx.Abs(z))
			}
			for i := range v {
				v[i] /= complex(vn, 0)
			}
			for h := j; h < n; h++ {
				var dot complex128
				for i := j; i < m; i++ {
					dot += cmplx.Conj(v[i-j]) * r[i*n+h]
				}
				for i := j; i < m; i++ {
					r[i*n+h] -= 2 * v[i-j] * dot
				}
			}
			for i := j + 1; i < m; i++ {
				r[i*n+j] = 0
			}
			if !opt.ROnly {
				for row := 0; row < m; row++ {
					var dot complex128
					for h := j; h < m; h++ {
						dot += q[row*m+h] * v[h-j]
					}
					for h := j; h < m; h++ {
						q[row*m+h] -= 2 * dot * cmplx.Conj(v[h-j])
					}
				}
			}
		}
		copy(rv[bi*qcols*n:(bi+1)*qcols*n], r[:qcols*n])
		if !opt.ROnly {
			for row := 0; row < m; row++ {
				copy(qv[bi*m*qcols+row*qcols:], q[row*m:row*m+qcols])
			}
		}
	}
	r, err := asComplexArray(rv, d, rs)
	if err != nil || opt.ROnly {
		return nil, r, err
	}
	q, err := asComplexArray(qv, d, qs)
	return q, r, err
}

// Pinv computes the Moore–Penrose pseudoinverse with the supplied relative cutoff.
// A negative cutoff selects NumPy's default rcond=1e-15.
func Pinv(a *np.Array, rcond float64) (*np.Array, error) {
	if rcond < 0 {
		rcond = 1e-15
	}
	return PinvWithOptions(a, PinvOptions{RCond: np.Scalar(rcond)})
}

// PinvOptions allows per-matrix tolerances. DefaultRTol is the Array API
// rtol=None default (max(M,N)*machine epsilon), distinct from NumPy's 1e-15.
type PinvOptions struct {
	RCond, RTol *np.Array
	DefaultRTol bool
	Hermitian   bool
}

func PinvWithOptions(a *np.Array, opt PinvOptions) (*np.Array, error) {
	if a.DType() == 0 {
		return nil, np.ErrDType
	}
	if a.NDim() < 2 {
		return nil, np.ErrShape
	}
	if opt.RCond != nil && (opt.RTol != nil || opt.DefaultRTol) || opt.RTol != nil && opt.DefaultRTol {
		return nil, np.ErrValue
	}
	shape := a.Shape()
	batch := shape[:len(shape)-2]
	m, n := shape[len(shape)-2], shape[len(shape)-1]
	tol := opt.RCond
	if tol == nil {
		tol = opt.RTol
	}
	if opt.DefaultRTol {
		if !a.DType().IsFloat() && !a.DType().IsComplex() {
			return nil, np.ErrDType
		}
		info, e := np.FInfo(a.DType())
		if e != nil {
			return nil, e
		}
		d := a.DType()
		if d == np.Complex64 {
			d = np.Float32
		} else if d == np.Complex128 {
			d = np.Float64
		}
		tol, e = np.Full(d, float64(max(m, n))*info.Eps)
		if e != nil {
			return nil, e
		}
	}
	if tol == nil {
		tol = np.Scalar(1e-15)
	}
	outShape := append(slices.Clone(batch), n, m)
	if m == 0 || n == 0 {
		return np.Zeros(a.DType(), outShape...)
	}
	thresholds, e := realTolerance(tol, batch)
	if e != nil {
		return nil, e
	}
	u, s, vh, err := SVD(a, SVDOptions{Reduced: true, Hermitian: opt.Hermitian})
	if err != nil {
		return nil, err
	}
	k := min(m, n)
	d := u.DType()
	size, err := np.ElementCount(d, outShape...)
	if err != nil {
		return nil, err
	}
	out := make([]complex128, size)
	if k == 0 || size == 0 {
		return asComplexArray(out, d, outShape)
	}
	uv, _ := u.Complex128s()
	sv, _ := s.Complex128s()
	if m == n && !opt.Hermitian {
		// With zero cutoff and no discarded singular values the
		// pseudoinverse is the ordinary inverse. An LU solve avoids
		// amplification of tiny singular-vector errors by 1/sigma in
		// strongly graded nonsingular square matrices.
		full := true
		for q, t := range thresholds {
			if t != 0 || !(real(sv[q*k+k-1]) > 0) {
				full = false
				break
			}
		}
		if full {
			if inv, e := Inv(a); e == nil {
				return inv, nil
			}
		}
	}
	vv, _ := vh.Complex128s()
	for q := 0; q < size/(m*n); q++ {
		threshold := cutoffProduct(thresholds[q], real(sv[q*k]), tol.DType(), s.DType())
		for i := 0; i < n; i++ {
			for j := 0; j < m; j++ {
				var z complex128
				for h := 0; h < k; h++ {
					sigma := real(sv[q*k+h])
					if sigma > threshold {
						inv := 1 / sigma
						if s.DType() == np.Float32 {
							inv = float64(float32(inv))
						}
						z += cmplx.Conj(vv[q*k*n+h*n+i]) * cmplx.Conj(uv[q*m*k+j*k+h]) * complex(inv, 0)
					}
				}
				out[q*n*m+i*m+j] = z
			}
		}
	}
	return asComplexArray(out, d, outShape)
}

// MatrixRank uses singular values and a NumPy-style scale-dependent tolerance.
// A negative tol uses max(m,n)*eps*largest singular value.
func MatrixRank(a *np.Array, tol float64) (*np.Array, error) {
	if tol < 0 {
		return MatrixRankWithOptions(a, RankOptions{})
	}
	return MatrixRankWithOptions(a, RankOptions{Tol: np.Scalar(tol)})
}

type RankOptions struct {
	Tol, RTol *np.Array // mutually exclusive absolute and relative cutoffs
	Hermitian bool
}

func MatrixRankWithOptions(a *np.Array, opt RankOptions) (*np.Array, error) {
	if opt.Tol != nil && opt.RTol != nil {
		return nil, np.ErrValue
	}
	if a.DType().ItemSize() == 0 {
		return nil, np.ErrDType
	}
	if a.NDim() < 2 {
		v, _ := a.Complex128s()
		var rank int64
		for _, z := range v {
			if z != 0 {
				rank = 1
				break
			}
		}
		return np.ScalarOf(rank), nil
	}
	_, s, _, err := SVD(a, SVDOptions{ValuesOnly: true, Hermitian: opt.Hermitian})
	if err != nil {
		return nil, err
	}
	shape := a.Shape()
	batch := shape[:len(shape)-2]
	m, n := shape[len(shape)-2], shape[len(shape)-1]
	k := min(m, n)
	tol := opt.Tol
	relative := tol == nil
	if relative {
		tol = opt.RTol
	}
	if tol == nil {
		info, e := np.FInfo(s.DType())
		if e != nil {
			return nil, e
		}
		tol, e = np.Full(s.DType(), float64(max(m, n))*info.Eps)
		if e != nil {
			return nil, e
		}
	}
	resultShape, err := np.BroadcastShapes(batch, tol.Shape())
	if err != nil {
		return nil, err
	}
	tolerances, err := realTolerance(tol, resultShape)
	if err != nil {
		return nil, err
	}
	count, err := np.ElementCount(np.Int64, resultShape...)
	if err != nil {
		return nil, err
	}
	out := make([]int64, count)
	sv, _ := s.Complex128s()
	for q := 0; q < count; q++ {
		if k == 0 {
			continue
		}
		si := matrixOffset(q, resultShape, batch, k)
		threshold := tolerances[q]
		if relative {
			threshold = cutoffProduct(threshold, real(sv[si]), tol.DType(), s.DType())
		}
		for j := 0; j < k; j++ {
			if real(sv[si+j]) > threshold {
				out[q]++
			}
		}
	}
	return np.WrapSlice(out, resultShape...)
}

func realTolerance(a *np.Array, shape []int) ([]float64, error) {
	if a.DType() == 0 || a.DType().IsComplex() {
		return nil, np.ErrDType
	}
	v, e := a.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	v, e = v.Astype(np.Float64)
	if e != nil {
		return nil, e
	}
	return v.Float64s()
}
func cutoffProduct(a, b float64, da, db np.DType) float64 {
	d, _ := np.PromoteTypes(da, db)
	if d == np.Float32 {
		return float64(float32(a) * float32(b))
	}
	return a * b
}
