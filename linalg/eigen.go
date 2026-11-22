package linalg

import (
	"math"
	"math/cmplx"
	"slices"

	"github.com/SamuelSupe/numpygo"
	"gonum.org/v1/gonum/blas/cblas128"
	"gonum.org/v1/gonum/mat"
)

func finiteMatrix(values []complex128) bool {
	for _, z := range values {
		if math.IsNaN(real(z)) || math.IsNaN(imag(z)) || math.IsInf(real(z), 0) || math.IsInf(imag(z), 0) {
			return false
		}
	}
	return true
}

// realEmbedding represents a complex matrix as [[Re(A),-Im(A)],[Im(A),Re(A)]].
func realEmbedding(a []complex128, n int) []float64 {
	b := make([]float64, 4*n*n)
	w := 2 * n
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			z := a[i*n+j]
			b[i*w+j] = real(z)
			b[i*w+n+j] = -imag(z)
			b[(i+n)*w+j] = imag(z)
			b[(i+n)*w+n+j] = real(z)
		}
	}
	return b
}
func vectorNorm(v []complex128) float64 {
	if len(v) >= 32 {
		return cblas128.Nrm2(cblas128.Vector{N: len(v), Inc: 1, Data: v})
	}
	var norm float64
	for _, x := range v {
		norm = math.Hypot(norm, cmplx.Abs(x))
	}
	return norm
}
func orthogonalize(v []complex128, basis [][]complex128) float64 {
	for pass := 0; pass < 2; pass++ {
		for _, u := range basis {
			if len(v) >= 32 {
				dot := cblas128.Dotc(contiguousVector(u), contiguousVector(v))
				cblas128.Axpy(-dot, contiguousVector(u), contiguousVector(v))
				continue
			}
			var dot complex128
			for i, z := range v {
				dot += cmplx.Conj(u[i]) * z
			}
			for i := range v {
				v[i] -= dot * u[i]
			}
		}
	}
	return vectorNorm(v)
}
func normalizeVector(v []complex128, norm float64) {
	for i := range v {
		v[i] = complex(real(v[i])/norm, imag(v[i])/norm)
	}
}

// Eigh computes ascending eigenvalues and orthonormal eigenvectors of a real
// symmetric or complex Hermitian matrix, using the lower triangle by default.
func Eigh(a *np.Array, upper ...bool) (*np.Array, *np.Array, error) {
	return eigh(a, true, upper...)
}
func eigh(a *np.Array, vectors bool, upper ...bool) (*np.Array, *np.Array, error) {
	if len(upper) > 1 {
		return nil, nil, np.ErrValue
	}
	up := len(upper) == 1 && upper[0]
	d, batch, n, count, err := squareShape(a)
	if err != nil {
		return nil, nil, err
	}
	shapeV := a.Shape()
	shapeE := append(slices.Clone(batch), n)
	sd := d
	if d == np.Complex128 {
		sd = np.Float64
	} else if d == np.Complex64 {
		sd = np.Float32
	}
	en, err := np.ElementCount(sd, shapeE...)
	if err != nil {
		return nil, nil, err
	}
	values := make([]float64, en)
	var vecs []complex128
	if vectors {
		vn, e := np.ElementCount(d, shapeV...)
		if e != nil {
			return nil, nil, e
		}
		vecs = make([]complex128, vn)
	}
	av, _ := a.Complex128s()
	if n != 0 {
		for q := 0; q < count; q++ {
			h := make([]complex128, n*n)
			src := av[q*n*n : (q+1)*n*n]
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					z := src[i*n+j]
					if up {
						z = cmplx.Conj(src[j*n+i])
					}
					if i == j {
						z = complex(real(z), 0)
					}
					h[i*n+j] = z
					h[j*n+i] = cmplx.Conj(z)
				}
			}
			if !finiteMatrix(h) {
				return nil, nil, ErrConvergence
			}
			vals, vec, err := hermitianEigen(h, n, d.IsComplex(), vectors)
			if err != nil {
				return nil, nil, err
			}
			copy(values[q*n:], vals)
			if vectors {
				copy(vecs[q*n*n:], vec)
			}
		}
	}
	s, err := np.WrapFloat64(values, shapeE...)
	if err != nil {
		return nil, nil, err
	}
	if sd != np.Float64 {
		s, err = s.Astype(sd)
		if err != nil {
			return nil, nil, err
		}
	}
	if !vectors {
		return s, nil, nil
	}
	v, err := asComplexArray(vecs, d, shapeV)
	return s, v, err
}
func Eigvalsh(a *np.Array, upper ...bool) (*np.Array, error) {
	s, _, err := eigh(a, false, upper...)
	return s, err
}

func hermitianEigen(h []complex128, n int, isComplex, vectors bool) ([]float64, []complex128, error) {
	if isComplex && n >= 16 {
		return complexHermitianEigen(h, n, vectors)
	}
	width := n
	var data []float64
	if isComplex {
		width = 2 * n
		data = realEmbedding(h, n)
	} else {
		data = make([]float64, n*n)
		for i, z := range h {
			data[i] = real(z)
		}
	}
	var eigen mat.EigenSym
	if !eigen.Factorize(mat.NewSymDense(width, data), vectors) {
		return nil, nil, ErrConvergence
	}
	e := eigen.Values(nil)
	if !vectors {
		if !isComplex {
			return e, nil, nil
		}
		// Each complex Hermitian eigenvalue occurs twice in the real
		// embedding. Sorted adjacent pairs give its multiplicity without
		// computing an orthogonal basis for an unrequested vector output.
		result := make([]float64, n)
		for i := range result {
			result[i] = e[2*i]
		}
		return result, nil, nil
	}
	var v mat.Dense
	eigen.VectorsTo(&v)
	out := make([]complex128, n*n)
	if !isComplex {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				out[i*n+j] = complex(v.At(i, j), 0)
			}
		}
		return e, out, nil
	}
	// The real embedding contains each eigenvalue twice; maintain a complex
	// orthonormal basis to select one representative per complex dimension.
	var basis [][]complex128
	result := make([]float64, 0, n)
	for j := 0; j < width && len(basis) < n; j++ {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(v.At(i, j), v.At(i+n, j))
		}
		norm := orthogonalize(x, basis)
		if norm < 1e-7 {
			continue
		}
		normalizeVector(x, norm)
		basis = append(basis, x)
		result = append(result, e[j])
	}
	if len(basis) != n {
		return nil, nil, ErrConvergence
	}
	for j, x := range basis {
		for i, z := range x {
			out[i*n+j] = z
		}
	}
	return result, out, nil
}

// Eig returns the eigenvalues and right eigenvectors of general square matrices.
// Eigenvalue order and eigenvector phase follow the underlying decomposition.
func Eig(a *np.Array) (*np.Array, *np.Array, error) {
	return eig(a, true)
}
func eig(a *np.Array, vectors bool) (*np.Array, *np.Array, error) {
	d, batch, n, count, err := squareShape(a)
	if err != nil {
		return nil, nil, err
	}
	es := append(slices.Clone(batch), n)
	vs := a.Shape()
	en, err := np.ElementCount(np.Complex128, es...)
	if err != nil {
		return nil, nil, err
	}
	ev := make([]complex128, en)
	var vv []complex128
	if vectors {
		vn, e := np.ElementCount(np.Complex128, vs...)
		if e != nil {
			return nil, nil, e
		}
		vv = make([]complex128, vn)
	}
	av, _ := a.Complex128s()
	if n != 0 {
		for q := 0; q < count; q++ {
			x := av[q*n*n : (q+1)*n*n]
			if !finiteMatrix(x) {
				return nil, nil, ErrConvergence
			}
			vals, vec, err := generalEigen(x, n, d.IsComplex(), vectors)
			if err != nil {
				return nil, nil, err
			}
			copy(ev[q*n:], vals)
			if vectors {
				copy(vv[q*n*n:], vec)
			}
		}
	}
	// NumPy 2.5 general eigen routines return complex precision even
	// when every eigenvalue is real (including empty matrices).
	if d == np.Float64 {
		d = np.Complex128
	} else if d == np.Float32 {
		d = np.Complex64
	}
	e, err := asComplexArray(ev, d, es)
	if err != nil {
		return nil, nil, err
	}
	if !vectors {
		return e, nil, nil
	}
	v, err := asComplexArray(vv, d, vs)
	return e, v, err
}
func Eigvals(a *np.Array) (*np.Array, error) { e, _, err := eig(a, false); return e, err }

func generalEigen(a []complex128, n int, isComplex, vectors bool) ([]complex128, []complex128, error) {
	if isComplex {
		return complexEigen(a, n, vectors)
	}
	data := make([]float64, n*n)
	for i, z := range a {
		data[i] = real(z)
	}
	var eigen mat.Eigen
	kind := mat.EigenNone
	if vectors {
		kind = mat.EigenRight
	}
	if !eigen.Factorize(mat.NewDense(n, n, data), kind) {
		return nil, nil, ErrConvergence
	}
	values := eigen.Values(nil)
	if !vectors {
		return values, nil, nil
	}
	var vec mat.CDense
	eigen.VectorsTo(&vec)
	out := make([]complex128, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out[i*n+j] = vec.At(i, j)
		}
	}
	return values, out, nil
}
