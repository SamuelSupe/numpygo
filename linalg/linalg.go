// Package linalg provides NumPy-style batched linear algebra.
package linalg

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
	"gonum.org/v1/gonum/mat"
)

var (
	ErrSingular    = errors.New("singular matrix")
	ErrConvergence = errors.New("matrix decomposition did not converge")
)

func matrixOffset(flat int, batch, operand []int, matrixSize int) int {
	index, stride := 0, matrixSize
	for i, j := len(batch)-1, len(operand)-1; j >= 0; i, j = i-1, j-1 {
		if operand[j] != 1 {
			index += (flat % batch[i]) * stride
		}
		flat /= batch[i]
		stride *= operand[j]
	}
	return index
}

func matrix(rows, cols int, values []float64) blas64.General {
	return blas64.General{Rows: rows, Cols: cols, Stride: cols, Data: values}
}

// MatMul implements NumPy matmul for numerical arrays, including rank-one
// operands and broadcastable leading batch dimensions. Rank-zero operands fail.
func MatMul(a, b *np.Array) (*np.Array, error) {
	dtype, err := np.PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if dtype.IsTemporal() || dtype == np.Object || dtype == np.RecordDType {
		return nil, np.ErrDType
	}
	ashape, bshape := a.Shape(), b.Shape()
	if len(ashape) == 0 || len(bshape) == 0 {
		return nil, np.ErrShape
	}
	aVector, bVector := len(ashape) == 1, len(bshape) == 1
	if aVector {
		ashape = []int{1, ashape[0]}
	}
	if bVector {
		bshape = []int{bshape[0], 1}
	}
	m, k := ashape[len(ashape)-2], ashape[len(ashape)-1]
	if k != bshape[len(bshape)-2] {
		return nil, fmt.Errorf("%w: matmul inner dimensions differ", np.ErrShape)
	}
	n := bshape[len(bshape)-1]
	abatch, bbatch := ashape[:len(ashape)-2], bshape[:len(bshape)-2]
	batch, err := np.BroadcastShapes(abatch, bbatch)
	if err != nil {
		return nil, err
	}
	shape := slices.Clone(batch)
	if !aVector {
		shape = append(shape, m)
	}
	if !bVector {
		shape = append(shape, n)
	}
	size, err := np.ElementCount(dtype, shape...)
	if err != nil {
		return nil, err
	}
	if size == 0 || k == 0 {
		return np.Zeros(dtype, shape...)
	}
	if a.DType() != dtype {
		a, err = a.Astype(dtype)
		if err != nil {
			return nil, err
		}
	}
	if b.DType() != dtype {
		b, err = b.Astype(dtype)
		if err != nil {
			return nil, err
		}
	}
	if dtype != np.Float64 {
		return matMulOther(a, b, dtype, shape, batch, abatch, bbatch, m, k, n, size)
	}
	return matMulBLAS[float64](a, b, shape, batch, abatch, bbatch, m, k, n, size)
}

// Solve solves a*x=b for numerical square matrices. As in NumPy 2.x, b is treated
// as a vector only when it is exactly one-dimensional; otherwise the last two
// axes form a matrix of right-hand sides. Leading axes broadcast.
// Solve does not change either input, and wraps ErrSingular for a zero pivot.
func Solve(a, b *np.Array) (*np.Array, error) {
	dtype, err := linearDType(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	ashape, bshape := a.Shape(), b.Shape()
	if len(ashape) < 2 || len(bshape) == 0 {
		return nil, np.ErrShape
	}
	n := ashape[len(ashape)-1]
	if ashape[len(ashape)-2] != n {
		return nil, fmt.Errorf("%w: solve requires square matrices", np.ErrShape)
	}
	vector := len(bshape) == 1
	if vector {
		bshape = []int{bshape[0], 1}
	}
	if bshape[len(bshape)-2] != n {
		return nil, fmt.Errorf("%w: incompatible right-hand side", np.ErrShape)
	}
	k := bshape[len(bshape)-1]
	abatch, bbatch := ashape[:len(ashape)-2], bshape[:len(bshape)-2]
	batch, err := np.BroadcastShapes(abatch, bbatch)
	if err != nil {
		return nil, err
	}
	shape := append(slices.Clone(batch), n)
	if !vector {
		shape = append(shape, k)
	}
	size, err := np.ElementCount(dtype, shape...)
	if err != nil {
		return nil, err
	}
	batches, err := np.ElementCount(np.Float64, batch...)
	if err != nil {
		return nil, err
	}
	if n == 0 || batches == 0 {
		return np.Zeros(dtype, shape...)
	}
	if dtype.IsComplex() {
		return solveComplex(a, b, dtype, shape, batch, abatch, bbatch, n, k, size, batches)
	}
	if a.DType() != np.Float64 {
		a, err = a.Astype(np.Float64)
		if err != nil {
			return nil, err
		}
	}
	if b.DType() != np.Float64 {
		b, err = b.Astype(np.Float64)
		if err != nil {
			return nil, err
		}
	}
	values := make([]float64, size)
	av, _ := a.Float64s()
	bv, _ := b.Float64s()
	lu := make([]float64, n*n)
	pivots := make([]int, n)
	last := -1
	for i := 0; i < batches; i++ {
		ai := matrixOffset(i, batch, abatch, n*n)
		am := matrix(n, n, lu)
		// Broadcasted RHS batches often share the same coefficient matrix.
		// Getrs reads the LU factors and pivots without changing them.
		if ai != last {
			copy(lu, av[ai:ai+n*n])
			if !lapack64.Getrf(am, pivots) {
				return nil, fmt.Errorf("%w: batch matrix %d", ErrSingular, i)
			}
			last = ai
		}
		// NumPy still factors A and checks singularity with zero RHS columns.
		if k == 0 {
			continue
		}
		bi := matrixOffset(i, batch, bbatch, n*k)
		rhs := values[i*n*k : (i+1)*n*k]
		copy(rhs, bv[bi:bi+n*k])
		lapack64.Getrs(blas.NoTrans, am, matrix(n, k, rhs), pivots)
	}
	out, err := np.WrapFloat64(values, shape...)
	if err != nil || dtype == np.Float64 {
		return out, err
	}
	return out.Astype(dtype)
}

// SVDOptions selects the shape of singular vector matrices and whether to
// compute them. The zero value matches NumPy full_matrices=True, compute_uv=True.
type SVDOptions struct {
	Reduced    bool
	ValuesOnly bool
	Hermitian  bool // assume symmetric/Hermitian and use the lower triangle
}

// SVD computes the decomposition A = U diag(S) Vh across the final two axes.
// S is descending. Singular vector signs and bases within degenerate singular
// subspaces may differ from other correct implementations. ValuesOnly returns
// nil U and Vh. The input must be a finite numerical array of rank at least two.
func SVD(a *np.Array, options ...SVDOptions) (u, s, vh *np.Array, err error) {
	if len(options) > 1 {
		return nil, nil, nil, np.ErrShape
	}
	if len(options) == 1 && options[0].Hermitian {
		return hermitianSVD(a, options[0])
	}
	dtype, err := linearDType(a.DType())
	if err != nil {
		return nil, nil, nil, err
	}
	if dtype.IsComplex() {
		return complexSVD(a, dtype, options)
	}
	if a.DType() != np.Float64 {
		a, err = a.Astype(np.Float64)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	u, s, vh, err = realSVD(a, options...)
	if err != nil || dtype == np.Float64 {
		return u, s, vh, err
	}
	s, err = s.Astype(dtype)
	if err != nil {
		return nil, nil, nil, err
	}
	if u != nil {
		u, err = u.Astype(dtype)
		if err != nil {
			return nil, nil, nil, err
		}
		vh, err = vh.Astype(dtype)
	}
	return u, s, vh, err
}

func realSVD(a *np.Array, options ...SVDOptions) (u, s, vh *np.Array, err error) {
	if len(options) > 1 || a.NDim() < 2 {
		return nil, nil, nil, np.ErrShape
	}
	var opt SVDOptions
	if len(options) == 1 {
		opt = options[0]
	}
	shape := a.Shape()
	batch := shape[:len(shape)-2]
	m, n := shape[len(shape)-2], shape[len(shape)-1]
	k := min(m, n)
	uc, vr := m, n
	kind := mat.SVDFull
	if opt.Reduced {
		uc, vr = k, k
		kind = mat.SVDThin
	}
	if opt.ValuesOnly {
		kind = mat.SVDNone
	}
	sshape := append(slices.Clone(batch), k)
	ushape := append(slices.Clone(batch), m, uc)
	vshape := append(slices.Clone(batch), vr, n)
	ssize, err := np.ElementCount(np.Float64, sshape...)
	if err != nil {
		return nil, nil, nil, err
	}
	sv := make([]float64, ssize)
	var uv, vv []float64
	if !opt.ValuesOnly {
		usize, err := np.ElementCount(np.Float64, ushape...)
		if err != nil {
			return nil, nil, nil, err
		}
		vsize, err := np.ElementCount(np.Float64, vshape...)
		if err != nil {
			return nil, nil, nil, err
		}
		uv, vv = make([]float64, usize), make([]float64, vsize)
	}
	if k == 0 && !opt.ValuesOnly && !opt.Reduced {
		// Empty rectangular matrices still have full orthonormal bases.
		for _, side := range []struct {
			data []float64
			n    int
		}{{uv, m}, {vv, n}} {
			if side.n > 0 {
				for off := 0; off < len(side.data); off += side.n * side.n {
					for i := 0; i < side.n; i++ {
						side.data[off+i*side.n+i] = 1
					}
				}
			}
		}
	}
	if k != 0 && ssize != 0 {
		av, _ := a.Float64s()
		for _, value := range av {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, nil, nil, fmt.Errorf("%w: non-finite input", ErrConvergence)
			}
		}
		var decomposition mat.SVD
		for i := 0; i < ssize/k; i++ {
			if !decomposition.Factorize(mat.NewDense(m, n, av[i*m*n:(i+1)*m*n]), kind) {
				return nil, nil, nil, fmt.Errorf("%w: batch matrix %d", ErrConvergence, i)
			}
			decomposition.Values(sv[i*k : (i+1)*k])
			if !opt.ValuesOnly {
				var left, right mat.Dense
				decomposition.UTo(&left)
				decomposition.VTo(&right)
				copy(uv[i*m*uc:(i+1)*m*uc], left.RawMatrix().Data)
				for row := 0; row < vr; row++ {
					for col := 0; col < n; col++ {
						vv[i*vr*n+row*n+col] = right.At(col, row)
					}
				}
			}
		}
	}
	s, err = np.WrapFloat64(sv, sshape...)
	if err != nil || opt.ValuesOnly {
		return nil, s, nil, err
	}
	u, err = np.WrapFloat64(uv, ushape...)
	if err != nil {
		return nil, nil, nil, err
	}
	vh, err = np.WrapFloat64(vv, vshape...)
	if err != nil {
		return nil, nil, nil, err
	}
	return u, s, vh, nil
}
