package linalg

import (
	"errors"
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

func SVDVals(a *np.Array) (*np.Array, error) {
	_, s, _, err := SVD(a, SVDOptions{ValuesOnly: true})
	return s, err
}
func MatrixTranspose(a *np.Array) (*np.Array, error) {
	if a.DType() == 0 || a.NDim() < 2 {
		return nil, np.ErrShape
	}
	return np.SwapAxes(a, -1, -2)
}

// MatVec multiplies matrices by vectors, broadcasting leading dimensions.
func MatVec(a, b *np.Array) (*np.Array, error) {
	if a.NDim() < 2 || b.NDim() < 1 {
		return nil, np.ErrShape
	}
	v, err := np.ExpandDims(b, -1)
	if err != nil {
		return nil, err
	}
	out, err := MatMul(a, v)
	if err != nil {
		return nil, err
	}
	return np.Squeeze(out, -1)
}

// VecMat multiplies the conjugate of each vector by its corresponding matrix.
func VecMat(a, b *np.Array) (*np.Array, error) {
	if a.NDim() < 1 || b.NDim() < 2 {
		return nil, np.ErrShape
	}
	var err error
	if a.DType().IsComplex() {
		a, err = np.Conjugate(a)
		if err != nil {
			return nil, err
		}
	}
	v, err := np.ExpandDims(a, -2)
	if err != nil {
		return nil, err
	}
	out, err := MatMul(v, b)
	if err != nil {
		return nil, err
	}
	return np.Squeeze(out, -2)
}

func Vecdot(a, b *np.Array, axis int) (*np.Array, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return nil, np.ErrDType
	}
	x, err := np.MoveAxis(a, axis, -1)
	if err != nil {
		return nil, err
	}
	y, err := np.MoveAxis(b, axis, -1)
	if err != nil {
		return nil, err
	}
	xs, ys := x.Shape(), y.Shape()
	if xs[len(xs)-1] != ys[len(ys)-1] {
		return nil, np.ErrShape
	}
	if x.DType().IsComplex() {
		x, err = np.Conjugate(x)
		if err != nil {
			return nil, err
		}
	}
	return np.Einsum("...i,...i->...", x, y)
}
func Cross(a, b *np.Array, axis int) (*np.Array, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return nil, np.ErrDType
	}
	ax, err := np.NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	bx, err := np.NormalizeAxis(axis, b.NDim())
	if err != nil {
		return nil, err
	}
	if a.Shape()[ax] != 3 || b.Shape()[bx] != 3 {
		return nil, np.ErrShape
	}
	return np.Cross(a, b, axis, axis, axis)
}
func MatrixNorm(a *np.Array, ord string, keepDims bool) (*np.Array, error) {
	return Norm(a, NormOptions{Ord: ord, Axes: []int{-2, -1}, KeepDims: keepDims})
}
func VectorNorm(a *np.Array, ord string, axes []int, keepDims bool) (*np.Array, error) {
	if a.DType() == 0 {
		return nil, np.ErrDType
	}
	if axes == nil {
		x, err := np.Ravel(a)
		if err != nil {
			return nil, err
		}
		out, err := Norm(x, NormOptions{Ord: ord, Axes: []int{0}})
		if err != nil {
			return nil, err
		}
		if keepDims {
			shape := make([]int, a.NDim())
			for i := range shape {
				shape[i] = 1
			}
			return out.Reshape(shape...)
		}
		return out, nil
	}
	if len(axes) == 1 {
		return Norm(a, NormOptions{Ord: ord, Axes: axes, KeepDims: keepDims})
	}
	shape := a.Shape()
	seen := make([]bool, a.NDim())
	normalized := make([]int, len(axes))
	for i, axis := range axes {
		ax, err := np.NormalizeAxis(axis, a.NDim())
		if err != nil || seen[ax] {
			return nil, np.ErrAxis
		}
		seen[ax] = true
		normalized[i] = ax
	}
	var order, batch, redShape []int
	for i, d := range shape {
		if !seen[i] {
			order = append(order, i)
			batch = append(batch, d)
		}
	}
	for _, ax := range normalized {
		redShape = append(redShape, shape[ax])
	}
	count, err := np.ElementCount(a.DType(), redShape...)
	if err != nil {
		return nil, err
	}
	order = append(order, normalized...)
	x, err := a.Transpose(order...)
	if err != nil {
		return nil, err
	}
	x, err = x.Reshape(append(slices.Clone(batch), count)...)
	if err != nil {
		return nil, err
	}
	out, err := Norm(x, NormOptions{Ord: ord, Axes: []int{-1}})
	if err != nil {
		return nil, err
	}
	if keepDims {
		for i := range shape {
			if seen[i] {
				shape[i] = 1
			}
		}
		return out.Reshape(shape...)
	}
	return out, nil
}

// Cond computes the condition number for the standard spectral and matrix norms.
func Cond(a *np.Array, ord string) (*np.Array, error) {
	d, err := linearDType(a.DType())
	if err != nil {
		return nil, err
	}
	if a.NDim() < 2 {
		return nil, np.ErrShape
	}
	sh := a.Shape()
	batch := sh[:len(sh)-2]
	m, n := sh[len(sh)-2], sh[len(sh)-1]
	k := min(m, n)
	if k == 0 {
		return nil, np.ErrValue
	}
	count, err := np.ElementCount(np.Float64, batch...)
	if err != nil {
		return nil, err
	}
	out := make([]float64, count)
	if ord == "" || ord == "2" || ord == "-2" {
		s, err := SVDVals(a)
		if err != nil {
			return nil, err
		}
		sv, _ := s.Complex128s()
		for i := range out {
			hi, lo := real(sv[i*k]), real(sv[(i+1)*k-1])
			if ord == "-2" {
				out[i] = lo / hi
			} else {
				out[i] = hi / lo
			}
			if math.IsNaN(out[i]) {
				out[i] = math.Inf(1)
			}
		}
	} else {
		if m != n {
			return nil, np.ErrShape
		}
		av, _ := a.Complex128s()
		for i := range out {
			x := av[i*n*n : (i+1)*n*n]
			l, err := matrixOrder(x, n, n, ord, np.Float64)
			if err != nil {
				return nil, err
			}
			y, err := asComplexArray(x, d, []int{n, n})
			if err != nil {
				return nil, err
			}
			inv, err := Inv(y)
			if errors.Is(err, ErrSingular) {
				out[i] = math.Inf(1)
				continue
			}
			if err != nil {
				return nil, err
			}
			iv, _ := inv.Complex128s()
			r, err := matrixOrder(iv, n, n, ord, np.Float64)
			if err != nil {
				return nil, err
			}
			out[i] = l * r
		}
	}
	res, err := np.WrapSlice(out, batch...)
	if err != nil {
		return nil, err
	}
	if d == np.Float32 || d == np.Complex64 {
		return res.Astype(np.Float32)
	}
	return res, nil
}

// MultiDot chooses a multiplication order minimizing scalar operations.
func MultiDot(arrays ...*np.Array) (*np.Array, error) {
	if len(arrays) < 2 {
		return nil, np.ErrShape
	}
	if len(arrays) == 2 {
		return dot2(arrays[0], arrays[1])
	}
	a := slices.Clone(arrays)
	first := a[0].NDim() == 1
	last := a[len(a)-1].NDim() == 1
	var err error
	if first {
		a[0], err = a[0].Reshape(1, a[0].Size())
		if err != nil {
			return nil, err
		}
	}
	if last {
		a[len(a)-1], err = a[len(a)-1].Reshape(a[len(a)-1].Size(), 1)
		if err != nil {
			return nil, err
		}
	}
	n := len(a)
	dims := make([]int, n+1)
	for i, x := range a {
		if x.DType() == 0 || x.NDim() != 2 {
			return nil, np.ErrShape
		}
		sh := x.Shape()
		if i == 0 {
			dims[0] = sh[0]
		} else if dims[i] != sh[0] {
			return nil, np.ErrShape
		}
		dims[i+1] = sh[1]
	}
	cost, split := make([]float64, n*n), make([]int, n*n)
	for length := 2; length <= n; length++ {
		for i := 0; i+length <= n; i++ {
			j := i + length - 1
			cost[i*n+j] = math.Inf(1)
			for k := i; k < j; k++ {
				v := cost[i*n+k] + cost[(k+1)*n+j] + float64(dims[i])*float64(dims[k+1])*float64(dims[j+1])
				if v < cost[i*n+j] {
					cost[i*n+j] = v
					split[i*n+j] = k
				}
			}
		}
	}
	var evaluate func(int, int) (*np.Array, error)
	evaluate = func(i, j int) (*np.Array, error) {
		if i == j {
			return a[i], nil
		}
		k := split[i*n+j]
		x, err := evaluate(i, k)
		if err != nil {
			return nil, err
		}
		y, err := evaluate(k+1, j)
		if err != nil {
			return nil, err
		}
		return MatMul(x, y)
	}
	out, err := evaluate(0, n-1)
	if err != nil {
		return nil, err
	}
	var shape []int
	if !first {
		shape = append(shape, dims[0])
	}
	if !last {
		shape = append(shape, dims[n])
	}
	return out.Reshape(shape...)
}
func dot2(a, b *np.Array) (*np.Array, error) {
	if a.NDim() < 1 || a.NDim() > 2 || b.NDim() < 1 || b.NDim() > 2 {
		return nil, np.ErrShape
	}
	return MatMul(a, b)
}

func TensorInv(a *np.Array, ind int) (*np.Array, error) {
	if a.DType() == 0 || ind <= 0 || ind > a.NDim() {
		return nil, np.ErrShape
	}
	sh := a.Shape()
	left, err := np.ElementCount(a.DType(), sh[:ind]...)
	if err != nil {
		return nil, err
	}
	right, err := np.ElementCount(a.DType(), sh[ind:]...)
	if err != nil || left != right {
		return nil, np.ErrShape
	}
	x, err := a.Reshape(left, right)
	if err != nil {
		return nil, err
	}
	x, err = Inv(x)
	if err != nil {
		return nil, err
	}
	shape := append(slices.Clone(sh[ind:]), sh[:ind]...)
	return x.Reshape(shape...)
}
func TensorSolve(a, b *np.Array, axes []int) (*np.Array, error) {
	if a.DType() == 0 || b.DType() == 0 || a.NDim() < b.NDim() {
		return nil, np.ErrShape
	}
	if len(axes) > 0 {
		seen := make([]bool, a.NDim())
		tail := make([]int, len(axes))
		for i, axis := range axes {
			ax, err := np.NormalizeAxis(axis, a.NDim())
			if err != nil || seen[ax] {
				return nil, np.ErrAxis
			}
			seen[ax] = true
			tail[i] = ax
		}
		var order []int
		for i := range seen {
			if !seen[i] {
				order = append(order, i)
			}
		}
		order = append(order, tail...)
		var err error
		a, err = a.Transpose(order...)
		if err != nil {
			return nil, err
		}
	}
	sh := a.Shape()
	resultShape := sh[b.NDim():]
	n, err := np.ElementCount(a.DType(), resultShape...)
	if err != nil {
		return nil, err
	}
	if b.Size() != n {
		return nil, np.ErrShape
	}
	x, err := a.Reshape(n, n)
	if err != nil {
		return nil, err
	}
	y, err := b.Reshape(-1)
	if err != nil {
		return nil, err
	}
	out, err := Solve(x, y)
	if err != nil {
		return nil, err
	}
	return out.Reshape(resultShape...)
}
