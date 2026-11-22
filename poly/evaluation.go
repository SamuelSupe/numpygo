package poly

import (
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/linalg"
)

// EvalND evaluates a coefficient tensor at matching point arrays, or on their
// Cartesian product if grid is true. Coefficient axes follow point order.
func (b Basis) EvalND(c *np.Array, grid bool, points ...*np.Array) (*np.Array, error) {
	if !b.valid() || c.DType() == 0 {
		return nil, np.ErrDType
	}
	if len(points) < 1 || len(points) > c.NDim() {
		return nil, np.ErrShape
	}
	if len(points) == 1 {
		return b.Eval(c, points[0])
	}
	shape := c.Shape()
	n := len(points)
	// NumPy permits trailing dimensions after the coefficient axes for multiple series.
	seriesShape := slices.Clone(shape[n:])
	seriesCount, err := np.ElementCount(c.DType(), seriesShape...)
	if err != nil {
		return nil, err
	}
	coeffCount, err := np.ElementCount(c.DType(), shape[:n]...)
	if err != nil {
		return nil, err
	}
	d := c.DType()
	outShape := slices.Clone(seriesShape)
	pvals := make([][]complex128, n)
	basis := make([][]complex128, n)
	for axis, p := range points {
		if p.DType() == 0 {
			return nil, np.ErrDType
		}
		d, err = numericType(d, p.DType())
		if err != nil {
			return nil, err
		}
		if grid {
			outShape = append(outShape, p.Shape()...)
		} else if axis == 0 {
			outShape = append(outShape, p.Shape()...)
		} else if !slices.Equal(points[0].Shape(), p.Shape()) {
			return nil, np.ErrShape
		}
		if shape[axis] == 0 {
			return nil, np.ErrShape
		}
		pvals[axis], _ = p.Complex128s()
		count, err := np.ElementCount(np.Complex128, p.Size(), shape[axis])
		if err != nil {
			return nil, err
		}
		basis[axis] = make([]complex128, count)
		for j, x := range pvals[axis] {
			fillBasis(basis[axis][j*shape[axis]:(j+1)*shape[axis]], x, b)
		}
	}
	size, err := np.ElementCount(d, outShape...)
	if err != nil {
		return nil, err
	}
	out := make([]complex128, size)
	if size == 0 {
		return array(out, d, outShape...)
	}
	cv, _ := c.Complex128s()
	pointCount := size / seriesCount
	pidx := make([]int, n)
	for j := 0; j < pointCount; j++ {
		if grid {
			flat := j
			for axis := n - 1; axis >= 0; axis-- {
				pidx[axis] = flat % len(pvals[axis])
				flat /= len(pvals[axis])
			}
		} else {
			for axis := range pidx {
				pidx[axis] = j
			}
		}
		for k := 0; k < coeffCount; k++ {
			factor := complex128(1)
			flat := k
			for axis := n - 1; axis >= 0; axis-- {
				i := flat % shape[axis]
				flat /= shape[axis]
				factor *= basis[axis][pidx[axis]*shape[axis]+i]
			}
			for s := 0; s < seriesCount; s++ {
				out[s*pointCount+j] += cv[k*seriesCount+s] * factor
			}
		}
	}
	return array(out, d, outShape...)
}

// VanderND builds the tensor-product Vandermonde matrix for equal-shape points,
// with the last dimension flattened in C order.
func (b Basis) VanderND(degrees []int, points ...*np.Array) (*np.Array, error) {
	if len(degrees) == 0 || len(degrees) != len(points) {
		return nil, np.ErrShape
	}
	var d np.DType
	var shape []int
	bases := make([][]complex128, len(points))
	terms := make([]int, len(points))
	for i, p := range points {
		if p.DType() == 0 || degrees[i] < 0 || degrees[i] == int(^uint(0)>>1) {
			return nil, np.ErrShape
		}
		v, err := b.Vander(p, degrees[i])
		if err != nil {
			return nil, err
		}
		x := v.Shape()
		x = x[:len(x)-1]
		if i == 0 {
			shape = x
			d = v.DType()
		} else if !slices.Equal(shape, x) {
			return nil, np.ErrShape
		}
		d, err = numericType(d, v.DType())
		if err != nil {
			return nil, err
		}
		bases[i], _ = v.Complex128s()
		terms[i] = degrees[i] + 1
	}
	width, err := np.ElementCount(d, terms...)
	if err != nil {
		return nil, err
	}
	outShape := append(shape, width)
	size, err := np.ElementCount(d, outShape...)
	if err != nil {
		return nil, err
	}
	out := make([]complex128, size)
	for row := 0; row < size/width; row++ {
		for j := 0; j < width; j++ {
			value := complex128(1)
			flat := j
			for axis := len(terms) - 1; axis >= 0; axis-- {
				idx := flat % terms[axis]
				flat /= terms[axis]
				value *= bases[axis][row*terms[axis]+idx]
			}
			out[row*width+j] = value
		}
	}
	return array(out, d, outShape...)
}

// ValFromRoots evaluates the monic power-basis polynomial from a root vector.
func ValFromRoots(roots, x *np.Array) (*np.Array, error) {
	if roots.DType() == 0 || x.DType() == 0 || roots.NDim() != 1 {
		return nil, np.ErrShape
	}
	d, err := numericType(roots.DType(), x.DType())
	if err != nil {
		return nil, err
	}
	r, _ := roots.Complex128s()
	v, _ := x.Complex128s()
	for i, z := range v {
		p := complex128(1)
		for _, root := range r {
			p *= z - root
		}
		v[i] = p
	}
	return array(v, d, x.Shape()...)
}

// Gauss uses the symmetric Jacobi matrix for Gaussian quadrature of the
// orthogonal basis. Weights are positive and sum to the basis measure.
func (b Basis) Gauss(degree int) (nodes, weights *np.Array, err error) {
	if !b.valid() || b == Power || degree < 1 {
		return nil, nil, np.ErrValue
	}
	if b == Chebyshev {
		if _, err := np.ElementCount(np.Float64, degree); err != nil {
			return nil, nil, err
		}
		x, w := make([]float64, degree), make([]float64, degree)
		for i := range x {
			x[i] = math.Cos(math.Pi * float64(2*i+1) / float64(2*degree))
			w[i] = math.Pi / float64(degree)
		}
		nodes, err = np.WrapFloat64(x, degree)
		if err != nil {
			return nil, nil, err
		}
		weights, err = np.WrapFloat64(w, degree)
		return
	}
	n := degree
	size, err := np.ElementCount(np.Float64, n, n)
	if err != nil {
		return nil, nil, err
	}
	jacobi := make([]float64, size)
	for j := 0; j < n; j++ {
		alpha, beta, _ := b.recurrence(j)
		jacobi[j*n+j] = beta
		if j+1 < n {
			_, _, gamma := b.recurrence(j + 1)
			v := math.Copysign(math.Sqrt(alpha*gamma), alpha)
			jacobi[(j+1)*n+j], jacobi[j*n+j+1] = v, v
		}
	}
	mat, err := np.WrapFloat64(jacobi, n, n)
	if err != nil {
		return nil, nil, err
	}
	nodes, v, err := linalg.Eigh(mat)
	if err != nil {
		return nil, nil, err
	}
	vec, _ := v.Float64s()
	mass := 1.0
	switch b {
	case Legendre:
		mass = 2
	case Hermite:
		mass = math.Sqrt(math.Pi)
	case HermiteE:
		mass = math.Sqrt(2 * math.Pi)
	}
	w := make([]float64, n)
	for i := range w {
		w[i] = mass * vec[i] * vec[i]
	}
	weights, err = np.WrapFloat64(w, n)
	return
}
func (b Basis) Weight(x *np.Array) (*np.Array, error) {
	if !b.valid() || b == Power || x.DType() == 0 || x.DType().IsComplex() {
		return nil, np.ErrDType
	}
	d, err := numericType(x.DType())
	if err != nil {
		return nil, err
	}
	v, _ := x.Complex128s()
	for i, z := range v {
		t := real(z)
		var y float64
		switch b {
		case Chebyshev:
			y = 1 / math.Sqrt(1-t*t)
		case Legendre:
			y = 1
		case Hermite:
			y = math.Exp(-t * t)
		case HermiteE:
			y = math.Exp(-t * t / 2)
		case Laguerre:
			y = math.Exp(-t)
		}
		v[i] = complex(y, 0)
	}
	return array(v, d, x.Shape()...)
}
func ChebPts1(n int) (*np.Array, error) {
	if n < 1 {
		return nil, np.ErrValue
	}
	if _, err := np.ElementCount(np.Float64, n); err != nil {
		return nil, err
	}
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Cos(math.Pi * float64(2*(n-i)-1) / float64(2*n))
	}
	return np.WrapFloat64(x, n)
}
func ChebPts2(n int) (*np.Array, error) {
	if n < 2 {
		return nil, np.ErrValue
	}
	if _, err := np.ElementCount(np.Float64, n); err != nil {
		return nil, err
	}
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Cos(math.Pi * float64(n-1-i) / float64(n-1))
	}
	return np.WrapFloat64(x, n)
}
func ChebInterpolate(f func(float64) complex128, degree int) (*np.Array, error) {
	if degree < 0 || f == nil || degree == int(^uint(0)>>1) {
		return nil, np.ErrValue
	}
	n := degree + 1
	nodes, err := ChebPts1(n)
	if err != nil {
		return nil, err
	}
	x, _ := nodes.Float64s()
	out := make([]complex128, n)
	values := make([]complex128, n)
	d := np.Float64
	for i, t := range x {
		values[i] = f(t)
		if imag(values[i]) != 0 {
			d = np.Complex128
		}
	}
	basis := make([]complex128, n)
	for i, t := range x {
		fillBasis(basis, complex(t, 0), Chebyshev)
		for k, v := range basis {
			out[k] += values[i] * v
		}
	}
	for k := range out {
		out[k] *= complex(2/float64(n), 0)
	}
	out[0] /= 2
	return coeffArray(out, d)
}
