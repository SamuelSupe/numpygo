package poly

import "github.com/SamuelSupe/numpygo"

// DerivativeAxis differentiates every coefficient lane along axis.
func (b Basis) DerivativeAxis(a *np.Array, order int, scale complex128, axis int) (*np.Array, error) {
	if order < 0 {
		return nil, np.ErrValue
	}
	ax, e := np.NormalizeAxis(axis, a.NDim())
	if e != nil {
		return nil, e
	}
	n := max(1, a.Shape()[ax]-order)
	return b.coefficientLanes(a, ax, n, func(v *np.Array) (*np.Array, error) { return b.Derivative(v, order, scale) })
}
func (b Basis) IntegralAxis(a *np.Array, order int, constants []complex128, lower, scale complex128, axis int) (*np.Array, error) {
	if order < 0 || len(constants) > order {
		return nil, np.ErrValue
	}
	ax, e := np.NormalizeAxis(axis, a.NDim())
	if e != nil {
		return nil, e
	}
	n := a.Shape()[ax]
	if order > int(^uint(0)>>1)-n {
		return nil, np.ErrShape
	}
	return b.coefficientLanes(a, ax, n+order, func(v *np.Array) (*np.Array, error) { return b.Integral(v, order, constants, lower, scale) })
}
func (b Basis) coefficientLanes(a *np.Array, ax, n int, fn func(*np.Array) (*np.Array, error)) (*np.Array, error) {
	if !b.valid() || a.DType() == 0 {
		return nil, np.ErrDType
	}
	v, e := np.MoveAxis(a, ax, 0)
	if e != nil {
		return nil, e
	}
	shape := v.Shape()
	if shape[0] == 0 {
		return nil, np.ErrShape
	}
	shape[0] = n
	d := a.DType()
	if d.IsInteger() || d == np.Bool {
		d = np.Float64
	}
	out, e := np.Zeros(d, shape...)
	if e != nil {
		return nil, e
	}
	it, e := np.NDIndex(shape[1:]...)
	if e != nil {
		return nil, e
	}
	for ix := range it {
		spec := make([]np.Index, len(shape))
		spec[0] = np.Slice{}
		for i, j := range ix {
			spec[i+1] = np.AtIndex(j)
		}
		x, e := v.Slice(spec...)
		if e != nil {
			return nil, e
		}
		y, e := fn(x)
		if e != nil {
			return nil, e
		}
		if y.Size() > n {
			return nil, np.ErrShape
		}
		spec[0] = np.Range(0, y.Size())
		if e = out.Assign(spec, y); e != nil {
			return nil, e
		}
	}
	return np.MoveAxis(out, 0, ax)
}
