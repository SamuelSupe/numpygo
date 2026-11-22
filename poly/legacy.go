package poly

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/linalg"
)

// Legacy polynomial routines use the descending coefficient order of
// numpy.poly1d and numpy.poly*. The Basis APIs use ascending order.
func reverse(a *np.Array) (*np.Array, error) {
	if a.DType() == 0 {
		return nil, np.ErrDType
	}
	if a.NDim() == 0 {
		var err error
		a, err = a.Reshape(1)
		if err != nil {
			return nil, err
		}
	}
	return a.Slice(np.Reverse())
}
func descending(a *np.Array) (*np.Array, error) {
	r, err := reverse(a)
	if err != nil {
		return nil, err
	}
	return r.Copy()
}
func LegacyAdd(a, b *np.Array) (*np.Array, error) { return legacyAdd(a, b, false) }
func LegacySub(a, b *np.Array) (*np.Array, error) { return legacyAdd(a, b, true) }
func legacyAdd(a, b *np.Array, subtract bool) (*np.Array, error) {
	if a.DType() == 0 || b.DType() == 0 || a.NDim() > 1 || b.NDim() > 1 {
		return nil, np.ErrShape
	}
	d, err := np.PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	n := max(a.Size(), b.Size())
	out, err := np.Zeros(d, n)
	if err != nil {
		return nil, err
	}
	x, err := np.Zeros(d, n)
	if err != nil {
		return nil, err
	}
	if err = out.Assign([]np.Index{np.Range(n-a.Size(), n)}, a); err != nil {
		return nil, err
	}
	if err = x.Assign([]np.Index{np.Range(n-b.Size(), n)}, b); err != nil {
		return nil, err
	}
	if subtract {
		return np.Subtract(out, x)
	}
	return np.Add(out, x)
}
func LegacyMul(a, b *np.Array) (*np.Array, error) {
	x, err := reverse(a)
	if err != nil {
		return nil, err
	}
	y, err := reverse(b)
	if err != nil {
		return nil, err
	}
	v, err := Power.Mul(x, y)
	if err != nil {
		return nil, err
	}
	d, err := np.PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	v, err = v.Astype(d)
	if err != nil {
		return nil, err
	}
	return descending(v)
}
func LegacyDiv(a, b *np.Array) (*np.Array, *np.Array, error) {
	x, err := reverse(a)
	if err != nil {
		return nil, nil, err
	}
	y, err := reverse(b)
	if err != nil {
		return nil, nil, err
	}
	q, r, err := Power.Div(x, y)
	if err != nil {
		return nil, nil, err
	}
	q, err = descending(q)
	if err != nil {
		return nil, nil, err
	}
	r, err = descending(r)
	return q, r, err
}
func LegacyDerivative(a *np.Array, order int) (*np.Array, error) {
	if a.DType() == 0 || a.NDim() > 1 || order < 0 {
		return nil, np.ErrShape
	}
	if a.NDim() == 0 {
		var err error
		a, err = a.Reshape(1)
		if err != nil {
			return nil, err
		}
	}
	out := a
	for j := 0; j < order; j++ {
		n := max(out.Size()-1, 0)
		c, err := out.Slice(np.Range(0, n))
		if err != nil {
			return nil, err
		}
		v := make([]int64, n)
		for i := range v {
			v[i] = int64(n - i)
		}
		factors, err := np.WrapSlice(v, n)
		if err != nil {
			return nil, err
		}
		out, err = np.Multiply(c, factors)
		if err != nil {
			return nil, err
		}
	}
	return out.Copy()
}
func LegacyIntegral(a *np.Array, order int, constants []complex128) (*np.Array, error) {
	x, err := reverse(a)
	if err != nil {
		return nil, err
	}
	out, err := Power.Integral(x, order, constants, 0, 1)
	if err != nil {
		return nil, err
	}
	return descending(out)
}
func LegacyEval(c, x *np.Array) (*np.Array, error) {
	if c.DType() == 0 || x.DType() == 0 || c.NDim() > 1 {
		return nil, np.ErrShape
	}
	if c.NDim() == 0 {
		var err error
		c, err = c.Reshape(1)
		if err != nil {
			return nil, err
		}
	}
	d, err := np.PromoteTypes(c.DType(), x.DType())
	if err != nil {
		return nil, err
	}
	out, err := np.Zeros(d, x.Shape()...)
	if err != nil {
		return nil, err
	}
	for i := 0; i < c.Size(); i++ {
		out, err = np.Multiply(out, x)
		if err != nil {
			return nil, err
		}
		k, err := c.Slice(np.AtIndex(i))
		if err != nil {
			return nil, err
		}
		out, err = np.Add(out, k)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
func LegacyRoots(a *np.Array) (*np.Array, error) {
	if a.DType() == 0 || a.NDim() > 1 {
		return nil, np.ErrShape
	}
	if a.Size() == 0 {
		return np.Zeros(np.Float64, 0)
	}
	x, err := reverse(a)
	if err != nil {
		return nil, err
	}
	return Power.Roots(x)
}
func LegacyFromRoots(roots *np.Array) (*np.Array, error) {
	if roots.DType() == 0 {
		return nil, np.ErrDType
	}
	if roots.NDim() == 2 {
		var err error
		roots, err = linalg.Eigvals(roots)
		if err != nil {
			return nil, err
		}
	}
	v, err := Power.FromRoots(roots)
	if err != nil {
		return nil, err
	}
	if v.DType().IsComplex() {
		c, _ := v.Complex128s()
		realOnly := true
		for _, z := range c {
			if imag(z) != 0 {
				realOnly = false
				break
			}
		}
		if realOnly {
			v, err = v.Astype(np.Float64)
			if err != nil {
				return nil, err
			}
		}
	}
	return descending(v)
}
func LegacyFit(x, y *np.Array, degree int, options ...FitOptions) (FitResult, error) {
	out, err := Power.Fit(x, y, degree, options...)
	if err != nil {
		return out, err
	}
	out.Coefficients, err = descending(out.Coefficients)
	return out, err
}
func NewLegacySeries(c *np.Array) (*Series, error) {
	x, err := reverse(c)
	if err != nil {
		return nil, err
	}
	return NewSeries(Power, x)
}
