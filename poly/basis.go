// Package poly implements NumPy's six polynomial bases. Coefficients are
// ordered from lowest to highest degree, as in numpy.polynomial.
package poly

import (
	"fmt"
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/linalg"
)

type Basis uint8

const (
	Power Basis = iota
	Chebyshev
	Legendre
	Hermite
	HermiteE
	Laguerre
)

func (b Basis) String() string {
	switch b {
	case Power:
		return "polynomial"
	case Chebyshev:
		return "chebyshev"
	case Legendre:
		return "legendre"
	case Hermite:
		return "hermite"
	case HermiteE:
		return "hermite_e"
	case Laguerre:
		return "laguerre"
	default:
		return "invalid"
	}
}
func (b Basis) valid() bool { return b <= Laguerre }

// recurrence gives x B_k(x) = alpha B_(k+1)(x) + beta B_k(x) + gamma B_(k-1)(x).
func (b Basis) recurrence(k int) (alpha, beta, gamma float64) {
	t := float64(k)
	switch b {
	case Power:
		return 1, 0, 0
	case Chebyshev:
		if k == 0 {
			return 1, 0, 0
		}
		return .5, 0, .5
	case Legendre:
		return (t + 1) / (2*t + 1), 0, t / (2*t + 1)
	case Hermite:
		return .5, 0, t
	case HermiteE:
		return 1, 0, t
	case Laguerre:
		return -(t + 1), 2*t + 1, -t
	default:
		return math.NaN(), 0, 0
	}
}
func numericType(types ...np.DType) (np.DType, error) {
	var d np.DType
	for _, t := range types {
		if !t.IsNativeNumeric() {
			return 0, np.ErrDType
		}
		if t.IsInteger() || t == np.Bool {
			t = np.Float64
		}
		if d == 0 {
			d = t
		} else {
			var err error
			d, err = np.PromoteTypes(d, t)
			if err != nil {
				return 0, err
			}
		}
	}
	if d == 0 {
		return 0, np.ErrDType
	}
	return d, nil
}
func coeffs(a *np.Array) ([]complex128, error) {
	if !a.DType().IsNativeNumeric() {
		return nil, np.ErrDType
	}
	if a.NDim() > 1 || a.Size() == 0 {
		return nil, np.ErrShape
	}
	return a.Complex128s()
}
func trim(a []complex128) []complex128 {
	n := len(a)
	for n > 1 && a[n-1] == 0 {
		n--
	}
	if n == 0 {
		return []complex128{0}
	}
	return a[:n]
}
func array(v []complex128, d np.DType, shape ...int) (*np.Array, error) {
	a, err := np.WrapComplex128(v, shape...)
	if err != nil {
		return nil, err
	}
	if d == np.Complex128 {
		return a, nil
	}
	return a.Astype(d)
}
func coeffArray(v []complex128, d np.DType) (*np.Array, error) { return array(v, d, len(v)) }
func add(a, b []complex128, scale complex128) []complex128 {
	out := make([]complex128, max(len(a), len(b)))
	copy(out, a)
	for i, v := range b {
		out[i] += scale * v
	}
	return trim(out)
}
func scaled(a []complex128, z complex128) []complex128 {
	out := make([]complex128, len(a))
	for i, v := range a {
		out[i] = v * z
	}
	return trim(out)
}
func (b Basis) mulX(a []complex128) []complex128 {
	out := make([]complex128, len(a)+1)
	for k, v := range a {
		alpha, beta, gamma := b.recurrence(k)
		out[k+1] += complex(alpha, 0) * v
		out[k] += complex(beta, 0) * v
		if k > 0 {
			out[k-1] += complex(gamma, 0) * v
		}
	}
	return trim(out)
}
func (b Basis) nextProduct(curr, prev []complex128, k int) []complex128 {
	alpha, beta, gamma := b.recurrence(k)
	next := b.mulX(curr)
	if beta != 0 {
		next = add(next, curr, complex(-beta, 0))
	}
	if gamma != 0 {
		next = add(next, prev, complex(-gamma, 0))
	}
	return scaled(next, complex(1/alpha, 0))
}
func (b Basis) mul(a, c []complex128) []complex128 {
	if len(a) < len(c) {
		a, c = c, a
	}
	out := scaled(a, c[0])
	prev := []complex128{0}
	curr := a
	for k := 1; k < len(c); k++ {
		next := b.nextProduct(curr, prev, k-1)
		out = add(out, next, c[k])
		prev, curr = curr, next
	}
	return trim(out)
}
func (b Basis) unary(a *np.Array, fn func([]complex128) ([]complex128, error)) (*np.Array, error) {
	if !b.valid() {
		return nil, np.ErrValue
	}
	v, err := coeffs(a)
	if err != nil {
		return nil, err
	}
	d, err := numericType(a.DType())
	if err != nil {
		return nil, err
	}
	result, err := fn(v)
	if err != nil {
		return nil, err
	}
	return coeffArray(result, d)
}
func (b Basis) binary(a, c *np.Array, fn func([]complex128, []complex128) []complex128) (*np.Array, error) {
	if !b.valid() {
		return nil, np.ErrValue
	}
	x, err := coeffs(a)
	if err != nil {
		return nil, err
	}
	y, err := coeffs(c)
	if err != nil {
		return nil, err
	}
	d, err := numericType(a.DType(), c.DType())
	if err != nil {
		return nil, err
	}
	return coeffArray(fn(trim(x), trim(y)), d)
}
func (b Basis) Add(a, c *np.Array) (*np.Array, error) {
	if hasExtended(a, c) {
		return b.xbinary(a, c, "add")
	}
	return b.binary(a, c, func(x, y []complex128) []complex128 { return add(x, y, 1) })
}
func (b Basis) Sub(a, c *np.Array) (*np.Array, error) {
	if hasExtended(a, c) {
		return b.xbinary(a, c, "sub")
	}
	return b.binary(a, c, func(x, y []complex128) []complex128 { return add(x, y, -1) })
}
func (b Basis) Mul(a, c *np.Array) (*np.Array, error) {
	if hasExtended(a, c) {
		return b.xbinary(a, c, "mul")
	}
	return b.binary(a, c, b.mul)
}
func (b Basis) MulX(a *np.Array) (*np.Array, error) {
	if hasExtended(a) {
		return b.xunary(a, func(x []xc) ([]xc, error) { return b.xmulX(xtrim(x)), nil })
	}
	return b.unary(a, func(x []complex128) ([]complex128, error) { return b.mulX(trim(x)), nil })
}

func (b Basis) Div(a, c *np.Array) (*np.Array, *np.Array, error) {
	if !b.valid() {
		return nil, nil, np.ErrValue
	}
	if hasExtended(a, c) {
		return b.xdivmod(a, c)
	}
	x, err := coeffs(a)
	if err != nil {
		return nil, nil, err
	}
	y, err := coeffs(c)
	if err != nil {
		return nil, nil, err
	}
	d, err := numericType(a.DType(), c.DType())
	if err != nil {
		return nil, nil, err
	}
	x, y = trim(x), trim(y)
	if len(y) == 1 && y[0] == 0 {
		return nil, nil, np.ErrValue
	}
	var q []complex128
	if len(x) < len(y) {
		q = []complex128{0}
	} else if len(y) == 1 {
		q = scaled(x, 1/y[0])
		x = []complex128{0}
	} else {
		q = make([]complex128, len(x)-len(y)+1)
		for j := len(q) - 1; j >= 0; j-- {
			basis := make([]complex128, j+1)
			basis[j] = 1
			product := b.mul(y, basis)
			top := len(y) + j - 1
			if len(product) != top+1 || product[top] == 0 {
				return nil, nil, np.ErrValue
			}
			q[j] = x[top] / product[top]
			for i := 0; i < top; i++ {
				x[i] -= q[j] * product[i]
			}
			x = x[:top]
		}
	}
	qa, err := coeffArray(trim(q), d)
	if err != nil {
		return nil, nil, err
	}
	ra, err := coeffArray(trim(x), d)
	return qa, ra, err
}

func (b Basis) Pow(a *np.Array, power int, maxPower ...int) (*np.Array, error) {
	if power < 0 || len(maxPower) > 1 {
		return nil, np.ErrValue
	}
	limit := -1
	if b != Power {
		limit = 16
	}
	if len(maxPower) == 1 {
		limit = maxPower[0]
	}
	if limit >= 0 && power > limit {
		return nil, np.ErrValue
	}
	if hasExtended(a) {
		return b.xpow(a, power)
	}
	return b.unary(a, func(x []complex128) ([]complex128, error) {
		x = trim(x)
		if len(x) > 1 && power > (int(^uint(0)>>1)/16-1)/(len(x)-1) {
			return nil, np.ErrShape
		}
		r := []complex128{1}
		for p := uint(power); p != 0; {
			if p&1 != 0 {
				r = b.mul(r, x)
			}
			p >>= 1
			if p != 0 {
				x = b.mul(x, x)
			}
		}
		return r, nil
	})
}

// Trim removes trailing coefficients whose absolute value is at most tol.
func (b Basis) Trim(a *np.Array, tol float64) (*np.Array, error) {
	if !b.valid() {
		return nil, np.ErrValue
	}
	return TrimCoefficients(a, tol)
}

func (b Basis) Line(off, slope complex128) (*np.Array, error) {
	if !b.valid() {
		return nil, np.ErrValue
	}
	alpha, beta, _ := b.recurrence(0)
	d := np.Float64
	if imag(off) != 0 || imag(slope) != 0 {
		d = np.Complex128
	}
	return coeffArray(trim([]complex128{off + complex(beta, 0)*slope, complex(alpha, 0) * slope}), d)
}

func (b Basis) FromRoots(roots *np.Array) (*np.Array, error) {
	if !b.valid() || roots.DType() == 0 {
		return nil, np.ErrDType
	}
	if roots.NDim() > 1 {
		return nil, np.ErrShape
	}
	if hasExtended(roots) {
		return b.xfromroots(roots)
	}
	d, err := numericType(roots.DType())
	if err != nil {
		return nil, err
	}
	values, _ := roots.Complex128s()
	alpha, beta, _ := b.recurrence(0)
	out := []complex128{1}
	for _, v := range values {
		out = b.mul(out, []complex128{complex(beta, 0) - v, complex(alpha, 0)})
	}
	return coeffArray(out, d)
}

// basisDerivatives computes B'_k in the same basis by differentiating the recurrence.
func (b Basis) basisDerivatives(n int) [][]complex128 {
	out := make([][]complex128, n+1)
	out[0] = []complex128{0}
	for k := 0; k < n; k++ {
		one := make([]complex128, k+1)
		one[k] = 1
		alpha, beta, gamma := b.recurrence(k)
		d := add(b.mulX(out[k]), one, 1)
		if beta != 0 {
			d = add(d, out[k], complex(-beta, 0))
		}
		if k > 0 && gamma != 0 {
			d = add(d, out[k-1], complex(-gamma, 0))
		}
		out[k+1] = scaled(d, complex(1/alpha, 0))
	}
	return out
}
func (b Basis) Derivative(a *np.Array, order int, scale complex128) (*np.Array, error) {
	if order < 0 {
		return nil, np.ErrValue
	}
	if hasExtended(a) {
		return b.xderivative(a, order, scale)
	}
	return b.unary(a, func(x []complex128) ([]complex128, error) {
		if order >= len(x) {
			return []complex128{0}, nil
		}
		if order == 0 {
			return x, nil
		}
		deriv := b.basisDerivatives(len(x) - 1)
		for j := 0; j < order; j++ {
			next := make([]complex128, len(x)-1)
			for k := 1; k < len(x); k++ {
				for i, v := range deriv[k] {
					next[i] += v * x[k] * scale
				}
			}
			x = next
		}
		return x, nil
	})
}
func (b Basis) Integral(a *np.Array, order int, constants []complex128, lower, scale complex128) (*np.Array, error) {
	if order < 0 || len(constants) > order {
		return nil, np.ErrValue
	}
	if hasExtended(a) {
		return b.xintegral(a, order, constants, lower, scale)
	}
	return b.unary(a, func(x []complex128) ([]complex128, error) {
		if order == 0 {
			return x, nil
		}
		if order > int(^uint(0)>>1)-len(x) {
			return nil, np.ErrShape
		}
		if _, err := np.ElementCount(np.Complex128, len(x)+order); err != nil {
			return nil, err
		}
		deriv := b.basisDerivatives(len(x) + order - 1)
		for j := 0; j < order; j++ {
			remain := slices.Clone(x)
			for i := range remain {
				remain[i] *= scale
			}
			target := complex128(0)
			if j < len(constants) {
				target = constants[j]
			}
			if len(remain) == 1 && remain[0] == 0 {
				x = []complex128{target}
				continue
			}
			next := make([]complex128, len(remain)+1)
			for k := len(remain) - 1; k >= 0; k-- {
				p := deriv[k+1]
				if len(p) != k+1 || p[k] == 0 {
					return nil, np.ErrValue
				}
				next[k+1] = remain[k] / p[k]
				for i := 0; i < k; i++ {
					remain[i] -= next[k+1] * p[i]
				}
			}
			next[0] = target - b.evaluate(next, lower)
			x = next
		}
		return x, nil
	})
}

// Convert changes coefficient basis by evaluating the source recurrence using
// polynomial multiplication by x in the target basis.
func (b Basis) Convert(a *np.Array, target Basis) (*np.Array, error) {
	if !target.valid() {
		return nil, np.ErrValue
	}
	if hasExtended(a) {
		return b.xunary(a, func(x []xc) ([]xc, error) {
			if target == b {
				return x, nil
			}
			a, be, _ := target.xrec(0)
			return b.xcompose(x, []xc{be, a}, target), nil
		})
	}
	return b.unary(a, func(x []complex128) ([]complex128, error) {
		if target == b {
			return x, nil
		}
		prev := []complex128{0}
		curr := []complex128{1}
		out := []complex128{x[0]}
		for k := 1; k < len(x); k++ {
			alpha, beta, gamma := b.recurrence(k - 1)
			next := target.mulX(curr)
			if beta != 0 {
				next = add(next, curr, complex(-beta, 0))
			}
			if gamma != 0 {
				next = add(next, prev, complex(-gamma, 0))
			}
			next = scaled(next, complex(1/alpha, 0))
			out = add(out, next, x[k])
			prev, curr = curr, next
		}
		return out, nil
	})
}

// Companion returns the scaled companion matrix used by numpy.polynomial.
// Orthogonal bases have symmetric off-diagonals before the final-column correction.
func (b Basis) Companion(a *np.Array) (*np.Array, error) {
	if !b.valid() {
		return nil, np.ErrValue
	}
	if hasExtended(a) {
		return b.xcompanion(a)
	}
	x, err := coeffs(a)
	if err != nil {
		return nil, err
	}
	x = trim(x)
	if len(x) < 2 {
		return nil, np.ErrShape
	}
	d, err := numericType(a.DType())
	if err != nil {
		return nil, err
	}
	n := len(x) - 1
	size, err := np.ElementCount(d, n, n)
	if err != nil {
		return nil, err
	}
	m := make([]complex128, size)
	scale := make([]float64, n)
	scale[0] = 1
	for j := 0; j < n; j++ {
		alpha, beta, gamma := b.recurrence(j)
		m[j*n+j] = complex(beta, 0)
		if j+1 < n {
			_, _, nextGamma := b.recurrence(j + 1)
			if b == Power {
				scale[j+1] = 1
			} else {
				scale[j+1] = scale[j] * math.Sqrt(alpha/nextGamma)
			}
			m[(j+1)*n+j] = complex(alpha*scale[j]/scale[j+1], 0)
			m[j*n+j+1] = complex(nextGamma*scale[j+1]/scale[j], 0)
		}
		if j > 0 && b == Power && gamma != 0 {
			m[(j-1)*n+j] = complex(gamma, 0)
		}
	}
	alpha, _, _ := b.recurrence(n - 1)
	for i := 0; i < n; i++ {
		m[i*n+n-1] -= complex(alpha*scale[n-1]/scale[i], 0) * x[i] / x[n]
	}
	return array(m, d, n, n)
}

func (b Basis) Roots(a *np.Array) (*np.Array, error) {
	if !b.valid() {
		return nil, np.ErrValue
	}
	if hasExtended(a) {
		return b.xroots(a)
	}
	x, err := coeffs(a)
	if err != nil {
		return nil, err
	}
	x = trim(x)
	d, err := numericType(a.DType())
	if err != nil {
		return nil, err
	}
	if len(x) == 1 {
		return np.Zeros(d, 0)
	}
	if len(x) == 2 {
		alpha, beta, _ := b.recurrence(0)
		val := complex(beta, 0) - complex(alpha, 0)*x[0]/x[1]
		return coeffArray([]complex128{val}, d)
	}
	c, err := b.Companion(a)
	if err != nil {
		return nil, err
	}
	c, err = c.Slice(np.Reverse(), np.Reverse())
	if err != nil {
		return nil, err
	}
	if c.DType() == np.Float16 {
		c, err = c.Astype(np.Float64)
		if err != nil {
			return nil, err
		}
	}
	val, err := linalg.Eigvals(c)
	if err != nil {
		return nil, fmt.Errorf("polynomial roots: %w", err)
	}
	// Unlike linalg.eigvals, NumPy's polynomial APIs preserve a real dtype
	// for an entirely real spectrum of a real companion matrix.
	if !c.DType().IsComplex() {
		values, err := val.Complex128s()
		if err != nil {
			return nil, err
		}
		realOnly := true
		for _, z := range values {
			if imag(z) != 0 {
				realOnly = false
				break
			}
		}
		if realOnly {
			val, err = np.Real(val)
			if err != nil {
				return nil, err
			}
		}
	}
	return np.Sort(val, np.Int(0))
}

func (b Basis) evaluate(c []complex128, x complex128) complex128 {
	// Generalized Clenshaw recursion for alpha_k B_(k+1) =
	// (x-beta_k) B_k - gamma_k B_(k-1).
	var u, v complex128
	for k := len(c) - 1; k >= 0; k-- {
		alpha, beta, _ := b.recurrence(k)
		nextAlpha, _, nextGamma := b.recurrence(k + 1)
		z := c[k] + (x-complex(beta, 0))/complex(alpha, 0)*u - complex(nextGamma/nextAlpha, 0)*v
		v, u = u, z
	}
	return u
}

func (b Basis) Eval(c, x *np.Array) (*np.Array, error) {
	if !b.valid() || c.DType() == 0 || x.DType() == 0 {
		return nil, np.ErrDType
	}
	if hasExtended(c, x) {
		return b.xeval(c, x)
	}
	if c.NDim() == 0 {
		var err error
		c, err = c.Reshape(1)
		if err != nil {
			return nil, err
		}
	}
	shape := c.Shape()
	if len(shape) == 0 || shape[0] == 0 {
		return nil, np.ErrShape
	}
	d, err := numericType(c.DType(), x.DType())
	if err != nil {
		return nil, err
	}
	outShape := append(slices.Clone(shape[1:]), x.Shape()...)
	sz, err := np.ElementCount(d, outShape...)
	if err != nil {
		return nil, err
	}
	cv, _ := c.Complex128s()
	xv, _ := x.Complex128s()
	count := 1
	for _, dim := range shape[1:] {
		count *= dim
	}
	out := make([]complex128, sz)
	coef := make([]complex128, shape[0])
	for k := 0; k < count; k++ {
		for j := range coef {
			coef[j] = cv[j*count+k]
		}
		for i, v := range xv {
			out[k*len(xv)+i] = b.evaluate(coef, v)
		}
	}
	return array(out, d, outShape...)
}

func (b Basis) Vander(x *np.Array, degree int) (*np.Array, error) {
	if !b.valid() || x.DType() == 0 {
		return nil, np.ErrDType
	}
	if degree < 0 || degree == int(^uint(0)>>1) {
		return nil, np.ErrValue
	}
	if hasExtended(x) {
		return b.xvander(x, degree)
	}
	if x.NDim() == 0 {
		var err error
		x, err = x.Reshape(1)
		if err != nil {
			return nil, err
		}
	}
	d, err := numericType(x.DType())
	if err != nil {
		return nil, err
	}
	shape := append(x.Shape(), degree+1)
	size, err := np.ElementCount(d, shape...)
	if err != nil {
		return nil, err
	}
	vals, _ := x.Complex128s()
	out := make([]complex128, size)
	for i, z := range vals {
		fillBasis(out[i*(degree+1):(i+1)*(degree+1)], z, b)
	}
	return array(out, d, shape...)
}

func fillBasis(dst []complex128, x complex128, b Basis) {
	if len(dst) == 0 {
		return
	}
	dst[0] = 1
	for k := 0; k < len(dst)-1; k++ {
		alpha, beta, gamma := b.recurrence(k)
		v := (x - complex(beta, 0)) * dst[k]
		if k > 0 {
			v -= complex(gamma, 0) * dst[k-1]
		}
		dst[k+1] = v / complex(alpha, 0)
	}
}
