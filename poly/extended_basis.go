package poly

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/x87"
	"math"
	"slices"
)

type xc = x87.Complex

func xi(i int) xc     { return xc{Real: x87.FromInt64(int64(i))} }
func xzero(x xc) bool { return x.Real.IsZero() && x.Imag.IsZero() }
func xmul(a, b xc) xc {
	if a.Imag.IsZero() && b.Imag.IsZero() {
		return xc{Real: a.Real.Mul(b.Real)}
	}
	return a.Mul(b)
}
func xdiv(a, b xc) xc {
	if a.Imag.IsZero() && b.Imag.IsZero() {
		return xc{Real: a.Real.Div(b.Real)}
	}
	return a.Div(b)
}
func xneg(a xc) xc { return xc{Real: a.Real.Neg(), Imag: a.Imag.Neg()} }
func hasExtended(a ...*np.Array) bool {
	for _, v := range a {
		if v.DType().IsExtended() {
			return true
		}
	}
	return false
}
func xdtype(a ...*np.Array) (np.DType, error) {
	var d np.DType
	for _, v := range a {
		t := v.DType()
		if !t.IsNativeNumeric() && !t.IsExtended() {
			return 0, np.ErrDType
		}
		if t.IsInteger() || t == np.Bool {
			t = np.Float64
		}
		if d == 0 {
			d = t
		} else {
			var e error
			d, e = np.PromoteTypes(d, t)
			if e != nil {
				return 0, e
			}
		}
	}
	return d, nil
}
func xvalues(a *np.Array) ([]xc, error) {
	v, e := a.Astype(np.Complex256)
	if e != nil {
		return nil, e
	}
	return np.Values[xc](v)
}
func xcoeff(a *np.Array) ([]xc, error) {
	if a.NDim() > 1 || a.Size() == 0 {
		return nil, np.ErrShape
	}
	return xvalues(a)
}
func xarray(v []xc, d np.DType, sh ...int) (*np.Array, error) {
	if d == np.Float128 {
		r := make([]x87.Float, len(v))
		for i, z := range v {
			r[i] = z.Real
		}
		return np.WrapSlice(r, sh...)
	}
	return np.WrapSlice(v, sh...)
}
func xtrim(a []xc) []xc {
	n := len(a)
	for n > 1 && xzero(a[n-1]) {
		n--
	}
	if n == 0 {
		return []xc{{}}
	}
	return a[:n]
}
func xadd(a, b []xc, scale xc) []xc {
	o := make([]xc, max(len(a), len(b)))
	copy(o, a)
	for i, v := range b {
		o[i] = o[i].Add(xmul(scale, v))
	}
	return xtrim(o)
}
func xscale(a []xc, v xc) []xc {
	o := make([]xc, len(a))
	for i, z := range a {
		o[i] = xmul(z, v)
	}
	return xtrim(o)
}
func (b Basis) xrec(k int) (a, be, g xc) {
	switch b {
	case Power:
		return xi(1), xi(0), xi(0)
	case Chebyshev:
		if k == 0 {
			return xi(1), xi(0), xi(0)
		}
		return xdiv(xi(1), xi(2)), xi(0), xdiv(xi(1), xi(2))
	case Legendre:
		return xdiv(xi(k+1), xi(2*k+1)), xi(0), xdiv(xi(k), xi(2*k+1))
	case Hermite:
		return xdiv(xi(1), xi(2)), xi(0), xi(k)
	case HermiteE:
		return xi(1), xi(0), xi(k)
	default:
		return xi(-k - 1), xi(2*k + 1), xi(-k)
	}
}
func (b Basis) xmulX(a []xc) []xc {
	o := make([]xc, len(a)+1)
	for k, v := range a {
		al, be, g := b.xrec(k)
		o[k+1] = o[k+1].Add(xmul(al, v))
		o[k] = o[k].Add(xmul(be, v))
		if k > 0 {
			o[k-1] = o[k-1].Add(xmul(g, v))
		}
	}
	return xtrim(o)
}
func (b Basis) xmul(a, c []xc) []xc {
	if len(a) < len(c) {
		a, c = c, a
	}
	o := xscale(a, c[0])
	prev, curr := []xc{{}}, a
	for k := 1; k < len(c); k++ {
		al, be, g := b.xrec(k - 1)
		next := xadd(xadd(b.xmulX(curr), curr, xneg(be)), prev, xneg(g))
		next = xscale(next, xdiv(xi(1), al))
		o = xadd(o, next, c[k])
		prev, curr = curr, next
	}
	return xtrim(o)
}
func (b Basis) xbinary(a, c *np.Array, op string) (*np.Array, error) {
	if !b.valid() {
		return nil, np.ErrValue
	}
	d, e := xdtype(a, c)
	if e != nil {
		return nil, e
	}
	x, e := xcoeff(a)
	if e != nil {
		return nil, e
	}
	y, e := xcoeff(c)
	if e != nil {
		return nil, e
	}
	x, y = xtrim(x), xtrim(y)
	var v []xc
	switch op {
	case "add":
		v = xadd(x, y, xi(1))
	case "sub":
		v = xadd(x, y, xi(-1))
	default:
		v = b.xmul(x, y)
	}
	return xarray(v, d, len(v))
}
func (b Basis) xunary(a *np.Array, fn func([]xc) ([]xc, error)) (*np.Array, error) {
	if !b.valid() {
		return nil, np.ErrValue
	}
	d, e := xdtype(a)
	if e != nil {
		return nil, e
	}
	x, e := xcoeff(a)
	if e != nil {
		return nil, e
	}
	v, e := fn(x)
	if e != nil {
		return nil, e
	}
	return xarray(v, d, len(v))
}
func (b Basis) xdivmod(a, c *np.Array) (*np.Array, *np.Array, error) {
	d, e := xdtype(a, c)
	if e != nil {
		return nil, nil, e
	}
	x, e := xcoeff(a)
	if e != nil {
		return nil, nil, e
	}
	y, e := xcoeff(c)
	if e != nil {
		return nil, nil, e
	}
	x, y = xtrim(x), xtrim(y)
	if len(y) == 1 && xzero(y[0]) {
		return nil, nil, np.ErrValue
	}
	q := []xc{{}}
	if len(x) >= len(y) {
		if len(y) == 1 {
			q = xscale(x, xdiv(xi(1), y[0]))
			x = []xc{{}}
		} else {
			q = make([]xc, len(x)-len(y)+1)
			for j := len(q) - 1; j >= 0; j-- {
				p := make([]xc, j+1)
				p[j] = xi(1)
				p = b.xmul(y, p)
				top := len(y) + j - 1
				if len(p) != top+1 || xzero(p[top]) {
					return nil, nil, np.ErrValue
				}
				q[j] = xdiv(x[top], p[top])
				for i := 0; i < top; i++ {
					x[i] = x[i].Sub(xmul(q[j], p[i]))
				}
				x = x[:top]
			}
		}
	}
	q, x = xtrim(q), xtrim(x)
	qa, e := xarray(q, d, len(q))
	if e != nil {
		return nil, nil, e
	}
	ra, e := xarray(x, d, len(x))
	return qa, ra, e
}
func (b Basis) xpow(a *np.Array, power int) (*np.Array, error) {
	return b.xunary(a, func(x []xc) ([]xc, error) {
		x = xtrim(x)
		if len(x) > 1 && power > (int(^uint(0)>>1)/32-1)/(len(x)-1) {
			return nil, np.ErrShape
		}
		r := []xc{xi(1)}
		for p := uint(power); p != 0; {
			if p&1 != 0 {
				r = b.xmul(r, x)
			}
			p >>= 1
			if p != 0 {
				x = b.xmul(x, x)
			}
		}
		return r, nil
	})
}
func (b Basis) xevaluate(c []xc, x xc) xc {
	var u, v xc
	for k := len(c) - 1; k >= 0; k-- {
		al, be, _ := b.xrec(k)
		na, _, ng := b.xrec(k + 1)
		z := c[k].Add(xmul(xdiv(x.Sub(be), al), u)).Sub(xmul(xdiv(ng, na), v))
		v, u = u, z
	}
	return u
}
func (b Basis) xeval(c, x *np.Array) (*np.Array, error) {
	d, e := xdtype(c, x)
	if e != nil {
		return nil, e
	}
	shape := c.Shape()
	if c.NDim() == 0 {
		shape = []int{1}
	}
	if shape[0] == 0 {
		return nil, np.ErrShape
	}
	outshape := append(slices.Clone(shape[1:]), x.Shape()...)
	n, e := np.ElementCount(d, outshape...)
	if e != nil {
		return nil, e
	}
	cv, e := xvalues(c)
	if e != nil {
		return nil, e
	}
	xv, e := xvalues(x)
	if e != nil {
		return nil, e
	}
	count := len(cv) / shape[0]
	out := make([]xc, n)
	coef := make([]xc, shape[0])
	for k := 0; k < count; k++ {
		for j := range coef {
			coef[j] = cv[j*count+k]
		}
		for i, v := range xv {
			out[k*len(xv)+i] = b.xevaluate(coef, v)
		}
	}
	return xarray(out, d, outshape...)
}
func (b Basis) xderivatives(n int) [][]xc {
	out := make([][]xc, n+1)
	out[0] = []xc{{}}
	for k := 0; k < n; k++ {
		one := make([]xc, k+1)
		one[k] = xi(1)
		a, be, g := b.xrec(k)
		d := xadd(b.xmulX(out[k]), one, xi(1))
		d = xadd(d, out[k], xneg(be))
		if k > 0 {
			d = xadd(d, out[k-1], xneg(g))
		}
		out[k+1] = xscale(d, xdiv(xi(1), a))
	}
	return out
}
func (b Basis) xderivative(a *np.Array, n int, scale complex128) (*np.Array, error) {
	return b.xunary(a, func(x []xc) ([]xc, error) {
		if n >= len(x) {
			return []xc{{}}, nil
		}
		if n == 0 {
			return x, nil
		}
		deriv := b.xderivatives(len(x) - 1)
		s := x87.FromComplex128(scale)
		for j := 0; j < n; j++ {
			next := make([]xc, len(x)-1)
			for k := 1; k < len(x); k++ {
				for i, v := range deriv[k] {
					next[i] = next[i].Add(xmul(xmul(v, x[k]), s))
				}
			}
			x = next
		}
		return x, nil
	})
}
func (b Basis) xintegral(a *np.Array, n int, constants []complex128, lower, scale complex128) (*np.Array, error) {
	return b.xunary(a, func(x []xc) ([]xc, error) {
		if n == 0 {
			return x, nil
		}
		if n > int(^uint(0)>>1)-len(x) {
			return nil, np.ErrShape
		}
		if _, e := np.ElementCount(np.Complex256, len(x)+n); e != nil {
			return nil, e
		}
		deriv := b.xderivatives(len(x) + n - 1)
		for j := 0; j < n; j++ {
			remain := slices.Clone(x)
			for i := range remain {
				remain[i] = xmul(remain[i], x87.FromComplex128(scale))
			}
			target := xc{}
			if j < len(constants) {
				target = x87.FromComplex128(constants[j])
			}
			if len(remain) == 1 && xzero(remain[0]) {
				x = []xc{target}
				continue
			}
			next := make([]xc, len(remain)+1)
			for k := len(remain) - 1; k >= 0; k-- {
				p := deriv[k+1]
				if len(p) != k+1 || xzero(p[k]) {
					return nil, np.ErrValue
				}
				next[k+1] = xdiv(remain[k], p[k])
				for i := 0; i < k; i++ {
					remain[i] = remain[i].Sub(xmul(next[k+1], p[i]))
				}
			}
			next[0] = target.Sub(b.xevaluate(next, x87.FromComplex128(lower)))
			x = next
		}
		return x, nil
	})
}
func (b Basis) xcompose(x, v []xc, target Basis) []xc {
	x, v = xtrim(x), xtrim(v)
	out, prev, curr := []xc{x[0]}, []xc{{}}, []xc{xi(1)}
	for k := 1; k < len(x); k++ {
		a, be, g := b.xrec(k - 1)
		next := target.xmul(v, curr)
		next = xadd(xadd(next, curr, xneg(be)), prev, xneg(g))
		next = xscale(next, xdiv(xi(1), a))
		out = xadd(out, next, x[k])
		prev, curr = curr, next
	}
	return out
}
func (b Basis) xfromroots(roots *np.Array) (*np.Array, error) {
	d, e := xdtype(roots)
	if e != nil {
		return nil, e
	}
	v, e := xvalues(roots)
	if e != nil {
		return nil, e
	}
	al, be, _ := b.xrec(0)
	o := []xc{xi(1)}
	for _, r := range v {
		o = b.xmul(o, []xc{be.Sub(r), al})
	}
	return xarray(o, d, len(o))
}
func (b Basis) xroots(a *np.Array) (*np.Array, error) {
	x, e := xcoeff(a)
	if e != nil {
		return nil, e
	}
	x = xtrim(x)
	d := a.DType()
	if len(x) == 1 {
		return np.Zeros(d, 0)
	}
	if len(x) != 2 {
		return nil, np.ErrDType
	} // NumPy's LAPACK rejects longdouble.
	al, be, _ := b.xrec(0)
	return xarray([]xc{be.Sub(xdiv(xmul(al, x[0]), x[1]))}, d, 1)
}

func (b Basis) xvander(a *np.Array, degree int) (*np.Array, error) {
	d := a.DType()
	shape := a.Shape()
	if a.NDim() == 0 {
		shape = []int{1}
	}
	shape = append(shape, degree+1)
	n, e := np.ElementCount(d, shape...)
	if e != nil {
		return nil, e
	}
	v, e := xvalues(a)
	if e != nil {
		return nil, e
	}
	out := make([]xc, n)
	for i, x := range v {
		r := out[i*(degree+1) : (i+1)*(degree+1)]
		r[0] = xi(1)
		for k := 0; k < degree; k++ {
			al, be, g := b.xrec(k)
			v := xmul(x.Sub(be), r[k])
			if k > 0 {
				v = v.Sub(xmul(g, r[k-1]))
			}
			r[k+1] = xdiv(v, al)
		}
	}
	return xarray(out, d, shape...)
}
func (b Basis) xcompanion(a *np.Array) (*np.Array, error) {
	x, e := xcoeff(a)
	if e != nil {
		return nil, e
	}
	x = xtrim(x)
	n := len(x) - 1
	if n < 1 {
		return nil, np.ErrShape
	}
	d := a.DType()
	size, e := np.ElementCount(d, n, n)
	if e != nil {
		return nil, e
	}
	m := make([]xc, size)
	if n == 1 {
		al, be, _ := b.xrec(0)
		m[0] = be.Sub(xdiv(xmul(al, x[0]), x[1]))
		return xarray(m, d, n, n)
	}
	// NumPy's orthogonal companion construction calculates the fixed
	// scaling factors in binary64 even when coefficients are longdouble.
	// Keep that rounding boundary, then apply coefficient corrections in x87.
	as := func(v float64) xc { return xc{Real: x87.FromFloat64(v)} }
	scale := make([]float64, n)
	switch b {
	case Chebyshev:
		scale[0] = 1
		for i := 1; i < n; i++ {
			scale[i] = math.Sqrt(.5)
		}
	case Legendre:
		for i := range scale {
			scale[i] = 1 / math.Sqrt(float64(2*i+1))
		}
	case Hermite, HermiteE:
		scale[n-1] = 1
		for i := n - 2; i >= 0; i-- {
			k := float64(i + 1)
			if b == Hermite {
				k *= 2
			}
			scale[i] = scale[i+1] * (1 / math.Sqrt(k))
		}
	}
	for j := 0; j < n; j++ {
		if b == Laguerre {
			m[j*n+j] = xi(2*j + 1)
		}
		if j == n-1 {
			continue
		}
		k := float64(j + 1)
		var v float64
		switch b {
		case Power:
			m[(j+1)*n+j] = xi(1)
			continue
		case Chebyshev:
			v = .5
			if j == 0 {
				v = math.Sqrt(.5)
			}
		case Legendre:
			v = k * scale[j] * scale[j+1]
		case Hermite:
			v = math.Sqrt(.5 * k)
		case HermiteE:
			v = math.Sqrt(k)
		case Laguerre:
			v = -k
		}
		m[(j+1)*n+j], m[j*n+j+1] = as(v), as(v)
	}
	for i := 0; i < n; i++ {
		v := xdiv(x[i], x[n])
		switch b {
		case Chebyshev:
			v = xmul(xmul(v, as(scale[i]/scale[n-1])), as(.5))
		case Legendre:
			v = xmul(xmul(v, as(scale[i]/scale[n-1])), as(float64(n)/float64(2*n-1)))
		case Hermite:
			v = xdiv(xmul(as(scale[i]), x[i]), xmul(xi(2), x[n]))
		case HermiteE:
			v = xdiv(xmul(as(scale[i]), x[i]), x[n])
		case Laguerre:
			v = xmul(v, xi(-n))
		}
		m[i*n+n-1] = m[i*n+n-1].Sub(v)
	}
	return xarray(m, d, n, n)
}
func (s *Series) xcomposeSeries(q *Series) (*Series, error) {
	c, e := xcoeff(s.coeff)
	if e != nil {
		return nil, e
	}
	v, e := xcoeff(q.coeff)
	if e != nil {
		return nil, e
	}
	c, v = xtrim(c), xtrim(v)
	if len(v) > 1 && len(c)-1 > (int(^uint(0)>>1)/32-1)/(len(v)-1) {
		return nil, np.ErrShape
	}
	d, e := xdtype(s.coeff, q.coeff)
	if e != nil {
		return nil, e
	}
	off, scale := s.complexMapping()
	if s.hasComplexMap() {
		d = np.Complex256
	}
	x := xadd(xscale(v, x87.FromComplex128(scale)), []xc{x87.FromComplex128(off)}, xi(1))
	out := s.Basis.xcompose(c, x, q.Basis)
	a, e := xarray(out, d, len(out))
	if e != nil {
		return nil, e
	}
	return q.withCoefficients(a)
}
