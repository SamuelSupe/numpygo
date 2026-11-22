package poly

import (
	"iter"
	"math"
	"math/cmplx"
	"strings"
	"unicode"

	"github.com/SamuelSupe/numpygo"
)

func validSymbol(s string) bool {
	for i, r := range s {
		start := r == '_' || unicode.IsLetter(r) || unicode.Is(unicode.Nl, r) || unicode.Is(unicode.Other_ID_Start, r)
		if !start && (i == 0 || !(unicode.IsDigit(r) || unicode.Is(unicode.Mn, r) || unicode.Is(unicode.Mc, r) ||
			unicode.Is(unicode.Pc, r) || unicode.Is(unicode.Other_ID_Continue, r))) {
			return false
		}
	}
	return s != ""
}

func (b Basis) DefaultDomain() [2]float64 {
	if b == Laguerre {
		return [2]float64{0, 1}
	}
	return [2]float64{-1, 1}
}
func (b Basis) DefaultWindow() [2]float64 { return b.DefaultDomain() }
func (b Basis) BasisName() string {
	switch b {
	case Chebyshev:
		return "T"
	case Legendre:
		return "P"
	case Hermite:
		return "H"
	case HermiteE:
		return "He"
	case Laguerre:
		return "L"
	}
	return ""
}
func (b Basis) className() string {
	switch b {
	case Power:
		return "Polynomial"
	case HermiteE:
		return "HermiteE"
	default:
		name := b.String()
		return strings.ToUpper(name[:1]) + name[1:]
	}
}
func (b Basis) MaxPower() int { return 100 }

func seriesIntervals(b Basis, opt SeriesOptions) (domain, window [2]float64, err error) {
	domain, window = b.DefaultDomain(), b.DefaultWindow()
	if opt.Domain != nil {
		domain = *opt.Domain
	}
	if opt.Window != nil {
		window = *opt.Window
	}
	if !b.valid() || domain[0] == domain[1] || window[0] == window[1] {
		err = np.ErrValue
	}
	return
}
func mapParameters(from, to [2]float64) (float64, float64) {
	delta := from[1] - from[0]
	return (from[1]*to[0] - from[0]*to[1]) / delta, (to[1] - to[0]) / delta
}
func seriesComplexIntervals(b Basis, opt SeriesOptions) (domain, window [2]complex128, err error) {
	if !b.valid() || opt.Domain != nil && opt.ComplexDomain != nil || opt.Window != nil && opt.ComplexWindow != nil {
		return domain, window, np.ErrValue
	}
	d, w := b.DefaultDomain(), b.DefaultWindow()
	if opt.Domain != nil {
		d = *opt.Domain
	}
	if opt.Window != nil {
		w = *opt.Window
	}
	domain = [2]complex128{complex(d[0], 0), complex(d[1], 0)}
	window = [2]complex128{complex(w[0], 0), complex(w[1], 0)}
	if opt.ComplexDomain != nil {
		domain = *opt.ComplexDomain
	}
	if opt.ComplexWindow != nil {
		window = *opt.ComplexWindow
	}
	if domain[0] == domain[1] || window[0] == window[1] {
		err = np.ErrValue
	}
	return
}
func mapComplexParameters(from, to [2]complex128) (complex128, complex128) {
	d := from[1] - from[0]
	return (from[1]*to[0] - from[0]*to[1]) / d, (to[1] - to[0]) / d
}
func mapComplexArray(x *np.Array, off, scale complex128) (*np.Array, error) {
	v, e := np.Multiply(x, np.ScalarOf(scale))
	if e != nil {
		return nil, e
	}
	return np.Add(v, np.ScalarOf(off))
}
func (s *Series) MapParametersComplex() (complex128, complex128) { return s.complexMapping() }
func (s *Series) MapParameters() (float64, float64) {
	if s.hasComplexMap() {
		o, k := s.complexMapping()
		return real(o), real(k)
	}
	return s.mapping()
}
func (s *Series) Symbol() string               { return s.symbol }
func (s *Series) Len() int                     { return s.coeff.Size() }
func (s *Series) Degree() int                  { return s.Len() - 1 }
func (s *Series) Iter() (iter.Seq[any], error) { return np.NDIter(s.coeff, "C") }

// CoefficientView shares the series' mutable coefficient storage without
// allowing the caller to resize its owning array. Coefficients returns a
// detached snapshot.
func (s *Series) CoefficientView() *np.Array {
	v, _ := s.coeff.Slice(np.Slice{})
	return v
}
func (s *Series) withCoefficients(c *np.Array) (*Series, error) {
	opt := SeriesOptions{Domain: &s.domain, Window: &s.window, Symbol: s.symbol}
	if s.domainIsComplex {
		opt.Domain = nil
		opt.ComplexDomain = &s.complexDomain
	}
	if s.windowIsComplex {
		opt.Window = nil
		opt.ComplexWindow = &s.complexWindow
	}
	return NewSeries(s.Basis, c, opt)
}
func (s *Series) Copy() (*Series, error) { return s.withCoefficients(s.coeff) }
func (s *Series) Truncate(size int) (*Series, error) {
	if size < 1 {
		return nil, np.ErrValue
	}
	c, e := s.coeff.Slice(np.Range(0, min(size, s.Len())))
	if e != nil {
		return nil, e
	}
	return s.withCoefficients(c)
}
func (s *Series) CutDegree(degree int) (*Series, error) {
	if degree < 0 {
		return nil, np.ErrValue
	}
	return s.Truncate(min(degree, s.Degree()) + 1)
}
func (s *Series) Trim(tol float64) (*Series, error) {
	c, e := s.Basis.Trim(s.coeff, tol)
	if e != nil {
		return nil, e
	}
	return s.withCoefficients(c)
}
func (s *Series) HasSameCoefficients(q *Series) (bool, error) {
	if q == nil {
		return false, nil
	}
	return np.ArrayEqual(s.coeff, q.coeff, false)
}
func (s *Series) HasSameDomain(q *Series) bool { return q != nil && s.complexDomain == q.complexDomain }
func (s *Series) HasSameWindow(q *Series) bool { return q != nil && s.complexWindow == q.complexWindow }
func (s *Series) HasSameType(q *Series) bool   { return q != nil && s.Basis == q.Basis }
func (s *Series) Equal(q *Series) (bool, error) {
	if !s.HasSameType(q) || !s.HasSameDomain(q) || !s.HasSameWindow(q) || s.symbol != q.symbol {
		return false, nil
	}
	return s.HasSameCoefficients(q)
}

// Binary operations accept a compatible Series or an explicit scalar/1-D
// coefficient array. Series operands must match basis, domain, window and symbol.
func (s *Series) operand(other any) (*np.Array, error) {
	switch q := other.(type) {
	case *Series:
		if !s.HasSameType(q) || !s.HasSameDomain(q) || !s.HasSameWindow(q) {
			return nil, np.ErrDType
		}
		if s.symbol != q.symbol {
			return nil, np.ErrValue
		}
		return q.coeff, nil
	case *np.Array:
		if q.DType().IsExtended() {
			if q.NDim() > 1 || q.Size() == 0 {
				return nil, np.ErrShape
			}
			return q, nil
		}
		if _, err := coeffs(q); err != nil {
			return nil, err
		}
		return q, nil
	default:
		return nil, np.ErrDType
	}
}
func (s *Series) binary(other any, op func(*np.Array, *np.Array) (*np.Array, error), reflected bool) (*Series, error) {
	q, e := s.operand(other)
	if e != nil {
		return nil, e
	}
	a := s.coeff
	if reflected {
		a, q = q, a
	}
	c, e := op(a, q)
	if e != nil {
		return nil, e
	}
	return s.withCoefficients(c)
}
func (s *Series) Add(q any) (*Series, error)  { return s.binary(q, s.Basis.Add, false) }
func (s *Series) Sub(q any) (*Series, error)  { return s.binary(q, s.Basis.Sub, false) }
func (s *Series) RSub(q any) (*Series, error) { return s.binary(q, s.Basis.Sub, true) }
func (s *Series) Mul(q any) (*Series, error)  { return s.binary(q, s.Basis.Mul, false) }
func (s *Series) Negative() (*Series, error) {
	c, err := np.Negative(s.coeff)
	if err != nil {
		return nil, err
	}
	return s.withCoefficients(c)
}
func (s *Series) Positive() *Series { return s }
func (s *Series) Pow(power int) (*Series, error) {
	c, e := s.Basis.Pow(s.coeff, power, s.Basis.MaxPower())
	if e != nil {
		return nil, e
	}
	return s.withCoefficients(c)
}
func (s *Series) DivMod(q any, reflected bool) (*Series, *Series, error) {
	b, e := s.operand(q)
	if e != nil {
		return nil, nil, e
	}
	a := s.coeff
	if reflected {
		a, b = b, a
	}
	quot, rem, e := s.Basis.Div(a, b)
	if e != nil {
		return nil, nil, e
	}
	x, e := s.withCoefficients(quot)
	if e != nil {
		return nil, nil, e
	}
	y, e := s.withCoefficients(rem)
	return x, y, e
}
func (s *Series) DivideScalar(q *np.Array) (*Series, error) {
	if q.DType() == 0 || q.NDim() != 0 || q.DType() == np.Bool {
		return nil, np.ErrDType
	}
	v, _, e := s.DivMod(q, false)
	return v, e
}

// Compose evaluates s on a series argument, retaining the argument's basis,
// domain, window and symbol. The source basis recurrence is evaluated directly
// in the argument basis, avoiding a lossy round-trip through the power basis.
func (s *Series) Compose(q *Series) (*Series, error) {
	if q == nil || !q.Basis.valid() || !s.Basis.valid() {
		return nil, np.ErrValue
	}
	if hasExtended(s.coeff, q.coeff) {
		return s.xcomposeSeries(q)
	}
	c, e := coeffs(s.coeff)
	if e != nil {
		return nil, e
	}
	v, e := coeffs(q.coeff)
	if e != nil {
		return nil, e
	}
	d, e := numericType(s.coeff.DType(), q.coeff.DType(), np.Float64)
	if e != nil {
		return nil, e
	}
	c, v = trim(c), trim(v)
	if len(v) > 1 && len(c)-1 > (int(^uint(0)>>1)/16-1)/(len(v)-1) {
		return nil, np.ErrShape
	}
	off, scale := s.complexMapping()
	if s.hasComplexMap() {
		d = np.Complex128
	}
	x := add(scaled(v, scale), []complex128{off}, 1)
	out, prev, curr := []complex128{c[0]}, []complex128{0}, []complex128{1}
	for k := 1; k < len(c); k++ {
		alpha, beta, gamma := s.Basis.recurrence(k - 1)
		next := q.Basis.mul(x, curr)
		if beta != 0 {
			next = add(next, curr, complex(-beta, 0))
		}
		if gamma != 0 {
			next = add(next, prev, complex(-gamma, 0))
		}
		next = scaled(next, complex(1/alpha, 0))
		out = add(out, next, c[k])
		prev, curr = curr, next
	}
	res, e := coeffArray(out, d)
	if e != nil {
		return nil, e
	}
	return q.withCoefficients(res)
}
func (b Basis) Identity(opt SeriesOptions) (*Series, error) {
	domain, window, e := seriesComplexIntervals(b, opt)
	if e != nil {
		return nil, e
	}
	off, scale := mapComplexParameters(window, domain)
	c, e := b.Line(off, scale)
	if e != nil {
		return nil, e
	}
	if opt.ComplexDomain != nil || opt.ComplexWindow != nil {
		c, e = c.Astype(np.Complex128)
		if e != nil {
			return nil, e
		}
	}
	return NewSeries(b, c, opt)
}
func (b Basis) SeriesBasis(degree int, opt SeriesOptions) (*Series, error) {
	if degree < 0 || degree == int(^uint(0)>>1) {
		return nil, np.ErrValue
	}
	c, e := np.Zeros(np.Float64, degree+1)
	if e != nil {
		return nil, e
	}
	if e = c.SetValue(1.0, degree); e != nil {
		return nil, e
	}
	return NewSeries(b, c, opt)
}
func (s *Series) Convert(target Basis, opt SeriesOptions) (*Series, error) {
	opt.Symbol = s.symbol
	q, e := target.Identity(opt)
	if e != nil {
		return nil, e
	}
	return s.Compose(q)
}
func (b Basis) Cast(s *Series, opt SeriesOptions) (*Series, error) {
	if s == nil {
		return nil, np.ErrValue
	}
	return s.Convert(b, opt)
}
func (s *Series) Linspace(n int, domain *[2]float64) (*np.Array, *np.Array, error) {
	if s.domainIsComplex && domain == nil {
		x, _, e := np.LinspaceArray(np.ScalarOf(s.complexDomain[0]), np.ScalarOf(s.complexDomain[1]), np.SpaceOptions{Num: n})
		if e != nil {
			return nil, nil, e
		}
		y, e := s.Eval(x)
		return x, y, e
	}
	d := s.domain
	if domain != nil {
		d = *domain
	}
	x, e := np.Linspace(d[0], d[1], n, true, np.Float64)
	if e != nil {
		return nil, nil, e
	}
	y, e := s.Eval(x)
	return x, y, e
}

func realDomain(x *np.Array) ([2]float64, error) {
	var domain [2]float64
	if !x.DType().IsNativeNumeric() || x.DType().IsComplex() || x.Size() == 0 {
		return domain, np.ErrDType
	}
	v, e := x.Complex128s()
	if e != nil {
		return domain, e
	}
	domain = [2]float64{real(v[0]), real(v[0])}
	for _, z := range v[1:] {
		domain[0], domain[1] = math.Min(domain[0], real(z)), math.Max(domain[1], real(z))
	}
	if math.IsNaN(domain[0]) || math.IsInf(domain[0], 0) || math.IsInf(domain[1], 0) {
		return domain, np.ErrValue
	}
	return domain, nil
}
func mapRealArray(x *np.Array, off, scale float64) (*np.Array, error) {
	v, e := np.Multiply(x, np.Scalar(scale))
	if e != nil {
		return nil, e
	}
	return np.Add(v, np.Scalar(off))
}

// SeriesFromRoots constructs a monic polynomial in physical coordinates.
// When inferDomain is true and Domain is nil, roots determine the real domain;
// otherwise the basis' default domain is used.
func (b Basis) SeriesFromRoots(roots *np.Array, opt SeriesOptions, inferDomain bool) (*Series, error) {
	if !roots.DType().IsNativeNumeric() && !roots.DType().IsExtended() || roots.NDim() > 1 || roots.Size() == 0 {
		return nil, np.ErrShape
	}
	if inferDomain && opt.Domain == nil && opt.ComplexDomain == nil {
		if roots.DType().IsComplex() {
			d, e := complexDomainOf(roots)
			if e != nil {
				return nil, e
			}
			opt.ComplexDomain = &d
		} else {
			d, e := realDomain(roots)
			if e != nil {
				return nil, e
			}
			opt.Domain = &d
		}
	}
	if opt.ComplexDomain != nil || opt.ComplexWindow != nil {
		d, w, e := seriesComplexIntervals(b, opt)
		if e != nil {
			return nil, e
		}
		o, k := mapComplexParameters(d, w)
		r, e := mapComplexArray(roots, o, k)
		if e != nil {
			return nil, e
		}
		c, e := b.FromRoots(r)
		if e != nil {
			return nil, e
		}
		c, e = np.Divide(c, np.ScalarOf(cmplx.Pow(k, complex(float64(roots.Size()), 0))))
		if e != nil {
			return nil, e
		}
		return NewSeries(b, c, opt)
	}
	domain, window, e := seriesIntervals(b, opt)
	if e != nil {
		return nil, e
	}
	off, scale := mapParameters(domain, window)
	r, e := mapRealArray(roots, off, scale)
	if e != nil {
		return nil, e
	}
	c, e := b.FromRoots(r)
	if e != nil {
		return nil, e
	}
	c, e = np.Divide(c, np.Scalar(math.Pow(scale, float64(roots.Size()))))
	if e != nil {
		return nil, e
	}
	return NewSeries(b, c, opt)
}

type SeriesFitOptions struct {
	SeriesOptions
	FitOptions
	UseDefaultDomain bool // otherwise nil Domain infers the real sample range
}

// FitSeries maps samples to the basis window before least squares. FitResult
// contains the residual, rank, singular values and rcond of that scaled fit.
func (b Basis) FitSeries(x, y *np.Array, degree int, opt SeriesFitOptions) (*Series, FitResult, error) {
	var result FitResult
	if x.NDim() != 1 || y.NDim() != 1 || x.Size() == 0 || x.Size() != y.Size() {
		return nil, result, np.ErrShape
	}
	if opt.Domain == nil && opt.ComplexDomain == nil && !opt.UseDefaultDomain && x.DType().IsComplex() {
		d, e := complexDomainOf(x)
		if e != nil {
			return nil, result, e
		}
		if d[0] == d[1] {
			d[0] -= 1
			d[1] += 1
		}
		opt.ComplexDomain = &d
	}
	if opt.Domain == nil && opt.ComplexDomain == nil && !opt.UseDefaultDomain {
		d, e := realDomain(x)
		if e != nil {
			return nil, result, e
		}
		if d[0] == d[1] {
			d[0]--
			d[1]++
		}
		opt.Domain = &d
	}
	if opt.ComplexDomain != nil || opt.ComplexWindow != nil {
		d, w, e := seriesComplexIntervals(b, opt.SeriesOptions)
		if e != nil {
			return nil, result, e
		}
		o, k := mapComplexParameters(d, w)
		mapped, e := mapComplexArray(x, o, k)
		if e != nil {
			return nil, result, e
		}
		result, e = b.Fit(mapped, y, degree, opt.FitOptions)
		if e != nil {
			return nil, result, e
		}
		s, e := NewSeries(b, result.Coefficients, opt.SeriesOptions)
		return s, result, e
	}
	domain, window, e := seriesIntervals(b, opt.SeriesOptions)
	if e != nil {
		return nil, result, e
	}
	off, scale := mapParameters(domain, window)
	mapped, e := mapRealArray(x, off, scale)
	if e != nil {
		return nil, result, e
	}
	result, e = b.Fit(mapped, y, degree, opt.FitOptions)
	if e != nil {
		return nil, result, e
	}
	s, e := NewSeries(b, result.Coefficients, opt.SeriesOptions)
	return s, result, e
}
func complexDomainOf(x *np.Array) ([2]complex128, error) {
	var out [2]complex128
	d, e := GetDomain(x)
	if e != nil {
		return out, e
	}
	v, e := d.Complex128s()
	if e != nil || len(v) != 2 {
		return out, np.ErrDType
	}
	return [2]complex128{v[0], v[1]}, nil
}

// InterpolateChebyshev constructs a Chebyshev series from a scalar callback
// on the scaled first-kind Chebyshev points of the requested real domain.
func InterpolateChebyshev(f func(float64) complex128, degree int, domain *[2]float64) (*Series, error) {
	if f == nil {
		return nil, np.ErrValue
	}
	opt := SeriesOptions{Domain: domain}
	d, w, e := seriesIntervals(Chebyshev, opt)
	if e != nil {
		return nil, e
	}
	off, scale := mapParameters(w, d)
	c, e := ChebInterpolate(func(x float64) complex128 { return f(off + scale*x) }, degree)
	if e != nil {
		return nil, e
	}
	return NewSeries(Chebyshev, c, opt)
}
