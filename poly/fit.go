package poly

import (
	"math"
	"math/cmplx"
	"slices"
	"sort"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/linalg"
)

type FitOptions struct {
	Weights *np.Array
	Rcond   *float64
	Degrees []int
}

type FitResult struct {
	Coefficients, Residuals, SingularValues *np.Array
	Rank                                    int64
	Rcond                                   float64
}

// Fit implements scaled Vandermonde least squares. Degrees, when supplied,
// selects a subset of basis terms, leaving other coefficients zero.
func (b Basis) Fit(x, y *np.Array, degree int, options ...FitOptions) (FitResult, error) {
	var out FitResult
	if !b.valid() || x.DType() == 0 || y.DType() == 0 {
		return out, np.ErrDType
	}
	if len(options) > 1 || x.NDim() != 1 || x.Size() == 0 || y.NDim() < 1 || y.NDim() > 2 || x.Size() != y.Shape()[0] || degree < 0 {
		return out, np.ErrShape
	}
	var opt FitOptions
	if len(options) == 1 {
		opt = options[0]
	}
	degrees := slices.Clone(opt.Degrees)
	if opt.Degrees == nil {
		if degree == int(^uint(0)>>1) {
			return out, np.ErrShape
		}
		degrees = make([]int, degree+1)
		for i := range degrees {
			degrees[i] = i
		}
	} else {
		if len(degrees) == 0 {
			return out, np.ErrValue
		}
		sort.Ints(degrees)
		for i, n := range degrees {
			if n < 0 || i > 0 && n == degrees[i-1] {
				return out, np.ErrValue
			}
		}
		degree = degrees[len(degrees)-1]
	}
	d, err := numericType(x.DType(), y.DType())
	if err != nil {
		return out, err
	}
	if opt.Weights != nil {
		if opt.Weights.DType() == 0 || opt.Weights.NDim() != 1 || opt.Weights.Size() != x.Size() {
			return out, np.ErrShape
		}
		d, err = numericType(d, opt.Weights.DType())
		if err != nil {
			return out, err
		}
	}
	if d == np.Float16 {
		return out, np.ErrDType
	}
	// NumPy uses double-precision LAPACK internally even for single-precision input.
	working := np.Float64
	if d.IsComplex() {
		working = np.Complex128
	}
	v, err := b.Vander(x, degree)
	if err != nil {
		return out, err
	}
	vv, _ := v.Complex128s()
	yv, _ := y.Complex128s()
	n := x.Size()
	columns := 1
	if y.NDim() == 2 {
		columns = y.Shape()[1]
	}
	width := len(degrees)
	size, err := np.ElementCount(working, n, width)
	if err != nil {
		return out, err
	}
	lhs := make([]complex128, size)
	scales := make([]float64, width)
	var weights []complex128
	if opt.Weights != nil {
		weights, _ = opt.Weights.Complex128s()
	}
	for i := 0; i < n; i++ {
		w := complex128(1)
		if weights != nil {
			w = weights[i]
		}
		for j, deg := range degrees {
			z := vv[i*(degree+1)+deg] * w
			lhs[i*width+j] = z
			scales[j] = math.Hypot(scales[j], cmplx.Abs(z))
		}
		for j := 0; j < columns; j++ {
			yv[i*columns+j] *= w
		}
	}
	for j, s := range scales {
		if s == 0 {
			scales[j] = 1
		}
	}
	for i := 0; i < n; i++ {
		for j, s := range scales {
			lhs[i*width+j] /= complex(s, 0)
		}
	}
	la, err := array(lhs, working, n, width)
	if err != nil {
		return out, err
	}
	ya, err := array(yv, working, y.Shape()...)
	if err != nil {
		return out, err
	}
	eps := 2.220446049250313e-16
	if x.DType() == np.Float32 || x.DType() == np.Complex64 {
		eps = 1.1920928955078125e-7
	}
	if x.DType() == np.Float16 {
		eps = 0.0009765625
	}
	out.Rcond = float64(n) * eps
	if opt.Rcond != nil {
		out.Rcond = *opt.Rcond
	}
	fit, res, rank, sv, err := linalg.Lstsq(la, ya, &out.Rcond)
	if err != nil {
		return out, err
	}
	fv, _ := fit.Complex128s()
	outShape := []int{degree + 1}
	if y.NDim() == 2 {
		outShape = append(outShape, columns)
	}
	size, err = np.ElementCount(d, outShape...)
	if err != nil {
		return out, err
	}
	coef := make([]complex128, size)
	for j, deg := range degrees {
		for i := 0; i < columns; i++ {
			coef[deg*columns+i] = fv[j*columns+i] / complex(scales[j], 0)
		}
	}
	out.Coefficients, err = array(coef, d, outShape...)
	if err != nil {
		return out, err
	}
	rd := np.Float64
	if d == np.Float32 || d == np.Complex64 {
		rd = np.Float32
	}
	out.Residuals, err = res.Astype(rd)
	if err != nil {
		return out, err
	}
	out.SingularValues, err = sv.Astype(rd)
	out.Rank = rank
	return out, err
}

// Series stores a polynomial together with the affine domain-to-window map.
type Series struct {
	Basis                            Basis
	coeff                            *np.Array
	domain, window                   [2]float64
	complexDomain, complexWindow     [2]complex128
	domainIsComplex, windowIsComplex bool
	symbol                           string
}
type SeriesOptions struct {
	Domain, Window               *[2]float64
	ComplexDomain, ComplexWindow *[2]complex128
	Symbol                       string // empty selects the default symbol "x"
}

func NewSeries(b Basis, c *np.Array, options ...SeriesOptions) (*Series, error) {
	if !b.valid() || len(options) > 1 {
		return nil, np.ErrValue
	}
	if !c.DType().IsNativeNumeric() && !c.DType().IsExtended() {
		return nil, np.ErrDType
	}
	arrays, err := AsSeries([]*np.Array{c}, false)
	if err != nil {
		return nil, err
	}
	coeff := arrays[0]
	var opt SeriesOptions
	if len(options) == 1 {
		opt = options[0]
	}
	domain, window, err := seriesComplexIntervals(b, opt)
	if err != nil {
		return nil, err
	}
	if opt.Symbol == "" {
		opt.Symbol = "x"
	}
	if !validSymbol(opt.Symbol) {
		return nil, np.ErrValue
	}
	return &Series{Basis: b, coeff: coeff,
		domain: [2]float64{real(domain[0]), real(domain[1])}, window: [2]float64{real(window[0]), real(window[1])},
		complexDomain: domain, complexWindow: window, domainIsComplex: opt.ComplexDomain != nil, windowIsComplex: opt.ComplexWindow != nil, symbol: opt.Symbol}, nil
}
func (s *Series) Coefficients() (*np.Array, error) { return s.coeff.Copy() }
func (s *Series) Domain() [2]float64               { return s.domain }
func (s *Series) Window() [2]float64               { return s.window }
func (s *Series) DomainComplex() [2]complex128     { return s.complexDomain }
func (s *Series) WindowComplex() [2]complex128     { return s.complexWindow }
func (s *Series) complexMapping() (complex128, complex128) {
	if !s.hasComplexMap() {
		o, k := s.mapping()
		return complex(o, 0), complex(k, 0)
	}
	return mapComplexParameters(s.complexDomain, s.complexWindow)
}
func (s *Series) hasComplexMap() bool { return s.domainIsComplex || s.windowIsComplex }
func (s *Series) mapping() (float64, float64) {
	return mapParameters(s.domain, s.window)
}
func (s *Series) Eval(x *np.Array) (*np.Array, error) {
	if s.hasComplexMap() {
		off, scale := s.complexMapping()
		v, e := mapComplexArray(x, off, scale)
		if e != nil {
			return nil, e
		}
		return s.Basis.Eval(s.coeff, v)
	}
	off, scale := s.mapping()
	v, err := mapRealArray(x, off, scale)
	if err != nil {
		return nil, err
	}
	return s.Basis.Eval(s.coeff, v)
}
func (s *Series) Derivative(order int) (*Series, error) {
	if s.hasComplexMap() {
		if order > 0 && !s.coeff.DType().IsComplex() {
			return nil, np.ErrDType
		}
		_, scale := s.complexMapping()
		c, e := s.Basis.Derivative(s.coeff, order, scale)
		if e != nil {
			return nil, e
		}
		return s.withCoefficients(c)
	}
	_, scale := s.mapping()
	c, err := s.Basis.Derivative(s.coeff, order, complex(scale, 0))
	if err != nil {
		return nil, err
	}
	return s.withCoefficients(c)
}
func (s *Series) Integral(order int, constants []complex128, lower float64) (*Series, error) {
	return s.Integrate(order, constants, &lower)
}

// Integrate uses zero in window coordinates when lower is nil, matching the
// default NumPy integration bound. A non-nil lower is in domain coordinates.
func (s *Series) Integrate(order int, constants []complex128, lower *float64) (*Series, error) {
	if s.hasComplexMap() {
		var lb *complex128
		if lower != nil {
			v := complex(*lower, 0)
			lb = &v
		}
		return s.IntegrateComplex(order, constants, lb)
	}
	off, scale := s.mapping()
	lbnd := 0.0
	if lower != nil {
		lbnd = off + *lower*scale
	}
	c, err := s.Basis.Integral(s.coeff, order, constants, complex(lbnd, 0), complex(1/scale, 0))
	if err != nil {
		return nil, err
	}
	return s.withCoefficients(c)
}
func (s *Series) IntegrateComplex(order int, constants []complex128, lower *complex128) (*Series, error) {
	if order > 0 && !s.coeff.DType().IsComplex() && (s.hasComplexMap() || lower != nil && imag(*lower) != 0) {
		return nil, np.ErrDType
	}
	off, scale := s.complexMapping()
	lb := complex128(0)
	if lower != nil {
		lb = off + *lower*scale
	}
	c, e := s.Basis.Integral(s.coeff, order, constants, lb, 1/scale)
	if e != nil {
		return nil, e
	}
	return s.withCoefficients(c)
}
func (s *Series) Roots() (*np.Array, error) {
	r, err := s.Basis.Roots(s.coeff)
	if err != nil {
		return nil, err
	}
	if s.hasComplexMap() {
		off, scale := mapComplexParameters(s.complexWindow, s.complexDomain)
		return mapComplexArray(r, off, scale)
	}
	off, scale := mapParameters(s.window, s.domain)
	return mapRealArray(r, off, scale)
}
