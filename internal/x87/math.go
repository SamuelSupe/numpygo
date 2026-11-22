package x87

import "math/big"

// Transcendentals evaluate with guard bits before rounding onto the x87
// lattice. Argument reduction uses extra precision for large angles.
const mathPrec uint = 192

func bf(p uint) *big.Float                 { return new(big.Float).SetPrec(p).SetMode(big.ToNearestEven) }
func bi(p uint, n int64) *big.Float        { return bf(p).SetInt64(n) }
func wide(x Float) *big.Float              { return bf(mathPrec).Set(x.BigFloat()) }
func add(a, b *big.Float) *big.Float       { return bf(a.Prec()).Add(a, b) }
func sub(a, b *big.Float) *big.Float       { return bf(a.Prec()).Sub(a, b) }
func mul(a, b *big.Float) *big.Float       { return bf(a.Prec()).Mul(a, b) }
func quo(a, b *big.Float) *big.Float       { return bf(a.Prec()).Quo(a, b) }
func neg(a *big.Float) *big.Float          { return bf(a.Prec()).Neg(a) }
func abs(a *big.Float) *big.Float          { return bf(a.Prec()).Abs(a) }
func sqrt(a *big.Float) *big.Float         { return bf(a.Prec()).Sqrt(a) }
func scale(a *big.Float, e int) *big.Float { return bf(a.Prec()).SetMantExp(a, e) }
func insignificant(term, sum *big.Float) bool {
	return term.Sign() == 0 || term.MantExp(nil) < sum.MantExp(nil)-int(sum.Prec())-8
}
func roundWide(x *big.Float) Float {
	if x.IsInf() || x.MantExp(nil) > 16384 {
		return Inf(x.Signbit())
	}
	if x.Sign() == 0 || x.MantExp(nil) < -16445 {
		return Zero(x.Signbit())
	}
	r, _ := x.Rat(nil)
	return roundRat(r, x.Signbit())
}

// Gauss-Legendre converges quadratically, including for reduction of the
// largest finite x87 arguments (roughly 16,384 extra bits of pi).
func calculatePi(p uint) *big.Float {
	a, b, t := bi(p, 1), sqrt(quo(bi(p, 1), bi(p, 2))), quo(bi(p, 1), bi(p, 4))
	power := bi(p, 1)
	for n := uint(1); n < p+8; n *= 2 {
		mean := scale(add(a, b), -1)
		next := sqrt(mul(a, b))
		d := sub(a, mean)
		t = sub(t, mul(power, mul(d, d)))
		a, b, power = mean, next, scale(power, 1)
	}
	s := add(a, b)
	return quo(mul(s, s), scale(t, 2))
}

var piWide = calculatePi(mathPrec)

func piAt(p uint) *big.Float {
	if p <= mathPrec {
		return bf(p).Set(piWide)
	}
	return calculatePi(p)
}
func Pi() Float { return roundWide(piWide) }

// log((1+x)/(1-x)) = 2 atanh(x), with |x| bounded by the
// range reduction. Relative termination also handles very small inputs.
func logRatio(x *big.Float) *big.Float {
	p := x.Prec()
	x2 := mul(x, x)
	term, sum := bf(p).Set(x), bf(p).Set(x)
	for n := int64(3); ; n += 2 {
		term = mul(term, x2)
		v := quo(term, bi(p, n))
		sum = add(sum, v)
		if insignificant(v, sum) {
			return scale(sum, 1)
		}
	}
}

var ln2Wide = logRatio(quo(bi(mathPrec, 1), bi(mathPrec, 3)))
var ln10Wide = logWide(bi(mathPrec, 10))

func logWide(x *big.Float) *big.Float {
	p := x.Prec()
	m := bf(p)
	e := x.MantExp(m)
	// Keep m near 1, particularly for inputs one ULP from 1.
	if m.Cmp(sqrt(quo(bi(p, 1), bi(p, 2)))) < 0 {
		m = scale(m, 1)
		e--
	}
	v := logRatio(quo(sub(m, bi(p, 1)), add(m, bi(p, 1))))
	return add(v, mul(bi(p, int64(e)), bf(p).Set(ln2Wide)))
}
func log1pWide(x *big.Float) *big.Float {
	p := x.Prec()
	if abs(x).Cmp(quo(bi(p, 1), bi(p, 2))) <= 0 {
		return logRatio(quo(x, add(bi(p, 2), x)))
	}
	return logWide(add(bi(p, 1), x))
}
func expSeries(x *big.Float, minusOne bool) *big.Float {
	p := x.Prec()
	term := bf(p).Set(x)
	sum := bf(p).Set(x)
	if !minusOne {
		sum = add(sum, bi(p, 1))
	}
	for n := int64(2); ; n++ {
		term = quo(mul(term, x), bi(p, n))
		sum = add(sum, term)
		if insignificant(term, sum) {
			return sum
		}
	}
}
func expWide(x *big.Float) *big.Float {
	p := x.Prec()
	// Beyond these bounds no x87 output is representable. Keep
	// a generous margin for scale factors in the complex formulas.
	if x.Cmp(bi(p, 24000)) > 0 {
		return bf(p).SetInf(false)
	}
	if x.Cmp(bi(p, -24000)) < 0 {
		return bf(p)
	}
	ln2 := bf(p).Set(ln2Wide)
	k, _ := quo(x, ln2).Int64()
	remainder := sub(x, mul(bi(p, k), ln2))
	return scale(expSeries(remainder, false), int(k))
}
func expm1Wide(x *big.Float) *big.Float {
	if abs(x).Cmp(bi(x.Prec(), 1)) <= 0 {
		return expSeries(x, true)
	}
	return sub(expWide(x), bi(x.Prec(), 1))
}
func (x Float) Exp() Float {
	if x.IsNaN() {
		return x
	}
	if x.IsInf() {
		if x.Signbit() {
			return Zero(false)
		}
		return x
	}
	return roundWide(expWide(wide(x)))
}
func (x Float) Exp2() Float {
	if !x.IsFinite() {
		return x.Exp()
	}
	return roundWide(expWide(mul(wide(x), ln2Wide)))
}
func (x Float) Expm1() Float {
	if x.IsZero() || x.IsNaN() {
		return x
	}
	if x.IsInf() {
		if x.Signbit() {
			return FromInt64(-1)
		}
		return x
	}
	return roundWide(expm1Wide(wide(x)))
}
func (x Float) Log() Float {
	if x.IsNaN() {
		return x
	}
	if x.IsZero() {
		return Inf(true)
	}
	if x.Signbit() {
		return NaN()
	}
	if x.IsInf() {
		return x
	}
	return roundWide(logWide(wide(x)))
}
func (x Float) Log2() Float {
	if !x.IsFinite() || x.IsZero() || x.Signbit() {
		return x.Log()
	}
	return roundWide(quo(logWide(wide(x)), ln2Wide))
}
func (x Float) Log10() Float {
	if !x.IsFinite() || x.IsZero() || x.Signbit() {
		return x.Log()
	}
	return roundWide(quo(logWide(wide(x)), ln10Wide))
}
func (x Float) Log1p() Float {
	if x.IsNaN() || x.IsZero() {
		return x
	}
	c, _ := x.Compare(FromInt64(-1))
	if c < 0 {
		return NaN()
	}
	if c == 0 {
		return Inf(true)
	}
	if x.IsInf() {
		return x
	}
	return roundWide(log1pWide(wide(x)))
}

// Taylor series on [-pi/4, pi/4] with no cancellation in the
// leading sine term. Reduce using an integer multiple of pi/2.
func smallSinCos(x *big.Float) (*big.Float, *big.Float) {
	p := x.Prec()
	x2 := neg(mul(x, x))
	s, c := bf(p).Set(x), bi(p, 1)
	st, ct := bf(p).Set(x), bi(p, 1)
	for n := int64(2); ; n += 2 {
		ct = quo(mul(ct, x2), bi(p, n*(n-1)))
		st = quo(mul(st, x2), bi(p, n*(n+1)))
		s, c = add(s, st), add(c, ct)
		if insignificant(st, s) && insignificant(ct, c) {
			return s, c
		}
	}
}
func nearestInteger(x *big.Float) *big.Int {
	n, _ := x.Int(nil)
	fraction := sub(x, bf(x.Prec()).SetInt(n))
	if abs(fraction).Cmp(quo(bi(x.Prec(), 1), bi(x.Prec(), 2))) >= 0 {
		n.Add(n, big.NewInt(int64(x.Sign())))
	}
	return n
}
func sinCosWide(x *big.Float) (*big.Float, *big.Float) {
	if x.Sign() == 0 {
		return bf(mathPrec).Set(x), bi(mathPrec, 1)
	}
	p := max(x.Prec(), mathPrec+uint(max(0, x.MantExp(nil))))
	// Small angles need only the usual guard bits; large ones need the
	// bits occupied by the integer part of the quadrant number as well.
	if p <= mathPrec+8 {
		p = mathPrec
	}
	r := bf(p).Set(x)
	halfPi := scale(piAt(p), -1)
	k := nearestInteger(quo(r, halfPi))
	r = sub(r, mul(bf(p).SetInt(k), halfPi))
	s, c := smallSinCos(bf(mathPrec).Set(r))
	switch k.And(k, big.NewInt(3)).Int64() {
	case 1:
		return c, neg(s)
	case 2:
		return neg(s), neg(c)
	case 3:
		return neg(c), s
	default:
		return s, c
	}
}
func (x Float) SinCos() (Float, Float) {
	if !x.IsFinite() {
		return NaN(), NaN()
	}
	if x.IsZero() {
		return x, FromInt64(1)
	}
	s, c := sinCosWide(wide(x))
	return roundWide(s), roundWide(c)
}
func (x Float) Sin() Float { s, _ := x.SinCos(); return s }
func (x Float) Cos() Float { _, c := x.SinCos(); return c }
func (x Float) Tan() Float {
	if !x.IsFinite() {
		return NaN()
	}
	if x.IsZero() {
		return x
	}
	s, c := sinCosWide(wide(x))
	return roundWide(quo(s, c))
}

// SinCosPiRatio supplies Fourier roots using the exact rational angle
// pi*numerator/denominator, rather than a rounded x87 angle. Exact axes
// are handled before evaluating the series.
func SinCosPiRatio(numerator, denominator int64) (Float, Float) {
	if denominator <= 0 {
		return NaN(), NaN()
	}
	// Reduce with integers, avoiding overflow in 2*numerator or 2*denominator.
	n, d := big.NewInt(numerator), big.NewInt(denominator)
	twoD := new(big.Int).Lsh(new(big.Int).Set(d), 1)
	n.Mod(n, twoD)
	twice := new(big.Int).Lsh(new(big.Int).Set(n), 1)
	q, r := new(big.Int), new(big.Int)
	q.QuoRem(twice, d, r)
	if r.Sign() == 0 {
		one := FromInt64(1)
		switch q.Int64() {
		case 0:
			return Zero(false), one
		case 1:
			return one, Zero(false)
		case 2:
			return Zero(false), one.Neg()
		case 3:
			return one.Neg(), Zero(false)
		}
	}
	p := mathPrec
	angle := mul(piAt(p), quo(bf(p).SetInt(n), bf(p).SetInt(d)))
	s, c := sinCosWide(angle)
	return roundWide(s), roundWide(c)
}

func atanSmall(x *big.Float) *big.Float {
	p := x.Prec()
	term, sum := bf(p).Set(x), bf(p).Set(x)
	x2 := neg(mul(x, x))
	for n := int64(3); ; n += 2 {
		term = mul(term, x2)
		v := quo(term, bi(p, n))
		sum = add(sum, v)
		if insignificant(v, sum) {
			return sum
		}
	}
}
func atanWide(x *big.Float) *big.Float {
	p := x.Prec()
	y := abs(x)
	var out *big.Float
	one := bi(p, 1)
	if y.Cmp(one) > 0 {
		out = sub(scale(piAt(p), -1), atanWide(quo(one, y)))
	} else if y.Cmp(scale(one, -1)) > 0 {
		out = add(scale(piAt(p), -2), atanSmall(quo(sub(y, one), add(y, one))))
	} else {
		out = atanSmall(y)
	}
	if x.Signbit() {
		out = neg(out)
	}
	return out
}
func (x Float) Atan() Float {
	if x.IsNaN() || x.IsZero() {
		return x
	}
	if x.IsInf() {
		p := roundWide(scale(piWide, -1))
		if x.Signbit() {
			return p.Neg()
		}
		return p
	}
	return roundWide(atanWide(wide(x)))
}
func Atan2(y, x Float) Float {
	if x.IsNaN() || y.IsNaN() {
		return NaN()
	}
	p := mathPrec
	var angle *big.Float
	switch {
	case y.IsZero():
		if !x.Signbit() {
			return y
		}
		angle = piAt(p)
	case x.IsZero():
		angle = scale(piAt(p), -1)
	case y.IsInf() && x.IsInf():
		angle = scale(piAt(p), -2)
		if x.Signbit() {
			angle = mul(angle, bi(p, 3))
		}
	case y.IsInf():
		angle = scale(piAt(p), -1)
	case x.IsInf():
		if !x.Signbit() {
			return Zero(y.Signbit())
		}
		angle = piAt(p)
	default:
		angle = atanWide(quo(abs(wide(y)), abs(wide(x))))
		if x.Signbit() {
			angle = sub(piAt(p), angle)
		}
	}
	if y.Signbit() {
		angle = neg(angle)
	}
	return roundWide(angle)
}
func (x Float) Asin() Float {
	if x.IsNaN() || x.IsZero() {
		return x
	}
	c, _ := x.Abs().Compare(FromInt64(1))
	if c > 0 {
		return NaN()
	}
	if c == 0 {
		return Atan2(x, Zero(false))
	}
	v := wide(x)
	one := bi(mathPrec, 1)
	r := atanWide(quo(v, sqrt(mul(sub(one, v), add(one, v)))))
	return roundWide(r)
}
func (x Float) Acos() Float {
	if x.IsNaN() {
		return x
	}
	c, _ := x.Abs().Compare(FromInt64(1))
	if c > 0 {
		return NaN()
	}
	if c == 0 {
		if x.Signbit() {
			return Pi()
		}
		return Zero(false)
	}
	if x.IsZero() {
		return roundWide(scale(piWide, -1))
	}
	v := wide(x)
	one := bi(mathPrec, 1)
	r := atanWide(quo(sqrt(mul(sub(one, v), add(one, v))), abs(v)))
	if x.Signbit() {
		r = sub(piWide, r)
	}
	return roundWide(r)
}
func sinhCoshWide(x *big.Float) (*big.Float, *big.Float) {
	p := x.Prec()
	a := abs(x)
	var s, c *big.Float
	if a.Cmp(bi(p, 1)) <= 0 {
		m, pos := expm1Wide(neg(a)), expm1Wide(a)
		s = scale(sub(pos, m), -1)
		c = add(bi(p, 1), scale(add(pos, m), -1))
	} else {
		e := expWide(a)
		inv := quo(bi(p, 1), e)
		s, c = scale(sub(e, inv), -1), scale(add(e, inv), -1)
	}
	if x.Signbit() {
		s = neg(s)
	}
	return s, c
}
func (x Float) Sinh() Float {
	if !x.IsFinite() || x.IsZero() {
		return x
	}
	s, _ := sinhCoshWide(wide(x))
	return roundWide(s)
}
func (x Float) Cosh() Float {
	if !x.IsFinite() {
		return x.Abs()
	}
	_, c := sinhCoshWide(wide(x))
	return roundWide(c)
}
func (x Float) Tanh() Float {
	if x.IsNaN() || x.IsZero() {
		return x
	}
	if x.IsInf() {
		if x.Signbit() {
			return FromInt64(-1)
		}
		return FromInt64(1)
	}
	v := wide(x.Abs())
	m := expm1Wide(neg(scale(v, 1)))
	t := quo(neg(m), add(bi(mathPrec, 2), m))
	if x.Signbit() {
		t = neg(t)
	}
	return roundWide(t)
}
func (x Float) Asinh() Float {
	if !x.IsFinite() || x.IsZero() {
		return x
	}
	v := wide(x.Abs())
	one := bi(mathPrec, 1)
	v2 := mul(v, v)
	y := log1pWide(add(v, quo(v2, add(one, sqrt(add(one, v2))))))
	if x.Signbit() {
		y = neg(y)
	}
	return roundWide(y)
}
func (x Float) Acosh() Float {
	if x.IsNaN() {
		return x
	}
	c, _ := x.Compare(FromInt64(1))
	if c < 0 {
		return NaN()
	}
	if c == 0 {
		return Zero(false)
	}
	if x.IsInf() {
		return x
	}
	v := wide(x)
	one := bi(mathPrec, 1)
	delta := sub(v, one)
	return roundWide(log1pWide(add(delta, sqrt(mul(delta, add(v, one))))))
}
func (x Float) Atanh() Float {
	if x.IsNaN() || x.IsZero() {
		return x
	}
	c, _ := x.Abs().Compare(FromInt64(1))
	if c > 0 {
		return NaN()
	}
	if c == 0 {
		return Inf(x.Signbit())
	}
	v := wide(x)
	return roundWide(scale(sub(log1pWide(v), log1pWide(neg(v))), -1))
}
func (x Float) Cbrt() Float {
	if !x.IsFinite() || x.IsZero() {
		return x
	}
	y := expWide(quo(logWide(wide(x.Abs())), bi(mathPrec, 3)))
	if x.Signbit() {
		y = neg(y)
	}
	return roundWide(y)
}
func (x Float) Degrees() Float {
	if !x.IsFinite() || x.IsZero() {
		return x
	}
	return roundWide(quo(mul(wide(x), bi(mathPrec, 180)), piWide))
}
func (x Float) Radians() Float {
	if !x.IsFinite() || x.IsZero() {
		return x
	}
	return roundWide(quo(mul(wide(x), piWide), bi(mathPrec, 180)))
}

func (x Float) I0() Float {
	if !x.IsFinite() {
		return NaN()
	}
	x = x.Abs()
	// Match NumPy's large-argument approximation, whose exp(x)
	// intermediate can overflow before the scaled magnitude would.
	if x.Exp().IsInf() {
		return Inf(false)
	}
	v := scale(wide(x), -1)
	v = mul(v, v)
	sum, term := bi(mathPrec, 1), bi(mathPrec, 1)
	for k := int64(1); ; k++ {
		term = quo(mul(term, v), bi(mathPrec, k*k))
		sum = add(sum, term)
		if insignificant(term, sum) {
			return roundWide(sum)
		}
	}
}
