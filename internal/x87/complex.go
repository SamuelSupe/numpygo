package x87

import "math/big"

// Complex represents the Linux/x86 clongdouble/complex256 numerical value
// as two independently rounded extended real components.
type Complex struct{ Real, Imag Float }

func FromComplex128(v complex128) Complex { return Complex{FromFloat64(real(v)), FromFloat64(imag(v))} }
func ParseComplex(re, im string) (Complex, error) {
	a, e := Parse(re)
	if e != nil {
		return Complex{}, e
	}
	b, e := Parse(im)
	return Complex{a, b}, e
}
func (z Complex) Add(w Complex) Complex { return Complex{z.Real.Add(w.Real), z.Imag.Add(w.Imag)} }
func (z Complex) Sub(w Complex) Complex { return Complex{z.Real.Sub(w.Real), z.Imag.Sub(w.Imag)} }
func (z Complex) Mul(w Complex) Complex {
	return Complex{z.Real.Mul(w.Real).Sub(z.Imag.Mul(w.Imag)), z.Real.Mul(w.Imag).Add(z.Imag.Mul(w.Real))}
}
func (z Complex) Div(w Complex) Complex {
	// Scaled complex division avoids squaring the larger denominator.
	a, b, c, d := z.Real, z.Imag, w.Real, w.Imag
	comparison, ordered := c.Abs().Compare(d.Abs())
	if !ordered {
		return Complex{NaN(), NaN()}
	}
	if c.IsZero() && d.IsZero() {
		return Complex{a.Div(c), b.Div(c)}
	}
	if comparison >= 0 {
		r := d.Div(c)
		s := FromInt64(1).Div(c.Add(d.Mul(r)))
		return Complex{a.Add(b.Mul(r)).Mul(s), b.Sub(a.Mul(r)).Mul(s)}
	}
	r := c.Div(d)
	s := FromInt64(1).Div(d.Add(c.Mul(r)))
	return Complex{a.Mul(r).Add(b).Mul(s), b.Mul(r).Sub(a).Mul(s)}
}
func (z Complex) Conj() Complex { return Complex{z.Real, z.Imag.Neg()} }
func (z Complex) Abs() Float {
	a, b := z.Real.Abs(), z.Imag.Abs()
	if a.IsInf() || b.IsInf() {
		return Inf(false)
	}
	if a.IsNaN() || b.IsNaN() {
		return NaN()
	}
	// Compute the magnitude with wider intermediates, avoiding premature
	// rounding of the ratio, sum, and final product.
	x, y := a.Rat(), b.Rat()
	x.Mul(x, x)
	y.Mul(y, y)
	x.Add(x, y)
	v := new(big.Float).SetPrec(256).SetRat(x)
	v.Sqrt(v)
	r, _ := v.Rat(nil)
	return roundRat(r, false)
}
func (z Complex) Equal(w Complex) bool   { return z.Real.Equal(w.Real) && z.Imag.Equal(w.Imag) }
func (z Complex) Complex128() complex128 { return complex(z.Real.Float64(), z.Imag.Float64()) }
