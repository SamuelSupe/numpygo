package x87

import "math/big"

func nanComplex() Complex      { return Complex{NaN(), NaN()} }
func (z Complex) finite() bool { return z.Real.IsFinite() && z.Imag.IsFinite() }
func (z Complex) zero() bool   { return z.Real.IsZero() && z.Imag.IsZero() }
func (z Complex) Neg() Complex { return Complex{z.Real.Neg(), z.Imag.Neg()} }
func (z Complex) Sign() Complex {
	if z.Real.IsNaN() || z.Imag.IsNaN() {
		return nanComplex()
	}
	if z.Real.IsInf() && z.Imag.IsInf() {
		return nanComplex()
	}
	if z.Real.IsInf() {
		return Complex{Real: FromInt64(signInt(z.Real))}
	}
	if z.Imag.IsInf() {
		return Complex{Imag: FromInt64(signInt(z.Imag))}
	}
	if z.zero() {
		return Complex{}
	}
	a, b := wide(z.Real), wide(z.Imag)
	r := sqrt(add(mul(a, a), mul(b, b)))
	return Complex{roundWide(quo(a, r)), roundWide(quo(b, r))}
}
func (z Complex) Sqrt() Complex {
	x, y := z.Real, z.Imag
	switch {
	case y.IsInf():
		return Complex{Inf(false), y}
	case x.IsInf():
		if x.Signbit() {
			return Complex{Zero(false), Inf(y.Signbit())}
		}
		if y.IsNaN() {
			return Complex{x, NaN()}
		}
		return Complex{x, Zero(y.Signbit())}
	case !z.finite():
		return nanComplex()
	case z.zero():
		return Complex{Zero(false), y}
	}
	a, b := wide(x), wide(y)
	r := sqrt(add(mul(a, a), mul(b, b)))
	t := sqrt(scale(add(r, abs(a)), -1))
	if x.Signbit() {
		v := roundWide(quo(abs(b), scale(t, 1)))
		w := roundWide(t)
		if y.Signbit() {
			w = w.Neg()
		}
		return Complex{v, w}
	}
	return Complex{roundWide(t), roundWide(quo(b, scale(t, 1)))}
}
func atan2Wide(y, x *big.Float) *big.Float {
	p := x.Prec()
	if y.Sign() == 0 {
		if !x.Signbit() {
			return bf(p).Set(y)
		}
		a := piAt(p)
		if y.Signbit() {
			a = neg(a)
		}
		return a
	}
	if x.Sign() == 0 {
		a := scale(piAt(p), -1)
		if y.Signbit() {
			a = neg(a)
		}
		return a
	}
	a := atanWide(quo(abs(y), abs(x)))
	if x.Signbit() {
		a = sub(piAt(p), a)
	}
	if y.Signbit() {
		a = neg(a)
	}
	return a
}

// log of the magnitude, using log1p near the unit circle to retain tiny
// imaginary components and near-one real components.
func logHypot(a, b *big.Float) *big.Float {
	one := bi(a.Prec(), 1)
	x, y := abs(a), abs(b)
	if y.Cmp(x) > 0 {
		x, y = y, x
	}
	d := add(mul(sub(x, one), add(x, one)), mul(y, y))
	if abs(d).Cmp(scale(one, -1)) <= 0 {
		return scale(log1pWide(d), -1)
	}
	return scale(logWide(add(mul(a, a), mul(b, b))), -1)
}
func (z Complex) Log() Complex {
	if !z.finite() {
		r := NaN()
		if z.Real.IsInf() || z.Imag.IsInf() {
			r = Inf(false)
		}
		return Complex{r, Atan2(z.Imag, z.Real)}
	}
	if z.zero() {
		return Complex{Inf(true), Atan2(z.Imag, z.Real)}
	}
	return Complex{roundWide(logHypot(wide(z.Real), wide(z.Imag))), Atan2(z.Imag, z.Real)}
}
func (z Complex) Log2() Complex {
	l := z.Log()
	// Use wide magnitude and angle for finite inputs, rounding just once.
	if z.finite() && !z.zero() {
		return Complex{roundWide(quo(logHypot(wide(z.Real), wide(z.Imag)), ln2Wide)), roundWide(quo(atan2Wide(wide(z.Imag), wide(z.Real)), ln2Wide))}
	}
	c := roundWide(ln2Wide)
	return Complex{l.Real.Div(c), l.Imag.Div(c)}
}
func (z Complex) Log10() Complex {
	l := z.Log()
	if z.finite() && !z.zero() {
		return Complex{roundWide(quo(logHypot(wide(z.Real), wide(z.Imag)), ln10Wide)), roundWide(quo(atan2Wide(wide(z.Imag), wide(z.Real)), ln10Wide))}
	}
	c := roundWide(ln10Wide)
	return Complex{l.Real.Div(c), l.Imag.Div(c)}
}
func (z Complex) Log1p() Complex {
	// NumPy's complex log1p rounds 1+x and hypot in the input
	// precision before taking the logarithm. Unlike real log1p,
	// tiny real components can intentionally round away.
	a := z.Real.Add(FromInt64(1))
	return Complex{Real: (Complex{a, z.Imag}).Abs().Log(), Imag: Atan2(z.Imag, a)}
}
func product(a, b *big.Float) *big.Float {
	if a.IsInf() && b.Sign() == 0 || b.IsInf() && a.Sign() == 0 {
		z := bf(a.Prec())
		if a.Signbit() != b.Signbit() {
			z.Neg(z)
		}
		return z
	}
	return mul(a, b)
}
func expComplexWide(x, y *big.Float) Complex {
	e := expWide(x)
	s, c := sinCosWide(y)
	return Complex{roundWide(product(e, c)), roundWide(product(e, s))}
}
func (z Complex) Exp() Complex {
	x, y := z.Real, z.Imag
	if !z.finite() {
		if y.IsZero() {
			return Complex{x.Exp(), y}
		}
		if x.IsInf() && x.Signbit() && !y.IsFinite() {
			return Complex{Zero(false), Zero(y.Signbit())}
		}
		if x.IsInf() && !x.Signbit() && !y.IsFinite() {
			return Complex{Inf(false), NaN()}
		}
		if !y.IsFinite() || x.IsNaN() {
			return nanComplex()
		}
		e := wide(x.Exp())
		s, c := sinCosWide(wide(y))
		return Complex{roundWide(product(e, c)), roundWide(product(e, s))}
	}
	if y.IsZero() {
		return Complex{x.Exp(), y}
	}
	return expComplexWide(wide(x), wide(y))
}
func (z Complex) Exp2() Complex {
	ln2 := roundWide(ln2Wide)
	return (Complex{z.Real.Mul(ln2), z.Imag.Mul(ln2)}).Exp()
}
func (z Complex) Expm1() Complex {
	if !z.finite() {
		v := z.Exp()
		v.Real = v.Real.Sub(FromInt64(1))
		return v
	}
	x, y := wide(z.Real), wide(z.Imag)
	if z.Imag.IsZero() {
		return Complex{z.Real.Expm1(), z.Imag}
	}
	s, c := sinCosWide(y)
	h, _ := sinCosWide(scale(y, -1))
	re := sub(product(expm1Wide(x), c), scale(mul(h, h), 1))
	return Complex{roundWide(re), roundWide(product(expWide(x), s))}
}

func (z Complex) Sin() Complex {
	x, y := z.Real, z.Imag
	if !x.IsFinite() {
		if y.IsZero() {
			return Complex{NaN(), y}
		}
		return nanComplex()
	}
	if y.IsNaN() {
		if x.IsZero() {
			return Complex{x, NaN()}
		}
		return nanComplex()
	}
	s, c := sinCosWide(wide(x))
	var h, k *big.Float
	if y.IsInf() {
		h = bf(mathPrec).SetInf(y.Signbit())
		k = bf(mathPrec).SetInf(false)
	} else {
		h, k = sinhCoshWide(wide(y))
	}
	return Complex{roundWide(product(s, k)), roundWide(product(c, h))}
}
func (z Complex) Cos() Complex {
	x, y := z.Real, z.Imag
	if !x.IsFinite() {
		if y.IsZero() {
			return Complex{NaN(), Zero(false)}
		}
		return nanComplex()
	}
	if y.IsNaN() {
		if x.IsZero() {
			return Complex{NaN(), Zero(false)}
		}
		return nanComplex()
	}
	s, c := sinCosWide(wide(x))
	var h, k *big.Float
	if y.IsInf() {
		h = bf(mathPrec).SetInf(y.Signbit())
		k = bf(mathPrec).SetInf(false)
	} else {
		h, k = sinhCoshWide(wide(y))
	}
	return Complex{roundWide(product(c, k)), roundWide(neg(product(s, h)))}
}
func (z Complex) Tan() Complex {
	if z.Imag.IsInf() {
		return Complex{Zero(z.Real.Signbit()), FromInt64(1).Mul(FromInt64(signInt(z.Imag)))}
	}
	if z.Imag.IsZero() {
		return Complex{z.Real.Tan(), z.Imag}
	}
	if !z.finite() {
		return nanComplex()
	}
	s, c := sinCosWide(wide(z.Real))
	y := wide(z.Imag.Abs())
	m := expm1Wide(neg(scale(y, 1)))
	t := add(bi(mathPrec, 1), m)
	// For large |y|, expm1 rounds to -1 at the working precision.
	// Evaluate t directly to preserve the tiny but representable real part.
	if t.Sign() == 0 {
		t = expWide(neg(scale(y, 1)))
	}
	den := add(mul(m, m), scale(mul(t, mul(c, c)), 2))
	re := quo(scale(mul(t, mul(s, c)), 2), den)
	im := quo(neg(mul(m, add(bi(mathPrec, 2), m))), den)
	if z.Imag.Signbit() {
		im = neg(im)
	}
	return Complex{roundWide(re), roundWide(im)}
}
func signInt(x Float) int64 {
	if x.Signbit() {
		return -1
	}
	return 1
}
func timesI(z Complex) Complex  { return Complex{z.Imag.Neg(), z.Real} }
func minusI(z Complex) Complex  { return Complex{z.Imag, z.Real.Neg()} }
func (z Complex) Sinh() Complex { return minusI(timesI(z).Sin()) }
func (z Complex) Cosh() Complex { return timesI(z).Cos() }
func (z Complex) Tanh() Complex { return minusI(timesI(z).Tan()) }

// For asin/acos, A=(|z+1|+|z-1|)/2. Compute A-1 and A-|Re z|
// by rationalization so that close-to-real inputs do not lose their
// small imaginary parts or their distance from a branch point.
func (z Complex) asinAcos(acos bool) Complex {
	x, y := z.Real, z.Imag
	if !z.finite() {
		if x.IsInf() || y.IsInf() {
			r := Zero(x.Signbit())
			switch {
			case x.IsNaN() || y.IsNaN():
				r = NaN()
			case x.IsInf() && y.IsInf():
				r = roundWide(scale(piWide, -2))
				if x.Signbit() {
					r = r.Neg()
				}
			case x.IsInf():
				r = roundWide(scale(piWide, -1))
				if x.Signbit() {
					r = r.Neg()
				}
			}
			i := Inf(y.Signbit())
			if acos {
				return Complex{roundWide(scale(piWide, -1)).Sub(r), i.Neg()}
			}
			return Complex{r, i}
		}
		if y.IsNaN() && x.IsZero() {
			if acos {
				return Complex{roundWide(scale(piWide, -1)), NaN()}
			}
			return Complex{x, NaN()}
		}
		return nanComplex()
	}
	a, b := abs(wide(x)), abs(wide(y))
	p := mathPrec
	one := bi(p, 1)
	b2 := mul(b, b)
	aplus, aminus := add(a, one), abs(sub(a, one))
	r, s := sqrt(add(mul(aplus, aplus), b2)), sqrt(add(mul(aminus, aminus), b2))
	correction := bf(p)
	if b.Sign() != 0 {
		correction = scale(add(quo(b2, add(r, aplus)), quo(b2, add(s, aminus))), -1)
	}
	var am1, amx *big.Float
	if a.Cmp(one) >= 0 {
		am1 = add(sub(a, one), correction)
		amx = correction
	} else {
		am1 = correction
		amx = add(sub(one, a), correction)
	}
	d := sqrt(mul(amx, add(add(am1, one), a)))
	angle := atan2Wide(a, d)
	if acos {
		angle = atan2Wide(d, a)
		if x.Signbit() {
			angle = sub(piAt(p), angle)
		}
	} else if x.Signbit() {
		angle = neg(angle)
	}
	im := log1pWide(add(am1, sqrt(mul(am1, add(am1, bi(p, 2))))))
	if y.Signbit() != acos {
		im = neg(im)
	}
	return Complex{roundWide(angle), roundWide(im)}
}
func (z Complex) Asin() Complex  { return z.asinAcos(false) }
func (z Complex) Acos() Complex  { return z.asinAcos(true) }
func (z Complex) Asinh() Complex { return minusI(timesI(z).Asin()) }
func (z Complex) Acosh() Complex {
	v := z.Acos()
	i := v.Real
	if z.Imag.Signbit() {
		i = i.Neg()
	}
	return Complex{v.Imag.Abs(), i}
}
func (z Complex) Atan() Complex {
	x, y := z.Real, z.Imag
	if x.IsInf() || y.IsInf() {
		r := roundWide(scale(piWide, -1))
		if x.Signbit() {
			r = r.Neg()
		}
		if x.IsNaN() {
			r = NaN()
		}
		return Complex{r, Zero(y.Signbit())}
	}
	if !z.finite() {
		if y.IsZero() {
			return Complex{NaN(), y}
		}
		return nanComplex()
	}
	a, b := wide(x), wide(y)
	one := bi(mathPrec, 1)
	// (1-y)*(1+y)-x*x avoids cancellation close to y=+/-1.
	den := sub(mul(sub(one, b), add(one, b)), mul(a, a))
	re := scale(atan2Wide(scale(a, 1), den), -1)
	t := sub(abs(b), one)
	m := add(mul(a, a), mul(t, t))
	if m.Sign() == 0 {
		return Complex{roundWide(re), Inf(y.Signbit())}
	}
	im := scale(log1pWide(quo(scale(abs(b), 2), m)), -2)
	if y.Signbit() {
		im = neg(im)
	}
	return Complex{roundWide(re), roundWide(im)}
}
func (z Complex) Atanh() Complex { return minusI(timesI(z).Atan()) }
func (z Complex) Pow(w Complex) Complex {
	one := Complex{Real: FromInt64(1)}
	if w.zero() {
		return one
	}
	if z.zero() {
		if !w.Imag.IsZero() || w.Real.Signbit() || w.Real.IsNaN() {
			return nanComplex()
		}
		return Complex{}
	}
	if z.finite() && w.finite() && w.Imag.IsZero() {
		r := w.Real.Rat()
		if r.IsInt() && r.Num().IsInt64() {
			k := r.Num().Int64()
			if k >= -32 && k <= 32 {
				n := k
				if n < 0 {
					n = -n
				}
				v, base := one, z
				first := true
				for n > 0 {
					if n&1 != 0 {
						// An initial multiplication by 1+0i can erase
						// the signed zero produced by a negative real base.
						if first {
							v = base
							first = false
						} else {
							v = v.Mul(base)
						}
					}
					n >>= 1
					if n > 0 {
						base = base.Mul(base)
					}
				}
				if k < 0 {
					v = one.Div(v)
				}
				return v
			}
		}
	}
	return w.Mul(z.Log()).Exp()
}
