package x87

import "math/big"

// Integral rounding is exact even for values beyond the float64 exponent
// and integer ranges; a rounded zero retains the input sign.
func (x Float) Round(mode string) Float {
	if !x.IsFinite() || x.IsZero() {
		return x
	}
	r := x.Rat()
	q, remainder := new(big.Int), new(big.Int)
	q.QuoRem(r.Num(), r.Denom(), remainder)
	switch mode {
	case "floor":
		if remainder.Sign() < 0 {
			q.Sub(q, big.NewInt(1))
		}
	case "ceil":
		if remainder.Sign() > 0 {
			q.Add(q, big.NewInt(1))
		}
	case "rint":
		remainder.Abs(remainder).Lsh(remainder, 1)
		c := remainder.Cmp(r.Denom())
		if c > 0 || c == 0 && q.Bit(0) == 1 {
			q.Add(q, big.NewInt(int64(r.Sign())))
		}
	}
	return roundRat(new(big.Rat).SetInt(q), x.Signbit())
}
func (x Float) RoundDecimal(decimals int) Float {
	if !x.IsFinite() || x.IsZero() {
		return x
	}
	if decimals == 0 {
		return x.Round("rint")
	}
	// NumPy implements decimal rounding by scaling in the selected dtype.
	// An unrepresentable scaling factor produces NaN, not integer overflow.
	if decimals > 4932 || decimals < -4932 {
		return NaN()
	}
	power := decimals
	if power < 0 {
		power = -power
	}
	factor := roundRat(new(big.Rat).SetInt(new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(power)), nil)), false)
	if decimals > 0 {
		return x.Mul(factor).Round("rint").Div(factor)
	}
	return x.Div(factor).Round("rint").Mul(factor)
}
func (x Float) Frexp() (Float, int) {
	if !x.IsFinite() || x.IsZero() {
		return x, 0
	}
	m := bf(mathPrec)
	e := wide(x).MantExp(m)
	return roundWide(m), e
}
func (x Float) Ldexp(ex int) Float {
	if !x.IsFinite() || x.IsZero() {
		return x
	}
	if ex > 40000 {
		return Inf(x.Signbit())
	}
	if ex < -40000 {
		return Zero(x.Signbit())
	}
	return roundWide(scale(wide(x), ex))
}
func (x Float) Fmod(y Float) Float {
	if x.IsNaN() || y.IsNaN() || x.IsInf() || y.IsZero() {
		return NaN()
	}
	if x.IsZero() || y.IsInf() {
		return x
	}
	// Remainder of exact rational inputs; the quotient may exceed
	// the floating-point range and is intentionally not rounded first.
	a, b := x.Rat(), y.Rat()
	q := new(big.Rat).Quo(a, b)
	n := new(big.Int).Quo(q.Num(), q.Denom())
	return roundRat(new(big.Rat).Sub(a, new(big.Rat).Mul(new(big.Rat).SetInt(n), b)), x.Signbit())
}
func (x Float) Divmod(y Float) (Float, Float) {
	if y.IsZero() {
		return x.Div(y), NaN()
	}
	r := x.Fmod(y)
	if r.IsNaN() {
		return NaN(), r
	}
	// Match NumPy's floating divmod: recover the quotient from fmod
	// and correct its sign before rounding the nearly integral result.
	quot := x.Sub(r).Div(y)
	if !r.IsZero() {
		if r.Signbit() != y.Signbit() {
			r = r.Add(y)
			quot = quot.Sub(FromInt64(1))
		}
	} else {
		r = Zero(y.Signbit())
	}
	if !quot.IsZero() {
		quot = quot.Round("rint")
	} else {
		quot = Zero(x.Signbit() != y.Signbit())
	}
	return quot, r
}
func (x Float) Pow(y Float) Float {
	one := FromInt64(1)
	if y.IsZero() || x.Equal(one) {
		return one
	}
	if x.IsNaN() || y.IsNaN() {
		return NaN()
	}
	negative := false
	if y.IsFinite() {
		r := y.Rat()
		integer := r.IsInt()
		negative = x.Signbit() && integer && r.Num().Bit(0) == 1
		if x.Signbit() && !x.IsZero() && !x.IsInf() && !integer {
			return NaN()
		}
	}
	applySign := func(v Float) Float {
		if negative {
			return v.Neg()
		}
		return v
	}
	if y.IsInf() {
		c, _ := x.Abs().Compare(one)
		if c == 0 {
			return one
		}
		if (c > 0) != y.Signbit() {
			return Inf(false)
		}
		return Zero(false)
	}
	if x.IsZero() || x.IsInf() {
		if x.IsInf() != y.Signbit() {
			return applySign(Inf(false))
		}
		return applySign(Zero(false))
	}
	// Preserve small integral powers without evaluating logarithms.
	r := y.Rat()
	if r.IsInt() && r.Num().IsInt64() {
		k := r.Num().Int64()
		if k >= -64 && k <= 64 {
			n := k
			if n < 0 {
				n = -n
			}
			base := x.Rat()
			v := new(big.Rat).SetInt64(1)
			for n > 0 {
				if n&1 != 0 {
					v.Mul(v, base)
				}
				n >>= 1
				if n > 0 {
					base.Mul(base, base)
				}
			}
			if k < 0 {
				v.Inv(v)
			}
			return roundRat(v, false)
		}
	}
	return applySign(roundWide(expWide(mul(wide(y), logWide(wide(x.Abs()))))))
}
func LogAddExp(x, y Float, base2 bool) Float {
	if x.IsNaN() || y.IsNaN() {
		return NaN()
	}
	if c, _ := x.Compare(y); c < 0 {
		x, y = y, x
	}
	if x.IsInf() {
		return x
	}
	if y.IsInf() {
		return x
	}
	a, b := wide(x), wide(y)
	delta := sub(b, a)
	if base2 {
		return roundWide(add(a, quo(log1pWide(expWide(mul(delta, ln2Wide))), ln2Wide)))
	}
	return roundWide(add(a, log1pWide(expWide(delta))))
}
