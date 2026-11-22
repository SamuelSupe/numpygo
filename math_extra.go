package np

import (
	"math"
	"math/bits"
	"math/cmplx"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

func Angle(a *Array, degrees bool) (*Array, error) {
	if a.DType().IsExtended() {
		return mapSpecialUnary(a, Float128, func(e element) (element, error) {
			z := extendedValue(e, a.DType())
			v := x87.Atan2(z.Imag, z.Real)
			if degrees {
				pi := float64(math.Pi)
				v = v.Mul(x87.FromFloat64(180 / pi))
			}
			return element{extra: v}, nil
		})
	}
	d := realDType(a.DType())
	if !d.IsFloat() {
		if a.DType() == Bool {
			d = Float64
		} else {
			var err error
			d, err = PromoteTypes(d, Float16)
			if err != nil {
				return nil, err
			}
		}
	}
	return unary(a, d, func(e element, from DType) (element, error) {
		z := e.number(from)
		v := math.Atan2(imag(z), real(z))
		if degrees {
			v *= 180 / math.Pi
		}
		return element{z: complex(v, 0)}, nil
	})
}
func Round(a *Array, decimals int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.DType() == Bool {
		return a.Astype(Float16)
	}
	if a.DType().IsExtended() {
		d := a.DType()
		return mapSpecialUnary(a, d, func(e element) (element, error) {
			z := extendedValue(e, d)
			z.Real = z.Real.RoundDecimal(decimals)
			z.Imag = z.Imag.RoundDecimal(decimals)
			return extendedElement(z, d), nil
		})
	}
	if a.DType().IsInteger() {
		if decimals >= 0 {
			return a.Copy()
		}
		return unary(a, a.DType(), func(e element, d DType) (element, error) {
			if decimals < -19 {
				return element{}, nil
			}
			factor := uint64(1)
			for j := 0; j < -decimals; j++ {
				factor *= 10
			}
			v := e.i
			negative := d.IsSigned() && int64(v) < 0
			if negative {
				v = 0 - v
			}
			q, r := v/factor, v%factor
			if r > factor/2 || r == factor/2 && q%2 != 0 {
				q++
			}
			v = q * factor
			if negative {
				v = 0 - v
			}
			return element{i: v}, nil
		})
	}
	factor := math.Pow10(decimals)
	round := func(v float64) float64 {
		if decimals == 0 {
			return math.RoundToEven(v)
		}
		if decimals > 0 {
			return math.RoundToEven(v*factor) / factor
		}
		f := math.Pow10(-decimals)
		return math.RoundToEven(v/f) * f
	}
	return unary(a, a.DType(), func(e element, d DType) (element, error) {
		return element{z: complex(round(real(e.z)), round(imag(e.z)))}, nil
	})
}
func Fabs(a *Array) (*Array, error) { return mathematical(a, "fabs", math.Abs, nil) }
func Fix(a *Array) (*Array, error)  { return Trunc(a) }
func FloatPower(a, b *Array) (*Array, error) {
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if d.IsExtended() {
		return Power(a, b)
	}
	if d.IsComplex() {
		d = Complex128
	} else {
		d = Float64
	}
	return binaryMap(a, b, d, d, func(x, y element) (element, error) {
		if d.IsComplex() {
			return element{z: complexPower(x.z, y.z)}, nil
		}
		return element{z: complex(math.Pow(real(x.z), real(y.z)), 0)}, nil
	})
}
func Fmod(a, b *Array) (*Array, error) {
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if d.IsComplex() {
		return nil, ErrDType
	}
	if d == Float128 {
		return extendedBinary(a, b, "fmod")
	}
	if d == Bool {
		d = Int8
	}
	return binaryMap(a, b, d, d, func(x, y element) (element, error) {
		if d.IsInteger() {
			if y.i == 0 {
				return element{}, nil
			}
			if d.IsSigned() {
				return element{i: uint64(int64(x.i) % int64(y.i))}, nil
			}
			return element{i: x.i % y.i}, nil
		}
		return element{z: complex(math.Mod(real(x.z), real(y.z)), 0)}, nil
	})
}
func Divmod(a, b *Array) (*Array, *Array, error) {
	q, err := FloorDivide(a, b)
	if err != nil {
		return nil, nil, err
	}
	r, err := Remainder(a, b)
	return q, r, err
}
func Frexp(a *Array) (*Array, *Array, error) {
	if a.DType() == 0 || a.DType().IsComplex() {
		return nil, nil, ErrDType
	}
	d, err := PromoteTypes(a.DType(), Float16)
	if err != nil {
		return nil, nil, err
	}
	m, err := Zeros(d, a.shape...)
	if err != nil {
		return nil, nil, err
	}
	ex, err := Zeros(Int32, a.shape...)
	if err != nil {
		return nil, nil, err
	}
	a.walk(func(i, off int) {
		if d == Float128 {
			v, e := extendedValue(a.elementAt(off), d).Real.Frexp()
			m.setElement(i, element{extra: v})
			ex.setElement(i, element{i: uint64(int64(e))})
			return
		}
		v, e := math.Frexp(real(a.elementAt(off).number(a.DType())))
		m.setElement(i, element{z: complex(v, 0)})
		ex.setElement(i, element{i: uint64(int64(e))})
	})
	return m, ex, nil
}
func Ldexp(a, exponents *Array) (*Array, error) {
	if a.DType() == 0 || a.DType().IsComplex() || (!exponents.DType().IsInteger() && exponents.DType() != Bool) || exponents.DType() == Uint64 {
		return nil, ErrDType
	}
	d, err := PromoteTypes(a.DType(), Float16)
	if err != nil {
		return nil, err
	}
	mapper := binaryMap
	if d == Float128 {
		mapper = binaryMapCore
	}
	return mapper(a, exponents, 0, d, func(x, y element) (element, error) {
		var ex int
		if exponents.DType().IsSigned() {
			v := int64(y.i)
			ex = int(max(int64(-1_000_000), min(int64(1_000_000), v)))
		} else {
			ex = int(min(y.i, 1_000_000))
		}
		if d == Float128 {
			return element{extra: extendedValue(x, d).Real.Ldexp(ex)}, nil
		}
		return element{z: complex(math.Ldexp(real(x.number(a.DType())), ex), 0)}, nil
	})
}
func absInteger(x uint64, d DType) uint64 {
	if d.IsSigned() && int64(x) < 0 {
		return 0 - x
	}
	return x
}
func integerGCD(x, y uint64) uint64 {
	for y != 0 {
		x, y = y, x%y
	}
	return x
}
func GCD(a, b *Array) (*Array, error) { return gcdOrLcm(a, b, false) }
func LCM(a, b *Array) (*Array, error) { return gcdOrLcm(a, b, true) }
func gcdOrLcm(a, b *Array, lcm bool) (*Array, error) {
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if !d.IsInteger() {
		return nil, ErrDType
	}
	return binaryMap(a, b, d, d, func(x, y element) (element, error) {
		xi, yi := absInteger(x.i, d), absInteger(y.i, d)
		g := integerGCD(xi, yi)
		if lcm {
			if g != 0 {
				g = (xi / g) * yi
			}
		}
		return element{i: g}, nil
	})
}
func Heaviside(a, b *Array) (*Array, error) {
	return realBinaryMath(a, b, "heaviside", func(x, y float64) float64 {
		if math.IsNaN(x) {
			return x
		}
		if x < 0 {
			return 0
		}
		if x > 0 {
			return 1
		}
		return y
	})
}
func I0(a *Array) (*Array, error) {
	if a.DType() == 0 || a.DType().IsComplex() {
		return nil, ErrDType
	}
	if a.DType() == Float128 {
		return mapSpecialUnary(a, Float128, func(e element) (element, error) { return element{extra: extendedValue(e, Float128).Real.I0()}, nil })
	}
	d := a.DType()
	if !d.IsFloat() {
		d = Float64
	}
	return unary(a, d, func(e element, from DType) (element, error) {
		return element{z: complex(besselI0(real(e.number(from))), 0)}, nil
	})
}
func besselI0(x float64) float64 {
	x = math.Abs(x)
	if math.IsNaN(x) {
		return x
	}
	if x > 1000 {
		return math.Inf(1)
	}
	sum, term := 1., 1.
	for k := 1; ; k++ {
		y := x / (2 * float64(k))
		term *= y * y
		sum += term
		if math.IsInf(sum, 1) || term <= sum*2.220446049250313e-16 {
			return sum
		}
	}
}
func Sinc(a *Array) (*Array, error) {
	return mathematical(a, "sinc", func(x float64) float64 {
		if x == 0 {
			return 1
		}
		z := math.Pi * x
		return math.Sin(z) / z
	}, func(x complex128) complex128 {
		if x == 0 {
			return 1
		}
		z := complex(math.Pi, 0) * x
		return cmplx.Sin(z) / z
	})
}
func LogAddExp2(a, b *Array) (*Array, error) {
	return realBinaryMath(a, b, "logaddexp2", func(x, y float64) float64 {
		if x == y {
			return x + 1
		}
		m := math.Max(x, y)
		return m + math.Log1p(math.Exp2(-math.Abs(x-y)))/math.Ln2
	})
}
func Modf(a *Array) (*Array, *Array, error) {
	if a.DType() == 0 || a.DType().IsComplex() {
		return nil, nil, ErrDType
	}
	d, err := PromoteTypes(a.DType(), Float16)
	if err != nil {
		return nil, nil, err
	}
	frac, err := Zeros(d, a.shape...)
	if err != nil {
		return nil, nil, err
	}
	integ, err := Zeros(d, a.shape...)
	if err != nil {
		return nil, nil, err
	}
	a.walk(func(i, off int) {
		if d == Float128 {
			x := extendedValue(a.elementAt(off), d).Real
			ip := x.Round("trunc")
			fp := x.Sub(ip)
			if x.IsInf() || fp.IsZero() {
				fp = x87.Zero(x.Signbit())
			}
			frac.setElement(i, element{extra: fp})
			integ.setElement(i, element{extra: ip})
			return
		}
		x := real(a.elementAt(off).number(a.DType()))
		ip, fp := math.Modf(x)
		if math.IsInf(x, 0) {
			fp = math.Copysign(0, x)
		}
		frac.setElement(i, element{z: complex(fp, 0)})
		integ.setElement(i, element{z: complex(ip, 0)})
	})
	return frac, integ, nil
}
func nextFloat(x, y float64, d DType) float64 {
	switch d {
	case Float16:
		h, k := HalfFromFloat64(x), HalfFromFloat64(y)
		x, y = h.Float64(), k.Float64()
		if math.IsNaN(x) || math.IsNaN(y) {
			return math.NaN()
		}
		if x == y {
			return y
		}
		if x == 0 {
			if math.Signbit(y) {
				return Half(0x8001).Float64()
			}
			return Half(1).Float64()
		}
		if (x > 0) == (y > x) {
			h++
		} else {
			h--
		}
		return h.Float64()
	case Float32:
		if float32(x) == float32(y) {
			return float64(float32(y))
		}
		return float64(math.Nextafter32(float32(x), float32(y)))
	default:
		if x == y {
			return y
		}
		return math.Nextafter(x, y)
	}
}
func Nextafter(a, b *Array) (*Array, error) {
	d, err := realMathDType(a.DType(), b.DType())
	if err != nil || d.IsComplex() {
		return nil, ErrDType
	}
	if d == Float128 {
		return binaryMapCore(a, b, d, d, func(x, y element) (element, error) {
			return element{extra: extendedValue(x, d).Real.Nextafter(extendedValue(y, d).Real)}, nil
		})
	}
	return binaryMap(a, b, d, d, func(x, y element) (element, error) {
		return element{z: complex(nextFloat(real(x.z), real(y.z), d), 0)}, nil
	})
}
func Spacing(a *Array) (*Array, error) {
	if a.DType().IsExtended() {
		return extendedMath(a, "spacing")
	}
	if a.DType() == 0 || a.DType().IsComplex() {
		return nil, ErrDType
	}
	d, err := PromoteTypes(a.DType(), Float16)
	if err != nil {
		return nil, err
	}
	return unary(a, d, func(e element, from DType) (element, error) {
		x := real(e.number(from))
		if math.IsInf(x, 0) {
			return element{z: complex(math.NaN(), 0)}, nil
		}
		toward := math.Inf(1)
		if x < 0 && d != Float16 {
			toward = math.Inf(-1)
		}
		return element{z: complex(nextFloat(x, toward, d)-x, 0)}, nil
	})
}

type NanToNumOptions struct{ NaN, PosInf, NegInf *float64 }

func NanToNum(a *Array, options ...NanToNumOptions) (*Array, error) {
	if len(options) > 1 {
		return nil, ErrValue
	}
	if a.DType() == 0 {
		return nil, ErrDType
	}
	d := a.DType()
	if !d.IsFloat() && !d.IsComplex() {
		return a.Copy()
	}
	if d.IsExtended() {
		n, pos, neg := x87.Zero(false), x87.Info().Max, x87.Info().Max.Neg()
		if len(options) == 1 {
			opt := options[0]
			if opt.NaN != nil {
				n = x87.FromFloat64(*opt.NaN)
			}
			if opt.PosInf != nil {
				pos = x87.FromFloat64(*opt.PosInf)
			}
			if opt.NegInf != nil {
				neg = x87.FromFloat64(*opt.NegInf)
			}
		}
		f := func(x x87.Float) x87.Float {
			if x.IsNaN() {
				return n
			}
			if x.IsInf() {
				if x.Signbit() {
					return neg
				}
				return pos
			}
			return x
		}
		return mapSpecialUnary(a, d, func(e element) (element, error) {
			z := extendedValue(e, d)
			return extendedElement(x87.Complex{Real: f(z.Real), Imag: f(z.Imag)}, d), nil
		})
	}
	maximum := math.MaxFloat64
	if d == Float32 || d == Complex64 {
		maximum = math.MaxFloat32
	}
	if d == Float16 {
		maximum = 65504
	}
	n, pos, neg := 0., maximum, -maximum
	if len(options) == 1 {
		opt := options[0]
		if opt.NaN != nil {
			n = *opt.NaN
		}
		if opt.PosInf != nil {
			pos = *opt.PosInf
		}
		if opt.NegInf != nil {
			neg = *opt.NegInf
		}
	}
	f := func(x float64) float64 {
		switch {
		case math.IsNaN(x):
			return n
		case math.IsInf(x, 1):
			return pos
		case math.IsInf(x, -1):
			return neg
		}
		return x
	}
	return unary(a, d, func(e element, _ DType) (element, error) { return element{z: complex(f(real(e.z)), f(imag(e.z)))}, nil })
}
func IsPosInf(a *Array) (*Array, error) { return isInfSign(a, 1) }
func IsNegInf(a *Array) (*Array, error) { return isInfSign(a, -1) }
func isInfSign(a *Array, sign int) (*Array, error) {
	if a.DType().IsComplex() {
		return nil, ErrDType
	}
	if a.DType() == Float128 {
		return mapSpecialUnary(a, Bool, func(e element) (element, error) {
			x := extendedValue(e, Float128).Real
			return boolElement(x.IsInf() && x.Signbit() == (sign < 0)), nil
		})
	}
	return unary(a, Bool, func(e element, d DType) (element, error) {
		return boolElement(d.IsFloat() && math.IsInf(real(e.z), sign)), nil
	})
}
func IsComplex(a *Array) (*Array, error) {
	if a.DType().IsExtended() {
		return mapSpecialUnary(a, Bool, func(e element) (element, error) { return boolElement(!extendedValue(e, a.DType()).Imag.IsZero()), nil })
	}
	return unary(a, Bool, func(e element, d DType) (element, error) { return boolElement(d.IsComplex() && imag(e.z) != 0), nil })
}
func IsReal(a *Array) (*Array, error) {
	if a.DType().IsExtended() {
		return mapSpecialUnary(a, Bool, func(e element) (element, error) { return boolElement(extendedValue(e, a.DType()).Imag.IsZero()), nil })
	}
	return unary(a, Bool, func(e element, d DType) (element, error) { return boolElement(!d.IsComplex() || imag(e.z) == 0), nil })
}
func IsComplexObj(a *Array) bool { return a.DType().IsComplex() }
func IsRealObj(a *Array) bool    { return a.DType() != 0 && !a.DType().IsComplex() }
func BitwiseCount(a *Array) (*Array, error) {
	if !a.DType().IsInteger() && a.DType() != Bool {
		return nil, ErrDType
	}
	return unary(a, Uint8, func(e element, d DType) (element, error) {
		return element{i: uint64(bits.OnesCount64(absInteger(e.i, d)))}, nil
	})
}
func LeftShift(a, b *Array) (*Array, error)  { return shiftBits(a, b, false) }
func RightShift(a, b *Array) (*Array, error) { return shiftBits(a, b, true) }
func shiftBits(a, b *Array, right bool) (*Array, error) {
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if d == Bool {
		d = Int8
	}
	if !d.IsInteger() {
		return nil, ErrDType
	}
	return binaryMap(a, b, d, d, func(x, y element) (element, error) {
		n := y.i
		if n >= uint64(d.ItemSize()*8) {
			if right && d.IsSigned() && int64(x.i) < 0 {
				return element{i: ^uint64(0)}, nil
			}
			return element{}, nil
		}
		if !right {
			return element{i: x.i << n}, nil
		}
		if d.IsSigned() {
			return element{i: uint64(int64(x.i) >> n)}, nil
		}
		return element{i: x.i >> n}, nil
	})
}
