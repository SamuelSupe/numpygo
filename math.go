package np

import (
	"math"
	"math/cmplx"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

func unary(a *Array, d DType, fn func(element, DType) (element, error)) (*Array, error) {
	if a.DType() == 0 || a.DType().IsSpecial() || d.IsSpecial() {
		return nil, ErrDType
	}
	out, err := Zeros(d, a.shape...)
	if err != nil {
		return nil, err
	}
	var failure error
	a.walk(func(i, off int) {
		if failure == nil {
			v, err := fn(a.elementAt(off), a.DType())
			if err != nil {
				failure = err
				return
			}
			out.setElement(i, v)
		}
	})
	if failure != nil {
		return nil, failure
	}
	return out, nil
}
func mathematical(a *Array, name string, realFn func(float64) float64, complexFn func(complex128) complex128) (*Array, error) {
	if a.DType().IsExtended() {
		return extendedMath(a, name)
	}
	d, err := PromoteTypes(a.DType(), Float16)
	if err != nil {
		return nil, err
	}
	if a.DType().IsComplex() && complexFn == nil {
		return nil, ErrDType
	}
	return unary(a, d, func(e element, from DType) (element, error) {
		z := e.number(from)
		if d.IsComplex() {
			return element{z: complexFn(z)}, nil
		}
		return element{z: complex(realFn(real(z)), 0)}, nil
	})
}
func Sqrt(a *Array) (*Array, error) {
	return mathematical(a, "sqrt", math.Sqrt, func(z complex128) complex128 {
		// NumPy follows the C99 complex square-root NaN rules, including
		// a NaN real component with a finite imaginary component.
		if math.IsNaN(real(z)) && !math.IsInf(imag(z), 0) {
			return complex(math.NaN(), math.NaN())
		}
		return cmplx.Sqrt(z)
	})
}
func Exp(a *Array) (*Array, error) {
	d := a.DType()
	if (d == Float32 || d == Float64) && nativeNumericStorage(a) && a.IsCContiguous() {
		out, e := Zeros(d, a.shape...)
		if e != nil || out.size == 0 {
			return out, e
		}
		extraNativeTo(d, out, []*Array{a}, nil, 7)
		return out, nil
	}
	return mathematical(a, "exp", math.Exp, cmplx.Exp)
}
func Exp2(a *Array) (*Array, error) {
	return mathematical(a, "exp2", math.Exp2, func(z complex128) complex128 { return cmplx.Pow(2, z) })
}
func Expm1(a *Array) (*Array, error) {
	return mathematical(a, "expm1", math.Expm1, func(z complex128) complex128 { return cmplx.Exp(z) - 1 })
}
func Log(a *Array) (*Array, error) { return mathematical(a, "log", math.Log, cmplx.Log) }
func Log2(a *Array) (*Array, error) {
	return mathematical(a, "log2", math.Log2, func(z complex128) complex128 {
		v := cmplx.Log(z)
		return complex(real(v)/math.Ln2, imag(v)/math.Ln2)
	})
}
func Log10(a *Array) (*Array, error) { return mathematical(a, "log10", math.Log10, cmplx.Log10) }
func Log1p(a *Array) (*Array, error) {
	return mathematical(a, "log1p", math.Log1p, func(z complex128) complex128 { return cmplx.Log(1 + z) })
}
func Sin(a *Array) (*Array, error)     { return mathematical(a, "sin", math.Sin, cmplx.Sin) }
func Cos(a *Array) (*Array, error)     { return mathematical(a, "cos", math.Cos, cmplx.Cos) }
func Tan(a *Array) (*Array, error)     { return mathematical(a, "tan", math.Tan, cmplx.Tan) }
func Arcsin(a *Array) (*Array, error)  { return mathematical(a, "arcsin", math.Asin, cmplx.Asin) }
func Arccos(a *Array) (*Array, error)  { return mathematical(a, "arccos", math.Acos, cmplx.Acos) }
func Arctan(a *Array) (*Array, error)  { return mathematical(a, "arctan", math.Atan, cmplx.Atan) }
func Sinh(a *Array) (*Array, error)    { return mathematical(a, "sinh", math.Sinh, cmplx.Sinh) }
func Cosh(a *Array) (*Array, error)    { return mathematical(a, "cosh", math.Cosh, cmplx.Cosh) }
func Tanh(a *Array) (*Array, error)    { return mathematical(a, "tanh", math.Tanh, cmplx.Tanh) }
func Arcsinh(a *Array) (*Array, error) { return mathematical(a, "arcsinh", math.Asinh, cmplx.Asinh) }
func Arccosh(a *Array) (*Array, error) { return mathematical(a, "arccosh", math.Acosh, cmplx.Acosh) }
func Arctanh(a *Array) (*Array, error) {
	return mathematical(a, "arctanh", math.Atanh, func(z complex128) complex128 {
		// Go's formula loses the side of the real branch cut when the
		// imaginary component is -0. NumPy preserves it on both cuts.
		if imag(z) == 0 && math.Abs(real(z)) > 1 {
			return complex(math.Atanh(1/real(z)), math.Copysign(math.Pi/2, imag(z)))
		}
		return cmplx.Atanh(z)
	})
}
func Cbrt(a *Array) (*Array, error) { return mathematical(a, "cbrt", math.Cbrt, nil) }
func Degrees(a *Array) (*Array, error) {
	return mathematical(a, "degrees", func(x float64) float64 { return x * 180 / math.Pi }, nil)
}
func Radians(a *Array) (*Array, error) {
	return mathematical(a, "radians", func(x float64) float64 { return x * math.Pi / 180 }, nil)
}
func Rint(a *Array) (*Array, error) {
	return mathematical(a, "rint", math.RoundToEven, func(z complex128) complex128 { return complex(math.RoundToEven(real(z)), math.RoundToEven(imag(z))) })
}
func rounding(a *Array, name string, fn func(float64) float64) (*Array, error) {
	if a.DType().IsExtended() {
		return extendedMath(a, name)
	}
	if a.DType().IsComplex() {
		return nil, ErrDType
	}
	return unary(a, a.DType(), func(x element, d DType) (element, error) {
		if d.IsFloat() {
			x.z = complex(fn(real(x.z)), 0)
		}
		return x, nil
	})
}
func Floor(a *Array) (*Array, error) { return rounding(a, "floor", math.Floor) }
func Ceil(a *Array) (*Array, error)  { return rounding(a, "ceil", math.Ceil) }
func Trunc(a *Array) (*Array, error) { return rounding(a, "trunc", math.Trunc) }

func Negative(a *Array) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialUnary(a, "negative")
	}
	if a.DType() == Bool {
		return nil, ErrDType
	}
	return unary(a, a.DType(), func(x element, d DType) (element, error) {
		if d.IsInteger() {
			x.i = 0 - x.i
		} else {
			x.z = -x.z
		}
		return x, nil
	})
}
func Positive(a *Array) (*Array, error) {
	if a.DType() == Bool {
		return nil, ErrDType
	}
	return a.Copy()
}
func Square(a *Array) (*Array, error) {
	if a.DType() == Bool {
		var err error
		a, err = a.Astype(Int8)
		if err != nil {
			return nil, err
		}
	}
	return Multiply(a, a)
}
func Reciprocal(a *Array) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialUnary(a, "reciprocal")
	}
	if a.DType() == Bool {
		var err error
		a, err = a.Astype(Int8)
		if err != nil {
			return nil, err
		}
	}
	return unary(a, a.DType(), func(x element, d DType) (element, error) {
		if d.IsInteger() || d == Bool {
			if x.i == 0 && (d == Int32 || d == Int64) {
				return element{i: uint64(1) << (8*d.ItemSize() - 1)}, nil
			}
			v, err := integerOp(1, x.i, d, floorOp)
			// integer reciprocal truncates toward zero.
			if d.IsSigned() && int64(x.i) < -1 {
				v = 0
			}
			return element{i: v}, err
		}
		if d.IsComplex() {
			if x.z == 0 {
				return element{z: complex(math.NaN(), math.NaN())}, nil
			}
			return element{z: complexDivide(1, x.z)}, nil
		}
		return element{z: complex(1/real(x.z), 0)}, nil
	})
}
func Absolute(a *Array) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialUnary(a, "absolute")
	}
	d := realDType(a.DType())
	return unary(a, d, func(x element, from DType) (element, error) {
		if from.IsSigned() {
			if int64(x.i) < 0 {
				x.i = 0 - x.i
			}
			return x, nil
		}
		if from.IsUnsigned() || from == Bool {
			return x, nil
		}
		if from.IsComplex() {
			return element{z: complex(cmplx.Abs(x.z), 0)}, nil
		}
		return element{z: complex(math.Abs(real(x.z)), 0)}, nil
	})
}
func Conjugate(a *Array) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialUnary(a, "conjugate")
	}
	if a.DType() == Bool {
		return a.Astype(Int8)
	}
	return unary(a, a.DType(), func(x element, d DType) (element, error) {
		if d.IsComplex() {
			x.z = cmplx.Conj(x.z)
		}
		return x, nil
	})
}
func Sign(a *Array) (*Array, error) {
	if a.DType().IsExtended() {
		return extendedMath(a, "sign")
	}
	if a.DType() == Bool {
		return nil, ErrDType
	}
	return unary(a, a.DType(), func(x element, d DType) (element, error) {
		if d.IsSigned() {
			switch {
			case int64(x.i) < 0:
				x.i = ^uint64(0)
			case x.i != 0:
				x.i = 1
			}
			return x, nil
		}
		if d.IsUnsigned() {
			if x.i != 0 {
				x.i = 1
			}
			return x, nil
		}
		if nanElement(x, d) {
			if d.IsComplex() {
				return element{z: complex(math.NaN(), math.NaN())}, nil
			}
			return x, nil
		}
		if x.z == 0 {
			return element{}, nil
		}
		if d.IsComplex() {
			return element{z: x.z / complex(cmplx.Abs(x.z), 0)}, nil
		}
		return element{z: complex(math.Copysign(1, real(x.z)), 0)}, nil
	})
}
func LogicalNot(a *Array) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialUnary(a, "logical_not")
	}
	return unary(a, Bool, func(x element, d DType) (element, error) { return boolElement(!x.truth(d)), nil })
}
func Invert(a *Array) (*Array, error) {
	if !a.DType().IsInteger() && a.DType() != Bool {
		return nil, ErrDType
	}
	return unary(a, a.DType(), func(x element, d DType) (element, error) {
		if d == Bool {
			return boolElement(x.i == 0), nil
		}
		return element{i: ^x.i}, nil
	})
}
func IsNaN(a *Array) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialUnary(a, "isnan")
	}
	return unary(a, Bool, func(x element, d DType) (element, error) { return boolElement(nanElement(x, d)), nil })
}
func IsInf(a *Array) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialUnary(a, "isinf")
	}
	return unary(a, Bool, func(x element, d DType) (element, error) {
		return boolElement((d.IsFloat() || d.IsComplex()) && (math.IsInf(real(x.z), 0) || math.IsInf(imag(x.z), 0))), nil
	})
}
func IsFinite(a *Array) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialUnary(a, "isfinite")
	}
	return unary(a, Bool, func(x element, d DType) (element, error) {
		return boolElement(!nanElement(x, d) && !math.IsInf(real(x.z), 0) && !math.IsInf(imag(x.z), 0)), nil
	})
}
func Signbit(a *Array) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialUnary(a, "signbit")
	}
	if a.DType().IsComplex() {
		return nil, ErrDType
	}
	return unary(a, Bool, func(x element, d DType) (element, error) {
		if d.IsSigned() {
			return boolElement(int64(x.i) < 0), nil
		}
		return boolElement(math.Signbit(real(x.z))), nil
	})
}
func realBinaryMath(a, b *Array, name string, fn func(float64, float64) float64) (*Array, error) {
	d, err := realMathDType(a.DType(), b.DType())
	if err != nil || d.IsComplex() {
		return nil, ErrDType
	}
	if d == Float128 {
		return extendedBinary(a, b, name)
	}
	return binaryMap(a, b, d, d, func(x, y element) (element, error) { return element{z: complex(fn(real(x.z), real(y.z)), 0)}, nil })
}

// These ufuncs resolve against real floating loops, not against integer
// addition loops. Promote each operand to its safe float loop first.
func realMathDType(a, b DType) (DType, error) {
	x, e := PromoteTypes(a, Float16)
	if e != nil {
		return 0, e
	}
	y, e := PromoteTypes(b, Float16)
	if e != nil {
		return 0, e
	}
	return PromoteTypes(x, y)
}
func Hypot(a, b *Array) (*Array, error)    { return realBinaryMath(a, b, "hypot", math.Hypot) }
func Arctan2(a, b *Array) (*Array, error)  { return realBinaryMath(a, b, "arctan2", math.Atan2) }
func Copysign(a, b *Array) (*Array, error) { return realBinaryMath(a, b, "copysign", math.Copysign) }
func LogAddExp(a, b *Array) (*Array, error) {
	return realBinaryMath(a, b, "logaddexp", func(x, y float64) float64 {
		if x == y {
			return x + math.Ln2
		}
		m := math.Max(x, y)
		return m + math.Log1p(math.Exp(-math.Abs(x-y)))
	})
}

func Clip(a, lo, hi *Array) (*Array, error) {
	var err error
	if lo != nil {
		a, err = Maximum(a, lo)
		if err != nil {
			return nil, err
		}
	}
	if hi != nil {
		return Minimum(a, hi)
	}
	if lo == nil {
		return nil, ErrValue
	}
	return a, nil
}

func IsClose(a, b *Array, rtol, atol float64, equalNaN bool) (*Array, error) {
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if d.IsInteger() || d == Bool {
		d = Float64
	}
	if d.IsExtended() {
		return binaryMapCore(a, b, d, Bool, func(x, y element) (element, error) {
			if equalElement(x, y, d) {
				return boolElement(true), nil
			}
			if nanElement(x, d) || nanElement(y, d) {
				return boolElement(equalNaN && nanElement(x, d) && nanElement(y, d)), nil
			}
			u, v := extendedValue(x, d), extendedValue(y, d)
			if !u.Real.IsFinite() || !u.Imag.IsFinite() || !v.Real.IsFinite() || !v.Imag.IsFinite() {
				return boolElement(false), nil
			}
			diff := u.Sub(v).Abs()
			tol := x87.FromFloat64(atol).Add(x87.FromFloat64(rtol).Mul(v.Abs()))
			c, ok := diff.Compare(tol)
			return boolElement(ok && c <= 0), nil
		})
	}
	return binaryMap(a, b, d, Bool, func(x, y element) (element, error) {
		if x.z == y.z {
			return element{i: 1}, nil
		}
		if nanElement(x, d) || nanElement(y, d) {
			return boolElement(equalNaN && nanElement(x, d) && nanElement(y, d)), nil
		}
		if math.IsInf(real(x.z), 0) || math.IsInf(imag(x.z), 0) || math.IsInf(real(y.z), 0) || math.IsInf(imag(y.z), 0) {
			return element{}, nil
		}
		return boolElement(cmplx.Abs(x.z-y.z) <= atol+rtol*cmplx.Abs(y.z)), nil
	})
}
func AllClose(a, b *Array, rtol, atol float64, equalNaN bool) (bool, error) {
	v, err := IsClose(a, b, rtol, atol, equalNaN)
	if err != nil {
		return false, err
	}
	r, err := All(v)
	if err != nil {
		return false, err
	}
	x, err := r.AtValue()
	if err != nil {
		return false, err
	}
	return x.(bool), nil
}
