package np

import (
	"github.com/SamuelSupe/numpygo/internal/x87"
	"math"
)

var extendedRealUnary = map[string]func(x87.Float) x87.Float{
	"sqrt": x87.Float.Sqrt, "exp": x87.Float.Exp, "exp2": x87.Float.Exp2, "expm1": x87.Float.Expm1,
	"log": x87.Float.Log, "log2": x87.Float.Log2, "log10": x87.Float.Log10, "log1p": x87.Float.Log1p,
	"sin": x87.Float.Sin, "cos": x87.Float.Cos, "tan": x87.Float.Tan,
	"arcsin": x87.Float.Asin, "arccos": x87.Float.Acos, "arctan": x87.Float.Atan,
	"sinh": x87.Float.Sinh, "cosh": x87.Float.Cosh, "tanh": x87.Float.Tanh,
	"arcsinh": x87.Float.Asinh, "arccosh": x87.Float.Acosh, "arctanh": x87.Float.Atanh,
	"cbrt": x87.Float.Cbrt, "degrees": x87.Float.Degrees, "radians": x87.Float.Radians, "fabs": x87.Float.Abs,
	"rint":  func(x x87.Float) x87.Float { return x.Round("rint") },
	"floor": func(x x87.Float) x87.Float { return x.Round("floor") },
	"ceil":  func(x x87.Float) x87.Float { return x.Round("ceil") },
	"trunc": func(x x87.Float) x87.Float { return x.Round("trunc") },
	"sign": func(x x87.Float) x87.Float {
		if x.IsNaN() {
			return x
		}
		if x.IsZero() {
			return x87.Zero(false)
		}
		if x.Signbit() {
			return x87.FromInt64(-1)
		}
		return x87.FromInt64(1)
	},
	"spacing": func(x x87.Float) x87.Float { return x.Nextafter(x87.Inf(x.Signbit() && !x.IsZero())).Sub(x) },
	"sinc": func(x x87.Float) x87.Float {
		if x.IsZero() {
			return x87.FromInt64(1)
		}
		// np.sinc multiplies by the binary64 np.pi constant.
		y := x.Mul(x87.FromFloat64(math.Pi))
		return y.Sin().Div(y)
	},
}
var extendedComplexUnary = map[string]func(x87.Complex) x87.Complex{
	"sqrt": x87.Complex.Sqrt, "exp": x87.Complex.Exp, "exp2": x87.Complex.Exp2, "expm1": x87.Complex.Expm1,
	"log": x87.Complex.Log, "log2": x87.Complex.Log2, "log10": x87.Complex.Log10, "log1p": x87.Complex.Log1p,
	"sin": x87.Complex.Sin, "cos": x87.Complex.Cos, "tan": x87.Complex.Tan,
	"arcsin": x87.Complex.Asin, "arccos": x87.Complex.Acos, "arctan": x87.Complex.Atan,
	"sinh": x87.Complex.Sinh, "cosh": x87.Complex.Cosh, "tanh": x87.Complex.Tanh,
	"arcsinh": x87.Complex.Asinh, "arccosh": x87.Complex.Acosh, "arctanh": x87.Complex.Atanh,
	"sign": x87.Complex.Sign,
	"rint": func(z x87.Complex) x87.Complex {
		return x87.Complex{Real: z.Real.Round("rint"), Imag: z.Imag.Round("rint")}
	},
	"sinc": func(z x87.Complex) x87.Complex {
		if z.Real.IsZero() && z.Imag.IsZero() {
			return x87.Complex{Real: x87.FromInt64(1)}
		}
		p := x87.FromFloat64(math.Pi)
		w := x87.Complex{Real: z.Real.Mul(p), Imag: z.Imag.Mul(p)}
		return w.Sin().Div(w)
	},
}

func extendedMath(a *Array, name string) (*Array, error) {
	d := a.DType()
	if d == Float128 {
		f, ok := extendedRealUnary[name]
		if !ok {
			return nil, ErrDType
		}
		return mapSpecialUnary(a, d, func(x element) (element, error) { return element{extra: f(extendedValue(x, d).Real)}, nil })
	}
	if d == Complex256 {
		f, ok := extendedComplexUnary[name]
		if !ok {
			return nil, ErrDType
		}
		return mapSpecialUnary(a, d, func(x element) (element, error) { return element{extra: f(extendedValue(x, d))}, nil })
	}
	return nil, ErrDType
}

var extendedRealBinary = map[string]func(x87.Float, x87.Float) x87.Float{
	"hypot":   func(x, y x87.Float) x87.Float { return (x87.Complex{Real: x, Imag: y}).Abs() },
	"arctan2": x87.Atan2,
	"copysign": func(x, y x87.Float) x87.Float {
		if x.Signbit() != y.Signbit() {
			return x.Neg()
		}
		return x
	},
	"logaddexp":  func(x, y x87.Float) x87.Float { return x87.LogAddExp(x, y, false) },
	"logaddexp2": func(x, y x87.Float) x87.Float { return x87.LogAddExp(x, y, true) },
	"heaviside": func(x, y x87.Float) x87.Float {
		if x.IsNaN() {
			return x
		}
		if x.IsZero() {
			return y
		}
		if x.Signbit() {
			return x87.Zero(false)
		}
		return x87.FromInt64(1)
	},
	"fmod": x87.Float.Fmod,
}

func extendedBinary(a, b *Array, name string) (*Array, error) {
	f, ok := extendedRealBinary[name]
	if !ok {
		return nil, ErrDType
	}
	return binaryMapCore(a, b, Float128, Float128, func(x, y element) (element, error) {
		return element{extra: f(extendedValue(x, Float128).Real, extendedValue(y, Float128).Real)}, nil
	})
}
