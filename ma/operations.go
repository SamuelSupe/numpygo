package ma

import (
	"github.com/SamuelSupe/numpygo"
)

type unaryOperation func(*np.Array) (*np.Array, error)
type binaryOperation func(*np.Array, *np.Array) (*np.Array, error)
type domainOperation func(*np.Array) (*np.Array, error)

func maskedUnary(a *Array, op unaryOperation, domain domainOperation) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	result, err := op(a.data)
	if err != nil {
		return nil, err
	}
	mask, err := a.mask.Copy()
	if err != nil {
		return nil, err
	}
	if domain != nil {
		finite, err := np.IsFinite(result)
		if err != nil {
			return nil, err
		}
		invalid, err := np.LogicalNot(finite)
		if err != nil {
			return nil, err
		}
		bad, err := domain(a.data)
		if err != nil {
			return nil, err
		}
		mask, err = np.LogicalOr(mask, invalid)
		if err != nil {
			return nil, err
		}
		mask, err = np.LogicalOr(mask, bad)
		if err != nil {
			return nil, err
		}
	}
	// NumPy restores the source at masked positions when same-kind assignment is possible.
	if result.DType() != np.Bool || a.DType() == np.Bool {
		if !(a.DType().IsComplex() && !result.DType().IsComplex()) {
			source, err := a.data.Astype(result.DType())
			if err != nil {
				return nil, err
			}
			result, err = np.Where(mask, source, result)
			if err != nil {
				return nil, err
			}
		}
	}
	out := a.with(result, mask)
	out.explicitMask = a.explicitMask || domain != nil
	return out, nil
}
func maskedBinary(a, b *Array, op binaryOperation, divideDomain, invalidDomain bool) (*Array, error) {
	if !valid(a) || !valid(b) {
		return nil, np.ErrDType
	}
	result, err := op(a.data, b.data)
	if err != nil {
		return nil, err
	}
	mask, err := np.LogicalOr(a.mask, b.mask)
	if err != nil {
		return nil, err
	}
	if divideDomain || invalidDomain {
		finite, err := np.IsFinite(result)
		if err != nil {
			return nil, err
		}
		bad, err := np.LogicalNot(finite)
		if err != nil {
			return nil, err
		}
		mask, err = np.LogicalOr(mask, bad)
		if err != nil {
			return nil, err
		}
		if divideDomain {
			x, err := np.Absolute(a.data)
			if err != nil {
				return nil, err
			}
			y, err := np.Absolute(b.data)
			if err != nil {
				return nil, err
			}
			threshold, err := np.Multiply(x, np.Scalar(2.2250738585072014e-308))
			if err != nil {
				return nil, err
			}
			bad, err = np.GreaterEqual(threshold, y)
			if err != nil {
				return nil, err
			}
			mask, err = np.LogicalOr(mask, bad)
			if err != nil {
				return nil, err
			}
		}
	}
	source, err := a.data.Astype(result.DType())
	if err != nil {
		return nil, err
	}
	result, err = np.Where(mask, source, result)
	if err != nil {
		return nil, err
	}
	out := a.with(result, mask)
	out.explicitMask = a.explicitMask || b.explicitMask || divideDomain || invalidDomain
	return out, nil
}
func lessThan(t float64, include bool) domainOperation {
	return func(a *np.Array) (*np.Array, error) {
		v := np.Scalar(t)
		if include {
			return np.LessEqual(a, v)
		}
		return np.Less(a, v)
	}
}
func outsideInterval(lo, hi float64) domainOperation {
	return func(a *np.Array) (*np.Array, error) {
		l, err := np.Less(a, np.Scalar(lo))
		if err != nil {
			return nil, err
		}
		h, err := np.Greater(a, np.Scalar(hi))
		if err != nil {
			return nil, err
		}
		return np.LogicalOr(l, h)
	}
}
func Sqrt(a *Array) (*Array, error)    { return maskedUnary(a, np.Sqrt, lessThan(0, false)) }
func Log(a *Array) (*Array, error)     { return maskedUnary(a, np.Log, lessThan(0, true)) }
func Log2(a *Array) (*Array, error)    { return maskedUnary(a, np.Log2, lessThan(0, true)) }
func Log10(a *Array) (*Array, error)   { return maskedUnary(a, np.Log10, lessThan(0, true)) }
func Arcsin(a *Array) (*Array, error)  { return maskedUnary(a, np.Arcsin, outsideInterval(-1, 1)) }
func Arccos(a *Array) (*Array, error)  { return maskedUnary(a, np.Arccos, outsideInterval(-1, 1)) }
func Arccosh(a *Array) (*Array, error) { return maskedUnary(a, np.Arccosh, lessThan(1, false)) }
func Arctanh(a *Array) (*Array, error) {
	return maskedUnary(a, np.Arctanh, outsideInterval(-1+1e-15, 1-1e-15))
}
func Tan(a *Array) (*Array, error) {
	domain := func(x *np.Array) (*np.Array, error) {
		v, err := np.Cos(x)
		if err != nil {
			return nil, err
		}
		v, err = np.Absolute(v)
		if err != nil {
			return nil, err
		}
		return np.Less(v, np.Scalar(1e-35))
	}
	return maskedUnary(a, np.Tan, domain)
}
func Angle(a *Array, degrees bool) (*Array, error) {
	return maskedUnary(a, func(x *np.Array) (*np.Array, error) { return np.Angle(x, degrees) }, nil)
}
func Round(a *Array, decimals int) (*Array, error) {
	return maskedUnary(a, func(x *np.Array) (*np.Array, error) { return np.Round(x, decimals) }, nil)
}
func Divide(a, b *Array) (*Array, error)      { return maskedBinary(a, b, np.Divide, true, false) }
func FloorDivide(a, b *Array) (*Array, error) { return maskedBinary(a, b, np.FloorDivide, true, false) }
func Remainder(a, b *Array) (*Array, error)   { return maskedBinary(a, b, np.Remainder, true, false) }
func Fmod(a, b *Array) (*Array, error)        { return maskedBinary(a, b, np.Fmod, true, false) }
func Power(a, b *Array) (*Array, error)       { return maskedBinary(a, b, np.Power, false, true) }
func Clip(a *Array, lo, hi *np.Array) (*Array, error) {
	return maskedUnary(a, func(x *np.Array) (*np.Array, error) { return np.Clip(x, lo, hi) }, nil)
}
func Exp(a *Array) (*Array, error)          { return maskedUnary(a, np.Exp, nil) }
func Conjugate(a *Array) (*Array, error)    { return maskedUnary(a, np.Conjugate, nil) }
func Sin(a *Array) (*Array, error)          { return maskedUnary(a, np.Sin, nil) }
func Cos(a *Array) (*Array, error)          { return maskedUnary(a, np.Cos, nil) }
func Arctan(a *Array) (*Array, error)       { return maskedUnary(a, np.Arctan, nil) }
func Arcsinh(a *Array) (*Array, error)      { return maskedUnary(a, np.Arcsinh, nil) }
func Sinh(a *Array) (*Array, error)         { return maskedUnary(a, np.Sinh, nil) }
func Cosh(a *Array) (*Array, error)         { return maskedUnary(a, np.Cosh, nil) }
func Tanh(a *Array) (*Array, error)         { return maskedUnary(a, np.Tanh, nil) }
func Absolute(a *Array) (*Array, error)     { return maskedUnary(a, np.Absolute, nil) }
func Fabs(a *Array) (*Array, error)         { return maskedUnary(a, np.Fabs, nil) }
func Negative(a *Array) (*Array, error)     { return maskedUnary(a, np.Negative, nil) }
func Floor(a *Array) (*Array, error)        { return maskedUnary(a, np.Floor, nil) }
func Ceil(a *Array) (*Array, error)         { return maskedUnary(a, np.Ceil, nil) }
func LogicalNot(a *Array) (*Array, error)   { return maskedUnary(a, np.LogicalNot, nil) }
func Add(a, b *Array) (*Array, error)       { return maskedBinary(a, b, np.Add, false, false) }
func Subtract(a, b *Array) (*Array, error)  { return maskedBinary(a, b, np.Subtract, false, false) }
func Multiply(a, b *Array) (*Array, error)  { return maskedBinary(a, b, np.Multiply, false, false) }
func Arctan2(a, b *Array) (*Array, error)   { return maskedBinary(a, b, np.Arctan2, false, false) }
func Equal(a, b *Array) (*Array, error)     { return maskedBinary(a, b, np.Equal, false, false) }
func NotEqual(a, b *Array) (*Array, error)  { return maskedBinary(a, b, np.NotEqual, false, false) }
func Less(a, b *Array) (*Array, error)      { return maskedBinary(a, b, np.Less, false, false) }
func LessEqual(a, b *Array) (*Array, error) { return maskedBinary(a, b, np.LessEqual, false, false) }
func Greater(a, b *Array) (*Array, error)   { return maskedBinary(a, b, np.Greater, false, false) }
func GreaterEqual(a, b *Array) (*Array, error) {
	return maskedBinary(a, b, np.GreaterEqual, false, false)
}
func LogicalAnd(a, b *Array) (*Array, error) { return maskedBinary(a, b, np.LogicalAnd, false, false) }
func LogicalOr(a, b *Array) (*Array, error)  { return maskedBinary(a, b, np.LogicalOr, false, false) }
func LogicalXor(a, b *Array) (*Array, error) { return maskedBinary(a, b, np.LogicalXor, false, false) }
func BitwiseAnd(a, b *Array) (*Array, error) { return maskedBinary(a, b, np.BitwiseAnd, false, false) }
func BitwiseOr(a, b *Array) (*Array, error)  { return maskedBinary(a, b, np.BitwiseOr, false, false) }
func BitwiseXor(a, b *Array) (*Array, error) { return maskedBinary(a, b, np.BitwiseXor, false, false) }
func Hypot(a, b *Array) (*Array, error)      { return maskedBinary(a, b, np.Hypot, false, false) }
func LeftShift(a, b *Array) (*Array, error)  { return maskedBinary(a, b, np.LeftShift, false, false) }
func RightShift(a, b *Array) (*Array, error) { return maskedBinary(a, b, np.RightShift, false, false) }
func Minimum(a, b *Array) (*Array, error)    { return maskedBinary(a, b, np.Minimum, false, false) }
func Maximum(a, b *Array) (*Array, error)    { return maskedBinary(a, b, np.Maximum, false, false) }
