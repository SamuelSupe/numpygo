// Package scalar exposes immutable, typed NumPy-style numeric scalars.
package scalar

import (
	"errors"
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"math"
	"math/big"
	"strconv"
	"strings"
)

type Value struct{ a *np.Array }

func New(a *np.Array) (Value, error) {
	if (!a.DType().IsNativeNumeric() && !a.DType().IsExtended()) || a.NDim() != 0 {
		return Value{}, np.ErrDType
	}
	v, e := a.Copy()
	if e != nil {
		return Value{}, e
	}
	return Value{v}, nil
}
func Of[T np.Number](v T) Value           { return Value{np.ScalarOf(v)} }
func (v Value) DType() np.DType           { return v.a.DType() }
func (v Value) Shape() []int              { return v.a.Shape() }
func (v Value) NDim() int                 { return 0 }
func (v Value) Size() int                 { return 1 }
func (v Value) ItemSize() int             { return v.a.ItemSize() }
func (v Value) NBytes() int               { return v.a.NBytes() }
func (v Value) Strides() []int            { return v.a.Strides() }
func (v Value) Device() string            { return "cpu" }
func (v Value) Item() (any, error)        { return v.a.AtValue() }
func (v Value) Array() (*np.Array, error) { return v.a.Copy() }
func (v Value) ToBytes() ([]byte, error)  { return v.a.ToBytes("C") }
func (v Value) Copy() (Value, error)      { return New(v.a) }
func (v Value) Astype(d np.DType) (Value, error) {
	a, e := v.a.Astype(d)
	if e != nil {
		return Value{}, e
	}
	return New(a)
}
func (v Value) ByteSwap() (Value, error) {
	a, e := v.a.ByteSwap(false)
	if e != nil {
		return Value{}, e
	}
	return New(a)
}
func (v Value) Real() (Value, error) {
	a, e := np.Real(v.a)
	if e != nil {
		return Value{}, e
	}
	return New(a)
}
func (v Value) Imag() (Value, error) {
	a, e := np.Imag(v.a)
	if e != nil {
		return Value{}, e
	}
	return New(a)
}
func (v Value) Conjugate() (Value, error) {
	if v.DType() == np.Bool {
		return v.Copy()
	}
	a, e := np.Conjugate(v.a)
	if e != nil {
		return Value{}, e
	}
	return New(a)
}
func (v Value) Reshape(shape ...int) (*np.Array, error) {
	a, e := v.Array()
	if e != nil {
		return nil, e
	}
	return a.Reshape(shape...)
}
func (v Value) Ravel() (*np.Array, error)   { return v.Reshape(1) }
func (v Value) Flatten() (*np.Array, error) { return v.Ravel() }
func (v Value) Transpose() (Value, error)   { return v.Copy() }
func (v Value) Squeeze() (Value, error)     { return v.Copy() }
func (v Value) View(d np.DType) (Value, error) {
	a, e := v.a.View(d)
	if e != nil {
		return Value{}, e
	}
	return New(a)
}
func (v Value) GetField(d np.DType, offset int) (Value, error) {
	a, e := v.a.GetField(d, offset)
	if e != nil {
		return Value{}, e
	}
	return New(a)
}

func (v Value) BitCount() (int, error) {
	if !v.DType().IsInteger() {
		return 0, np.ErrDType
	}
	a, e := np.BitwiseCount(v.a)
	if e != nil {
		return 0, e
	}
	x, e := a.AtValue()
	if e != nil {
		return 0, e
	}
	return int(x.(uint8)), nil
}
func (v Value) Numerator() (Value, error) {
	if !v.DType().IsInteger() {
		return Value{}, np.ErrDType
	}
	return v.Copy()
}
func (v Value) Denominator() (int, error) {
	if !v.DType().IsInteger() {
		return 0, np.ErrDType
	}
	return 1, nil
}

// AsIntegerRatio returns exact, reduced arbitrary-size integers. Neither
// binary16 nor x87 values are rounded through an intermediate float64.
func (v Value) AsIntegerRatio() (*big.Int, *big.Int, error) {
	if !v.DType().IsFloat() {
		return nil, nil, np.ErrDType
	}
	x, e := v.Item()
	if e != nil {
		return nil, nil, e
	}
	var r *big.Rat
	if f, ok := x.(np.ExtendedFloat); ok {
		if f.IsNaN() {
			return nil, nil, np.ErrValue
		}
		if f.IsInf() {
			return nil, nil, np.ErrOverflow
		}
		r = f.Rat()
	} else {
		var f float64
		switch z := x.(type) {
		case np.Half:
			f = z.Float64()
		case float32:
			f = float64(z)
		case float64:
			f = z
		default:
			return nil, nil, np.ErrDType
		}
		if math.IsNaN(f) {
			return nil, nil, np.ErrValue
		}
		if math.IsInf(f, 0) {
			return nil, nil, np.ErrOverflow
		}
		r = new(big.Rat).SetFloat64(f)
	}
	return new(big.Int).Set(r.Num()), new(big.Int).Set(r.Denom()), nil
}
func (v Value) IsInteger() (bool, error) {
	if v.DType().IsInteger() {
		return true, nil
	}
	_, d, e := v.AsIntegerRatio()
	if errors.Is(e, np.ErrOverflow) || errors.Is(e, np.ErrValue) {
		return false, nil
	}
	if e != nil {
		return false, e
	}
	return d.Cmp(big.NewInt(1)) == 0, nil
}

// Hex matches Python's float.hex representation for the float64 scalar,
// including signed zero and the fixed exponent used for subnormals.
func (v Value) Hex() (string, error) {
	if v.DType() != np.Float64 {
		return "", np.ErrDType
	}
	x, e := v.Item()
	if e != nil {
		return "", e
	}
	f := x.(float64)
	if math.IsNaN(f) {
		return "nan", nil
	}
	sign := ""
	if math.Signbit(f) {
		sign = "-"
	}
	if math.IsInf(f, 0) {
		return sign + "inf", nil
	}
	if f == 0 {
		return sign + "0x0.0p+0", nil
	}
	b := math.Float64bits(f)
	exp := int((b >> 52) & 0x7ff)
	frac := b & ((uint64(1) << 52) - 1)
	lead := 1
	if exp == 0 {
		lead = 0
		exp = 1
	}
	return fmt.Sprintf("%s0x%d.%013xp%+d", sign, lead, frac, exp-1023), nil
}
func FromHex(s string) (Value, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" || strings.Contains(s, "_") {
		return Value{}, np.ErrValue
	}
	if s == "nan" || s == "+nan" || s == "-nan" {
		return Of(math.NaN()), nil
	}
	if s == "inf" || s == "+inf" || s == "infinity" || s == "+infinity" {
		return Of(math.Inf(1)), nil
	}
	if s == "-inf" || s == "-infinity" {
		return Of(math.Inf(-1)), nil
	}
	sign := ""
	if s[0] == '-' || s[0] == '+' {
		sign = s[:1]
		s = s[1:]
	}
	if !strings.HasPrefix(s, "0x") {
		s = "0x" + s
	}
	if !strings.Contains(s, "p") {
		s += "p0"
	}
	x, e := strconv.ParseFloat(sign+s, 64)
	if e != nil {
		if errors.Is(e, strconv.ErrRange) {
			return Value{}, np.ErrOverflow
		}
		return Value{}, np.ErrValue
	}
	return Of(x), nil
}
