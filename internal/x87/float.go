// Package extended implements NumPy's Linux/x86 extended-real representation
// (80-bit encoding, 64-bit significand, in a 16-byte slot) without using the
// host C long double.
package x87

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/big"
	"strings"
)

// Float stores the x87 explicit significand and sign/exponent. Padding bytes
// in NumPy's 16-byte representation are not part of the numerical value.
type Float struct {
	Significand  uint64
	SignExponent uint16
}

const (
	bias              = 16383
	integerBit uint64 = 1 << 63
)

func Zero(negative bool) Float {
	if negative {
		return Float{SignExponent: 0x8000}
	}
	return Float{}
}
func Inf(negative bool) Float {
	x := Float{integerBit, 0x7fff}
	if negative {
		x.SignExponent |= 0x8000
	}
	return x
}
func NaN() Float               { return Float{integerBit | 1<<62, 0x7fff} }
func (x Float) Signbit() bool  { return x.SignExponent&0x8000 != 0 }
func (x Float) exponent() int  { return int(x.SignExponent & 0x7fff) }
func (x Float) IsNaN() bool    { return x.exponent() == 0x7fff && x.Significand != integerBit }
func (x Float) IsInf() bool    { return x.exponent() == 0x7fff && x.Significand == integerBit }
func (x Float) IsFinite() bool { return x.exponent() != 0x7fff }
func (x Float) IsZero() bool   { return x.exponent() == 0 && x.Significand == 0 }
func (x Float) Neg() Float     { x.SignExponent ^= 0x8000; return x }
func (x Float) Abs() Float     { x.SignExponent &= 0x7fff; return x }
func (x Float) Raw() [10]byte {
	var b [10]byte
	binary.LittleEndian.PutUint64(b[:], x.Significand)
	binary.LittleEndian.PutUint16(b[8:], x.SignExponent)
	return b
}
func FromRaw(b []byte) (Float, error) {
	if len(b) != 10 && len(b) != 16 {
		return Float{}, fmt.Errorf("invalid x87 byte count")
	}
	x := Float{binary.LittleEndian.Uint64(b), binary.LittleEndian.Uint16(b[8:])}
	if x.exponent() != 0 && x.Significand&integerBit == 0 {
		return Float{}, fmt.Errorf("unsupported x87 noncanonical encoding")
	}
	if x.exponent() == 0 && x.Significand&integerBit != 0 {
		x.SignExponent |= 1
	}
	return x, nil
}
func (x Float) Bytes16() [16]byte { var b [16]byte; r := x.Raw(); copy(b[:], r[:]); return b }

func (x Float) Rat() *big.Rat {
	if !x.IsFinite() {
		return nil
	}
	n := new(big.Int).SetUint64(x.Significand)
	if x.Signbit() {
		n.Neg(n)
	}
	e := max(1, x.exponent()) - bias - 63
	if e >= 0 {
		return new(big.Rat).SetInt(n.Lsh(n, uint(e)))
	}
	d := new(big.Int).Lsh(big.NewInt(1), uint(-e))
	return new(big.Rat).SetFrac(n, d)
}
func (x Float) BigFloat() *big.Float {
	if x.IsNaN() {
		return nil
	}
	f := new(big.Float).SetPrec(64).SetMode(big.ToNearestEven)
	if x.IsInf() {
		return f.SetInf(x.Signbit())
	}
	f.SetRat(x.Rat())
	if x.IsZero() && x.Signbit() {
		f.Neg(f)
	}
	return f
}
func (x Float) Float64() float64 {
	if x.IsNaN() {
		return math.NaN()
	}
	v, _ := x.BigFloat().Float64()
	return v
}
func FromFloat64(v float64) Float {
	if math.IsNaN(v) {
		return NaN()
	}
	if math.IsInf(v, 0) {
		return Inf(math.Signbit(v))
	}
	r := new(big.Rat).SetFloat64(v)
	return roundRat(r, math.Signbit(v))
}
func FromInt64(v int64) Float { return roundRat(new(big.Rat).SetInt64(v), v < 0) }
func FromUint64(v uint64) Float {
	return roundRat(new(big.Rat).SetInt(new(big.Int).SetUint64(v)), false)
}

// roundRat rounds the exact mathematical value directly to the x87 lattice,
// including gradual underflow. Avoiding intermediate float64 or normal-only
// rounding matters for subnormal ties and for integers above 2^53.
func roundRat(r *big.Rat, negativeZero bool) Float {
	if r.Sign() == 0 {
		return Zero(negativeZero)
	}
	negative := r.Sign() < 0
	n := new(big.Int).Abs(r.Num())
	d := new(big.Int).Set(r.Denom())
	e := n.BitLen() - d.BitLen()
	var cmp int
	if e >= 0 {
		cmp = n.Cmp(new(big.Int).Lsh(new(big.Int).Set(d), uint(e)))
	} else {
		cmp = new(big.Int).Lsh(new(big.Int).Set(n), uint(-e)).Cmp(d)
	}
	if cmp < 0 {
		e--
	}
	if e > 16383 {
		return Inf(negative)
	}
	shift := max(e-63, -16445)
	if shift >= 0 {
		d.Lsh(d, uint(shift))
	} else {
		n.Lsh(n, uint(-shift))
	}
	q, rem := new(big.Int), new(big.Int)
	q.QuoRem(n, d, rem)
	twice := rem.Lsh(rem, 1).Cmp(d)
	if twice > 0 || twice == 0 && q.Bit(0) == 1 {
		q.Add(q, big.NewInt(1))
	}
	if q.Sign() == 0 {
		return Zero(negative)
	}
	if q.BitLen() > 64 {
		q.Rsh(q, 1)
		shift++
	}
	exponent := shift + 63 + bias
	if exponent >= 0x7fff {
		return Inf(negative)
	}
	if exponent < 1 || exponent == 1 && q.BitLen() < 64 {
		exponent = 0
	}
	x := Float{q.Uint64(), uint16(exponent)}
	if negative {
		x.SignExponent |= 0x8000
	}
	return x
}

// Parse accepts decimal strings (including exponents, infinities and NaN).
func Parse(s string) (Float, error) {
	s = strings.TrimSpace(s)
	negative := strings.HasPrefix(s, "-")
	switch strings.ToLower(strings.TrimLeft(s, "+-")) {
	case "nan":
		return NaN(), nil
	case "inf", "infinity":
		return Inf(negative), nil
	}
	// ParseFloat first bounds pathological exponents before Rat's exact
	// decimal conversion could allocate enormous integer powers.
	if len(s) > 100000 {
		return Float{}, fmt.Errorf("extended literal too long")
	}
	f, _, e := big.ParseFloat(s, 10, 128, big.ToNearestEven)
	if e != nil {
		return Float{}, fmt.Errorf("invalid extended literal %q: %w", s, e)
	}
	exp := f.MantExp(nil)
	if exp > 16500 {
		return Inf(negative), nil
	}
	if exp < -16500 {
		return Zero(negative), nil
	}
	r, ok := new(big.Rat).SetString(s)
	if !ok {
		return Float{}, fmt.Errorf("invalid extended literal %q", s)
	}
	return roundRat(r, negative), nil
}
func MustParse(s string) Float {
	x, e := Parse(s)
	if e != nil {
		panic(e)
	}
	return x
}
func (x Float) String() string {
	if x.IsNaN() {
		return "nan"
	}
	if x.IsInf() {
		if x.Signbit() {
			return "-inf"
		}
		return "inf"
	}
	if x.IsZero() && x.Signbit() {
		return "-0"
	}
	return x.BigFloat().Text('g', -1)
}
func (x Float) Format(format byte, precision int) string {
	if !x.IsFinite() {
		return x.String()
	}
	return x.BigFloat().Text(format, precision)
}
func (x Float) Equal(y Float) bool {
	if x.IsNaN() || y.IsNaN() {
		return false
	}
	return x.IsZero() && y.IsZero() || x == y
}

// Compare returns -1,0,+1 and false for an unordered NaN comparison.
func (x Float) Compare(y Float) (int, bool) {
	if x.IsNaN() || y.IsNaN() {
		return 0, false
	}
	if x.Equal(y) {
		return 0, true
	}
	if x.IsInf() {
		if x.Signbit() {
			return -1, true
		}
		return 1, true
	}
	if y.IsInf() {
		if y.Signbit() {
			return 1, true
		}
		return -1, true
	}
	return x.Rat().Cmp(y.Rat()), true
}
func (x Float) Add(y Float) Float {
	if x.IsNaN() || y.IsNaN() {
		return NaN()
	}
	if x.IsInf() {
		if y.IsInf() && x.Signbit() != y.Signbit() {
			return NaN()
		}
		return x
	}
	if y.IsInf() {
		return y
	}
	return roundRat(new(big.Rat).Add(x.Rat(), y.Rat()), x.Signbit() && y.Signbit())
}
func (x Float) Sub(y Float) Float { return x.Add(y.Neg()) }
func (x Float) Mul(y Float) Float {
	if x.IsNaN() || y.IsNaN() {
		return NaN()
	}
	sign := x.Signbit() != y.Signbit()
	if x.IsInf() || y.IsInf() {
		if x.IsZero() || y.IsZero() {
			return NaN()
		}
		return Inf(sign)
	}
	return roundRat(new(big.Rat).Mul(x.Rat(), y.Rat()), sign)
}
func (x Float) Div(y Float) Float {
	if x.IsNaN() || y.IsNaN() || x.IsInf() && y.IsInf() || x.IsZero() && y.IsZero() {
		return NaN()
	}
	sign := x.Signbit() != y.Signbit()
	if x.IsInf() || y.IsZero() {
		return Inf(sign)
	}
	if y.IsInf() || x.IsZero() {
		return Zero(sign)
	}
	return roundRat(new(big.Rat).Quo(x.Rat(), y.Rat()), sign)
}
func (x Float) Sqrt() Float {
	if x.IsNaN() || x.Signbit() && !x.IsZero() {
		return NaN()
	}
	if x.IsInf() || x.IsZero() {
		return x
	}
	// A high precision square root is rounded to the extended lattice.
	v := new(big.Float).SetPrec(256).SetRat(x.Rat())
	v.Sqrt(v)
	r, _ := v.Rat(nil)
	return roundRat(r, false)
}
func (x Float) Nextafter(y Float) Float {
	c, ok := x.Compare(y)
	if !ok {
		return NaN()
	}
	if c == 0 {
		return y
	}
	if x.IsZero() {
		s := uint16(0)
		if c > 0 {
			s = 0x8000
		}
		return Float{1, s}
	}
	negative := x.Signbit()
	up := c < 0
	if negative {
		up = !up
	}
	e := x.exponent()
	m := x.Significand
	if up {
		if e == 0x7fff {
			return x
		}
		m++
		if m == 0 {
			m = integerBit
			e++
		} else if e == 0 && m == integerBit {
			e = 1
		}
		if e == 0x7fff {
			m = integerBit
		}
	} else {
		if e == 0x7fff {
			e = 0x7ffe
			m = ^uint64(0)
		} else if m == integerBit && e > 0 {
			e--
			if e == 0 {
				m = integerBit - 1
			} else {
				m = ^uint64(0)
			}
		} else {
			m--
		}
	}
	x = Float{m, uint16(e)}
	if negative {
		x.SignExponent |= 0x8000
	}
	return x
}

type FloatInfo struct {
	Bits, Precision, Mantissa, MinExp, MaxExp int
	Eps, Tiny, Max, SmallestSubnormal         Float
}

func Info() FloatInfo {
	return FloatInfo{128, 18, 63, -16382, 16384, Float{integerBit, uint16(bias - 63)}, Float{integerBit, 1}, Float{^uint64(0), 0x7ffe}, Float{1, 0}}
}
