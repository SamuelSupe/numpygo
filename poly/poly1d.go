package poly

import (
	"github.com/SamuelSupe/numpygo"
	"iter"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Poly1D is the legacy descending-coefficient polynomial class. Unlike
// Series, coefficients retain their numeric dtype and share storage with
// a nonzero input array after trimming leading zeros.
type Poly1D struct {
	coeff    *np.Array
	variable string
}

func NewPoly1D(c *np.Array, fromRoots bool, variable string) (*Poly1D, error) {
	if c.DType() == 0 {
		return nil, np.ErrDType
	}
	var e error
	if fromRoots {
		c, e = LegacyFromRoots(c)
		if e != nil {
			return nil, e
		}
	}
	if c.NDim() == 0 {
		c, e = c.Reshape(1)
		if e != nil {
			return nil, e
		}
	}
	if c.NDim() != 1 {
		return nil, np.ErrShape
	}
	if !c.DType().IsNativeNumeric() && !c.DType().IsExtended() {
		return nil, np.ErrDType
	}
	z, e := np.Zeros(c.DType())
	if e != nil {
		return nil, e
	}
	eq, e := np.Equal(c, z)
	if e != nil {
		return nil, e
	}
	v, e := np.Values[bool](eq)
	if e != nil {
		return nil, e
	}
	first := 0
	for first < len(v) && v[first] {
		first++
	}
	if first == len(v) {
		c, e = np.Zeros(c.DType(), 1)
	} else {
		c, e = c.Slice(np.Range(first, c.Size()))
	}
	if e != nil {
		return nil, e
	}
	if variable == "" {
		variable = "x"
	}
	return &Poly1D{c, variable}, nil
}
func (p *Poly1D) Alias(variable string) *Poly1D {
	if variable == "" {
		variable = p.variable
	}
	return &Poly1D{p.coeff, variable}
}
func (p *Poly1D) Coefficients() *np.Array   { return p.coeff }
func (p *Poly1D) Coeffs() *np.Array         { return p.coeff }
func (p *Poly1D) Order() int                { return p.coeff.Size() - 1 }
func (p *Poly1D) Len() int                  { return p.Order() }
func (p *Poly1D) Variable() string          { return p.variable }
func (p *Poly1D) Roots() (*np.Array, error) { return LegacyRoots(p.coeff) }
func (p *Poly1D) Coefficient(power int) (any, error) {
	if power < 0 || power > p.Order() {
		z, e := np.Zeros(p.coeff.DType())
		if e != nil {
			return nil, e
		}
		return z.AtValue()
	}
	return p.coeff.AtValue(p.Order() - power)
}
func (p *Poly1D) SetCoefficient(power int, value any) error {
	if power < 0 {
		return np.ErrValue
	}
	if power <= p.Order() {
		return p.coeff.SetValue(value, p.Order()-power)
	}
	if power == int(^uint(0)>>1) {
		return np.ErrShape
	}
	out, e := np.Zeros(p.coeff.DType(), power+1)
	if e != nil {
		return e
	}
	if e = out.Assign([]np.Index{np.Range(power-p.Order(), power+1)}, p.coeff); e != nil {
		return e
	}
	if e = out.SetValue(value, 0); e != nil {
		return e
	}
	p.coeff = out
	return nil
}
func (p *Poly1D) Iter() (iter.Seq[any], error)        { return np.NDIter(p.coeff, "C") }
func (p *Poly1D) Eval(x *np.Array) (*np.Array, error) { return LegacyEval(p.coeff, x) }
func polyResult(a *np.Array, e error) (*Poly1D, error) {
	if e != nil {
		return nil, e
	}
	return NewPoly1D(a, false, "")
}
func (p *Poly1D) Add(q *Poly1D) (*Poly1D, error) { return polyResult(LegacyAdd(p.coeff, q.coeff)) }
func (p *Poly1D) Sub(q *Poly1D) (*Poly1D, error) { return polyResult(LegacySub(p.coeff, q.coeff)) }
func (p *Poly1D) Mul(q *Poly1D) (*Poly1D, error) { return polyResult(LegacyMul(p.coeff, q.coeff)) }
func (p *Poly1D) Divide(q *Poly1D) (*Poly1D, *Poly1D, error) {
	x, y, e := LegacyDiv(p.coeff, q.coeff)
	if e != nil {
		return nil, nil, e
	}
	a, e := NewPoly1D(x, false, "")
	if e != nil {
		return nil, nil, e
	}
	b, e := NewPoly1D(y, false, "")
	return a, b, e
}
func (p *Poly1D) Scale(x *np.Array) (*Poly1D, error) {
	if x.NDim() != 0 {
		return nil, np.ErrShape
	}
	return polyResult(np.Multiply(p.coeff, x))
}
func (p *Poly1D) DivideScalar(x *np.Array, reflected bool) (*Poly1D, error) {
	if x.NDim() != 0 {
		return nil, np.ErrShape
	}
	if reflected {
		return polyResult(np.Divide(x, p.coeff))
	}
	return polyResult(np.Divide(p.coeff, x))
}
func (p *Poly1D) Negative() (*Poly1D, error)    { return polyResult(np.Negative(p.coeff)) }
func (p *Poly1D) Positive() *Poly1D             { return p }
func (p *Poly1D) Equal(q *Poly1D) (bool, error) { return np.ArrayEqual(p.coeff, q.coeff, false) }
func (p *Poly1D) Pow(power int) (*Poly1D, error) {
	if power < 0 {
		return nil, np.ErrValue
	}
	v, e := np.FromSlice([]int64{1}, 1)
	if e != nil {
		return nil, e
	}
	for j := 0; j < power; j++ {
		v, e = LegacyMul(p.coeff, v)
		if e != nil {
			return nil, e
		}
	}
	return NewPoly1D(v, false, "")
}
func (p *Poly1D) Compose(q *Poly1D) (*Poly1D, error) {
	v, e := np.Zeros(p.coeff.DType(), 1)
	if e != nil {
		return nil, e
	}
	for i := 0; i < p.coeff.Size(); i++ {
		v, e = LegacyMul(v, q.coeff)
		if e != nil {
			return nil, e
		}
		c, e := p.coeff.Slice(np.Range(i, i+1))
		if e != nil {
			return nil, e
		}
		v, e = LegacyAdd(v, c)
		if e != nil {
			return nil, e
		}
	}
	return NewPoly1D(v, false, "")
}
func (p *Poly1D) Deriv(order int) (*Poly1D, error) {
	return polyResult(LegacyDerivative(p.coeff, order))
}
func (p *Poly1D) Integ(order int, k []complex128) (*Poly1D, error) {
	return polyResult(LegacyIntegral(p.coeff, order, k))
}
func (p *Poly1D) Repr() (string, error) {
	s, e := np.ArrayRepr(p.coeff)
	if e != nil {
		return "", e
	}
	return "poly1d(" + strings.TrimSuffix(strings.TrimPrefix(s, "array("), ")") + ")", nil
}

var powerText = regexp.MustCompile(`\*\*([0-9]+)`)

func raisePower(s string) string {
	n := 0
	line1, line2, out := "", "", " "
	for _, p := range powerText.FindAllStringSubmatchIndex(s, -1) {
		pow := s[p[2]:p[3]]
		part := s[n:p[0]]
		n = p[1]
		add2 := part + strings.Repeat(" ", len(pow)-1)
		add1 := strings.Repeat(" ", max(utf8.RuneCountInString(part)-1, 0)) + pow
		if utf8.RuneCountInString(line2+add2) > 70 || utf8.RuneCountInString(line1+add1) > 70 {
			out += line1 + "\n" + line2 + "\n "
			line1, line2 = add1, add2
		} else {
			line1 += add1
			line2 += add2
		}
	}
	return out + line1 + "\n" + line2 + s[n:]
}
func (p *Poly1D) Format() (string, error) {
	c, e := p.coeff.Complex128s()
	if e != nil {
		return "", e
	}
	first := 0
	for first < len(c) && c[first] == 0 {
		first++
	}
	c = c[first:]
	s := "0"
	fmtFloat := func(v float64) string { return strconv.FormatFloat(v, 'g', 4, 64) }
	for i, z := range c {
		power := len(c) - 1 - i
		text := ""
		if imag(z) == 0 {
			text = fmtFloat(real(z))
		} else if real(z) == 0 {
			text = fmtFloat(imag(z)) + "j"
		} else {
			text = "(" + fmtFloat(real(z)) + " + " + fmtFloat(imag(z)) + "j)"
		}
		if text == "0" && i > 0 {
			continue
		}
		if power > 0 {
			text += " " + p.variable
			if power > 1 {
				text += "**" + strconv.Itoa(power)
			}
		}
		if i == 0 {
			s = text
		} else if strings.HasPrefix(text, "-") {
			s += " - " + text[1:]
		} else {
			s += " + " + text
		}
	}
	return raisePower(s), nil
}
func (p *Poly1D) String() string {
	s, e := p.Format()
	if e != nil {
		return "<poly1d: " + e.Error() + ">"
	}
	return s
}
