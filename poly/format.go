package poly

import (
	"math"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"unicode/utf8"

	"github.com/SamuelSupe/numpygo"
)

var asciiStyle atomic.Bool

func init() { asciiStyle.Store(runtime.GOOS == "windows") }
func SetDefaultPrintStyle(style string) error {
	if style != "ascii" && style != "unicode" {
		return np.ErrValue
	}
	asciiStyle.Store(style == "ascii")
	return nil
}
func DefaultPrintStyle() string {
	if asciiStyle.Load() {
		return "ascii"
	}
	return "unicode"
}
func polynomialFloat(x float64, parens bool, opt np.PrintOptions) string {
	if math.IsNaN(x) {
		return opt.NaN
	}
	if math.IsInf(x, 0) {
		return opt.Inf
	}
	p := opt.Precision
	f := np.FloatFormatOptions{Precision: &p, Sign: opt.Sign == "+", Fixed: opt.FloatMode == "fixed", Trim: "0"}
	if f.Fixed {
		f.Trim = "k"
	}
	var text string
	if x != 0 && (math.Abs(x) >= 1e8 || math.Abs(x) < math.Pow10(-p/2)) {
		text, _ = np.FormatFloatScientific(x, f)
		if parens {
			text = "(" + text + ")"
		}
	} else {
		text, _ = np.FormatFloatPositional(x, f)
	}
	return text
}
func translateNumber(s string, alphabet string) string {
	runes := []rune(alphabet)
	var b strings.Builder
	for _, c := range s {
		b.WriteRune(runes[c-'0'])
	}
	return b.String()
}
func (s *Series) Format(style string) (string, error) {
	if style == "" {
		style = DefaultPrintStyle()
	}
	if style != "ascii" && style != "unicode" {
		return "", np.ErrValue
	}
	if s.coeff.DType().IsComplex() || s.hasComplexMap() {
		return "", np.ErrDType
	}
	c, e := s.coeff.Astype(np.Float64)
	if e != nil {
		return "", e
	}
	coefficients, e := c.Float64s()
	if e != nil {
		return "", e
	}
	opt := np.GetPrintOptions()
	off, scale := s.mapping()
	argument := s.symbol
	if off != 0 || scale != 1 {
		switch {
		case scale == 1:
			argument = polynomialFloat(off, false, opt) + " + " + s.symbol
		case off == 0:
			argument = polynomialFloat(scale, false, opt) + s.symbol
		default:
			argument = polynomialFloat(off, false, opt) + " + " + polynomialFloat(scale, false, opt) + s.symbol
		}
		argument = "(" + argument + ")"
	}
	out := polynomialFloat(coefficients[0], false, opt)
	separator := " "
	if style == "unicode" {
		separator = "·"
	}
	for i, v := range coefficients[1:] {
		degree := strconv.Itoa(i + 1)
		term := ""
		if v >= 0 {
			term = "+ " + polynomialFloat(v, true, opt)
		} else {
			term = "- " + polynomialFloat(-v, true, opt)
		}
		term += separator
		if s.Basis == Power {
			term += argument
			if i > 0 {
				if style == "ascii" {
					term += "**" + degree
				} else {
					term += translateNumber(degree, "⁰¹²³⁴⁵⁶⁷⁸⁹")
				}
			}
		} else {
			term += s.Basis.BasisName()
			if style == "ascii" {
				term += "_" + degree
			} else {
				term += translateNumber(degree, "₀₁₂₃₄₅₆₇₈₉")
			}
			term += "(" + argument + ")"
		}
		out += " "
		last := out[strings.LastIndexByte(out, '\n')+1:]
		length := utf8.RuneCountInString(last) + utf8.RuneCountInString(term)
		if i < len(coefficients)-2 {
			length += 2
		}
		if length >= opt.LineWidth {
			term = strings.Replace(term, " ", "\n", 1)
		}
		out += term
	}
	return out, nil
}

// Repr renders a numerical Series with its basis, coefficients, intervals
// and independent variable using the current ndarray print settings.
func (s *Series) Repr() (string, error) {
	d, e := np.FromSlice(s.domain[:], 2)
	if s.domainIsComplex {
		d, e = np.FromSlice(s.complexDomain[:], 2)
	}
	if e != nil {
		return "", e
	}
	w, e := np.FromSlice(s.window[:], 2)
	if s.windowIsComplex {
		w, e = np.FromSlice(s.complexWindow[:], 2)
	}
	if e != nil {
		return "", e
	}
	parts := make([]string, 3)
	for i, a := range []*np.Array{s.coeff, d, w} {
		text, e := np.ArrayRepr(a)
		if e != nil {
			return "", e
		}
		parts[i] = strings.TrimSuffix(strings.TrimPrefix(text, "array("), ")")
	}
	return s.Basis.className() + "(" + parts[0] + ", domain=" + parts[1] +
		", window=" + parts[2] + ", symbol='" + s.symbol + "')", nil
}
func (s *Series) String() string {
	v, e := s.Format("")
	if e != nil {
		return "<polynomial: " + e.Error() + ">"
	}
	return v
}
