package np

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
)

type PrintOptions struct {
	Precision, Threshold, EdgeItems, LineWidth int
	Suppress                                   bool
	Sign, FloatMode, NaN, Inf                  string
	Formatter                                  map[string]func(any) string
	OverrideRepr                               func(*Array) string
}

func DefaultPrintOptions() PrintOptions {
	return PrintOptions{Precision: 8, Threshold: 1000, EdgeItems: 3, LineWidth: 75, Sign: "-", FloatMode: "maxprec", NaN: "nan", Inf: "inf"}
}

var printMu sync.RWMutex
var printSettings = DefaultPrintOptions()

func cloneOptions(p PrintOptions) PrintOptions {
	if p.Formatter != nil {
		m := make(map[string]func(any) string, len(p.Formatter))
		for k, v := range p.Formatter {
			m[k] = v
		}
		p.Formatter = m
	}
	return p
}
func validatePrint(p PrintOptions) error {
	if p.Precision < 0 || p.Threshold < 0 || p.EdgeItems < 0 || p.LineWidth < 1 {
		return ErrValue
	}
	switch p.Sign {
	case "-", "+", " ":
	default:
		return ErrValue
	}
	switch p.FloatMode {
	case "fixed", "unique", "maxprec", "maxprec_equal":
	default:
		return ErrValue
	}
	return nil
}
func GetPrintOptions() PrintOptions {
	printMu.RLock()
	defer printMu.RUnlock()
	return cloneOptions(printSettings)
}
func SetPrintOptions(p PrintOptions) error {
	if e := validatePrint(p); e != nil {
		return e
	}
	printMu.Lock()
	printSettings = cloneOptions(p)
	printMu.Unlock()
	return nil
}

type Formatter struct{ options PrintOptions }

func NewFormatter(p PrintOptions) (*Formatter, error) {
	if e := validatePrint(p); e != nil {
		return nil, e
	}
	return &Formatter{cloneOptions(p)}, nil
}
func (f *Formatter) GetPrintOptions() PrintOptions { return cloneOptions(f.options) }

// WithPrintOptions provides scoped formatting without changing process-wide
// defaults used by other goroutines.
func WithPrintOptions(p PrintOptions, fn func(*Formatter) error) error {
	if fn == nil {
		return ErrValue
	}
	f, e := NewFormatter(p)
	if e != nil {
		return e
	}
	return fn(f)
}

type ArrayStringOptions struct{ Separator, Prefix, Suffix string }

func paddedLeft(s string, n int) string { return strings.Repeat(" ", max(0, n-len(s))) + s }
func paddedRight(s string, n int, fill string) string {
	return s + strings.Repeat(fill, max(0, n-len(s)))
}
func visibleIndices(size, edge int, summary bool) []int {
	if summary && 2*edge < size {
		out := make([]int, 0, 2*edge+1)
		for i := 0; i < edge; i++ {
			out = append(out, i)
		}
		out = append(out, -1)
		for i := size - edge; i < size; i++ {
			out = append(out, i)
		}
		if edge == 0 {
			out = append(out, size-1)
		}
		return out
	}
	out := make([]int, size)
	for i := range out {
		out[i] = i
	}
	return out
}
func finiteParts(value float64, d DType, scientific bool, p PrintOptions, sign string) (string, string, string) {
	form := byte('f')
	if scientific {
		form = 'e'
	}
	bits := 64
	if d == Float32 || d == Complex64 || d == Float16 {
		bits = 32
	}
	s := strconv.FormatFloat(value, form, -1, bits)
	if d == Float16 {
		// Find the shortest decimal that returns the same binary16 value.
		for k := 0; k <= 8; k++ {
			test := strconv.FormatFloat(value, form, k, 64)
			v, e := strconv.ParseFloat(test, 64)
			if e == nil && HalfFromFloat64(v) == HalfFromFloat64(value) {
				s = test
				break
			}
		}
	}
	if p.FloatMode != "unique" {
		m := strings.SplitN(s, "e", 2)[0]
		n := 0
		if i := strings.IndexByte(m, '.'); i >= 0 {
			n = len(m) - i - 1
		}
		if p.FloatMode == "fixed" || n > p.Precision {
			s = strconv.FormatFloat(value, form, p.Precision, bits)
		}
	}
	mantissa, exponent := s, ""
	if scientific {
		j := strings.IndexByte(s, 'e')
		mantissa, exponent = s[:j], s[j+1:]
	}
	if sign == "+" && !strings.HasPrefix(mantissa, "-") {
		mantissa = "+" + mantissa
	}
	ip, frac := mantissa, ""
	if i := strings.IndexByte(mantissa, '.'); i >= 0 {
		ip, frac = mantissa[:i], mantissa[i+1:]
	}
	if p.FloatMode != "fixed" {
		frac = strings.TrimRight(frac, "0")
	}
	return ip, frac, exponent
}
func floatFormatter(data []float64, d DType, p PrintOptions, sign string) func(float64) string {
	minVal, maxVal := math.Inf(1), 0.
	for _, x := range data {
		if math.IsInf(x, 0) || math.IsNaN(x) || x == 0 {
			continue
		}
		abs := math.Abs(x)
		minVal = math.Min(minVal, abs)
		maxVal = math.Max(maxVal, abs)
	}
	precision := 15
	if d == Float32 || d == Complex64 {
		precision = 6
	}
	if d == Float16 {
		precision = 3
	}
	scientific := maxVal >= math.Pow10(min(8, precision)) || maxVal > 0 && !p.Suppress && (minVal < 1e-4 || maxVal/minVal > 1000)
	left, right, expWidth := 0, 0, 0
	hasNegative := false
	for _, x := range data {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			continue
		}
		if math.Signbit(x) {
			hasNegative = true
		}
		ip, frac, exp := finiteParts(x, d, scientific, p, sign)
		left = max(left, len(ip))
		right = max(right, len(frac))
		if scientific {
			expWidth = max(expWidth, len(exp))
		}
	}
	if sign == " " && !hasNegative {
		left++
	}
	width := left + 1 + right
	if scientific {
		width += expWidth + 1
	}
	for _, x := range data {
		if math.IsNaN(x) {
			s := p.NaN
			if sign == "+" {
				s = "+" + s
			}
			width = max(width, len(s))
		}
		if math.IsInf(x, 0) {
			s := p.Inf
			if math.Signbit(x) || sign == "+" {
				s = "-" + s
			}
			width = max(width, len(s))
		}
	}
	formatter := func(x float64) string {
		if math.IsNaN(x) {
			s := p.NaN
			if sign == "+" {
				s = "+" + s
			}
			return paddedLeft(s, width)
		}
		if math.IsInf(x, 0) {
			s := p.Inf
			if math.Signbit(x) {
				s = "-" + s
			} else if sign == "+" {
				s = "+" + s
			}
			return paddedLeft(s, width)
		}
		ip, frac, exp := finiteParts(x, d, scientific, p, sign)
		ip = paddedLeft(ip, left)
		if scientific || p.FloatMode == "fixed" || p.FloatMode == "maxprec_equal" {
			frac = paddedRight(frac, right, "0")
		} else {
			frac = paddedRight(frac, right, " ")
		}
		if scientific {
			eSign, digits := exp[:1], exp[1:]
			exp = eSign + strings.Repeat("0", max(0, expWidth-1-len(digits))) + digits
			return paddedLeft(ip+"."+frac+"e"+exp, width)
		}
		return paddedLeft(ip+"."+frac, width)
	}
	return formatter
}
func scalarFormat(d DType, values []any, p PrintOptions, rank int) (func(any) string, error) {
	kind := ""
	switch {
	case d == Bool:
		kind = "bool"
	case d.IsInteger():
		kind = "int_kind"
	case d.IsFloat():
		kind = "float_kind"
	case d.IsComplex():
		kind = "complex_kind"
	}
	if p.Formatter != nil {
		specific := ""
		switch {
		case d.IsInteger():
			specific = "int"
		case d.IsFloat():
			specific = "float"
		case d.IsComplex():
			specific = "complexfloat"
		}
		for _, k := range []string{d.String(), specific, kind, "all"} {
			if f := p.Formatter[k]; f != nil {
				return f, nil
			}
		}
	}
	if d == StringDType {
		return stringScalarRepr, nil
	}
	if d.IsFlexible() {
		return flexibleRepr, nil
	}
	if d == Bool {
		return func(v any) string {
			if v.(bool) {
				if rank > 0 {
					return " True"
				}
				return "True"
			}
			return "False"
		}, nil
	}
	if d.IsInteger() {
		text := func(v any) string {
			var s string
			if d.IsSigned() {
				s = strconv.FormatInt(int64(elementFrom(v).i), 10)
			} else {
				s = strconv.FormatUint(elementFrom(v).i, 10)
			}
			if !strings.HasPrefix(s, "-") && p.Sign == "+" {
				s = "+" + s
			}
			return s
		}
		width := 0
		negative := false
		for _, v := range values {
			s := text(v)
			width = max(width, len(s))
			negative = negative || strings.HasPrefix(s, "-")
		}
		if p.Sign == " " && !negative {
			width++
		}
		return func(v any) string { return paddedLeft(text(v), width) }, nil
	}
	if d.IsFloat() || d.IsComplex() {
		re, im := make([]float64, len(values)), make([]float64, len(values))
		for i, v := range values {
			z := elementFrom(v).z
			re[i], im[i] = real(z), imag(z)
		}
		rf := floatFormatter(re, d, p, p.Sign)
		if d.IsFloat() {
			return func(v any) string { return rf(real(elementFrom(v).z)) }, nil
		}
		iF := floatFormatter(im, d, p, "+")
		return func(v any) string {
			z := elementFrom(v).z
			i := iF(imag(z))
			trim := strings.TrimRight(i, " ")
			return rf(real(z)) + trim + "j" + i[len(trim):]
		}, nil
	}
	return nil, ErrDType
}
func (f *Formatter) Array2String(a *Array, options ...ArrayStringOptions) (string, error) {
	if a.DType() == 0 || len(options) > 1 {
		return "", ErrValue
	}
	opt := ArrayStringOptions{Separator: " "}
	if len(options) == 1 {
		opt = options[0]
		if opt.Separator == "" {
			opt.Separator = " "
		}
	}
	if a.Size() == 0 {
		return "[]", nil
	}
	p := f.options
	summary := a.Size() > p.Threshold
	shape := a.Shape()
	coord := make([]int, len(shape))
	var values []any
	var collect func(int)
	collect = func(axis int) {
		if axis == len(shape) {
			v, _ := a.AtValue(coord...)
			values = append(values, v)
			return
		}
		for _, i := range visibleIndices(shape[axis], p.EdgeItems, summary) {
			if i < 0 {
				continue
			}
			coord[axis] = i
			collect(axis + 1)
		}
	}
	collect(0)
	format, e := scalarFormat(a.DType(), values, p, len(shape))
	if e != nil {
		return "", e
	}
	var render func(int, int, int) string
	render = func(axis, indent, width int) string {
		if axis == len(shape) {
			v, _ := a.AtValue(coord...)
			return format(v)
		}
		pad := strings.Repeat(" ", indent)
		var words []string
		for _, i := range visibleIndices(shape[axis], p.EdgeItems, summary) {
			if i < 0 {
				words = append(words, "...")
				continue
			}
			coord[axis] = i
			words = append(words, render(axis+1, indent+1, width-1))
		}
		text := ""
		if axis == len(shape)-1 {
			line := pad
			bound := width - max(len(strings.TrimRight(opt.Separator, " ")), 1)
			for i, w := range words {
				if len(line) > len(pad) && len(line)+len(w) > bound {
					text += strings.TrimRight(line, " ") + "\n"
					line = pad
				}
				line += w
				if i < len(words)-1 {
					line += opt.Separator
				}
			}
			text += line
		} else {
			sep := strings.TrimRight(opt.Separator, " ") + strings.Repeat("\n", len(shape)-axis-1)
			for i, w := range words {
				if i > 0 {
					text += sep
				}
				text += pad + w
			}
		}
		return "[" + strings.TrimPrefix(text, pad) + "]"
	}
	return render(0, len(opt.Prefix)+1, p.LineWidth-len(opt.Suffix)), nil
}
func tupleShape(v []int) string {
	parts := make([]string, len(v))
	for i, n := range v {
		parts[i] = strconv.Itoa(n)
	}
	if len(v) == 1 {
		return "(" + parts[0] + ",)"
	}
	return "(" + strings.Join(parts, ", ") + ")"
}
func (f *Formatter) ArrayRepr(a *Array) (string, error) {
	if a.DType() == 0 {
		return "", ErrDType
	}
	if f.options.OverrideRepr != nil {
		return f.options.OverrideRepr(a), nil
	}
	s, e := f.Array2String(a, ArrayStringOptions{Separator: ", ", Prefix: "array(", Suffix: ")"})
	if e != nil {
		return "", e
	}
	extra := []string{}
	if a.Size() == 0 && !(a.NDim() == 1 && a.Shape()[0] == 0) || a.Size() > f.options.Threshold {
		extra = append(extra, "shape="+tupleShape(a.Shape()))
	}
	d := a.DType()
	if a.Size() == 0 || d != Bool && d != Int64 && d != Float64 && d != Complex128 {
		if d.IsFlexible() {
			extra = append(extra, "dtype='"+a.data.flex.String()+"'")
		} else if d == StringDType {
			extra = append(extra, "dtype="+a.stringDescriptor().String())
		} else {
			extra = append(extra, "dtype="+d.String())
		}
	}
	out := "array(" + s
	if len(extra) > 0 {
		out += ","
		rest := strings.Join(extra, ", ") + ")"
		last := len(out) - strings.LastIndexByte(out, '\n') - 1
		if last+1+len(rest) > f.options.LineWidth {
			return out + "\n      " + rest, nil
		}
		return out + " " + rest, nil
	}
	return out + ")", nil
}
func (f *Formatter) ArrayStr(a *Array) (string, error) {
	if a.DType() == 0 {
		return "", ErrDType
	}
	if a.NDim() == 0 {
		v, e := a.AtValue()
		if e != nil {
			return "", e
		}
		d := a.DType()
		if d == StringDType {
			return stringValue(v, false)
		}
		if d.IsFlexible() {
			switch d {
			case BytesDType:
				return "np.bytes_(" + flexibleRepr(v) + ")", nil
			case UnicodeDType:
				return string(v.(TextScalar)), nil
			default:
				return flexibleRepr(v), nil
			}
		}
		if d.IsFloat() {
			v := real(elementFrom(v).z)
			bits := 64
			if d != Float64 {
				bits = 32
			}
			s := strconv.FormatFloat(v, 'g', -1, bits)
			if math.IsNaN(v) {
				return "nan", nil
			}
			if math.IsInf(v, 1) {
				return "inf", nil
			}
			if math.IsInf(v, -1) {
				return "-inf", nil
			}
			if !strings.ContainsAny(s, ".e") {
				s += ".0"
			}
			return s, nil
		}
		if d.IsComplex() {
			z := elementFrom(v).z
			s := strings.ReplaceAll(fmt.Sprint(z), "i", "j")
			if real(z) == 0 && !math.Signbit(real(z)) {
				return strconv.FormatFloat(imag(z), 'g', -1, 64) + "j", nil
			}
			return s, nil
		}
		format, e := scalarFormat(d, []any{v}, DefaultPrintOptions(), 0)
		if e != nil {
			return "", e
		}
		return strings.TrimSpace(format(v)), nil
	}
	return f.Array2String(a)
}
func Array2String(a *Array, options ...ArrayStringOptions) (string, error) {
	f, _ := NewFormatter(GetPrintOptions())
	return f.Array2String(a, options...)
}
func ArrayRepr(a *Array) (string, error) {
	f, _ := NewFormatter(GetPrintOptions())
	return f.ArrayRepr(a)
}
func ArrayStr(a *Array) (string, error) {
	f, _ := NewFormatter(GetPrintOptions())
	return f.ArrayStr(a)
}
