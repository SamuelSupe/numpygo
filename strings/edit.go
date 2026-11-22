package npstrings

import (
	"github.com/SamuelSupe/numpygo"
	"strings"
)

// Strip methods accept zero or one explicit set of characters.
func strip(a *np.Array, left, right bool, chars ...string) (*np.Array, error) {
	if len(chars) > 1 {
		return nil, np.ErrValue
	}
	if len(chars) == 1 {
		arg, e := scalarText(chars[0], a.DType())
		if e != nil {
			return nil, e
		}
		return stripArrays(a, arg, left, right)
	}
	return mapTextNullable(a, true, true, func(s string, d np.DType) (string, error) { return stripText(s, d, left, right, nil), nil })
}
func stripText(s string, d np.DType, left, right bool, chars *string) string {
	r := units(s, d)
	predicate := func(x rune) bool { return isSpace(x, d == np.BytesDType) }
	if chars != nil {
		set := units(*chars, d)
		predicate = func(x rune) bool {
			for _, y := range set {
				if x == y {
					return true
				}
			}
			return false
		}
	}
	i, j := 0, len(r)
	if left {
		for i < j && predicate(r[i]) {
			i++
		}
	}
	if right {
		for i < j && predicate(r[j-1]) {
			j--
		}
	}
	return join(r[i:j], d)
}
func stripArrays(a, chars *np.Array, left, right bool) (*np.Array, error) {
	c, e := textInputs(a, chars)
	if e != nil {
		return nil, e
	}
	v := c.data[0]
	c.missing = make([]bool, len(v))
	for i, s := range v {
		if c.null[0][i] || c.null[1][i] {
			if !c.nanSentinel() {
				return nil, np.ErrValue
			}
			c.missing[i] = true
			continue
		}
		v[i] = stripText(s, c.dtype, left, right, &c.data[1][i])
	}
	return result(c, v, c.width)
}
func StripWithChars(a, chars *np.Array) (*np.Array, error) { return stripArrays(a, chars, true, true) }
func LStripWithChars(a, chars *np.Array) (*np.Array, error) {
	return stripArrays(a, chars, true, false)
}
func RStripWithChars(a, chars *np.Array) (*np.Array, error) {
	return stripArrays(a, chars, false, true)
}
func Strip(a *np.Array, chars ...string) (*np.Array, error)  { return strip(a, true, true, chars...) }
func LStrip(a *np.Array, chars ...string) (*np.Array, error) { return strip(a, true, false, chars...) }
func RStrip(a *np.Array, chars ...string) (*np.Array, error) { return strip(a, false, true, chars...) }

func justify(a *np.Array, width int, mode string, fill []string) (*np.Array, error) {
	if len(fill) > 1 {
		return nil, np.ErrValue
	}
	ch := " "
	if len(fill) == 1 {
		ch = fill[0]
	}
	f, e := scalarText(ch, a.DType())
	if e != nil {
		return nil, e
	}
	return justifyArrays(a, np.ScalarOf(int64(width)), f, mode)
}
func justifyArrays(a, width, fill *np.Array, mode string) (*np.Array, error) {
	c, opts, e := textWithInts([]*np.Array{a, fill}, width)
	if e != nil {
		return nil, e
	}
	v := c.data[0]
	c.missing = make([]bool, len(v))
	for i, s := range v {
		if c.null[0][i] || c.null[1][i] {
			if !c.nanSentinel() {
				if mode == "zero" {
					return nil, np.ErrDType
				}
				return nil, np.ErrValue
			}
			c.missing[i] = true
			continue
		}
		v[i], e = padText(s, c.data[1][i], opts[0][i], c.dtype, mode)
		if e != nil {
			return nil, e
		}
	}
	return result(c, v, -1)
}
func padText(s, ch string, width int, d np.DType, mode string) (string, error) {
	if length(ch, d) != 1 {
		return "", np.ErrValue
	}
	n := length(s, d)
	if width <= n {
		return s, nil
	}
	p := width - n
	if p > (int(^uint(0)>>1)-len(s))/len(ch) {
		return "", np.ErrShape
	}
	switch mode {
	case "left":
		return s + strings.Repeat(ch, p), nil
	case "right":
		return strings.Repeat(ch, p) + s, nil
	case "center":
		// Python centers odd padding according to the target width parity.
		l := p/2 + (p & width & 1)
		return strings.Repeat(ch, l) + s + strings.Repeat(ch, p-l), nil
	case "zero":
		pad := strings.Repeat("0", p)
		if len(s) > 0 && (s[0] == '+' || s[0] == '-') {
			return s[:1] + pad + s[1:], nil
		}
		return pad + s, nil
	}
	return "", np.ErrValue
}
func CenterWithWidths(a, width, fill *np.Array) (*np.Array, error) {
	return justifyArrays(a, width, fill, "center")
}
func LJustWithWidths(a, width, fill *np.Array) (*np.Array, error) {
	return justifyArrays(a, width, fill, "left")
}
func RJustWithWidths(a, width, fill *np.Array) (*np.Array, error) {
	return justifyArrays(a, width, fill, "right")
}
func ZFillWithWidths(a, width *np.Array) (*np.Array, error) {
	f, e := scalarText(" ", a.DType())
	if e != nil {
		return nil, e
	}
	return justifyArrays(a, width, f, "zero")
}
func Center(a *np.Array, width int, fill ...string) (*np.Array, error) {
	return justify(a, width, "center", fill)
}
func LJust(a *np.Array, width int, fill ...string) (*np.Array, error) {
	return justify(a, width, "left", fill)
}
func RJust(a *np.Array, width int, fill ...string) (*np.Array, error) {
	return justify(a, width, "right", fill)
}
func ZFill(a *np.Array, width int) (*np.Array, error) { return justify(a, width, "zero", nil) }

func ExpandTabs(a *np.Array, tabsize int) (*np.Array, error) {
	return ExpandTabsWithSizes(a, np.ScalarOf(int64(tabsize)))
}
func ExpandTabsWithSizes(a, sizes *np.Array) (*np.Array, error) {
	c, opts, err := textWithInts([]*np.Array{a}, sizes)
	if err != nil {
		return nil, err
	}
	if c.hasNull() {
		return nil, np.ErrValue
	}
	v := c.data[0]
	width := 1
	for i, s := range v {
		tabsize := opts[0][i]
		var buf strings.Builder
		col, capacity := 0, 0
		for _, r := range units(s, c.dtype) {
			if r == '\t' {
				if tabsize > 0 {
					n := tabsize - col%tabsize
					buf.WriteString(strings.Repeat(" ", n))
					capacity += n
					col += n
				}
			} else {
				if c.dtype == np.BytesDType {
					buf.WriteByte(byte(r))
					capacity++
				} else {
					buf.WriteRune(r)
					capacity += 4
				}
				col++
				if r == '\n' || r == '\r' {
					col = 0
				}
			}
		}
		v[i] = buf.String()
		width = max(width, capacity)
	}
	if len(v) == 0 && c.dtype != np.StringDType {
		return nil, np.ErrValue
	}
	return result(c, v, width)
}

// Slice applies a Python slice to each string rather than to array axes.
func Slice(a *np.Array, spec np.Slice) (*np.Array, error) {
	if spec.Step != nil && *spec.Step == 0 {
		return nil, np.ErrValue
	}
	return mapText(a, true, func(s string, d np.DType) (string, error) { return sliceText(s, d, spec) })
}
func SliceWithBounds(a, start, stop, step *np.Array) (*np.Array, error) {
	c, opts, e := textWithInts([]*np.Array{a}, start, stop, step)
	if e != nil {
		return nil, e
	}
	if c.hasNull() {
		return nil, np.ErrDType
	}
	v := c.data[0]
	for i, s := range v {
		v[i], e = sliceText(s, c.dtype, np.Slice{Start: np.Int(opts[0][i]), Stop: np.Int(opts[1][i]), Step: np.Int(opts[2][i])})
		if e != nil {
			return nil, e
		}
	}
	return result(c, v, c.width)
}
func sliceText(s string, d np.DType, spec np.Slice) (string, error) {
	step := 1
	if spec.Step != nil {
		step = *spec.Step
	}
	if step == 0 {
		return "", np.ErrValue
	}
	r := units(s, d)
	lo, hi, start, stop := 0, len(r), 0, len(r)
	if step < 0 {
		lo, hi, start, stop = -1, len(r)-1, len(r)-1, -1
	}
	if spec.Start != nil {
		start = *spec.Start
		if start < 0 {
			start += len(r)
		}
		start = min(hi, max(lo, start))
	}
	if spec.Stop != nil {
		stop = *spec.Stop
		if stop < 0 {
			stop += len(r)
		}
		stop = min(hi, max(lo, stop))
	}
	out := make([]rune, 0, len(r))
	for i := start; step > 0 && i < stop || step < 0 && i > stop; {
		out = append(out, r[i])
		if step > 0 && step >= stop-i || step < 0 && step <= stop-i {
			break
		}
		i += step
	}
	return join(out, d), nil
}
