// Package char provides NumPy's legacy character-array operations.
package char

import (
	"github.com/SamuelSupe/numpygo"
	npstrings "github.com/SamuelSupe/numpygo/strings"
	"strings"
	"unicode"
	"unicode/utf8"
)

// CharArray is the Go equivalent of the deprecated NumPy chararray subclass.
// Array retains the underlying fixed-width ndarray; scalar access trims
// trailing whitespace, without changing the underlying storage.
type CharArray struct{ *np.Array }

type Options struct {
	DType np.DType // zero preserves a fixed-width input, otherwise bytes
	Width *int     // nil preserves or infers the width
}

func New(shape []int, dtype np.DType, width int) (*CharArray, error) {
	var d np.FlexibleDType
	switch dtype {
	case np.BytesDType:
		d = np.FixedBytes(width)
	case np.UnicodeDType:
		d = np.FixedUnicode(width, nil)
	default:
		return nil, np.ErrDType
	}
	a, e := np.ZerosFlexible(d, shape...)
	if e != nil {
		return nil, e
	}
	return &CharArray{a}, nil
}
func Array(a *np.Array, opt Options) (*CharArray, error)   { return fromArray(a, opt, true) }
func AsArray(a *np.Array, opt Options) (*CharArray, error) { return fromArray(a, opt, false) }
func fromArray(a *np.Array, opt Options, copyData bool) (*CharArray, error) {
	if !a.DType().IsCharacter() && a.DType() != np.StringDType {
		return nil, np.ErrDType
	}
	d := opt.DType
	if d == 0 {
		d = a.DType()
		if d == np.StringDType {
			d = np.BytesDType
		}
	}
	if !d.IsCharacter() {
		return nil, np.ErrDType
	}
	width := 1
	if f, e := a.FlexibleType(); e == nil {
		width = f.Length
	} else {
		v, e := a.CharacterValues()
		if e != nil {
			return nil, e
		}
		for _, s := range v {
			width = max(width, charLen(s, d))
		}
	}
	if opt.Width != nil {
		width = *opt.Width
	}
	f := np.FixedBytes(width)
	if d == np.UnicodeDType {
		f = np.FixedUnicode(width, nil)
	}
	// NumPy 2.5's legacy constructor compares requested character count
	// against the existing byte itemsize. Unicode arrays therefore copy
	// even with their usual unchanged logical width.
	if _, e := a.FlexibleType(); e == nil && a.DType() == d && width == a.ItemSize() && !copyData {
		return &CharArray{a}, nil
	}
	r, e := a.CastFlexible(f)
	if e != nil {
		return nil, e
	}
	return &CharArray{r}, nil
}
func (a *CharArray) AtValue(indices ...int) (any, error) {
	v, e := a.Array.AtValue(indices...)
	if e != nil {
		return nil, e
	}
	switch x := v.(type) {
	case np.BytesScalar:
		return np.BytesScalar(strings.TrimRight(string(x), " \t\r\n\v\f")), nil
	case np.TextScalar:
		return np.TextScalar(strings.TrimRightFunc(string(x), func(r rune) bool { return pySpace(r, false) })), nil
	}
	return nil, np.ErrDType
}

func CompareCharArrays(a, b *np.Array, operator string, rstrip bool) (*np.Array, error) {
	if !a.DType().IsCharacter() || !b.DType().IsCharacter() {
		return nil, np.ErrDType
	}
	if rstrip {
		var e error
		a, e = npstrings.RStrip(a)
		if e != nil {
			return nil, e
		}
		b, e = npstrings.RStrip(b)
		if e != nil {
			return nil, e
		}
	}
	switch operator {
	case "==":
		return np.Equal(a, b)
	case "!=":
		return np.NotEqual(a, b)
	case "<":
		return np.Less(a, b)
	case "<=":
		return np.LessEqual(a, b)
	case ">":
		return np.Greater(a, b)
	case ">=":
		return np.GreaterEqual(a, b)
	default:
		return nil, np.ErrValue
	}
}
func Equal(a, b *np.Array) (*np.Array, error)        { return CompareCharArrays(a, b, "==", true) }
func NotEqual(a, b *np.Array) (*np.Array, error)     { return CompareCharArrays(a, b, "!=", true) }
func Less(a, b *np.Array) (*np.Array, error)         { return CompareCharArrays(a, b, "<", true) }
func LessEqual(a, b *np.Array) (*np.Array, error)    { return CompareCharArrays(a, b, "<=", true) }
func Greater(a, b *np.Array) (*np.Array, error)      { return CompareCharArrays(a, b, ">", true) }
func GreaterEqual(a, b *np.Array) (*np.Array, error) { return CompareCharArrays(a, b, ">=", true) }

func Partition(a, sep *np.Array) (*np.Array, error) {
	p, e := npstrings.Partition(a, sep)
	if e != nil {
		return nil, e
	}
	return np.Stack(p[:], -1)
}
func RPartition(a, sep *np.Array) (*np.Array, error) {
	p, e := npstrings.RPartition(a, sep)
	if e != nil {
		return nil, e
	}
	return np.Stack(p[:], -1)
}

func charLen(s string, d np.DType) int {
	if d == np.BytesDType {
		return len(s)
	}
	return utf8.RuneCountInString(s)
}
func pySpace(r rune, bytes bool) bool {
	if bytes {
		return r == ' ' || r >= 9 && r <= 13
	}
	return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f
}
func stringResult(v []string, d np.DType, shape []int) (*np.Array, error) {
	if d == np.StringDType {
		return np.FromStrings(v, shape...)
	}
	width := 1
	for _, s := range v {
		width = max(width, charLen(s, d))
	}
	if d == np.BytesDType {
		return np.FromByteStrings(v, width, shape...)
	}
	return np.FromUnicodeStrings(v, width, shape...)
}

// Join treats each Unicode string as a sequence of code points, matching
// str.join. Nonempty byte strings are integer sequences in Python and error.
func Join(sep, seq *np.Array) (*np.Array, error) {
	a, e := np.BroadcastArrays(sep, seq)
	if e != nil {
		return nil, e
	}
	s, e := a[0].CharacterValues()
	if e != nil {
		return nil, e
	}
	v, e := a[1].CharacterValues()
	if e != nil {
		return nil, e
	}
	d := seq.DType()
	if sep.DType() == np.BytesDType && d != np.BytesDType || sep.DType() != np.BytesDType && d == np.BytesDType {
		return nil, np.ErrDType
	}
	for i, x := range v {
		if d == np.BytesDType && x != "" {
			return nil, np.ErrDType
		}
		v[i] = strings.Join(strings.Split(x, ""), s[i])
	}
	return stringResult(v, d, a[1].Shape())
}
