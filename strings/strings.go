// Package npstrings implements elementwise NumPy character operations on
// fixed bytes, fixed Unicode, and variable UTF-8 ndarrays.
package npstrings

import (
	"github.com/SamuelSupe/numpygo"
	"math"
	"strings"
	"unicode/utf8"
)

func Add(a, b *np.Array) (*np.Array, error)          { return np.Add(a, b) }
func Multiply(a, count *np.Array) (*np.Array, error) { return np.RepeatStrings(a, count) }
func Equal(a, b *np.Array) (*np.Array, error)        { return np.Equal(a, b) }
func NotEqual(a, b *np.Array) (*np.Array, error)     { return np.NotEqual(a, b) }
func Less(a, b *np.Array) (*np.Array, error)         { return np.Less(a, b) }
func LessEqual(a, b *np.Array) (*np.Array, error)    { return np.LessEqual(a, b) }
func Greater(a, b *np.Array) (*np.Array, error)      { return np.Greater(a, b) }
func GreaterEqual(a, b *np.Array) (*np.Array, error) { return np.GreaterEqual(a, b) }

type characters struct {
	dtype   np.DType
	shape   []int
	width   int
	data    [][]string
	desc    np.StringDescriptor
	null    [][]bool // non-string missing slots; string sentinels have text
	missing []bool   // missing slots in the result
}

func textInputs(arrays ...*np.Array) (characters, error) {
	var c characters
	views, err := np.BroadcastArrays(arrays...)
	if err != nil {
		return c, err
	}
	c.dtype = arrays[0].DType()
	for _, a := range arrays {
		d := a.DType()
		if !d.IsCharacter() && d != np.StringDType {
			return c, np.ErrDType
		}
		if d != c.dtype {
			if d == np.BytesDType || c.dtype == np.BytesDType {
				return c, np.ErrDType
			}
			c.dtype = np.StringDType
		}
	}
	if f, e := arrays[0].FlexibleType(); e == nil {
		c.width = f.Length
	}
	c.shape = views[0].Shape()
	c.desc, err = np.PromoteStringDescriptors(arrays...)
	if err != nil {
		return c, err
	}
	for _, a := range views {
		mask := make([]bool, a.Size())
		var v []string
		if a.DType() == np.StringDType {
			values, missing, e := a.StringValues()
			if e != nil {
				return c, e
			}
			v = make([]string, len(values))
			for i, x := range values {
				if s, ok := x.(string); ok {
					v[i] = s
				} else if missing[i] {
					mask[i] = true
				} else {
					return c, np.ErrDType
				}
			}
		} else {
			v, err = a.CharacterValues()
			if err != nil {
				return c, err
			}
		}
		c.data = append(c.data, v)
		c.null = append(c.null, mask)
	}
	return c, nil
}
func (c characters) hasNull() bool {
	for _, v := range c.null {
		for _, m := range v {
			if m {
				return true
			}
		}
	}
	return false
}
func (c characters) nanSentinel() bool {
	switch v := c.desc.NA.(type) {
	case float64:
		return math.IsNaN(v)
	case float32:
		return math.IsNaN(float64(v))
	}
	return false
}

func length(s string, d np.DType) int {
	if d == np.BytesDType {
		return len(s)
	}
	return utf8.RuneCountInString(s)
}

// Use rune slots for Unicode and byte slots for bytes, even for invalid UTF-8.
func units(s string, d np.DType) []rune {
	if d != np.BytesDType {
		return []rune(s)
	}
	r := make([]rune, len(s))
	for i := range r {
		r[i] = rune(s[i])
	}
	return r
}
func join(r []rune, d np.DType) string {
	if d != np.BytesDType {
		return string(r)
	}
	b := make([]byte, len(r))
	for i, x := range r {
		b[i] = byte(x)
	}
	return string(b)
}

func result(c characters, v []string, width int) (*np.Array, error) {
	if c.dtype == np.StringDType {
		return np.FromStringSlots(v, c.missing, c.desc, c.shape...)
	}
	if width < 0 {
		if len(v) == 0 {
			return nil, np.ErrValue
		}
		width = 1
		for _, s := range v {
			width = max(width, length(s, c.dtype))
		}
	}
	if c.dtype == np.BytesDType {
		return np.FromByteStrings(v, width, c.shape...)
	}
	return np.FromUnicodeStrings(v, width, c.shape...)
}

func mapText(a *np.Array, preserve bool, fn func(string, np.DType) (string, error)) (*np.Array, error) {
	return mapTextNullable(a, preserve, false, fn)
}
func mapTextNullable(a *np.Array, preserve, allowNaN bool, fn func(string, np.DType) (string, error)) (*np.Array, error) {
	c, err := textInputs(a)
	if err != nil {
		return nil, err
	}
	v := c.data[0]
	c.missing = make([]bool, len(v))
	for i, s := range v {
		if c.null[0][i] {
			if !allowNaN {
				return nil, np.ErrDType
			}
			if !c.nanSentinel() {
				return nil, np.ErrValue
			}
			c.missing[i] = true
			continue
		}
		v[i], err = fn(s, c.dtype)
		if err != nil {
			return nil, err
		}
		if !allowNaN && c.desc.HasNA {
			if na, ok := c.desc.NA.(string); ok && v[i] == na {
				c.missing[i] = true
			}
		}
	}
	width := -1
	if preserve {
		width = c.width
	}
	return result(c, v, width)
}

func StrLen(a *np.Array) (*np.Array, error) {
	c, err := textInputs(a)
	if err != nil {
		return nil, err
	}
	if c.hasNull() {
		return nil, np.ErrValue
	}
	v := make([]int64, len(c.data[0]))
	for i, s := range c.data[0] {
		v[i] = int64(length(s, c.dtype))
	}
	return np.WrapSlice(v, c.shape...)
}

// Replace permits broadcasting old/new and a scalar count (negative means all).
func Replace(a, old, newText *np.Array, count int) (*np.Array, error) {
	return ReplaceWithCounts(a, old, newText, np.ScalarOf(int64(count)))
}
func ReplaceWithCounts(a, old, newText, counts *np.Array) (*np.Array, error) {
	c, opts, err := textWithInts([]*np.Array{a, old, newText}, counts)
	if err != nil {
		return nil, err
	}
	v := c.data[0]
	c.missing = make([]bool, len(v))
	for i, s := range v {
		count := opts[0][i]
		if c.null[0][i] || c.null[1][i] || c.null[2][i] {
			if !c.nanSentinel() {
				return nil, np.ErrValue
			}
			c.missing[i] = true
			continue
		}
		if c.dtype == np.BytesDType && c.data[1][i] == "" {
			// strings.Replace inserts at UTF-8 boundaries; bytes require every byte position.
			var b strings.Builder
			n := count
			for k := 0; k <= len(s); k++ {
				if n != 0 {
					b.WriteString(c.data[2][i])
					if n > 0 {
						n--
					}
				}
				if k < len(s) {
					b.WriteByte(s[k])
				}
			}
			v[i] = b.String()
		} else {
			v[i] = strings.Replace(s, c.data[1][i], c.data[2][i], count)
		}
	}
	return result(c, v, -1)
}

// Partition and RPartition return the three independent output arrays.
func Partition(a, sep *np.Array) ([3]*np.Array, error)  { return partition(a, sep, false) }
func RPartition(a, sep *np.Array) ([3]*np.Array, error) { return partition(a, sep, true) }
func partition(a, sep *np.Array, right bool) ([3]*np.Array, error) {
	var out [3]*np.Array
	c, err := textInputs(a, sep)
	if err != nil {
		return out, err
	}
	if c.dtype == np.StringDType {
		return out, np.ErrDType
	}
	if len(c.data[0]) == 0 {
		return out, np.ErrValue
	}
	var parts [3][]string
	var widths [3]int
	found := false
	for i := range parts {
		parts[i] = make([]string, len(c.data[0]))
	}
	for i, s := range c.data[0] {
		t := c.data[1][i]
		if t == "" {
			return out, np.ErrValue
		}
		j := strings.Index(s, t)
		if right {
			j = strings.LastIndex(s, t)
		}
		widths[1] = max(widths[1], length(t, c.dtype))
		if j >= 0 {
			found = true
			parts[0][i], parts[1][i], parts[2][i] = s[:j], t, s[j+len(t):]
		} else if right {
			parts[2][i] = s
		} else {
			parts[0][i] = s
		}
		for k := range parts {
			widths[k] = max(widths[k], length(parts[k][i], c.dtype))
		}
	}
	if !found {
		widths[1] = 1
	}
	for i := range out {
		out[i], err = result(c, parts[i], widths[i])
		if err != nil {
			return out, err
		}
	}
	return out, nil
}
