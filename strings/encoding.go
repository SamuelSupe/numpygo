package npstrings

import (
	"github.com/SamuelSupe/numpygo"
	"strings"
	"unicode/utf8"
)

func encoding(name string) string {
	switch strings.ReplaceAll(strings.ToLower(name), "_", "-") {
	case "", "utf-8", "utf8":
		return "utf8"
	case "ascii", "us-ascii":
		return "ascii"
	case "latin1", "latin-1", "iso-8859-1":
		return "latin1"
	}
	return ""
}
func validPolicy(policy string) bool {
	return policy == "" || policy == "strict" || policy == "replace" || policy == "ignore"
}

// Encode supports UTF-8, ASCII and Latin-1 with strict, replace or ignore errors.
func Encode(a *np.Array, codec, policy string) (*np.Array, error) {
	c, e := textInputs(a)
	if e != nil {
		return nil, e
	}
	if c.hasNull() {
		return nil, np.ErrDType
	}
	if c.dtype == np.BytesDType {
		return nil, np.ErrDType
	}
	codec = encoding(codec)
	if codec == "" || !validPolicy(policy) {
		return nil, np.ErrValue
	}
	v := c.data[0]
	if codec != "utf8" {
		limit := rune(255)
		if codec == "ascii" {
			limit = 127
		}
		for i, s := range v {
			var b strings.Builder
			for _, r := range s {
				if r <= limit {
					b.WriteByte(byte(r))
				} else if policy == "replace" {
					b.WriteByte('?')
				} else if policy != "ignore" {
					return nil, np.ErrValue
				}
			}
			v[i] = b.String()
		}
	}
	c.dtype = np.BytesDType
	width := -1
	if len(v) == 0 {
		width = 1
	}
	return result(c, v, width)
}

func badDecode(b *strings.Builder, policy string) error {
	if policy == "replace" {
		b.WriteRune(utf8.RuneError)
		return nil
	}
	if policy == "ignore" {
		return nil
	}
	return np.ErrValue
}
func Decode(a *np.Array, codec, policy string) (*np.Array, error) {
	c, e := textInputs(a)
	if e != nil {
		return nil, e
	}
	if c.dtype != np.BytesDType {
		return nil, np.ErrDType
	}
	codec = encoding(codec)
	if codec == "" || !validPolicy(policy) {
		return nil, np.ErrValue
	}
	v := c.data[0]
	for i, s := range v {
		var b strings.Builder
		for len(s) > 0 {
			if codec == "utf8" {
				r, n := utf8.DecodeRuneInString(s)
				if r == utf8.RuneError && n == 1 {
					if e = badDecode(&b, policy); e != nil {
						return nil, e
					}
					// Consume a valid prefix of an incomplete UTF-8 sequence.
					first := s[0]
					expected := 1
					switch {
					case first >= 0xC2 && first <= 0xDF:
						expected = 2
					case first >= 0xE0 && first <= 0xEF:
						expected = 3
					case first >= 0xF0 && first <= 0xF4:
						expected = 4
					}
					for n < expected && n < len(s) && s[n] >= 0x80 && s[n] <= 0xBF {
						if n == 1 && (first == 0xE0 && s[n] < 0xA0 || first == 0xED && s[n] >= 0xA0 || first == 0xF0 && s[n] < 0x90 || first == 0xF4 && s[n] >= 0x90) {
							break
						}
						n++
					}
				} else {
					b.WriteRune(r)
				}
				s = s[n:]
			} else {
				x := s[0]
				s = s[1:]
				if codec == "ascii" && x >= 128 {
					if e = badDecode(&b, policy); e != nil {
						return nil, e
					}
				} else {
					b.WriteRune(rune(x))
				}
			}
		}
		v[i] = b.String()
	}
	c.dtype = np.UnicodeDType
	width := -1
	if len(v) == 0 {
		width = 1
	}
	return result(c, v, width)
}

// Translate maps Unicode code points; an empty replacement deletes a point.
func Translate(a *np.Array, table map[rune]string) (*np.Array, error) {
	if a.DType() == np.BytesDType {
		return nil, np.ErrDType
	}
	return mapText(a, true, func(s string, _ np.DType) (string, error) {
		var b strings.Builder
		for _, r := range s {
			if v, ok := table[r]; ok {
				b.WriteString(v)
			} else {
				b.WriteRune(r)
			}
		}
		return b.String(), nil
	})
}

// TranslateBytes implements Python's 256-byte table and deletechars form.
// A nil table leaves all retained bytes unchanged.
func TranslateBytes(a *np.Array, table []byte, deletechars string) (*np.Array, error) {
	if a.DType() != np.BytesDType {
		return nil, np.ErrDType
	}
	if table != nil && len(table) != 256 {
		return nil, np.ErrValue
	}
	return mapText(a, true, func(s string, _ np.DType) (string, error) {
		b := make([]byte, 0, len(s))
		for i := 0; i < len(s); i++ {
			x := s[i]
			if strings.IndexByte(deletechars, x) >= 0 {
				continue
			}
			if table != nil {
				x = table[x]
			}
			b = append(b, x)
		}
		return string(b), nil
	})
}
