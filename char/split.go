package char

import (
	"github.com/SamuelSupe/numpygo"
	"slices"
	"strings"
	"unicode/utf8"
)

// Split and RSplit accept nil for Python whitespace splitting. maxsplit < 0
// has no limit. Each output object slot contains a Go slice of pieces.
func Split(a *np.Array, sep *string, maxsplit int) (*np.Array, error) {
	return split(a, sep, maxsplit, false)
}
func RSplit(a *np.Array, sep *string, maxsplit int) (*np.Array, error) {
	return split(a, sep, maxsplit, true)
}
func split(a *np.Array, sep *string, maxsplit int, right bool) (*np.Array, error) {
	v, e := a.CharacterValues()
	if e != nil {
		return nil, e
	}
	if sep != nil && *sep == "" {
		return nil, np.ErrValue
	}
	out := make([]any, len(v))
	for i, s := range v {
		var parts []string
		if sep != nil {
			if !right || maxsplit < 0 {
				n := -1
				if maxsplit >= 0 && maxsplit < int(^uint(0)>>1) {
					n = maxsplit + 1
				}
				parts = strings.SplitN(s, *sep, n)
			} else {
				parts = []string{}
				for j := 0; j < maxsplit; j++ {
					idx := strings.LastIndex(s, *sep)
					if idx < 0 {
						break
					}
					parts = append(parts, s[idx+len(*sep):])
					s = s[:idx]
				}
				parts = append(parts, s)
				slices.Reverse(parts)
			}
		} else {
			parts = splitSpace(s, a.DType() == np.BytesDType, maxsplit, right)
		}
		out[i] = pieces(parts, a.DType())
	}
	return np.FromObjects(out, a.Shape()...)
}
func pieces(v []string, d np.DType) any {
	if d == np.BytesDType {
		b := make([]np.BytesScalar, len(v))
		for i, s := range v {
			b[i] = np.BytesScalar(s)
		}
		return b
	}
	return v
}
func runeLength(s string, bytes bool) (rune, int) {
	if bytes {
		return rune(s[0]), 1
	}
	return utf8.DecodeRuneInString(s)
}
func lastRune(s string, bytes bool) (rune, int) {
	if bytes {
		return rune(s[len(s)-1]), 1
	}
	return utf8.DecodeLastRuneInString(s)
}
func splitSpace(s string, bytes bool, maxsplit int, right bool) []string {
	out := []string{}
	for len(s) > 0 {
		if right {
			for len(s) > 0 {
				r, n := lastRune(s, bytes)
				if !pySpace(r, bytes) {
					break
				}
				s = s[:len(s)-n]
			}
		} else {
			for len(s) > 0 {
				r, n := runeLength(s, bytes)
				if !pySpace(r, bytes) {
					break
				}
				s = s[n:]
			}
		}
		if s == "" {
			break
		}
		if maxsplit >= 0 && len(out) == maxsplit {
			out = append(out, s)
			break
		}
		if right {
			i := len(s)
			for i > 0 {
				r, n := lastRune(s[:i], bytes)
				if pySpace(r, bytes) {
					break
				}
				i -= n
			}
			out = append(out, s[i:])
			s = s[:i]
		} else {
			i := 0
			for i < len(s) {
				r, n := runeLength(s[i:], bytes)
				if pySpace(r, bytes) {
					break
				}
				i += n
			}
			out = append(out, s[:i])
			s = s[i:]
		}
	}
	if right {
		slices.Reverse(out)
	}
	return out
}
func SplitLines(a *np.Array, keepEnds bool) (*np.Array, error) {
	v, e := a.CharacterValues()
	if e != nil {
		return nil, e
	}
	b := a.DType() == np.BytesDType
	out := make([]any, len(v))
	for i, s := range v {
		parts := []string{}
		start := 0
		for j := 0; j < len(s); {
			r, n := runeLength(s[j:], b)
			line := r == '\n' || r == '\r' || !b && (r == '\v' || r == '\f' || r >= 0x1c && r <= 0x1e || r == 0x85 || r == 0x2028 || r == 0x2029)
			if !line {
				j += n
				continue
			}
			end := j + n
			if r == '\r' && end < len(s) && s[end] == '\n' {
				end++
			}
			stop := j
			if keepEnds {
				stop = end
			}
			parts = append(parts, s[start:stop])
			j = end
			start = end
		}
		if start < len(s) {
			parts = append(parts, s[start:])
		}
		out[i] = pieces(parts, a.DType())
	}
	return np.FromObjects(out, a.Shape()...)
}
