package npstrings

import (
	"github.com/SamuelSupe/numpygo"
	"slices"
	"strings"
)

type runeRange struct{ lo, hi rune }

func contains(ranges []runeRange, r rune) bool {
	i, _ := slices.BinarySearchFunc(ranges, r, func(p runeRange, r rune) int {
		if p.hi < r {
			return -1
		}
		if p.lo > r {
			return 1
		}
		return 0
	})
	return i < len(ranges) && ranges[i].lo <= r && r <= ranges[i].hi
}
func isLower(r rune, b bool) bool {
	if b {
		return 'a' <= r && r <= 'z'
	}
	return contains(lowerRanges[:], r)
}
func isUpper(r rune, b bool) bool {
	if b {
		return 'A' <= r && r <= 'Z'
	}
	return contains(upperRanges[:], r)
}
func isTitle(r rune, b bool) bool {
	if b {
		return isUpper(r, b)
	}
	return contains(titleRanges[:], r)
}
func isCased(r rune, b bool) bool { return isLower(r, b) || isUpper(r, b) || isTitle(r, b) }
func isAlpha(r rune, b bool) bool {
	if b {
		return isCased(r, true)
	}
	return contains(alphaRanges[:], r)
}
func isDigit(r rune, b bool) bool {
	if b {
		return '0' <= r && r <= '9'
	}
	return contains(digitRanges[:], r)
}
func isSpace(r rune, b bool) bool {
	if b {
		return r == ' ' || r >= 9 && r <= 13
	}
	return contains(spaceRanges[:], r)
}
func isNumeric(r rune, b bool) bool {
	if b {
		return isDigit(r, true)
	}
	return contains(numericRanges[:], r)
}
func isDecimal(r rune, b bool) bool { return contains(decimalRanges[:], r) }

func classify(a *np.Array, name string) (*np.Array, error) {
	c, err := textInputs(a)
	if err != nil {
		return nil, err
	}
	b := c.dtype == np.BytesDType
	if b && (name == "decimal" || name == "numeric") {
		return nil, np.ErrDType
	}
	v := make([]bool, len(c.data[0]))
	for i, s := range c.data[0] {
		if c.null[0][i] {
			if !c.nanSentinel() {
				return nil, np.ErrValue
			}
			continue
		}
		r := units(s, c.dtype)
		ok, seen, prev := len(r) > 0, false, false
		for _, x := range r {
			switch name {
			case "alpha":
				ok = ok && isAlpha(x, b)
			case "digit":
				ok = ok && isDigit(x, b)
			case "space":
				ok = ok && isSpace(x, b)
			case "alnum":
				ok = ok && (isAlpha(x, b) || isNumeric(x, b))
			case "numeric":
				ok = ok && isNumeric(x, b)
			case "decimal":
				ok = ok && isDecimal(x, b)
			case "lower":
				if isCased(x, b) {
					seen = true
					ok = ok && isLower(x, b)
				}
			case "upper":
				if isCased(x, b) {
					seen = true
					ok = ok && isUpper(x, b)
				}
			case "title":
				if isCased(x, b) {
					seen = true
					ok = ok && ((prev && isLower(x, b)) || (!prev && isTitle(x, b)))
					prev = true
				} else {
					prev = false
				}
			}
		}
		if name == "lower" || name == "upper" || name == "title" {
			ok = ok && seen
		}
		v[i] = ok
	}
	return np.WrapSlice(v, c.shape...)
}
func IsAlpha(a *np.Array) (*np.Array, error)   { return classify(a, "alpha") }
func IsDigit(a *np.Array) (*np.Array, error)   { return classify(a, "digit") }
func IsSpace(a *np.Array) (*np.Array, error)   { return classify(a, "space") }
func IsAlnum(a *np.Array) (*np.Array, error)   { return classify(a, "alnum") }
func IsLower(a *np.Array) (*np.Array, error)   { return classify(a, "lower") }
func IsUpper(a *np.Array) (*np.Array, error)   { return classify(a, "upper") }
func IsTitle(a *np.Array) (*np.Array, error)   { return classify(a, "title") }
func IsDecimal(a *np.Array) (*np.Array, error) { return classify(a, "decimal") }
func IsNumeric(a *np.Array) (*np.Array, error) { return classify(a, "numeric") }

func finalSigma(r []rune, i int) bool {
	before := false
	for j := i - 1; j >= 0; j-- {
		if !contains(ignorableRanges[:], r[j]) {
			before = isCased(r[j], false)
			break
		}
	}
	if !before {
		return false
	}
	for j := i + 1; j < len(r); j++ {
		if !contains(ignorableRanges[:], r[j]) {
			return !isCased(r[j], false)
		}
	}
	return true
}

func caseMap(a *np.Array, mode string) (*np.Array, error) {
	return mapText(a, true, func(s string, d np.DType) (string, error) {
		r := units(s, d)
		var out strings.Builder
		prev := false
		for i, x := range r {
			m := mode
			if mode == "capitalize" {
				if i == 0 {
					m = "title"
				} else {
					m = "lower"
				}
			}
			if mode == "title" && prev {
				m = "lower"
			}
			b := d == np.BytesDType
			prev = isCased(x, b)
			if b {
				if (m == "upper" || m == "title" || m == "swap") && isLower(x, true) {
					x -= 32
				} else if (m == "lower" || m == "swap") && isUpper(x, true) {
					x += 32
				}
				out.WriteByte(byte(x))
				continue
			}
			var table map[rune]string
			switch m {
			case "upper":
				table = upperMappings
			case "lower":
				table = lowerMappings
			case "title":
				table = titleMappings
			case "swap":
				table = swapMappings
			}
			if (m == "lower" || m == "swap") && x == 'Σ' && finalSigma(r, i) {
				out.WriteRune('ς')
			} else if v, ok := table[x]; ok {
				out.WriteString(v)
			} else {
				out.WriteRune(x)
			}
		}
		return out.String(), nil
	})
}
func Upper(a *np.Array) (*np.Array, error)      { return caseMap(a, "upper") }
func Lower(a *np.Array) (*np.Array, error)      { return caseMap(a, "lower") }
func Title(a *np.Array) (*np.Array, error)      { return caseMap(a, "title") }
func Capitalize(a *np.Array) (*np.Array, error) { return caseMap(a, "capitalize") }
func SwapCase(a *np.Array) (*np.Array, error)   { return caseMap(a, "swap") }
