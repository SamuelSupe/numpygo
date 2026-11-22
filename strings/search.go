package npstrings

import (
	"github.com/SamuelSupe/numpygo"
	"strings"
)

// Search operations use Python's start/end code-point (or byte) offsets,
// including the distinction between an empty slice and start > len(s).
func search(a, sub *np.Array, kind string, bounds []int) (*np.Array, error) {
	if len(bounds) > 2 {
		return nil, np.ErrValue
	}
	start, end := 0, int(^uint(0)>>1)
	if len(bounds) > 0 {
		start = bounds[0]
	}
	if len(bounds) > 1 {
		end = bounds[1]
	}
	return searchArrays(a, sub, np.ScalarOf(int64(start)), np.ScalarOf(int64(end)), kind)
}
func searchArrays(a, sub, start, end *np.Array, kind string) (*np.Array, error) {
	c, options, err := textWithInts([]*np.Array{a, sub}, start, end)
	if err != nil {
		return nil, err
	}
	if c.hasNull() {
		return nil, np.ErrValue
	}
	flags := kind == "startswith" || kind == "endswith"
	n := len(c.data[0])
	v := make([]int64, n)
	ok := make([]bool, n)
	for i, s := range c.data[0] {
		r := units(s, c.dtype)
		start, end := options[0][i], options[1][i]
		if start < 0 {
			start = max(0, start+len(r))
		}
		if end < 0 {
			end += len(r)
		}
		end = min(len(r), max(0, end))
		t := c.data[1][i]
		invalid := start > len(r) || start > end
		if invalid {
			if kind != "count" {
				v[i] = -1
			}
		} else {
			span := join(r[start:end], c.dtype)
			switch kind {
			case "startswith":
				ok[i] = strings.HasPrefix(span, t)
			case "endswith":
				ok[i] = strings.HasSuffix(span, t)
			case "count":
				if t == "" {
					v[i] = int64(end - start + 1)
				} else {
					v[i] = int64(strings.Count(span, t))
				}
			default:
				pos := strings.Index(span, t)
				if kind == "rfind" || kind == "rindex" {
					pos = strings.LastIndex(span, t)
				}
				v[i] = -1
				if pos >= 0 {
					v[i] = int64(start + length(span[:pos], c.dtype))
				}
			}
		}
		if (kind == "index" || kind == "rindex") && v[i] < 0 {
			return nil, np.ErrValue
		}
	}
	if flags {
		return np.WrapSlice(ok, c.shape...)
	}
	return np.WrapSlice(v, c.shape...)
}
func FindWithBounds(a, sub, start, end *np.Array) (*np.Array, error) {
	return searchArrays(a, sub, start, end, "find")
}
func RFindWithBounds(a, sub, start, end *np.Array) (*np.Array, error) {
	return searchArrays(a, sub, start, end, "rfind")
}
func IndexWithBounds(a, sub, start, end *np.Array) (*np.Array, error) {
	return searchArrays(a, sub, start, end, "index")
}
func RIndexWithBounds(a, sub, start, end *np.Array) (*np.Array, error) {
	return searchArrays(a, sub, start, end, "rindex")
}
func CountWithBounds(a, sub, start, end *np.Array) (*np.Array, error) {
	return searchArrays(a, sub, start, end, "count")
}
func StartsWithBounds(a, sub, start, end *np.Array) (*np.Array, error) {
	return searchArrays(a, sub, start, end, "startswith")
}
func EndsWithBounds(a, sub, start, end *np.Array) (*np.Array, error) {
	return searchArrays(a, sub, start, end, "endswith")
}
func Find(a, sub *np.Array, bounds ...int) (*np.Array, error) { return search(a, sub, "find", bounds) }
func RFind(a, sub *np.Array, bounds ...int) (*np.Array, error) {
	return search(a, sub, "rfind", bounds)
}
func Index(a, sub *np.Array, bounds ...int) (*np.Array, error) {
	return search(a, sub, "index", bounds)
}
func RIndex(a, sub *np.Array, bounds ...int) (*np.Array, error) {
	return search(a, sub, "rindex", bounds)
}
func Count(a, sub *np.Array, bounds ...int) (*np.Array, error) {
	return search(a, sub, "count", bounds)
}
func StartsWith(a, sub *np.Array, bounds ...int) (*np.Array, error) {
	return search(a, sub, "startswith", bounds)
}
func EndsWith(a, sub *np.Array, bounds ...int) (*np.Array, error) {
	return search(a, sub, "endswith", bounds)
}
