package npstrings

import (
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"regexp"
	"strings"
)

var percent = regexp.MustCompile(`^%([-+ #0]*[0-9]*(?:\.[0-9]+)?)([diuoxXeEfFgGcs%])`)

// Mod implements scalar percent formatting with positional conversions.
// A Go []any object slot can supply multiple positional values. Mapping
// placeholders and dynamic * widths are not part of this typed subset.
func Mod(a, values *np.Array) (*np.Array, error) {
	v, e := np.BroadcastArrays(a, values)
	if e != nil {
		return nil, e
	}
	c, e := textInputs(v[0])
	if e != nil {
		return nil, e
	}
	if c.hasNull() {
		return nil, np.ErrDType
	}
	flat, e := v[1].Reshape(v[1].Size())
	if e != nil {
		return nil, e
	}
	for i, s := range c.data[0] {
		x, e := flat.AtValue(i)
		if e != nil {
			return nil, e
		}
		args, ok := x.([]any)
		if !ok {
			args = []any{x}
		}
		var out strings.Builder
		n := 0
		for len(s) > 0 {
			j := strings.IndexByte(s, '%')
			if j < 0 {
				out.WriteString(s)
				break
			}
			out.WriteString(s[:j])
			s = s[j:]
			m := percent.FindStringSubmatch(s)
			if m == nil {
				return nil, np.ErrValue
			}
			if m[2] == "%" {
				out.WriteByte('%')
				s = s[len(m[0]):]
				continue
			}
			if n == len(args) {
				return nil, np.ErrValue
			}
			arg := args[n]
			n++
			verb := m[2]
			if verb == "i" || verb == "u" {
				verb = "d"
			}
			switch verb {
			case "s":
				switch x := arg.(type) {
				case np.BytesScalar:
					arg = string(x)
				case np.TextScalar:
					arg = string(x)
				case bool:
					if x {
						arg = "True"
					} else {
						arg = "False"
					}
				case nil:
					arg = "None"
				default:
					arg = fmt.Sprint(arg)
				}
			case "d", "o", "x", "X", "c":
				switch x := arg.(type) {
				case float64:
					arg = int64(x)
				case float32:
					arg = int64(x)
				case bool:
					if x {
						arg = 1
					} else {
						arg = 0
					}
				}
			case "e", "E", "f", "F", "g", "G":
				switch x := arg.(type) {
				case int64:
					arg = float64(x)
				case int32:
					arg = float64(x)
				case int:
					arg = float64(x)
				case uint64:
					arg = float64(x)
				case uint32:
					arg = float64(x)
				}
				if verb == "F" {
					verb = "f"
				}
			}
			formatted := fmt.Sprintf("%"+m[1]+verb, arg)
			if strings.Contains(formatted, "%!") {
				return nil, np.ErrDType
			}
			out.WriteString(formatted)
			s = s[len(m[0]):]
		}
		c.data[0][i] = out.String()
	}
	c.desc = np.StringDescriptor{}
	return result(c, c.data[0], -1)
}
