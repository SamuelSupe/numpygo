package npstrings

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestStringNamespaceNumPy(t *testing.T) {
	for _, kind := range []string{"s", "u", "t"} {
		t.Run(kind, func(t *testing.T) {
			m := refcheck.Require(t)
			makeText := func(s []string, width int, shape ...int) *np.Array {
				switch kind {
				case "s":
					return m(np.FromByteStrings(s, width, shape...))
				case "u":
					return m(np.FromUnicodeStrings(s, width, shape...))
				default:
					return m(np.FromStrings(s, shape...))
				}
			}
			raw := []string{" Ab ", "éß", "a\x00b", "", "ΟΣ", "İ", "ﬃ", "1²½", " \t", "中文", "a\tb", "-7"}
			if kind == "s" {
				raw = []string{" Ab ", "A\xff", "a\x00b", "", "ABC", "xy", "\xff\t", "123", " \t", "\x80", "a\tb", "-7"}
			}
			a := m(makeText(raw, 12, 3, 4).Slice(np.Slice{}, np.Reverse()))
			sub := makeText([]string{"a", "b", " ", "-"}, 3, 4)
			newText := makeText([]string{"xyz", "Q", "", "+"}, 3, 4)
			empty := makeText([]string{""}, 3)
			name := func(s string) string { return "strings_" + kind + "_" + s }
			check := func(s string, v *np.Array) { t.Helper(); refcheck.Check(t, name(s), v) }
			char := func(s string, v *np.Array) { t.Helper(); refcheck.CheckCharacter(t, name(s), v, false) }
			for _, p := range []struct {
				name string
				fn   func(*np.Array, *np.Array) (*np.Array, error)
			}{
				{"equal", Equal}, {"not_equal", NotEqual}, {"less", Less}, {"less_equal", LessEqual}, {"greater", Greater}, {"greater_equal", GreaterEqual},
			} {
				t.Run(p.name, func(t *testing.T) { check(p.name, m(p.fn(a, sub))) })
			}
			char("add", m(Add(a, sub)))
			char("multiply", m(Multiply(a, m(np.FromSlice([]int64{2, 0, -2, 3}, 4)))))
			check("str_len", m(StrLen(a)))
			for _, p := range []struct {
				name string
				fn   func(*np.Array) (*np.Array, error)
			}{
				{"isalpha", IsAlpha}, {"isdigit", IsDigit}, {"isspace", IsSpace}, {"isalnum", IsAlnum}, {"islower", IsLower}, {"isupper", IsUpper}, {"istitle", IsTitle}, {"isdecimal", IsDecimal}, {"isnumeric", IsNumeric},
			} {
				t.Run(p.name, func(t *testing.T) {
					v, e := p.fn(a)
					if kind == "s" && (p.name == "isdecimal" || p.name == "isnumeric") {
						refcheck.Case(t, name(p.name), v, e)
					} else {
						check(p.name, m(v, e))
					}
				})
			}
			for _, p := range []struct {
				name string
				fn   func(*np.Array) (*np.Array, error)
			}{
				{"lower", Lower}, {"upper", Upper}, {"swapcase", SwapCase}, {"capitalize", Capitalize}, {"title", Title},
			} {
				t.Run(p.name, func(t *testing.T) { char(p.name, m(p.fn(a))) })
			}
			for _, p := range []struct {
				name string
				fn   func(*np.Array, ...string) (*np.Array, error)
			}{
				{"strip", Strip}, {"lstrip", LStrip}, {"rstrip", RStrip},
			} {
				t.Run(p.name, func(t *testing.T) { char(p.name, m(p.fn(a))) })
			}
			for _, p := range []struct {
				name string
				fn   func(*np.Array, *np.Array, ...int) (*np.Array, error)
			}{
				{"find", Find}, {"rfind", RFind}, {"count", Count}, {"startswith", StartsWith}, {"endswith", EndsWith},
			} {
				t.Run(p.name, func(t *testing.T) { check(p.name, m(p.fn(a, sub, -4, -1))) })
			}
			check("index", m(Index(a, empty)))
			check("rindex", m(RIndex(a, empty)))
			v, e := Index(a, makeText([]string{"not-found"}, 9))
			refcheck.Case(t, name("missing_index"), v, e)
			char("replace", m(Replace(a, sub, newText, 1)))
			char("replace_empty", m(Replace(a, empty, newText, 2)))
			char("slice", m(Slice(a, np.Slice{Step: np.Int(-2)})))
			char("expandtabs", m(ExpandTabs(a, 4)))
			for _, p := range []struct {
				name string
				fn   func(*np.Array, int, ...string) (*np.Array, error)
			}{
				{"center", Center}, {"ljust", LJust}, {"rjust", RJust},
			} {
				t.Run(p.name, func(t *testing.T) { char(p.name, m(p.fn(a, 6, "*"))) })
			}
			char("zfill", m(ZFill(a, 6)))
			for _, p := range []struct {
				name string
				fn   func(*np.Array, *np.Array) ([3]*np.Array, error)
			}{
				{"partition", Partition}, {"rpartition", RPartition},
			} {
				t.Run(p.name, func(t *testing.T) {
					v, e := p.fn(a, sub)
					if kind == "t" {
						refcheck.Case(t, name(p.name), nil, e)
						return
					}
					if e != nil {
						t.Fatal(e)
					}
					for i, x := range v {
						char(p.name+string(rune('0'+i)), x)
					}
				})
			}
			if kind == "s" {
				table := make([]byte, 256)
				for i := range table {
					table[i] = byte(i)
				}
				table['A'], table['b'] = 'X', 'Y'
				char("translate", m(TranslateBytes(a, table, " ")))
				for _, p := range [][2]string{{"latin-1", "strict"}, {"utf-8", "replace"}, {"ascii", "ignore"}} {
					char("decode_"+p[0], m(Decode(a, p[0], p[1])))
				}
				_, e := Decode(a, "", "")
				refcheck.Error(t, name("decode_strict"), "UnicodeDecodeError", e, np.ErrValue)
			} else {
				char("translate", m(Translate(a, map[rune]string{'a': "XY", 'b': ""})))
				for _, p := range [][2]string{{"utf-8", "strict"}, {"latin-1", "replace"}, {"ascii", "ignore"}} {
					char("encode_"+p[0], m(Encode(a, p[0], p[1])))
				}
				_, e := Encode(a, "ascii", "")
				refcheck.Error(t, name("encode_strict"), "UnicodeEncodeError", e, np.ErrValue)
			}
			format := makeText([]string{"%04d", "%+d", "value=%d", "%d%%"}, 12, 4)
			char("mod", m(Mod(format, m(np.FromSlice([]int64{7, -12}, 2, 1)))))
			char("mod_multi", m(Mod(makeText([]string{"%s %.2f"}, 12), np.ObjectScalar([]any{"hi", 1.25}))))
		})
	}
	m := refcheck.Require(t)
	a := m(np.FromUnicodeStrings([]string{"ß", "İ", "ﬃ"}, 1, 3))
	refcheck.CheckCharacter(t, "strings_u_narrow_upper", m(Upper(a)), false)
	refcheck.CheckCharacter(t, "strings_u_narrow_lower", m(Lower(a)), false)
	e := m(np.ZerosFlexible(np.FixedUnicode(4, nil), 2, 0))
	refcheck.CheckCharacter(t, "strings_empty_upper", m(Upper(e)), false)
	v, err := Multiply(e, np.ScalarOf(int64(2)))
	refcheck.Case(t, "strings_empty_multiply", v, err)
	strict := m(np.FromStringValues([]any{"x", "y"}, true, 2))
	refcheck.CheckCharacter(t, "strings_t_strict_upper", m(Upper(strict)), false)
}
