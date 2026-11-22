package char

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"strings"
	"testing"
)

func checkLists(t *testing.T, name string, a *np.Array) {
	t.Helper()
	m := refcheck.Require(t)
	values := refcheck.RequireValue[[]any](t)(a.ObjectValues())
	counts := make([]int64, len(values))
	off := []int64{0}
	var b strings.Builder
	for i, v := range values {
		var s []string
		switch x := v.(type) {
		case []string:
			s = x
		case []np.BytesScalar:
			for _, p := range x {
				s = append(s, string(p))
			}
		default:
			t.Fatalf("bad list slot %T", v)
		}
		counts[i] = int64(len(s))
		for _, p := range s {
			b.WriteString(p)
			off = append(off, int64(b.Len()))
		}
	}
	refcheck.Check(t, name+"_counts", m(np.WrapSlice(counts, a.Shape()...)))
	refcheck.Check(t, name+"_offsets", m(np.WrapSlice(off, len(off))))
	refcheck.Bytes(t, name, []byte(b.String()))
}
func TestLegacyCharNamespaceNumPy(t *testing.T) {
	for _, kind := range []string{"s", "u", "t"} {
		t.Run(kind, func(t *testing.T) {
			m := refcheck.Require(t)
			mc := refcheck.RequireValue[*CharArray](t)
			makeText := func(v []string, w int, shape ...int) *np.Array {
				switch kind {
				case "s":
					return m(np.FromByteStrings(v, w, shape...))
				case "u":
					return m(np.FromUnicodeStrings(v, w, shape...))
				default:
					return m(np.FromStrings(v, shape...))
				}
			}
			a := m(makeText([]string{" a ", "a  ", "A", "a b", "xy\r\nz", " \t", "-7", ""}, 12, 2, 4).Slice(np.Slice{}, np.Reverse()))
			b := makeText([]string{"a", "A ", "a", ""}, 3, 4)
			newText := makeText([]string{"Q", "XY", "", "+ "}, 3, 4)
			empty := makeText([]string{""}, 3)
			sep := makeText([]string{" "}, 3)
			name := func(n string) string { return "char_" + kind + "_" + n }
			check := func(n string, v *np.Array) { t.Helper(); refcheck.Check(t, name(n), v) }
			char := func(n string, v *np.Array) { t.Helper(); refcheck.CheckCharacter(t, name(n), v, false) }
			for _, p := range []struct {
				name string
				fn   func(*np.Array, *np.Array) (*np.Array, error)
			}{
				{"equal", Equal}, {"not_equal", NotEqual}, {"less", Less}, {"less_equal", LessEqual}, {"greater", Greater}, {"greater_equal", GreaterEqual},
			} {
				t.Run(p.name, func(t *testing.T) {
					v, e := p.fn(a, b)
					if kind == "t" {
						refcheck.Case(t, name(p.name), v, e)
					} else {
						check(p.name, m(v, e))
					}
				})
			}
			v, e := CompareCharArrays(a, b, "==", false)
			if kind == "t" {
				refcheck.Case(t, name("compare_chararrays"), v, e)
				char("array", mc(Array(a, Options{})).Array)
			} else {
				check("compare_chararrays", m(v, e))
				copyA, view := mc(Array(a, Options{})), mc(AsArray(a, Options{}))
				char("array", copyA.Array)
				char("asarray", view.Array)
				check("array_copy", np.ScalarOf(np.SharesMemory(a, copyA.Array)))
				check("asarray_share", np.ScalarOf(np.SharesMemory(a, view.Array)))
				x := refcheck.RequireValue[any](t)(view.AtValue(0, 3))
				var st string
				switch x := x.(type) {
				case np.BytesScalar:
					st = string(x)
				case np.TextScalar:
					st = string(x)
				default:
					t.Fatal(x)
				}
				refcheck.Bytes(t, name("scalar"), []byte(st))
				c := mc(New([]int{2, 2}, a.DType(), 4))
				for i, s := range []string{"abc", "ab", "x", "1234"} {
					if e := c.SetValue(s, i/2, i%2); e != nil {
						t.Fatal(e)
					}
				}
				char("chararray", c.Array)
			}
			for _, p := range []struct {
				name string
				fn   func(*np.Array) (*np.Array, error)
			}{{"upper", Upper}, {"lower", Lower}, {"capitalize", Capitalize}, {"title", Title}, {"swapcase", SwapCase}} {
				t.Run(p.name, func(t *testing.T) { char(p.name, m(p.fn(a))) })
			}
			for _, p := range []struct {
				name string
				fn   func(*np.Array, ...string) (*np.Array, error)
			}{{"strip", Strip}, {"lstrip", LStrip}, {"rstrip", RStrip}} {
				t.Run(p.name, func(t *testing.T) { char(p.name, m(p.fn(a))) })
			}
			for _, p := range []struct {
				name string
				fn   func(*np.Array) (*np.Array, error)
			}{{"isalnum", IsAlnum}, {"isalpha", IsAlpha}, {"isdigit", IsDigit}, {"islower", IsLower}, {"isupper", IsUpper}, {"isspace", IsSpace}, {"istitle", IsTitle}, {"isnumeric", IsNumeric}, {"isdecimal", IsDecimal}} {
				t.Run(p.name, func(t *testing.T) {
					v, e := p.fn(a)
					if kind == "s" && (p.name == "isnumeric" || p.name == "isdecimal") {
						refcheck.Case(t, name(p.name), v, e)
					} else {
						check(p.name, m(v, e))
					}
				})
			}
			check("str_len", m(StrLen(a)))
			char("add", m(Add(a, b)))
			char("multiply", m(Multiply(a, m(np.FromSlice([]int64{2, 0, 1, -1}, 4)))))
			char("mod", m(Mod(makeText([]string{"%03d", "%d%%"}, 12, 2), m(np.FromSlice([]int64{7, -4}, 2)))))
			for _, p := range []struct {
				name string
				fn   func(*np.Array, *np.Array, ...int) (*np.Array, error)
			}{{"find", Find}, {"rfind", RFind}, {"count", Count}, {"startswith", StartsWith}, {"endswith", EndsWith}} {
				t.Run(p.name, func(t *testing.T) { check(p.name, m(p.fn(a, b, -3, 12))) })
			}
			check("index", m(Index(a, empty)))
			check("rindex", m(RIndex(a, empty)))
			char("replace", m(Replace(a, b, newText, 1)))
			char("expandtabs", m(ExpandTabs(a, 4)))
			for _, p := range []struct {
				name string
				fn   func(*np.Array, int, ...string) (*np.Array, error)
			}{{"center", Center}, {"ljust", LJust}, {"rjust", RJust}} {
				t.Run(p.name, func(t *testing.T) { char(p.name, m(p.fn(a, 6, "*"))) })
			}
			char("zfill", m(ZFill(a, 6)))
			for _, p := range []struct {
				name string
				fn   func(*np.Array, *np.Array) (*np.Array, error)
			}{{"partition", Partition}, {"rpartition", RPartition}} {
				t.Run(p.name, func(t *testing.T) {
					v, e := p.fn(a, sep)
					if kind == "t" {
						refcheck.Case(t, name(p.name), v, e)
					} else {
						char(p.name, m(v, e))
					}
				})
			}
			space := " "
			checkLists(t, name("split"), m(Split(a, nil, 1)))
			checkLists(t, name("split_explicit"), m(Split(a, &space, 1)))
			checkLists(t, name("rsplit"), m(RSplit(a, nil, 1)))
			checkLists(t, name("rsplit_explicit"), m(RSplit(a, &space, 1)))
			checkLists(t, name("splitlines"), m(SplitLines(a, true)))
			if kind == "s" {
				v, e := Join(makeText([]string{"-"}, 1), a)
				refcheck.Case(t, name("join"), v, e)
				char("decode", m(Decode(a, "utf-8", "strict")))
				table := make([]byte, 256)
				for i := range table {
					table[i] = byte(i)
				}
				table['a'], table['A'] = 'X', 'Y'
				char("translate", m(TranslateBytes(a, table, " ")))
			} else {
				char("join", m(Join(m(np.FromUnicodeStrings([]string{"-", "+"}, 1, 2, 1)), a)))
				char("encode", m(Encode(a, "utf-8", "strict")))
				char("translate", m(Translate(a, map[rune]string{'a': "xy", ' ': ""})))
			}
		})
	}
}
