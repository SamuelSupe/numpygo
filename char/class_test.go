package char

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestCharArraySubclassNumPy(t *testing.T) {
	for _, kind := range []string{"u", "s"} {
		t.Run(kind, func(t *testing.T) {
			m := refcheck.Require(t)
			cm := refcheck.RequireValue[*CharArray](t)
			makeText := func(v []string, w int, shape ...int) *np.Array {
				if kind == "u" {
					return m(np.FromUnicodeStrings(v, w, shape...))
				}
				return m(np.FromByteStrings(v, w, shape...))
			}
			scalar := func(s string) *np.Array { return makeText([]string{s}, 1) }
			a := cm(Array(makeText([]string{" a ", "Bb  ", "ab", "a  ", "xyz", "12", "\t", ""}, 5, 2, 4), Options{}))
			prefix := "subclass_char_" + kind + "_"
			check := func(n string, v *np.Array) { t.Helper(); refcheck.Check(t, prefix+n, v) }
			charCheck := func(n string, v *np.Array, isChar bool) {
				t.Helper()
				refcheck.CheckCharacter(t, prefix+n, v, false)
				check(n+"_subclass", np.ScalarOf(isChar))
			}
			for n, v := range map[string]*CharArray{
				"capitalize": cm(a.Capitalize()), "upper": cm(a.Upper()), "lower": cm(a.Lower()),
				"title": cm(a.Title()), "swapcase": cm(a.SwapCase()), "strip": cm(a.Strip()), "lstrip": cm(a.LStrip()), "rstrip": cm(a.RStrip()),
				"center": cm(a.Center(8, "-")), "ljust": cm(a.LJust(8, "-")), "rjust": cm(a.RJust(8, "-")),
				"zfill": cm(a.ZFill(8)), "expandtabs": cm(a.ExpandTabs(4)),
				"replace":   cm(a.Replace(scalar("a"), scalar("Q"), 1)),
				"partition": cm(a.Partition(scalar("b"))), "rpartition": cm(a.RPartition(scalar("b"))),
			} {
				charCheck(n, v.Array, true)
			}
			for n, v := range map[string]*np.Array{
				"isalpha": m(a.IsAlpha()), "isalnum": m(a.IsAlnum()), "isdigit": m(a.IsDigit()),
				"islower": m(a.IsLower()), "isupper": m(a.IsUpper()), "istitle": m(a.IsTitle()), "isspace": m(a.IsSpace()),
				"find": m(a.Find(scalar("a"), 0, 5)), "rfind": m(a.RFind(scalar("a"), 0, 5)), "count": m(a.Count(scalar("a"), 0, 5)),
				"startswith": m(a.StartsWith(scalar("a"), 0, 5)), "endswith": m(a.EndsWith(scalar("a"), 0, 5)),
				"index": m(a.FindIndex(scalar(""))), "rindex": m(a.RFindIndex(scalar(""))),
			} {
				check(n, v)
			}
			if kind == "u" {
				check("isdecimal", m(a.IsDecimal()))
				check("isnumeric", m(a.IsNumeric()))
			} else {
				v, e := a.IsDecimal()
				refcheck.Case(t, prefix+"isdecimal", v, e)
				v, e = a.IsNumeric()
				refcheck.Case(t, prefix+"isnumeric", v, e)
			}
			fmt := cm(Array(makeText([]string{"%03d", "%d%%"}, 5, 2), Options{}))
			for n, v := range map[string]*CharArray{
				"add": cm(a.Add(scalar("!"), false)), "radd": cm(a.Add(scalar("!"), true)),
				"multiply": cm(a.Multiply(np.ScalarOf(int64(2)))), "rmul": cm(a.Multiply(np.ScalarOf(int64(2)))),
				"mod": cm(fmt.Mod(m(np.FromSlice([]int64{7, -4}, 2)))),
			} {
				charCheck(n, v.Array, true)
			}
			for n, op := range map[string]string{"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="} {
				check(n, m(a.Compare(makeText([]string{"a"}, 5), op)))
			}
			check("argsort", m(a.Argsort(np.Int(1))))
			trailing := cm(Array(makeText([]string{"a ", "a", "a  "}, 5, 3), Options{}))
			check("argsort_trailing", m(trailing.Argsort(np.Int(0))))
			checkLists(t, prefix+"split", m(a.Split(nil, 1)))
			checkLists(t, prefix+"rsplit", m(a.RSplit(nil, 1)))
			checkLists(t, prefix+"splitlines", m(a.SplitLines(true)))
			if kind == "u" {
				charCheck("encode", m(a.Encode("utf-8", "strict")), false)
				sep := cm(Array(makeText([]string{"-"}, 1, 1), Options{}))
				charCheck("join", m(sep.Join(makeText([]string{"ab"}, 2, 1))), false)
				charCheck("translate", cm(a.Translate(map[rune]string{'a': "XY", 'b': ""})).Array, true)
			} else {
				charCheck("decode", m(a.Decode("utf-8", "strict")), false)
				table := make([]byte, 256)
				for i := range table {
					table[i] = byte(i)
				}
				table['a'] = 'X'
				table['b'] = 'Y'
				charCheck("translate", cm(a.TranslateBytes(table, "")).Array, true)
			}
			indices := m(np.FromSlice([]int64{2, 0}, 2))
			condition := m(np.FromSlice([]bool{true, false, true, false}, 4))
			ex := cm(a.IndexChar(np.NewAxis, np.Ellipsis))
			f, e := a.FlexibleType()
			if e != nil {
				t.Fatal(e)
			}
			target := np.FixedBytes(3)
			if kind == "u" {
				target = np.FixedUnicode(3, nil)
			}
			for n, v := range map[string]*CharArray{
				"slice": cm(a.SliceChar(np.Slice{}, np.Reverse())), "gather": cm(a.IndexChar(np.Slice{}, indices)),
				"copy": cm(a.CopyChar()), "reshape": cm(a.ReshapeChar(4, 2)), "transpose": cm(a.TransposeChar()),
				"ravel": cm(a.Ravel()), "flatten": cm(a.Flatten()), "take": cm(a.Take(indices, np.Int(1))),
				"repeat": cm(a.Repeat(2, np.Int(1))), "squeeze": cm(ex.Squeeze(0)), "swapaxes": cm(a.SwapAxes(0, 1)),
				"compress": cm(a.Compress(condition, np.Int(1))), "diagonal": cm(a.Diagonal(0, 0, 1)),
				"cast": cm(a.CastChar(target, np.CastOptions{})), "view": cm(a.ViewChar(f)),
			} {
				charCheck(n, v.Array, true)
			}
			v := refcheck.RequireValue[any](t)(a.AtValue(0, 0))
			var st string
			switch s := v.(type) {
			case np.BytesScalar:
				st = string(s)
			case np.TextScalar:
				st = string(s)
			default:
				t.Fatal(v)
			}
			refcheck.Text(t, prefix+"scalar", st)
			view := cm(a.SliceChar(np.Slice{}, np.Reverse()))
			if e := view.SetValue("Go", 0, 0); e != nil {
				t.Fatal(e)
			}
			charCheck("view_write", a.Array, true)
		})
	}
}
