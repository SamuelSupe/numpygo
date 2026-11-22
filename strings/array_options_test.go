package npstrings

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestArrayStringOptionsNumPy(t *testing.T) {
	for _, kind := range []string{"s", "u", "t"} {
		t.Run(kind, func(t *testing.T) {
			m := refcheck.Require(t)
			text := func(v []string, w int, shape ...int) *np.Array {
				switch kind {
				case "s":
					return m(np.FromByteStrings(v, w, shape...))
				case "u":
					return m(np.FromUnicodeStrings(v, w, shape...))
				default:
					return m(np.FromStrings(v, shape...))
				}
			}
			ints := func(v ...int64) *np.Array { return m(np.FromSlice(v, len(v))) }
			last, tabLast := "éx", "é\t"
			if kind == "s" {
				last, tabLast = "zx", "z\t"
			}
			a := text([]string{"a b", "xyab", last}, 6, 3, 1)
			b := text([]string{"a", "b", "x"}, 1, 3)
			newText := text([]string{"Q", "Z", "-"}, 1, 3)
			start, end, width := ints(-2, 0, 1), ints(4, 3, 6), ints(3, 5, 6)
			name := func(s string) string { return "strarray_" + kind + "_" + s }
			check := func(n string, a *np.Array) { refcheck.Check(t, name(n), a) }
			char := func(n string, a *np.Array) { refcheck.CheckCharacter(t, name(n), a, false) }
			for _, p := range []struct {
				n  string
				fn func(*np.Array, *np.Array, *np.Array, *np.Array) (*np.Array, error)
			}{
				{"find", FindWithBounds}, {"rfind", RFindWithBounds}, {"count", CountWithBounds}, {"startswith", StartsWithBounds}, {"endswith", EndsWithBounds},
			} {
				t.Run(p.n, func(t *testing.T) { check(p.n, m(p.fn(a, b, start, end))) })
			}
			empty := text([]string{""}, 1)
			check("index", m(IndexWithBounds(a, empty, start, end)))
			check("rindex", m(RIndexWithBounds(a, empty, start, end)))
			char("replace", m(ReplaceWithCounts(a, b, newText, ints(0, 1, -1))))
			for _, p := range []struct {
				n  string
				fn func(*np.Array, *np.Array, *np.Array) (*np.Array, error)
			}{
				{"center", CenterWithWidths}, {"ljust", LJustWithWidths}, {"rjust", RJustWithWidths},
			} {
				t.Run(p.n, func(t *testing.T) { char(p.n, m(p.fn(a, width, b))) })
			}
			char("zfill", m(ZFillWithWidths(a, width)))
			char("strip", m(StripWithChars(a, b)))
			char("lstrip", m(LStripWithChars(a, b)))
			char("rstrip", m(RStripWithChars(a, b)))
			tab := text([]string{"a\tb", "xy\n\t", tabLast}, 6, 3, 1)
			char("expandtabs", m(ExpandTabsWithSizes(tab, ints(0, 2, 4))))
			char("slice", m(SliceWithBounds(a, ints(0, -1, 0), ints(3, -3, 4), ints(1, -1, 2))))
			v, e := SliceWithBounds(a, np.ScalarOf(int64(0)), np.ScalarOf(int64(3)), ints(1, 0, 1))
			refcheck.Case(t, name("slice_step"), v, e)
		})
	}
}
