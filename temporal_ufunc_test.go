package np_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestTemporalUFuncOptionsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	lookup := refcheck.RequireValue[*np.UFunc](t)
	time := func(v []int64, d np.DType, unit string) *np.Array {
		return m(np.FromDateTicks(m(np.FromSlice(v, len(v))), d, unit))
	}
	a := m(time([]int64{-2, 1, 0}, np.DateTime64, "D").Slice(np.Reverse()))
	b := time([]int64{2, 3, 4}, np.TimeDelta64, "h")
	mask := m(np.FromSlice([]bool{true, false, true}, 3))
	add := lookup(np.LookupUFunc("add"))
	for _, unit := range []string{"h", "D", "m"} {
		for _, rule := range []string{"no", "equiv", "safe", "same_kind", "unsafe"} {
			t.Run(unit+"_"+rule, func(t *testing.T) {
				out := m(time([]int64{99, 99, 99}, np.DateTime64, unit).Slice(np.Reverse()))
				name := "temporal_ufunc_add_" + unit + "_" + rule
				v, e := add.CallWithOptions(np.UFuncOptions{Out: out, Where: mask, Casting: rule}, a, b)
				if rule == "no" || rule == "equiv" || unit == "D" && rule == "safe" {
					refcheck.Case(t, name, v, e)
					return
				}
				if m(v, e) != out {
					t.Fatal("out identity changed")
				}
				refcheck.CheckLayout(t, name, m(out.DateTicks()))
			})
		}
	}
	a = time([]int64{-2, np.NaT, 3}, np.TimeDelta64, "3h")
	for _, name := range []string{"negative", "absolute", "isnan", "isfinite", "isinf", "isnat"} {
		t.Run(name, func(t *testing.T) {
			var out *np.Array
			ticks := name == "negative" || name == "absolute"
			if ticks {
				out = m(time([]int64{9, 9, 9}, np.TimeDelta64, "h").Slice(np.Reverse()))
			} else {
				out = m(m(np.FromSlice([]bool{true, true, true}, 3)).Slice(np.Reverse()))
			}
			if v := m(lookup(np.LookupUFunc(name)).CallWithOptions(np.UFuncOptions{Out: out, Where: mask}, a)); v != out {
				t.Fatal("out identity")
			}
			if ticks {
				out = m(out.DateTicks())
			}
			refcheck.CheckLayout(t, "temporal_ufunc_"+name, out)
		})
	}
}
