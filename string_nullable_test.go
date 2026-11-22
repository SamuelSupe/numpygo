package np_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	s "github.com/SamuelSupe/numpygo/strings"
	"math"
	"testing"
)

func TestNullableStringNumPy(t *testing.T) {
	for _, p := range []struct {
		name string
		na   any
	}{{"none", nil}, {"nan", math.NaN()}, {"text", "NA"}, {"int", 0}} {
		t.Run(p.name, func(t *testing.T) {
			m := refcheck.Require(t)
			d := np.StringDescriptor{HasNA: true, NA: p.na}
			a := m(np.FromStringDescriptor([]any{"a", p.na, "", "N", p.na, "A"}, d, 2, 3))
			name := func(n string) string { return "strna_" + p.name + "_" + n }
			char := func(n string, a *np.Array, layout bool) {
				t.Helper()
				refcheck.CheckNullableString(t, name(n), a, layout)
			}
			check := func(n string, a *np.Array) { t.Helper(); refcheck.Check(t, name(n), a) }
			char("base", a, true)
			rev := m(a.Slice(np.Slice{}, np.Reverse()))
			char("reverse", rev, true)
			if !np.SharesMemory(a, rev) {
				t.Fatal("nullable view lost storage")
			}
			char("copy", m(a.Copy()), false)
			char("concat", m(np.Concatenate([]*np.Array{a, a}, 0)), false)
			char("where", m(np.Where(m(np.FromSlice([]bool{true, false}, 2, 1)), a, rev)), false)
			char("zeros", m(np.ZerosLike(a)), false)
			char("ones", m(np.OnesLike(a)), false)
			char("take", m(np.Take(a, m(np.FromSlice([]int64{1, 0}, 2)), np.Int(1))), false)
			changed := m(a.Copy())
			if e := changed.SetValue(p.na, 0, 0); e != nil {
				t.Fatal(e)
			}
			if e := changed.SetValue("real", 1, 1); e != nil {
				t.Fatal(e)
			}
			char("assign", changed, false)
			check("bool", m(a.Astype(np.Bool)))
			check("isnan", m(np.IsNaN(a)))
			check("any", m(np.Any(a, np.SumOptions{Axes: []int{1}})))
			check("eq", m(np.Equal(a, rev)))
			check("ne", m(np.NotEqual(a, rev)))
			check("isin", m(np.Isin(a, a, false)))
			refcheck.CheckCharacter(t, name("cast_u"), m(a.CastFlexible(np.FixedUnicode(8, nil))), false)
			refcheck.CheckCharacter(t, name("cast_default"), m(a.CastStrings(false)), false)
			v, e := m(np.FromStringDescriptor([]any{"1", p.na}, d, 2)).Astype(np.Float64)
			refcheck.Case(t, name("cast_num"), v, e)
			opt := np.SumOptions{Axes: []int{1}}
			ao := np.ArgOptions{Axis: np.Int(1)}
			nonnan := p.name == "none" || p.name == "int"
			textArg := m(np.FromStrings([]string{"a"}))
			repl := m(np.FromStrings([]string{"X"}))
			for _, f := range []struct {
				n  string
				fn func() (*np.Array, error)
			}{
				{"add", func() (*np.Array, error) { return np.Add(a, a) }},
				{"multiply", func() (*np.Array, error) { return np.Multiply(a, np.ScalarOf(int64(2))) }},
				{"sum", func() (*np.Array, error) { return np.Sum(a, opt) }},
				{"min", func() (*np.Array, error) { return np.Min(a, opt) }},
				{"max", func() (*np.Array, error) { return np.Max(a, opt) }},
				{"sort", func() (*np.Array, error) { return np.Sort(a, np.Int(1)) }},
				{"unique", func() (*np.Array, error) { v, e := np.Unique(a); return v.Values, e }},
				{"strip", func() (*np.Array, error) { return s.Strip(a) }},
				{"replace", func() (*np.Array, error) { return s.Replace(a, textArg, repl, -1) }},
				{"upper", func() (*np.Array, error) { return s.Upper(a) }},
			} {
				t.Run(f.n, func(t *testing.T) {
					v, e := f.fn()
					if nonnan || p.name == "nan" && f.n == "upper" {
						refcheck.Case(t, name(f.n), v, e)
					} else {
						char(f.n, m(v, e), false)
					}
				})
			}
			for _, f := range []struct {
				n  string
				fn func() (*np.Array, error)
			}{
				{"less", func() (*np.Array, error) { return np.Less(a, rev) }},
				{"argmin", func() (*np.Array, error) { return np.ArgMin(a, ao) }},
				{"argmax", func() (*np.Array, error) { return np.ArgMax(a, ao) }},
				{"alpha", func() (*np.Array, error) { return s.IsAlpha(a) }},
				{"strlen", func() (*np.Array, error) { return s.StrLen(a) }},
			} {
				t.Run(f.n, func(t *testing.T) {
					v, e := f.fn()
					if nonnan || p.name == "nan" && f.n == "strlen" {
						refcheck.Case(t, name(f.n), v, e)
					} else {
						check(f.n, m(v, e))
					}
				})
			}
			if p.name == "text" {
				char("same_text", m(np.Add(m(np.FromStringDescriptor([]any{"N"}, d, 1)), m(np.FromStringDescriptor([]any{"A"}, d, 1)))), false)
			}
			d.Strict = true
			strict := m(np.FromStringDescriptor([]any{p.na, "x"}, d, 2))
			char("strict", strict, false)
			refcheck.Error(t, name("strict_value"), "ValueError", strict.SetValue(123, 1), np.ErrValue)
			refcheck.Text(t, name("repr"), refcheck.RequireValue[string](t)(np.ArrayRepr(a)))
		})
	}
	m := refcheck.Require(t)
	a := m(np.FromStringDescriptor([]any{"x"}, np.StringDescriptor{HasNA: true, NA: nil}, 1))
	b := m(np.FromStringDescriptor([]any{"x"}, np.StringDescriptor{HasNA: true, NA: math.NaN()}, 1))
	v, e := np.Add(a, b)
	refcheck.Case(t, "strna_different_sentinel", v, e)
}
