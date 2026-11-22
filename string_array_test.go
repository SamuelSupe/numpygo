package np_test

import (
	"bytes"
	"errors"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"strings"
	"testing"
)

func TestVariableStringNDArrayNumPy(t *testing.T) {
	m := refcheck.Require(t)
	a := m(np.FromStrings([]string{strings.Repeat("long", 30), "é🙂", "a\x00", "", "12", "-7"}, 2, 3))
	c := func(n string, v *np.Array, layout bool) {
		t.Helper()
		refcheck.CheckCharacter(t, "varstr_"+n, v, layout)
	}
	check := func(n string, v *np.Array) { t.Helper(); refcheck.Check(t, "varstr_"+n, v) }
	c("base", a, true)
	reverse := m(a.Slice(np.Reverse(), np.Reverse()))
	c("reverse", reverse, true)
	if !np.SharesMemory(a, reverse) {
		t.Fatal("variable string view lost storage")
	}
	trans := m(a.Transpose())
	c("transpose", trans, true)
	c("reshape", m(trans.Reshape(6)), false)
	c("advanced", m(a.Index(m(np.FromSlice([]int64{1, 0}, 2)), m(np.FromSlice([]int64{2, 0}, 2)))), false)
	c("broadcast", m(m(a.Slice(np.Range(0, 1))).BroadcastTo(2, 3)), true)
	rowReverse := m(a.Slice(np.Slice{}, np.Reverse()))
	c("copy", m(rowReverse.Copy()), false)
	c("concat", m(np.Concatenate([]*np.Array{a, m(np.FromUnicodeStrings([]string{"u", "V", ""}, 3, 1, 3))}, 0)), false)
	c("where", m(np.Where(m(np.FromSlice([]bool{true, false}, 2, 1)), a, rowReverse)), false)
	c("zero", m(np.ZerosLike(a)), false)
	c("ones", m(np.OnesLike(a)), false)
	c("sort", m(np.Sort(a, np.Int(1))), false)
	check("argsort", m(np.Argsort(a, np.Int(1))))
	c("unique", refcheck.RequireValue[np.UniqueResult](t)(np.Unique(a)).Values, false)
	check("isin", m(np.Isin(a, m(np.FromStrings([]string{"a\x00", "-7"}, 2)), false)))
	trailing := m(np.FromStrings([]string{"a\x00", "a", "\x00", ""}, 4))
	check("isin_nul", m(np.Isin(trailing, m(np.FromStrings([]string{"a\x00", "\x00"}, 2)), false)))
	check("equal_nul", m(np.Equal(trailing, m(np.FromStrings([]string{"a\x00"}, 1)))))
	check("bool", m(a.Astype(np.Bool)))
	opt := np.SumOptions{Axes: []int{1}}
	check("any", m(np.Any(a, opt)))
	check("count", m(np.CountNonzero(a)))
	check("isnan", m(np.IsNaN(a)))
	c("min", m(np.Min(a, opt)), false)
	c("max", m(np.Max(a, opt)), false)
	c("sum", m(np.Sum(a, opt)), false)
	c("cast_u", m(a.CastFlexible(np.FixedUnicode(4, nil))), false)
	c("cast_from_u", m(m(np.FromUnicodeStrings([]string{"é", "abc"}, 4, 2)).Astype(np.StringDType)), false)
	c("cast_from_num", m(m(np.FromSlice([]int64{1, -7, 23}, 3)).Astype(np.StringDType)), false)
	check("cast_num", m(m(np.FromStrings([]string{"12", "-7", " 23 "}, 3)).Astype(np.Int64)))
	c("coerce", m(np.FromStringValues([]any{np.BytesScalar("a"), nil, 1.2, true}, false, 4)), false)
	strict := m(np.FromStringValues([]any{"x", "y"}, true, 2))
	c("strict", m(strict.Copy()), false)
	c("strict_ones", m(np.OnesLike(strict)), false)
	c("strict_add", m(np.Add(strict, m(np.FromStrings([]string{"!", "?"}, 2)))), false)
	c("strict_mul", m(np.Multiply(strict, np.ScalarOf(int64(2)))), false)
	c("strict_concat", m(np.Concatenate([]*np.Array{strict, m(np.FromStrings([]string{"z"}, 1))}, 0)), false)
	c("strict_cast", m(strict.CastStrings(false)), false)
	refcheck.Error(t, "varstr_strict_error", "ValueError", strict.SetValue(3, 0), np.ErrValue)
	changed := m(a.Copy())
	rhs := m(m(changed.Slice(np.Slice{}, np.Reverse())).Slice(np.Slice{}, np.Range(0, 2)))
	if e := changed.Assign([]np.Index{np.Slice{}, np.Range(1, 3)}, rhs); e != nil {
		t.Fatal(e)
	}
	if e := changed.SetValue(strings.Repeat("replacement", 20), 0, 0); e != nil {
		t.Fatal(e)
	}
	c("assign", changed, false)
	_, e := np.PromoteTypes(a.DType(), np.Int64)
	refcheck.Error(t, "varstr_bad_promote", "DTypePromotionError", e, np.ErrDType)
	_, e = a.View(np.Uint8)
	refcheck.Error(t, "varstr_bad_view", "TypeError", e, np.ErrDType)
	_, e = a.Astype(np.UnicodeDType)
	refcheck.Error(t, "varstr_bad_astype", "TypeError", e, np.ErrDType)
	refcheck.Text(t, "varstr_repr", refcheck.RequireValue[string](t)(np.ArrayRepr(m(np.FromStrings([]string{"a\x00", "é", ""}, 3)))))
	for _, fn := range []func() error{
		func() error { _, e := a.ToBytes("C"); return e },
		func() error { _, e := a.DataBytes(); return e },
		func() error { _, e := a.ViewFlexible(np.FixedBytes(16)); return e },
		func() error { return np.SaveNPY(new(bytes.Buffer), a) },
		func() error { _, e := np.FromBuffer(make([]byte, 16), np.StringDType, 1, 0, nil); return e },
	} {
		if e := fn(); !errors.Is(e, np.ErrDType) {
			t.Fatalf("pointer-managed dtype exposed binary storage: %v", e)
		}
	}
}

func TestCharacterDirectUFuncNumPy(t *testing.T) {
	m := refcheck.Require(t)
	lookup := refcheck.RequireValue[*np.UFunc](t)
	mul := lookup(np.LookupUFunc("multiply"))
	add := lookup(np.LookupUFunc("add"))
	for _, kind := range []string{"u", "t"} {
		t.Run(kind, func(t *testing.T) {
			makeText := func(v []string) *np.Array {
				if kind == "u" {
					return m(np.FromUnicodeStrings(v, 4, len(v)))
				}
				return m(np.FromStrings(v, len(v)))
			}
			a := makeText([]string{"abc", "x", "é"})
			out := m(makeText([]string{"old", "keep", "old"}).Slice(np.Reverse()))
			mask := m(np.FromSlice([]bool{true, false, true}, 3))
			if v := m(mul.CallWithOptions(np.UFuncOptions{Out: out, Where: mask}, a, m(np.FromSlice([]int64{3, 100, 2}, 3)))); v != out {
				t.Fatal("out identity")
			}
			refcheck.CheckCharacter(t, "strings_"+kind+"_ufunc_multiply", out, true)
			v, e := np.Multiply(a, np.ScalarOf(int64(2)))
			if kind == "u" {
				refcheck.Case(t, "strings_u_direct_multiply", v, e)
				return
			}
			refcheck.CheckCharacter(t, "strings_t_direct_multiply", m(v, e), false)
			mask = m(np.FromSlice([]bool{false, true, true}, 3))
			b := m(np.FromUnicodeStrings([]string{"!", "?", ":"}, 1, 3))
			refcheck.CheckCharacter(t, "strings_t_ufunc_add", m(add.CallWithOptions(np.UFuncOptions{Out: out, Where: mask}, a, b)), true)
		})
	}
}
