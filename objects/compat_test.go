package objects

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestObjectUFunc(t *testing.T) {
	m := refcheck.Require(t)
	arr := refcheck.RequireValue[*Array](t)
	uf := refcheck.RequireValue[*UFunc](t)
	outs := refcheck.RequireValue[[]*Array](t)
	compare := func(name string, a *Array) {
		v := make([]int64, a.Size())
		for i, x := range a.Values() {
			v[i] = x.(int64)
		}
		refcheck.Check(t, "final_object_"+name, m(np.WrapSlice(v, a.Shape()...)))
	}
	f := uf(FromPyFunc("pair", 2, 2, func(v ...any) ([]any, error) { x, y := v[0].(int64), v[1].(int64); return []any{x + y, x * y}, nil }))
	x := arr(FromSlice([]any{int64(1), int64(2)}, 2, 1))
	y := arr(FromSlice([]any{int64(10), int64(20), int64(30)}, 3))
	o := outs(f.Apply(x, y))
	compare("callback_sum", o[0])
	compare("callback_product", o[1])
	refcheck.Check(t, "final_object_callback_meta", m(np.WrapSlice([]int64{int64(f.NIn), int64(f.NOut)}, 2)))
	g := uf(FromPyFunc("add", 2, 1, func(v ...any) ([]any, error) { return []any{v[0].(int64) + v[1].(int64)}, nil }))
	a := arr(FromSlice([]any{int64(1), int64(2), int64(3)}, 3))
	compare("reduce", arr(g.Reduce(a, 0, false)))
	compare("accumulate", arr(g.Accumulate(a, 0)))
	compare("outer", arr(g.Outer(a, arr(FromSlice([]any{int64(10), int64(20)}, 2)))))
	if e := g.At(a, []int{0, 0, 2}, arr(FromSlice([]any{int64(4), int64(5), int64(6)}, 3))); e != nil {
		t.Fatal(e)
	}
	compare("at", a)
	refcheck.Text(t, "final_object_type", Scalar("red").Values()[0].(string))
}
