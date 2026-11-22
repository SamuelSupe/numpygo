package np_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestUFuncReductionOptions(t *testing.T) {
	m := refcheck.Require(t)
	lookup := refcheck.RequireValue[*np.UFunc](t)
	a := m(refcheck.Input(t, "behavior_reduceopt_input").Transpose(2, 1, 0))
	a = m(a.Slice(np.Reverse()))
	mask := refcheck.Input(t, "behavior_reduceopt_mask")
	for _, tc := range []struct {
		name    string
		initial any
	}{
		{"add", int64(3)}, {"multiply", int64(2)}, {"maximum", int64(-100)}, {"minimum", int64(100)}, {"bitwise_xor", int64(1)}, {"gcd", int64(6)}, {"logical_or", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			u := lookup(np.LookupUFunc(tc.name))
			refcheck.Check(t, "behavior_reduceopt_"+tc.name, m(u.ReduceWithOptions(a, np.ReduceOptions{Axes: []int{0, 2}, KeepDims: true, Initial: tc.initial, Where: mask})))
		})
	}
	sub := lookup(np.LookupUFunc("subtract"))
	add := lookup(np.LookupUFunc("add"))
	maximum := lookup(np.LookupUFunc("maximum"))
	refcheck.Check(t, "behavior_reduceopt_subtract", m(sub.ReduceWithOptions(a, np.ReduceOptions{Axes: []int{0}, Initial: int64(100), Where: mask})))
	refcheck.Check(t, "behavior_reduceopt_divide", m(lookup(np.LookupUFunc("divide")).ReduceWithOptions(m(np.WrapSlice([]int32{2, 4, 8}, 3)), np.ReduceOptions{})))
	base := m(np.Full(np.Float32, float32(-1), 3))
	out := m(base.Slice(np.Reverse()))
	if got := m(add.ReduceWithOptions(a, np.ReduceOptions{Axes: []int{0, 2}, Where: mask, Out: out})); got != out {
		t.Fatal("out replaced")
	}
	refcheck.Check(t, "behavior_reduceopt_out_base", base)
	refcheck.Check(t, "behavior_reduceopt_scalar", m(add.ReduceWithOptions(np.ScalarOf(int16(7)), np.ReduceOptions{Axes: []int{0}, KeepDims: true, Initial: int64(3)})))
	empty := m(np.Zeros(np.Float64, 0, 3))
	refcheck.Check(t, "behavior_reduceopt_empty", m(maximum.ReduceWithOptions(empty, np.ReduceOptions{Axes: []int{0}, Initial: float64(-2)})))
	refcheck.Check(t, "behavior_reduceopt_empty_mask", m(add.ReduceWithOptions(a, np.ReduceOptions{Axes: []int{2}, Where: np.ScalarOf(false), Initial: int64(5)})))
	_, e := maximum.ReduceWithOptions(a, np.ReduceOptions{Axes: []int{0}, Where: mask})
	refcheck.Error(t, "behavior_reduceopt_missing_initial", "ValueError", e, np.ErrValue)
	_, e = sub.ReduceWithOptions(a, np.ReduceOptions{Axes: []int{0, 2}})
	refcheck.Error(t, "behavior_reduceopt_nonreorderable", "ValueError", e, np.ErrValue)
	_, e = lookup(np.LookupUFunc("minimum")).ReduceWithOptions(empty, np.ReduceOptions{Axes: []int{0}})
	refcheck.Error(t, "behavior_reduceopt_empty_no_identity", "ValueError", e, np.ErrValue)
}
func TestUFuncAccumulateAndReduceAt(t *testing.T) {
	m := refcheck.Require(t)
	lookup := refcheck.RequireValue[*np.UFunc](t)
	a := m(refcheck.Input(t, "behavior_reduceopt_input").Transpose(2, 1, 0))
	a = m(a.Slice(np.Reverse()))
	for _, name := range []string{"maximum", "minimum", "bitwise_xor", "gcd", "subtract", "logical_xor"} {
		t.Run("accumulate/"+name, func(t *testing.T) {
			refcheck.Check(t, "behavior_accumulate_"+name, m(lookup(np.LookupUFunc(name)).Accumulate(a, -2, 0)))
		})
	}
	for _, name := range []string{"add", "multiply", "maximum", "subtract"} {
		t.Run("reduceat/"+name, func(t *testing.T) {
			refcheck.Check(t, "behavior_reduceat_"+name, m(lookup(np.LookupUFunc(name)).ReduceAt(a, []int{0, 2, 2, 1}, 1, 0, nil)))
		})
	}
	_, e := lookup(np.LookupUFunc("add")).ReduceAt(a, []int{-1}, 1, 0, nil)
	refcheck.Error(t, "behavior_reduceat_index", "IndexError", e, np.ErrIndex)
}
