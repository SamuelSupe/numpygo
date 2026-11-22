package ma_test

import (
	"bytes"
	"encoding/json"
	"iter"
	"strconv"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/ma"
	"github.com/SamuelSupe/numpygo/records"
)

func TestMaskedArrayClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mm := refcheck.RequireValue[*ma.Array](t)
	ok := func(e error) {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
	}
	check := func(n string, a *np.Array) { t.Helper(); refcheck.Check(t, "subclass_ma_"+n, a) }
	yes := func(v bool) int64 {
		if v {
			return 1
		}
		return 0
	}
	ints := func(n string, v ...int64) { t.Helper(); check(n, m(np.FromSlice(v, len(v)))) }
	mask := func(v []bool, sh ...int) *np.Array { return m(np.FromSlice(v, sh...)) }
	number := func(v ...int64) *np.Array { return m(np.FromSlice(v, len(v))) }
	maCheck := func(n string, a *ma.Array, raw, meta bool) {
		t.Helper()
		check(n+"_mask", a.Mask())
		check(n+"_values", m(a.Filled(int64(0))))
		if raw {
			check(n+"_raw", a.Data())
		}
		if meta {
			ints(n+"_flags", yes(a.HardMask()), yes(a.SharedMask()), yes(ma.GetMask(a) == nil))
			check(n+"_fill", m(np.Full(np.Complex128, a.FillValue())))
		}
	}
	masked := func() *ma.Array {
		return mm(ma.New(refcheck.Input(t, "subclass_ma_input_raw"), refcheck.Input(t, "subclass_ma_input_mask"), ma.Options{FillValue: int16(-7), HardMask: true}))
	}
	a := masked()
	maCheck("input", a, true, true)
	length := refcheck.RequireValue[int](t)(a.Len())
	shape := []int64{int64(a.NDim()), int64(a.Size()), int64(a.ItemSize()), int64(a.NBytes()), int64(length), yes(a.IsContiguous())}
	for _, v := range append(a.Shape(), a.Strides()...) {
		shape = append(shape, int64(v))
	}
	ints("meta", shape...)
	refcheck.Text(t, "subclass_ma_baseclass", a.BaseClass())
	check("recordmask", a.RecordMask())
	check("filled", m(a.Filled()))
	check("filled_explicit", m(a.Filled(-99)))
	check("compressed", m(ma.Compressed(a)))
	item := refcheck.RequireValue[any](t)(a.Item(1))
	check("item", np.ScalarOf(int64(item.(int16))))
	for _, p := range []struct {
		n string
		v any
	}{
		{"tolist", refcheck.RequireValue[any](t)(a.ToList())}, {"tolist_fill", refcheck.RequireValue[any](t)(a.ToList(-99))},
	} {
		b, e := json.Marshal(p.v)
		ok(e)
		refcheck.Text(t, "subclass_ma_"+p.n, string(b))
	}
	for _, p := range []struct{ n, order string }{{"tobytes", "C"}, {"tobytes_f", "F"}} {
		b, e := a.ToBytes(p.order)
		ok(e)
		refcheck.Bytes(t, "subclass_ma_"+p.n, b)
	}
	for n, v := range map[string]*ma.Array{
		"copy": mm(a.Copy()), "astype": mm(a.Astype(np.Float32, np.CastOptions{})), "byteswap": mm(a.ByteSwap(false)),
		"real": mm(a.Real()), "imag": mm(a.Imag()), "view": mm(a.View()),
		"slice": mm(a.Index(np.Slice{}, np.Range(1, 3))), "advanced": mm(a.Index(number(0, 0))),
		"flatten": mm(a.Flatten()), "ravel": mm(a.Ravel()),
	} {
		maCheck(n, v, true, true)
	}
	c := mm(ma.New(m(np.FromSlice([]complex64{1 + 2i, 3 - 4i, 5 + 6i}, 3)), mask([]bool{false, true, false}, 3), ma.Options{FillValue: complex64(9 + 2i), HardMask: true}))
	maCheck("complex_real", mm(c.Real()), true, true)
	maCheck("complex_imag", mm(c.Imag()), true, true)
	maCheck("complex_conj", mm(ma.Conjugate(c)), false, false)
	axis := 1
	for n, v := range map[string]*ma.Array{
		"reshape": mm(a.Reshape(3, 2)), "transpose": mm(a.Transpose()), "mT": mm(a.MatrixTranspose()),
		"swapped": mm(ma.SwapAxes(a, 0, 1)), "squeeze": mm(ma.Squeeze(mm(a.Index(np.NewAxis, np.Ellipsis)), 0)),
		"diagonal": mm(ma.Diagonal(a, 0, 0, 1)), "repeat": mm(ma.Repeat(a, 2, &axis)),
		"take":     mm(ma.Take(a, number(2, 0), &axis)),
		"compress": mm(ma.Compress(mask([]bool{true, false, true}, 3), a, &axis)),
		"trace":    mm(ma.Trace(a, 0, 0, 1, 0)),
	} {
		maCheck(n, v, false, false)
	}
	i := 0
	for v, e := range refcheck.RequireValue[iter.Seq2[*ma.Array, error]](t)(a.Iter()) {
		ok(e)
		maCheck("iter_"+strconv.Itoa(i), v, false, false)
		i++
	}
	vv := mm(a.Index(np.Slice{}))
	av := mm(a.Index(number(0, 1)))
	d, mk := a.IDs()
	vd, vm := vv.IDs()
	ad, am := av.IDs()
	ints("ids", yes(d == vd), yes(mk == vm), yes(d == ad), yes(mk == am))
	v := mm(a.View())
	ok(v.Data().SetValue(int16(77), 0, 0))
	check("data_view_write", a.Data())
	a = masked()
	ok(a.Assign([]np.Index{np.AtIndex(0), np.Slice{}}, mm(ma.New(number(80, 90, 100), mask([]bool{true, false, false}, 3)))))
	maCheck("hard_set_masked_rhs", a, true, false)
	a = masked()
	ok(a.SetMask(np.ScalarOf(false)))
	maCheck("hard_mask_clear", a, true, true)
	ok(a.SetMask(mask([]bool{true, false, true, false, false, false}, 2, 3)))
	maCheck("hard_mask_or", a, true, true)
	refcheck.Error(t, "subclass_ma_hard_mask_bad_dtype", "UFuncTypeError", a.SetMask(m(np.Ones(np.Int64, 2, 3))), np.ErrDType)
	a.SoftenMask()
	ok(a.SetMask(nil))
	maCheck("soft_mask_clear", a, true, true)
	ok(a.ShrinkMask())
	ints("shrink_nomask", yes(ma.GetMask(a) == nil))
	check("shrink_recordmask", a.RecordMask())
	a = masked()
	a.SoftenMask()
	v = mm(a.Index(np.Slice{}))
	ok(v.UnshareMask())
	ok(v.SetMask(np.ScalarOf(false)))
	maCheck("unshare_view", v, true, true)
	maCheck("unshare_base", a, true, true)
	ok(v.ShrinkMask())
	ints("unshare_shrink", yes(ma.GetMask(v) == nil), yes(v.SharedMask()))
	nm := mm(ma.New(number(1, 2), nil))
	ints("nomask", yes(ma.GetMask(nm) == nil), yes(nm.SharedMask()))
	ok(nm.SetMask(np.ScalarOf(false)))
	ints("nomask_explicit", yes(ma.GetMask(nm) == nil), yes(nm.SharedMask()))
	a = masked()
	ok(a.SetFillValue(int16(-21)))
	check("get_fill", m(np.Full(np.Complex128, a.FillValue())))
	ok(a.Fill(9))
	maCheck("fill_data", a, true, true)
	a = masked()
	ok(ma.Put(mm(a.Ravel()), number(1, 2), mm(ma.New(number(80, 90), nil)), "raise"))
	maCheck("put", a, true, false)
	a.SoftenMask()
	ok(a.Assign([]np.Index{np.AtIndex(0), np.AtIndex(1)}, mm(ma.New(np.ScalarOf(int64(24)), nil))))
	maCheck("setitem", a, true, false)
	a.HardenMask()
	ints("harden", yes(a.HardMask()))
	for i, h := range []bool{false, true} {
		b := mm(ma.New(number(3, 4, 1, 2), mask([]bool{false, true, false, true}, 4), ma.Options{HardMask: h}))
		ok(b.SortInPlace(np.Int(0), true))
		maCheck("sort_"+strconv.Itoa(i), b, true, false)
	}
	b := mm(ma.New(number(3, 4, 1, 2), mask([]bool{false, true, false, true}, 4)))
	check("argsort", m(ma.Argsort(b, np.Int(0), true)))
	check("argpartition", m(b.ArgPartition([]int{0, 1, 2, 3}, np.Int(0))))
	ok(b.PartitionInPlace([]int{0, 1, 2, 3}, np.Int(0)))
	maCheck("partition", b, true, false)
	check("searchsorted", m(mm(ma.New(number(1, 3, 5, 7), mask([]bool{false, true, false, false}, 4))).SearchSorted(number(2, 5, 8), false)))
	a = masked()
	for _, n := range []string{"toflex", "torecords"} {
		r := refcheck.RequireValue[*records.Array](t)(a.ToFlex())
		check(n+"_data", m(r.NDArray().Field("_data")))
		check(n+"_mask", m(r.NDArray().Field("_mask")))
		layout := []int64{int64(r.Schema().ItemSize())}
		for _, f := range r.Schema().Fields() {
			layout = append(layout, int64(f.Offset))
		}
		ints(n+"_layout", layout...)
		refcheck.Text(t, "subclass_ma_"+n+"_names", strings.Join(r.Schema().Names(), "|"))
		refcheck.Bytes(t, "subclass_ma_"+n+"_bytes", r.Bytes())
	}
	refcheck.Error(t, "subclass_ma_resize", "ValueError", a.ResizeInPlace(6), np.ErrValue)
	var buf bytes.Buffer
	refcheck.Error(t, "subclass_ma_tofile", "NotImplementedError", a.ToFile(&buf), ma.ErrMA)
	if buf.Len() != 0 {
		t.Fatal("unsupported tofile wrote data")
	}
	g := mm(ma.New(m(np.FromSlice([]float64{3, -2, 1, 5, 4, 0}, 2, 3)), mask([]bool{false, true, false, true, false, false}, 2, 3)))
	o := np.SumOptions{Axes: []int{1}, KeepDims: true}
	for n, fn := range map[string]func(*ma.Array, ...np.SumOptions) (*ma.Array, error){
		"all": ma.All, "any": ma.Any, "sum": ma.Sum, "prod": ma.Prod, "product": ma.Prod, "min": ma.Min, "max": ma.Max, "mean": ma.Mean, "ptp": ma.PTP,
	} {
		maCheck("reduce_"+n, mm(fn(g, o)), false, false)
	}
	vo := np.VarOptions{Axes: []int{1}, KeepDims: true, DDOF: 1}
	maCheck("reduce_var", mm(ma.Var(g, vo)), false, false)
	maCheck("reduce_std", mm(ma.Std(g, vo)), false, false)
	check("argmin", m(ma.ArgMin(g, np.ArgOptions{Axis: &axis})))
	check("argmax", m(ma.ArgMax(g, np.ArgOptions{Axis: &axis})))
	check("count", m(ma.Count(g, o)))
	maCheck("cumsum", mm(ma.Cumsum(g, &axis, 0)), false, false)
	maCheck("cumprod", mm(ma.Cumprod(g, &axis, 0)), false, false)
	maCheck("anom", mm(ma.Anomalies(g, np.SumOptions{Axes: []int{1}})), false, false)
	maCheck("round", mm(ma.Round(g, 1)), false, false)
	maCheck("clip", mm(ma.Clip(g, np.Scalar(1), np.Scalar(3))), false, false)
	maCheck("dot", mm(ma.Dot(g, mm(ma.New(m(np.FromSlice([]float64{1, 2, 3}, 3)), mask([]bool{false, true, false}, 3))))), false, false)
	for i, v := range refcheck.RequireValue[[]*np.Array](t)(ma.Nonzero(g)) {
		check("nonzero_"+strconv.Itoa(i), v)
	}
}

func TestMaskedIteratorClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mm := refcheck.RequireValue[*ma.Array](t)
	a := mm(ma.New(m(np.FromSlice([]int64{1, 2, 3, 4, 5, 6}, 2, 3)), m(np.FromSlice([]bool{false, true, false, false, false, true}, 2, 3))))
	fm := refcheck.RequireValue[*ma.MaskedIterator](t)
	f := fm(a.FlatIterator())
	check := func(n string, v *np.Array) { t.Helper(); refcheck.Check(t, "subclass_ma_"+n, v) }
	yes := int64(0)
	if f.Base() == a {
		yes = 1
	}
	check("flat_iterators", m(np.FromSlice([]int64{yes, int64(f.DataIterator().Index()), int64(f.MaskIterator().Index())}, 3)))
	values := []int64{}
	mask := []bool{}
	for {
		v, masked, ok, e := f.Next()
		if e != nil {
			t.Fatal(e)
		}
		if !ok {
			break
		}
		n := v.(int64)
		if masked {
			n = 0
		}
		values = append(values, n)
		mask = append(mask, masked)
	}
	check("flat_values", m(np.FromSlice(values, len(values))))
	check("flat_mask", m(np.FromSlice(mask, len(mask))))
	f = fm(a.FlatIterator())
	maCheck := func(n string, v *ma.Array, meta bool) {
		check(n+"_raw", v.Data())
		check(n+"_mask", v.Mask())
		check(n+"_values", m(v.Filled(int64(0))))
		if meta {
			flags := []int64{0, 0, 0}
			if v.HardMask() {
				flags[0] = 1
			}
			if v.SharedMask() {
				flags[1] = 1
			}
			if ma.GetMask(v) == nil {
				flags[2] = 1
			}
			check(n+"_flags", m(np.FromSlice(flags, 3)))
			check(n+"_fill", m(np.Full(np.Complex128, v.FillValue())))
		}
	}
	maCheck("flat_slice", mm(f.Get(np.Slice{Start: np.Int(1), Stop: np.Int(5), Step: np.Int(2)})), true)
	maCheck("flat_gather", mm(f.Get(m(np.FromSlice([]int64{0, 3, 5}, 3)))), true)
	if e := f.Assign(np.Range(1, 5), mm(ma.New(m(np.FromSlice([]int64{9, 8}, 2)), nil))); e != nil {
		t.Fatal(e)
	}
	maCheck("flat_assign", a, false)
	if e := f.Assign(m(np.FromSlice([]int64{0, 0, 5}, 3)), mm(ma.New(m(np.FromSlice([]int64{1, 2, 3}, 3)), m(np.FromSlice([]bool{false, false, true}, 3))))); e != nil {
		t.Fatal(e)
	}
	maCheck("flat_repeat", a, false)
}
