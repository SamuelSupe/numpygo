package ma_test

import (
	"errors"
	"strconv"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/ma"
)

func remaining(t *testing.T) (func(string) *np.Array, func(*ma.Array, error) *ma.Array, func(string, *ma.Array), func(string, *np.Array)) {
	in := func(n string) *np.Array { return refcheck.Input(t, "ma_remaining_"+n) }
	must := func(a *ma.Array, e error) *ma.Array {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
		return a
	}
	check := func(n string, a *np.Array) { refcheck.Check(t, "ma_remaining_"+n, a) }
	masked := func(n string, a *ma.Array) {
		t.Helper()
		check(n+"_mask", a.Mask())
		check(n+"_values", refcheck.Require(t)(a.Filled(int64(0))))
	}
	return in, must, masked, check
}
func TestRemainingMaskMetadata(t *testing.T) {
	in, must, check, raw := remaining(t)
	plain := refcheck.Require(t)
	a := must(ma.New(in("a_data"), in("a_mask"), ma.Options{FillValue: int64(-99)}))
	_, maskErr := ma.New(plain(np.FromSlice([]int64{1, 2, 3}, 3)), plain(np.FromSlice([]bool{true, false}, 2)))
	refcheck.Error(t, "ma_remaining_mask_error", "MaskError", maskErr, ma.ErrMask)
	raw("mask_error_inheritance", np.ScalarOf(errors.Is(maskErr, ma.ErrMA)))
	b := must(a.Copy())
	check("copy", b)
	if e := b.Assign([]np.Index{np.AtIndex(0), np.AtIndex(0)}, must(ma.New(np.ScalarOf(int64(41)), nil))); e != nil {
		t.Fatal(e)
	}
	check("copy_mutation", b)
	check("copy_original", a)
	b = must(a.Copy())
	if e := b.SetFillValue(-17); e != nil {
		t.Fatal(e)
	}
	raw("set_fill_value", plain(b.Filled()))
	raw("fill_shared", plain(a.Filled()))
	y := must(ma.New(plain(np.FromSlice([]int64{1, 2}, 2)), nil, ma.Options{FillValue: int64(-17)}))
	v, e := ma.CommonFillValue(a, y)
	if e != nil {
		t.Fatal(e)
	}
	raw("common_fill_value", plain(np.Full(np.Int64, v)))
	if e = y.SetFillValue(42); e != nil {
		t.Fatal(e)
	}
	v, e = ma.CommonFillValue(a, y)
	if e != nil {
		t.Fatal(e)
	}
	raw("common_fill_mismatch", np.ScalarOf(v == nil))
	for _, d := range []np.DType{np.Int8, np.Int64, np.Uint8, np.Float32, np.Complex64, np.Bool} {
		name := d.String()
		w := in("default_fill_" + name)
		raw("default_fill_"+name, plain(np.Full(w.DType(), ma.DefaultFillValue(d))))
		raw("minimum_fill_"+name, plain(np.Full(d, ma.MinimumFillValue(d))))
		raw("maximum_fill_"+name, plain(np.Full(d, ma.MaximumFillValue(d))))
	}
	raw("flatten_mask", plain(ma.FlattenMask(a.Mask())))
	for _, fn := range []string{"isMA", "isMaskedArray", "isarray"} {
		raw(fn, plain(np.FromSlice([]bool{ma.IsMaskedArray(a), ma.IsMaskedArray(a.Data())}, 2)))
	}
	for _, fn := range []string{"MaskType", "bool_"} {
		raw(fn, plain(plain(np.FromSlice([]int64{0, 3}, 2)).Astype(np.Bool)))
	}
	sh := []int64{}
	for _, n := range a.Shape() {
		sh = append(sh, int64(n))
	}
	raw("shape", plain(np.FromSlice(sh, len(sh))))
	raw("ndim", np.ScalarOf(int64(a.NDim())))
	raw("size", np.ScalarOf(int64(a.Size())))
	ea := must(ma.Empty(np.Int32, 2, 3))
	if e = ea.Data().Fill(7); e != nil {
		t.Fatal(e)
	}
	check("empty", ea)
	ea = must(ma.EmptyLike(a, np.Float32, "K"))
	if e = ea.Data().Fill(6.); e != nil {
		t.Fatal(e)
	}
	check("empty_like", ea)
	data := plain(np.FromSlice([]int32{0, 1, 2, 3, 4}, 5))
	bytes, e := data.ToBytes("C")
	if e != nil {
		t.Fatal(e)
	}
	check("frombuffer", must(ma.FromBuffer(bytes, np.Int32, 3, 0, nil)))
	check("fromfunction", must(ma.FromFunction([]int{2, 3}, np.Int64, func(grid ...*np.Array) (*np.Array, error) {
		return np.Add(plain(np.Multiply(grid[0], np.ScalarOf(int64(4)))), grid[1])
	})))
	check("indices", must(ma.Indices([]int{2, 3})))
}

func TestRemainingMaskSets(t *testing.T) {
	in, must, check, raw := remaining(t)
	a := must(ma.New(in("x_data"), in("x_mask")))
	b := must(ma.New(in("y_data"), in("y_mask")))
	u, e := ma.Unique(a)
	if e != nil {
		t.Fatal(e)
	}
	check("unique", u.Values)
	raw("unique_indices", u.Indices)
	raw("unique_inverse", u.Inverse)
	check("isin", must(ma.Isin(a, b, false)))
	check("in1d", must(ma.In1D(a, b, false)))
	check("isin_invert", must(ma.Isin(a, b, true)))
	for name, fn := range map[string]maskedBinary{"intersect1d": ma.Intersect1D, "union1d": ma.Union1D, "setdiff1d": ma.Setdiff1D, "setxor1d": ma.Setxor1D} {
		check(name, must(fn(a, b)))
	}
	plain := refcheck.Require(t)
	check("setdiff1d_one_mask", must(ma.Setdiff1D(a, must(ma.New(plain(np.FromSlice([]int64{1, 2}, 2)), nil)))))
}

func TestRemainingMaskManipulation(t *testing.T) {
	in, must, check, raw := remaining(t)
	plain := refcheck.Require(t)
	a := must(ma.New(in("a_data"), in("a_mask")))
	cube := must(ma.New(in("cube_data"), in("cube_mask")))
	raw("compress_nd", plain(ma.CompressND(cube, []int{1, 2})))
	raw("compress_rowcols", plain(ma.CompressRowCols(a, nil)))
	raw("compress_all_masked", plain(ma.CompressND(must(ma.MaskedAll(np.Int32, 2, 3)), nil)))
	check("mask_rowcols", must(ma.MaskRowCols(must(a.Copy()), nil)))
	raw("flatnotmasked_edges", plain(ma.FlatNotMaskedEdges(a)))
	ranges := func(name string, v [][2]int) {
		out := []int64{}
		for _, p := range v {
			out = append(out, int64(p[0]), int64(p[1]))
		}
		raw(name, plain(np.FromSlice(out, len(v), 2)))
	}
	flat, e := ma.FlatNotMaskedContiguous(a)
	if e != nil {
		t.Fatal(e)
	}
	ranges("flatnotmasked_contiguous", flat)
	groups, e := ma.NotMaskedContiguous(a, np.Int(1))
	if e != nil {
		t.Fatal(e)
	}
	for j, v := range groups {
		ranges("notmasked_contiguous_"+strconv.Itoa(j), v)
	}
	first, last, e := ma.NotMaskedEdges(a, 1)
	if e != nil {
		t.Fatal(e)
	}
	for i, group := range [][]*np.Array{first, last} {
		for j, v := range group {
			raw("notmasked_edges_"+strconv.Itoa(i)+"_"+strconv.Itoa(j), v)
		}
	}
	it, e := ma.NDEnumerate(a, true)
	if e != nil {
		t.Fatal(e)
	}
	var indices, values []int64
	for idx, v := range it {
		for _, i := range idx {
			indices = append(indices, int64(i))
		}
		values = append(values, v.Value.(int64))
	}
	raw("ndenumerate_indices", plain(np.FromSlice(indices, len(values), a.NDim())))
	raw("ndenumerate_values", plain(np.FromSlice(values, len(values))))
	it, e = ma.NDEnumerate(a, false)
	if e != nil {
		t.Fatal(e)
	}
	var masks []bool
	for _, v := range it {
		masks = append(masks, v.Masked)
	}
	raw("ndenumerate_all_mask", plain(np.FromSlice(masks, len(masks))))
	check("apply_along_axis", must(ma.ApplyAlongAxis(a, 1, func(x *ma.Array) (*ma.Array, error) { return ma.Sum(x) })))
	check("apply_along_axis_vector", must(ma.ApplyAlongAxis(a, 1, func(x *ma.Array) (*ma.Array, error) {
		return ma.Stack([]*ma.Array{must(ma.Min(x)), must(ma.Max(x))}, 0)
	})))
	check("apply_over_axes", must(ma.ApplyOverAxes(cube, []int{0, 2}, func(x *ma.Array, axis int) (*ma.Array, error) { return ma.Sum(x, np.SumOptions{Axes: []int{axis}}) })))
	v := must(ma.New(plain(np.FromSlice([]int64{20, 30, 40}, 3)), plain(np.FromSlice([]bool{false, true, false}, 3))))
	idx := plain(np.FromSlice([]int64{1, 5, 7}, 3))
	b := must(a.Copy())
	if e = ma.Put(b, idx, v, "raise"); e != nil {
		t.Fatal(e)
	}
	check("put", b)
	b = must(ma.Ravel(must(a.Copy())))
	b.HardenMask()
	if e = ma.Put(b, plain(np.FromSlice([]int64{1, 2}, 2)), must(ma.New(plain(np.FromSlice([]int64{70, 80}, 2)), nil)), "raise"); e != nil {
		t.Fatal(e)
	}
	check("put_hard", b)
	mask := plain(np.Equal(plain(np.Remainder(a.Data(), np.ScalarOf(int64(2)))), np.ScalarOf(int64(0))))
	v = must(ma.New(plain(np.FromSlice([]int64{100, 200, 300}, 3)), plain(np.FromSlice([]bool{false, true, false}, 3))))
	b = must(a.Copy())
	if e = ma.Putmask(b, mask, v); e != nil {
		t.Fatal(e)
	}
	check("putmask", b)
	b = must(a.Copy())
	b.HardenMask()
	if e = ma.Putmask(b, mask, v); e != nil {
		t.Fatal(e)
	}
	check("putmask_hard", b)
}

func TestRemainingMaskNumerical(t *testing.T) {
	in, must, check, raw := remaining(t)
	plain := refcheck.Require(t)
	a := must(ma.New(in("signal_lhs_data"), in("signal_lhs_mask")))
	b := must(ma.New(in("signal_rhs_data"), in("signal_rhs_mask")))
	for _, propagate := range []bool{true, false} {
		suffix := "_0"
		if propagate {
			suffix = "_1"
		}
		check("convolve"+suffix, must(ma.Convolve(a, b, np.FullMode, propagate)))
		check("correlate"+suffix, must(ma.Correlate(a, b, np.SameMode, propagate)))
	}
	mat := must(ma.New(in("cov_data"), in("cov_input_mask")))
	check("cov", must(ma.Cov(mat, nil, true, true, 1)))
	check("cov_rowvar_false", must(ma.Cov(must(ma.Transpose(mat)), nil, false, true, 0)))
	check("corrcoef", must(ma.Corrcoef(mat, nil, true, true)))
	r0, r1 := must(mat.Index(np.AtIndex(0))), must(mat.Index(np.AtIndex(1)))
	check("corrcoef_scalar", must(ma.Corrcoef(r0, nil, true, true)))
	check("cov_pair", must(ma.Cov(r0, r1, true, true, 1)))
	_, e := ma.Cov(mat, nil, true, false, 1)
	refcheck.Error(t, "ma_remaining_cov_masked", "ValueError", e, np.ErrValue)
	raw("vander", plain(ma.Vander(must(ma.New(in("x_data"), in("x_mask"))), 4)))
	x := must(ma.New(in("fit_x_data"), in("fit_x_mask")))
	y := must(ma.New(in("fit_y_data"), in("fit_y_mask")))
	w := must(ma.New(in("fit_w_data"), in("fit_w_mask")))
	fit, e := ma.Polyfit(x, y, w, 2)
	if e != nil {
		t.Fatal(e)
	}
	raw("polyfit", fit.Coefficients)
}
