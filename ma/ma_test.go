package ma_test

import (
	"strconv"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/ma"
)

type maskedUnary func(*ma.Array) (*ma.Array, error)
type maskedBinary func(*ma.Array, *ma.Array) (*ma.Array, error)

func helpers(t *testing.T) (func(string) *np.Array, func(*ma.Array, error) *ma.Array, func(string, *ma.Array), func(string, *np.Array)) {
	in := func(name string) *np.Array { return refcheck.Input(t, "ma_"+name) }
	must := func(a *ma.Array, e error) *ma.Array {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
		return a
	}
	raw := func(name string, a *np.Array) { t.Helper(); refcheck.Check(t, "ma_"+name, a) }
	check := func(name string, a *ma.Array) {
		t.Helper()
		raw(name+"_mask", a.Mask())
		raw(name+"_values", refcheck.Require(t)(a.Filled(int64(0))))
	}
	return in, must, check, raw
}

func TestMaskedOperations(t *testing.T) {
	in, must, check, _ := helpers(t)
	x := must(ma.New(in("x_data"), in("x_input_mask")))
	a := must(ma.New(in("a_data"), in("a_input_mask")))
	b := must(ma.New(in("b_data"), in("b_input_mask")))
	g := must(ma.New(in("g_data"), in("g_input_mask")))
	for name, fn := range map[string]maskedUnary{
		"sqrt": ma.Sqrt, "log": ma.Log, "log2": ma.Log2, "log10": ma.Log10, "arcsin": ma.Arcsin, "arccos": ma.Arccos, "arccosh": ma.Arccosh, "arctanh": ma.Arctanh, "tan": ma.Tan,
		"exp": ma.Exp, "conjugate": ma.Conjugate, "sin": ma.Sin, "cos": ma.Cos, "arctan": ma.Arctan, "arcsinh": ma.Arcsinh, "sinh": ma.Sinh, "cosh": ma.Cosh, "tanh": ma.Tanh, "absolute": ma.Absolute, "fabs": ma.Fabs, "negative": ma.Negative, "floor": ma.Floor, "ceil": ma.Ceil, "logical_not": ma.LogicalNot,
	} {
		t.Run(name, func(t *testing.T) { _, m, c, _ := helpers(t); c(name, m(fn(x))) })
	}
	for name, fn := range map[string]maskedBinary{
		"add": ma.Add, "subtract": ma.Subtract, "multiply": ma.Multiply, "divide": ma.Divide, "floor_divide": ma.FloorDivide, "remainder": ma.Remainder, "fmod": ma.Fmod, "power": ma.Power, "arctan2": ma.Arctan2, "equal": ma.Equal, "not_equal": ma.NotEqual, "less": ma.Less, "less_equal": ma.LessEqual, "greater": ma.Greater, "greater_equal": ma.GreaterEqual,
		"logical_and": ma.LogicalAnd, "logical_or": ma.LogicalOr, "logical_xor": ma.LogicalXor, "hypot": ma.Hypot, "minimum": ma.Minimum, "maximum": ma.Maximum,
	} {
		t.Run(name, func(t *testing.T) { _, m, c, _ := helpers(t); c(name, m(fn(a, b))) })
	}
	other := must(ma.New(refcheck.Require(t)(np.FromSlice([]int64{1, 2, 0}, 3)), nil))
	for name, fn := range map[string]maskedBinary{"bitwise_and": ma.BitwiseAnd, "bitwise_or": ma.BitwiseOr, "bitwise_xor": ma.BitwiseXor, "left_shift": ma.LeftShift, "right_shift": ma.RightShift} {
		t.Run(name, func(t *testing.T) { _, m, c, _ := helpers(t); c(name, m(fn(g, other))) })
	}
	check("angle", must(ma.Angle(x, true)))
	check("round", must(ma.Round(a, 1)))
	check("clip", must(ma.Clip(a, np.ScalarOf(-1.), np.ScalarOf(2.))))
}

func TestMaskedReduction(t *testing.T) {
	in, must, check, raw := helpers(t)
	plain := refcheck.Require(t)
	a := must(ma.New(in("a_data"), in("a_input_mask")))
	o := np.SumOptions{Axes: []int{1}, KeepDims: true}
	for name, fn := range map[string]func(*ma.Array, ...np.SumOptions) (*ma.Array, error){
		"sum": ma.Sum, "prod": ma.Prod, "min": ma.Min, "max": ma.Max, "all": ma.All, "any": ma.Any, "ptp": ma.PTP, "mean": ma.Mean, "median": ma.Median,
	} {
		t.Run(name, func(t *testing.T) { _, m, c, _ := helpers(t); c(name, m(fn(a, o))) })
	}
	v := np.VarOptions{Axes: []int{1}, DDOF: 1, KeepDims: true}
	check("var", must(ma.Var(a, v)))
	check("std", must(ma.Std(a, v)))
	for _, d := range []string{"float16", "float32", "complex64"} {
		z := must(ma.New(in("typed_"+d+"_data"), in("typed_"+d+"_mask")))
		o := np.SumOptions{Axes: []int{1}}
		v := np.VarOptions{Axes: []int{1}}
		t.Run(d, func(t *testing.T) {
			_, m, c, _ := helpers(t)
			c("mean_"+d, m(ma.Mean(z, o)))
			c("median_"+d, m(ma.Median(z, o)))
			c("var_"+d, m(ma.Var(z, v)))
			c("std_"+d, m(ma.Std(z, v)))
		})
	}
	raw("count", plain(ma.Count(a, o)))
	raw("count_masked", plain(ma.CountMasked(a, np.SumOptions{Axes: []int{1}})))
	axis := 1
	raw("argmin", plain(ma.ArgMin(a, np.ArgOptions{Axis: &axis})))
	raw("argmax", plain(ma.ArgMax(a, np.ArgOptions{Axis: &axis})))
	check("cumsum", must(ma.Cumsum(a, &axis, 0)))
	check("cumprod", must(ma.Cumprod(a, &axis, 0)))
	check("anomalies", must(ma.Anomalies(a, np.SumOptions{Axes: []int{1}})))
	w := must(ma.New(in("weights_data"), in("weights_mask")))
	avg, total, e := ma.Average(a, w, np.SumOptions{Axes: []int{1}})
	if e != nil {
		t.Fatal(e)
	}
	check("average", avg)
	check("average_weight", total)
	eq, e := ma.AllClose(a, a, 1e-5, 1e-8, true)
	if e != nil {
		t.Fatal(e)
	}
	raw("allclose", np.ScalarOf(eq))
	eq, e = ma.AllEqual(a, a, true)
	if e != nil {
		t.Fatal(e)
	}
	raw("allequal", np.ScalarOf(eq))
}

func TestMaskedConstructionAndAssignment(t *testing.T) {
	in, must, check, raw := helpers(t)
	plain := refcheck.Require(t)
	g := must(ma.New(in("g_data"), in("g_input_mask")))
	x := must(ma.New(in("x_data"), in("x_input_mask")))
	a := must(ma.New(in("a_data"), in("a_input_mask")))
	raw("filled", plain(g.Filled(-9)))
	raw("compressed", plain(ma.Compressed(g)))
	raw("getdata", ma.GetData(g))
	raw("getmask", ma.GetMask(g))
	raw("getmaskarray", ma.GetMaskArray(g))
	raw("is_masked", np.ScalarOf(ma.IsMasked(g)))
	raw("is_mask", np.ScalarOf(ma.IsMask(g.Mask())))
	ints := func(v ...int64) *np.Array { return plain(np.FromSlice(v, len(v))) }
	raw("make_mask", plain(ma.MakeMask(ints(0, 1, 2))))
	raw("make_mask_none", plain(ma.MakeMaskNone(2, 3)))
	raw("mask_or", plain(ma.MaskOr(ints(0, 1, 0), ints(1, 0, 0))))
	check("masked_all", must(ma.MaskedAll(np.Int32, 2, 3)))
	check("masked_all_like", must(ma.MaskedAllLike(g)))
	check("masked_where", must(ma.MaskedWhere(plain(np.Greater(g.Data(), np.ScalarOf(int64(4)))), g)))
	for name, fn := range map[string]func(*ma.Array, *np.Array) (*ma.Array, error){
		"masked_equal": ma.MaskedEqual, "masked_not_equal": ma.MaskedNotEqual, "masked_greater": ma.MaskedGreater, "masked_greater_equal": ma.MaskedGreaterEqual, "masked_less": ma.MaskedLess, "masked_less_equal": ma.MaskedLessEqual,
	} {
		check(name, must(fn(g, np.ScalarOf(int64(4)))))
	}
	lo, hi := np.ScalarOf(int64(2)), np.ScalarOf(int64(5))
	check("masked_inside", must(ma.MaskedInside(g, lo, hi)))
	check("masked_outside", must(ma.MaskedOutside(g, lo, hi)))
	check("masked_values", must(ma.MaskedValues(a, np.ScalarOf(4.), 1e-5, 1e-8)))
	check("masked_invalid", must(ma.MaskedInvalid(x)))
	check("fix_invalid", must(ma.FixInvalid(x, 99.)))
	h := must(g.Copy())
	h.HardenMask()
	idx := []np.Index{np.AtIndex(0), np.Slice{}}
	if e := h.Assign(idx, must(ma.New(np.ScalarOf(int64(9)), nil))); e != nil {
		t.Fatal(e)
	}
	check("hardmask", h)
	h.SoftenMask()
	if e := h.Assign(idx, must(ma.New(np.ScalarOf(int64(8)), nil))); e != nil {
		t.Fatal(e)
	}
	check("softmask", h)
}

func TestMaskedManipulation(t *testing.T) {
	in, must, check, raw := helpers(t)
	plain := refcheck.Require(t)
	g := must(ma.New(in("g_data"), in("g_input_mask")))
	ints := func(v ...int64) *np.Array { return plain(np.FromSlice(v, len(v))) }
	axis := 1
	r0, r1 := must(g.Index(np.AtIndex(0))), must(g.Index(np.AtIndex(1)))
	check("reshape", must(ma.Reshape(g, 3, 2)))
	check("transpose", must(ma.Transpose(g)))
	check("ravel", must(ma.Ravel(g)))
	for k, f := range []maskedUnary{ma.Atleast1D, ma.Atleast2D, ma.Atleast3D} {
		check("atleast"+strconv.Itoa(k+1), must(f(r0)))
	}
	expanded := must(ma.ExpandDims(g, 0))
	check("expand_dims", expanded)
	check("squeeze", must(ma.Squeeze(expanded)))
	check("swapaxes", must(ma.SwapAxes(g, 0, 1)))
	check("resize", must(ma.Resize(g, 3, 4)))
	check("repeat", must(ma.Repeat(g, 2, &axis)))
	check("take", must(ma.Take(g, ints(2, 0), &axis)))
	check("compress", must(ma.Compress(plain(np.FromSlice([]bool{true, false, true}, 3)), g, &axis)))
	check("diag", must(ma.Diag(g, 1)))
	check("diagflat", must(ma.Diagflat(g, 1)))
	check("diagonal", must(ma.Diagonal(g, 1, 0, 1)))
	check("trace", must(ma.Trace(g, 1, 0, 1, 0)))
	check("concatenate", must(ma.Concatenate([]*ma.Array{g, g}, 0)))
	check("stack", must(ma.Stack([]*ma.Array{g, g}, 1)))
	check("append", must(ma.Append(g, g, &axis)))
	for name, fn := range map[string]func(...*ma.Array) (*ma.Array, error){"hstack": ma.HStack, "vstack": ma.VStack, "dstack": ma.DStack, "column_stack": ma.ColumnStack} {
		check(name, must(fn(r0, r1)))
	}
	pair := func(name string, as []*ma.Array, e error) {
		if e != nil {
			t.Fatal(e)
		}
		for k, b := range as {
			check(name+"_"+strconv.Itoa(k), b)
		}
	}
	as, e := ma.Split(g, 2, 0)
	pair("split", as, e)
	as, e = ma.HSplit(g, 3)
	pair("hsplit", as, e)
	as, e = ma.VSplit(g, 2)
	pair("vsplit", as, e)
	check("diff", must(ma.Diff(g, 1, 1)))
	check("ediff1d", must(ma.Ediff1D(g, must(ma.New(np.ScalarOf(int64(-1)), nil)), must(ma.New(np.ScalarOf(int64(9)), nil)))))
	cond := must(ma.Greater(g, must(ma.New(np.ScalarOf(int64(3)), nil))))
	check("where", must(ma.Where(cond, g, must(ma.New(np.ScalarOf(int64(0)), nil)))))
	check("choose", must(ma.Choose(must(ma.New(ints(1, 0, 1), nil)), []*ma.Array{r0, r1})))
	check("sort", must(ma.Sort(g, &axis, true)))
	raw("argsort", plain(ma.Argsort(g, &axis, true)))
	ns, e := ma.Nonzero(g)
	if e != nil {
		t.Fatal(e)
	}
	for k, v := range ns {
		raw("nonzero_"+strconv.Itoa(k), v)
	}
	check("dot", must(ma.Dot(g, must(ma.Transpose(g)))))
	check("inner", must(ma.Inner(g, g)))
	check("outer", must(ma.Outer(r0, r1)))
	for name, fn := range map[string]func(*ma.Array) ([][2]int, error){"clump_masked": ma.ClumpMasked, "clump_unmasked": ma.ClumpUnmasked} {
		v, e := fn(g)
		if e != nil {
			t.Fatal(e)
		}
		flat := []int64{}
		for _, p := range v {
			flat = append(flat, int64(p[0]), int64(p[1]))
		}
		raw(name, plain(np.FromSlice(flat, len(v), 2)))
	}
	raw("compress_rows", plain(ma.CompressRows(g)))
	raw("compress_cols", plain(ma.CompressCols(g)))
	check("mask_rows", must(ma.MaskRows(must(g.Copy()))))
	check("mask_cols", must(ma.MaskCols(must(g.Copy()))))
	check("zeros", must(ma.Zeros(np.Int32, 2, 3)))
	check("ones", must(ma.Ones(np.Int32, 2, 3)))
	check("identity", must(ma.Identity(3, np.Int32)))
	check("arange", must(ma.Arange(1, 5, 2, np.Int32)))
	check("zeros_like", must(ma.ZerosLike(g)))
	check("ones_like", must(ma.OnesLike(g)))
}
