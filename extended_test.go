package np_test

import (
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

type unaryFn func(*np.Array) (*np.Array, error)
type binaryFn func(*np.Array, *np.Array) (*np.Array, error)

func TestDTypePromotion(t *testing.T) {
	must := refcheck.Require(t)
	types := []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64, np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128}
	for _, a := range types {
		for _, b := range types {
			name := "promote_" + a.String() + "_" + b.String()
			t.Run(name, func(t *testing.T) {
				left := must(np.Ones(a))
				right := must(np.Ones(b))
				out := must(np.Add(left, right))
				refcheck.Check(t, name, out)
				d, err := np.PromoteTypes(a, b)
				if err != nil || out.DType() != d {
					t.Fatalf("promotion=%v %v", d, err)
				}
			})
		}
	}
}

func TestIntegerArithmetic(t *testing.T) {
	must := refcheck.Require(t)
	ops := map[string]binaryFn{"add": np.Add, "subtract": np.Subtract, "multiply": np.Multiply, "floor_divide": np.FloorDivide, "remainder": np.Remainder, "bitwise_and": np.BitwiseAnd, "bitwise_or": np.BitwiseOr, "bitwise_xor": np.BitwiseXor, "divide": np.Divide}
	for _, d := range []np.DType{np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64} {
		t.Run(d.String(), func(t *testing.T) {
			a := refcheck.Input(t, "integer_input_"+d.String())
			b := refcheck.Input(t, "integer_other_"+d.String())
			for name, fn := range ops {
				refcheck.Check(t, "integer_"+name+"_"+d.String(), must(fn(a, b)))
			}
			refcheck.Check(t, "integer_sum_"+d.String(), must(np.Sum(a)))
			refcheck.Check(t, "integer_prod_"+d.String(), must(np.Prod(a)))
			refcheck.Check(t, "integer_reciprocal_"+d.String(), must(np.Reciprocal(refcheck.Input(t, "integer_reciprocal_input_"+d.String()))))
		})
	}
	a := must(np.FromSlice([]int64{-8, -7, 7, 8}, 4))
	b := must(np.FromSlice([]int64{3, -3, -3, 3}, 4))
	refcheck.Check(t, "integer_floor_negative", must(np.FloorDivide(a, b)))
	refcheck.Check(t, "integer_mod_negative", must(np.Remainder(a, b)))
	refcheck.Check(t, "integer_pow", must(np.Power(must(np.FromSlice([]int8{-3, 3, 100}, 3)), must(np.FromSlice([]int8{3, 4, 2}, 3)))))
	_, err := np.Power(np.ScalarOf(int64(2)), np.ScalarOf(int64(-1)))
	refcheck.Error(t, "integer_negative_power", "ValueError", err, np.ErrValue)
	_, err = np.Subtract(np.ScalarOf(true), np.ScalarOf(false))
	refcheck.Error(t, "bool_subtract", "TypeError", err, np.ErrDType)
	refcheck.Check(t, "bool_sum", must(np.Sum(must(np.FromSlice([]bool{true, false, true}, 3)))))
	refcheck.Check(t, "bool_prod", must(np.Prod(must(np.FromSlice([]bool{true, false, true}, 3)))))
	refcheck.Check(t, "bool_logic", must(np.LogicalXor(must(np.FromSlice([]bool{true, false}, 2)), must(np.FromSlice([]bool{true, true}, 2)))))
	bools := must(np.FromSlice([]bool{true, false}, 2))
	refcheck.Check(t, "bool_square", must(np.Square(bools)))
	refcheck.Check(t, "bool_reciprocal", must(np.Reciprocal(bools)))
	refcheck.Check(t, "bool_conjugate", must(np.Conjugate(bools)))
	signed := must(np.FromSlice([]int64{math.MaxInt64, -1, 7}, 3))
	unsigned := must(np.FromSlice([]uint64{1 << 63, math.MaxUint64, 7}, 3))
	refcheck.Check(t, "integer_cross_equal", must(np.Equal(signed, unsigned)))
	refcheck.Check(t, "integer_cross_less", must(np.Less(signed, unsigned)))
	refcheck.Check(t, "integer_cross_greater", must(np.Greater(unsigned, signed)))
	refcheck.Check(t, "integer_cross_isin", must(np.Isin(signed, unsigned, false)))
}
func TestExactIntegerStorageAndFloat16(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "uint64_input")
	refcheck.Check(t, "uint64_input", a)
	refcheck.Check(t, "uint64_cast_int64", must(a.Astype(np.Int64)))
	refcheck.Check(t, "uint64_cast_float64", must(a.Astype(np.Float64)))
	view := must(a.Slice(np.Reverse()))
	refcheck.CheckLayout(t, "uint64_negative_slice", view)
	values, _ := np.Values[uint64](a)
	if values[0] != 1<<63+123 || values[1] != math.MaxUint64 || values[2] != 1<<53+1 {
		t.Fatal("lossy integer conversion", values)
	}
	if err := view.SetValue(uint64(1<<63+55), 0); err != nil {
		t.Fatal(err)
	}
	if v, _ := a.AtValue(-1); v != uint64(1<<63+55) {
		t.Fatal("view did not propagate exact integer")
	}
	backing := []int8{1, -2, 3}
	wrap := must(np.WrapSlice(backing, 3))
	if err := must(wrap.Slice(np.Reverse())).SetValue(int8(99), 0); err != nil {
		t.Fatal(err)
	}
	if backing[2] != 99 {
		t.Fatal("wrap lost shared storage")
	}
	refcheck.Check(t, "half_cast", must(refcheck.Input(t, "half_input").Astype(np.Float16)))
	if np.HalfFromFloat64(math.Copysign(0, -1)) != 0x8000 || np.HalfFromFloat64(math.Ldexp(1, -24)) != 1 || np.HalfFromFloat64(65520) != 0x7c00 {
		t.Fatal("half encoding edge")
	}
	half := must(must(np.FromFloat64([]float64{.1, .2, -.3, 5}, 4)).Astype(np.Float16))
	refcheck.Check(t, "half_sum", must(np.Sum(half)))
	refcheck.Check(t, "complex_cast", must(must(np.FromComplex128([]complex128{1 + 2i, 3 - 4i}, 2)).Astype(np.Complex64)))
}

func TestAdvancedIndexing(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "advanced_input")
	i := refcheck.Input(t, "advanced_i")
	j := refcheck.Input(t, "advanced_j")
	all := np.Slice{}
	refcheck.Check(t, "advanced_adjacent", must(a.Index(all, i, j, all)))
	refcheck.Check(t, "advanced_separated", must(a.Index(all, i, all, j)))
	refcheck.Check(t, "advanced_scalar", must(a.Index(np.AtIndex(1), all, j, all)))
	refcheck.Check(t, "advanced_ellipsis", must(a.Index(must(np.FromSlice([]int64{1, 0}, 2)), np.Ellipsis, must(np.FromSlice([]int64{4, 1}, 2)))))
	refcheck.Check(t, "advanced_newaxis", must(a.Index(all, i, np.NewAxis, j, all)))
	mask := refcheck.Input(t, "advanced_mask_input")
	refcheck.Check(t, "advanced_mask", must(a.Index(mask)))
	refcheck.Check(t, "advanced_mask_true", must(a.Index(np.ScalarOf(true))))
	refcheck.Check(t, "advanced_mask_false", must(a.Index(np.ScalarOf(false))))
	refcheck.Check(t, "advanced_negative", must(a.Index(must(np.FromSlice([]int64{-1, 0}, 2)))))
	refcheck.Check(t, "advanced_take", must(np.Take(a, must(np.FromSlice([]int64{3, 1, 0, -1}, 2, 2)), np.Int(2))))
	coords, err := np.Nonzero(mask)
	if err != nil {
		t.Fatal(err)
	}
	for k, name := range []string{"advanced_nonzero_0", "advanced_nonzero_1"} {
		refcheck.Check(t, name, coords[k])
	}
	refcheck.Check(t, "advanced_flatnonzero", must(np.FlatNonzero(mask)))
	refcheck.Check(t, "advanced_where", must(np.Where(mask, must(np.ArangeInt64(0, 3, 1)), np.ScalarOf(int64(-1)))))
	copy := must(a.Index(mask))
	if err := copy.SetValue(int64(999), 0, 0, 0); err != nil {
		t.Fatal(err)
	}
	if v, _ := a.AtValue(0, 0, 0, 0); v != int64(0) {
		t.Fatal("advanced indexing produced alias")
	}
	target := must(must(np.ArangeInt64(0, 12, 1)).Reshape(3, 4))
	ix := must(np.FromSlice([]int64{0, 0, 2}, 3))
	iy := must(np.FromSlice([]int64{1, 1, 0}, 3))
	if err := target.Assign([]np.Index{ix, iy}, must(np.FromSlice([]int64{100, 200, 300}, 3))); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "advanced_assignment", target)
	alias := must(np.ArangeInt64(0, 6, 1))
	if err := alias.Assign([]np.Index{np.Range(1, 6)}, must(alias.Slice(np.Range(0, 5)))); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "advanced_assign_alias", alias)
	_, err = a.Index(must(np.Ones(np.Bool, 3, 3)))
	refcheck.Error(t, "advanced_bad_mask", "IndexError", err, np.ErrIndex)
	_, err = a.Index(must(np.FromSlice([]int64{2}, 1)))
	refcheck.Error(t, "advanced_index_oob", "IndexError", err, np.ErrIndex)
	_, err = a.Index(must(np.FromFloat64([]float64{0}, 1)))
	refcheck.Error(t, "advanced_index_type", "IndexError", err, np.ErrIndex)
	_, err = a.Index(all, must(np.FromSlice([]int64{0, 1}, 2)), must(np.FromSlice([]int64{0, 1, 2}, 3)), all)
	refcheck.Error(t, "advanced_broadcast_error", "IndexError", err, np.ErrIndex)
}

func TestStatistics(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "stats_input")
	opt := np.SumOptions{Axes: []int{1}, KeepDims: true}
	fns := map[string]func(*np.Array, ...np.SumOptions) (*np.Array, error){
		"mean": np.Mean, "nanmean": np.NanMean, "sum": np.Sum, "nansum": np.NanSum, "prod": np.Prod, "nanprod": np.NanProd,
		"min": np.Min, "max": np.Max, "nanmin": np.NanMin, "nanmax": np.NanMax, "median": np.Median, "nanmedian": np.NanMedian,
		"ptp": np.PTP, "all": np.All, "any": np.Any, "count_nonzero": np.CountNonzero,
	}
	for name, fn := range fns {
		t.Run(name, func(t *testing.T) { refcheck.Check(t, "stats_"+name, must(fn(a, opt))) })
	}
	vopt := np.VarOptions{Axes: []int{1}, KeepDims: true}
	for name, fn := range map[string]func(*np.Array, ...np.VarOptions) (*np.Array, error){"var": np.Var, "nanvar": np.NanVar, "std": np.Std, "nanstd": np.NanStd} {
		refcheck.Check(t, "stats_"+name, must(fn(a, vopt)))
	}
	refcheck.Check(t, "stats_var_ddof", must(np.NanVar(a, np.VarOptions{Axes: []int{0}, DDOF: 1})))
	refcheck.Check(t, "stats_std_ddof", must(np.NanStd(a, np.VarOptions{Axes: []int{1}, DDOF: 2})))
	axis := np.SumOptions{Axes: []int{1}}
	refcheck.Check(t, "stats_quantile", must(np.Quantile(a, .2, axis)))
	refcheck.Check(t, "stats_nanquantile", must(np.NanQuantile(a, .75, axis)))
	refcheck.Check(t, "stats_percentile", must(np.Percentile(a, 35, axis)))
	refcheck.Check(t, "stats_nanpercentile", must(np.NanPercentile(a, 60, axis)))
	for name, fn := range map[string]func(*np.Array, ...np.ArgOptions) (*np.Array, error){"argmin": np.ArgMin, "argmax": np.ArgMax, "nanargmin": np.NanArgMin, "nanargmax": np.NanArgMax} {
		refcheck.Check(t, "stats_"+name, must(fn(a, np.ArgOptions{Axis: np.Int(1)})))
	}
	refcheck.Check(t, "stats_arg_flat_transpose", must(np.ArgMax(must(a.Transpose()))))
	refcheck.Check(t, "stats_argsort", must(np.Argsort(a, np.Int(1))))
	refcheck.Check(t, "stats_sort", must(np.Sort(a, np.Int(-1))))
	for name, fn := range map[string]func(*np.Array, *int, np.DType) (*np.Array, error){"cumsum": np.Cumsum, "cumprod": np.Cumprod, "nancumsum": np.NanCumsum, "nancumprod": np.NanCumprod} {
		refcheck.Check(t, "stats_"+name, must(fn(a, np.Int(1), 0)))
	}
	refcheck.Check(t, "stats_empty_mean", must(np.Mean(must(np.Zeros(np.Float64, 2, 0)), axis)))
	refcheck.Check(t, "stats_empty_sum", must(np.Sum(must(np.Zeros(np.Int32, 2, 0)), axis)))
	_, err := np.Min(must(np.Zeros(np.Float64, 2, 0)), axis)
	refcheck.Error(t, "stats_empty_min", "ValueError", err, np.ErrValue)
	_, err = np.NanArgMax(must(np.FromFloat64([]float64{math.NaN(), math.NaN()}, 2)))
	refcheck.Error(t, "stats_all_nan_arg", "ValueError", err, np.ErrValue)
	_, err = np.Quantile(a, 1.1)
	refcheck.Error(t, "stats_invalid_quantile", "ValueError", err, np.ErrValue)
	refcheck.Check(t, "stats_low_precision_quantile", must(np.Quantile(must(np.FromSlice([]float32{1, 2, 3}, 3)), .7)))
	c := refcheck.Input(t, "stats_complex_input")
	refcheck.Check(t, "stats_complex_mean", must(np.Mean(c, axis)))
	refcheck.Check(t, "stats_complex_var", must(np.Var(c, np.VarOptions{Axes: []int{1}})))
	refcheck.Check(t, "stats_complex_std", must(np.Std(c, np.VarOptions{Axes: []int{1}})))
	refcheck.Check(t, "stats_cov", must(np.Cov(c, 1)))
	refcheck.Check(t, "stats_corrcoef", must(np.Corrcoef(c)))
	refcheck.Check(t, "stats_complex_corrcoef_scalar", must(np.Corrcoef(must(np.FromComplex128([]complex128{1 + 2i, 3 - 1i}, 2)))))
}
func TestMathFunctions(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "math_input")
	c := refcheck.Input(t, "math_complex_input")
	fns := map[string]unaryFn{"sqrt": np.Sqrt, "exp": np.Exp, "exp2": np.Exp2, "expm1": np.Expm1, "log": np.Log, "log2": np.Log2, "log10": np.Log10, "log1p": np.Log1p,
		"sin": np.Sin, "cos": np.Cos, "tan": np.Tan, "arcsin": np.Arcsin, "arccos": np.Arccos, "arctan": np.Arctan, "sinh": np.Sinh, "cosh": np.Cosh, "tanh": np.Tanh, "arcsinh": np.Arcsinh, "arccosh": np.Arccosh, "arctanh": np.Arctanh,
		"cbrt": np.Cbrt, "degrees": np.Degrees, "radians": np.Radians, "rint": np.Rint, "floor": np.Floor, "ceil": np.Ceil, "trunc": np.Trunc, "negative": np.Negative, "positive": np.Positive, "square": np.Square, "reciprocal": np.Reciprocal, "absolute": np.Absolute, "conjugate": np.Conjugate, "sign": np.Sign}
	for name, fn := range fns {
		t.Run(name, func(t *testing.T) { refcheck.Check(t, "math_"+name, must(fn(a))) })
	}
	for _, name := range []string{"sqrt", "exp", "log", "log1p", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "absolute", "conjugate", "sign", "rint", "reciprocal", "square"} {
		t.Run("complex_"+name, func(t *testing.T) { refcheck.Check(t, "math_complex_"+name, must(fns[name](c))) })
	}
	special := refcheck.Input(t, "math_special")
	for name, fn := range map[string]unaryFn{"isnan": np.IsNaN, "isinf": np.IsInf, "isfinite": np.IsFinite, "signbit": np.Signbit, "logical_not": np.LogicalNot} {
		refcheck.Check(t, "math_"+name, must(fn(special)))
	}
	refcheck.Check(t, "math_invert", must(np.Invert(must(np.FromSlice([]int8{-3, 0, 3}, 3)))))
	refcheck.Check(t, "math_sqrt_int", must(np.Sqrt(must(np.FromSlice([]int16{1, 4, 9}, 3)))))
	refcheck.Check(t, "math_clip", must(np.Clip(a, np.Scalar(-.25), np.Scalar(1))))
	for name, fn := range map[string]binaryFn{"equal": np.Equal, "not_equal": np.NotEqual, "less": np.Less, "less_equal": np.LessEqual, "greater": np.Greater, "greater_equal": np.GreaterEqual, "minimum": np.Minimum, "maximum": np.Maximum, "fmin": np.FMin, "fmax": np.FMax, "logical_and": np.LogicalAnd, "logical_or": np.LogicalOr, "logical_xor": np.LogicalXor, "hypot": np.Hypot, "arctan2": np.Arctan2, "copysign": np.Copysign, "logaddexp": np.LogAddExp} {
		t.Run(name, func(t *testing.T) { refcheck.Check(t, "math_"+name, must(fn(a, np.Scalar(.25)))) })
	}
	refcheck.Check(t, "math_isclose", must(np.IsClose(special, special, 1e-5, 1e-8, true)))
	ok, err := np.AllClose(special, special, 1e-5, 1e-8, true)
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "math_allclose", np.ScalarOf(ok))
}
func TestCreationAndManipulation(t *testing.T) {
	must := refcheck.Require(t)
	refcheck.Check(t, "create_ones", must(np.Ones(np.Int16, 2, 3)))
	refcheck.Check(t, "create_full", must(np.Full(np.Complex64, complex128(3+2i), 2, 2)))
	refcheck.Check(t, "create_eye", must(np.Eye(3, 4, 1, np.Uint8)))
	refcheck.Check(t, "create_identity", must(np.Identity(3, np.Float32)))
	refcheck.Check(t, "create_arange_int", must(np.ArangeInt64(9, -4, -3)))
	refcheck.Check(t, "create_arange_float", must(np.Arange(.5, 2.6, .3, np.Float32)))
	refcheck.Check(t, "create_linspace", must(np.Linspace(-2.5, 3.5, 7, true, np.Int64)))
	refcheck.Check(t, "create_logspace", must(np.Logspace(-2, 2, 5, true, 10, np.Float64)))
	a := refcheck.Input(t, "manip_input")
	refcheck.Check(t, "manip_zeros_like", must(np.ZerosLike(a)))
	refcheck.Check(t, "manip_ones_like", must(np.OnesLike(a)))
	refcheck.Check(t, "manip_full_like", must(np.FullLike(a, int64(7))))
	refcheck.Check(t, "manip_ravel", must(np.Ravel(must(a.Transpose()))))
	refcheck.Check(t, "manip_flatten", must(np.Flatten(must(a.Transpose()))))
	e := must(np.ExpandDims(a, 1))
	refcheck.CheckLayout(t, "manip_expand", e)
	refcheck.CheckLayout(t, "manip_squeeze", must(np.Squeeze(e, 1)))
	refcheck.CheckLayout(t, "manip_swap", must(np.SwapAxes(a, 0, -1)))
	refcheck.CheckLayout(t, "manip_move", must(np.MoveAxis(a, 0, 1)))
	refcheck.CheckLayout(t, "manip_flip", must(np.Flip(a, 1)))
	refcheck.Check(t, "manip_roll", must(np.Roll(a, 2, np.Int(1))))
	refcheck.Check(t, "manip_repeat", must(np.Repeat(a, 2, np.Int(0))))
	refcheck.Check(t, "manip_tile", must(np.Tile(a, 2, 1, 2)))
	refcheck.Check(t, "manip_diff", must(np.Diff(a, 2, 1)))
	refcheck.Check(t, "manip_bool_diff", must(np.Diff(must(np.FromSlice([]bool{true, false, false, true}, 4)), 1, 0)))
	diag := must(np.Diagonal(a, 1, 0, 1))
	refcheck.CheckLayout(t, "manip_diagonal", diag)
	if diag.Writable() {
		t.Fatal("diagonal writable")
	}
	refcheck.Check(t, "manip_trace", must(np.Trace(a, 1, 0, 1, 0)))
	refcheck.Check(t, "manip_concat", must(np.Concatenate([]*np.Array{a, a}, 1)))
	refcheck.Check(t, "manip_stack", must(np.Stack([]*np.Array{a, a}, -1)))
}
