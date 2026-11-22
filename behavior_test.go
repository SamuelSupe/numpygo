package np_test

import (
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

var behaviorDTypes = []np.DType{np.Bool, np.Int8, np.Int16, np.Int32, np.Int64, np.Uint8, np.Uint16, np.Uint32, np.Uint64, np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128}

var behaviorBinary = map[string]binaryFn{
	"add": np.Add, "subtract": np.Subtract, "multiply": np.Multiply, "divide": np.Divide,
	"floor_divide": np.FloorDivide, "remainder": np.Remainder, "fmod": np.Fmod, "power": np.Power,
	"float_power": np.FloatPower, "bitwise_and": np.BitwiseAnd, "bitwise_or": np.BitwiseOr, "bitwise_xor": np.BitwiseXor,
	"left_shift": np.LeftShift, "right_shift": np.RightShift, "gcd": np.GCD, "lcm": np.LCM,
	"equal": np.Equal, "not_equal": np.NotEqual, "less": np.Less, "less_equal": np.LessEqual, "greater": np.Greater, "greater_equal": np.GreaterEqual,
	"minimum": np.Minimum, "maximum": np.Maximum, "fmin": np.FMin, "fmax": np.FMax,
	"logical_and": np.LogicalAnd, "logical_or": np.LogicalOr, "logical_xor": np.LogicalXor,
	"hypot": np.Hypot, "arctan2": np.Arctan2, "copysign": np.Copysign, "logaddexp": np.LogAddExp, "logaddexp2": np.LogAddExp2,
	"heaviside": np.Heaviside, "nextafter": np.Nextafter, "ldexp": np.Ldexp,
}
var behaviorUnary = map[string]unaryFn{
	"sqrt": np.Sqrt, "exp": np.Exp, "exp2": np.Exp2, "expm1": np.Expm1, "log": np.Log, "log2": np.Log2, "log10": np.Log10, "log1p": np.Log1p,
	"sin": np.Sin, "cos": np.Cos, "tan": np.Tan, "arcsin": np.Arcsin, "arccos": np.Arccos, "arctan": np.Arctan, "sinh": np.Sinh, "cosh": np.Cosh, "tanh": np.Tanh,
	"arcsinh": np.Arcsinh, "arccosh": np.Arccosh, "arctanh": np.Arctanh, "cbrt": np.Cbrt, "degrees": np.Degrees, "radians": np.Radians,
	"rint": np.Rint, "floor": np.Floor, "ceil": np.Ceil, "trunc": np.Trunc, "fix": np.Fix, "negative": np.Negative, "positive": np.Positive,
	"square": np.Square, "reciprocal": np.Reciprocal, "absolute": np.Absolute, "fabs": np.Fabs, "conjugate": np.Conjugate, "sign": np.Sign,
	"logical_not": np.LogicalNot, "invert": np.Invert, "isnan": np.IsNaN, "isinf": np.IsInf, "isfinite": np.IsFinite, "signbit": np.Signbit, "spacing": np.Spacing,
}

func TestBehaviorBinaryGrid(t *testing.T) {
	m := refcheck.Require(t)
	left, right := map[np.DType]*np.Array{}, map[np.DType]*np.Array{}
	for _, d := range behaviorDTypes {
		left[d] = m(refcheck.Input(t, "behavior_left_"+d.String()).Slice(np.Reverse()))
		right[d] = m(refcheck.Input(t, "behavior_right_"+d.String()).Slice(np.Slice{}, np.Reverse()))
	}
	for name, fn := range behaviorBinary {
		t.Run(name, func(t *testing.T) {
			for _, a := range behaviorDTypes {
				for _, b := range behaviorDTypes {
					t.Run(a.String()+"_"+b.String(), func(t *testing.T) {
						v, e := fn(left[a], right[b])
						refcheck.Case(t, "behavior_binary_"+name+"_"+a.String()+"_"+b.String(), v, e)
					})
				}
			}
		})
	}
}
func TestBehaviorUnaryGrid(t *testing.T) {
	m := refcheck.Require(t)
	for _, d := range behaviorDTypes {
		a := m(refcheck.Input(t, "behavior_left_"+d.String()).Slice(np.Reverse()))
		t.Run(d.String(), func(t *testing.T) {
			for name, fn := range behaviorUnary {
				t.Run(name, func(t *testing.T) {
					v, e := fn(a)
					refcheck.Case(t, "behavior_unary_"+name+"_"+d.String(), v, e)
				})
			}
		})
	}
}

type reductionFn func(*np.Array, ...np.SumOptions) (*np.Array, error)

var behaviorReductions = map[string]reductionFn{
	"sum": np.Sum, "prod": np.Prod, "mean": np.Mean, "min": np.Min, "max": np.Max,
	"nansum": np.NanSum, "nanprod": np.NanProd, "nanmean": np.NanMean, "nanmin": np.NanMin, "nanmax": np.NanMax,
	"all": np.All, "any": np.Any, "count_nonzero": np.CountNonzero,
}
var behaviorVariances = map[string]func(*np.Array, ...np.VarOptions) (*np.Array, error){"var": np.Var, "std": np.Std, "nanvar": np.NanVar, "nanstd": np.NanStd}

func TestBehaviorReductionGrid(t *testing.T) {
	m := refcheck.Require(t)
	opt := np.SumOptions{Axes: []int{0, -1}, KeepDims: true}
	vo := np.VarOptions{Axes: opt.Axes, KeepDims: true}
	for _, d := range behaviorDTypes {
		t.Run(d.String(), func(t *testing.T) {
			a := refcheck.Input(t, "behavior_left_"+d.String())
			b := m(refcheck.Input(t, "behavior_right_"+d.String()).Transpose())
			v := m(m(np.Concatenate([]*np.Array{a, b}, 1)).Reshape(3, 1, 2))
			v = m(v.Slice(np.Reverse()))
			for name, fn := range behaviorReductions {
				t.Run(name, func(t *testing.T) {
					out, e := fn(v, opt)
					refcheck.Case(t, "behavior_reduce_"+name+"_"+d.String(), out, e)
				})
			}
			for name, fn := range behaviorVariances {
				t.Run(name, func(t *testing.T) {
					out, e := fn(v, vo)
					refcheck.Case(t, "behavior_reduce_"+name+"_"+d.String(), out, e)
				})
			}
			a = m(a.Slice(np.Reverse()))
			for name, fn := range map[string]func(*np.Array, ...np.ArgOptions) (*np.Array, error){"argmin": np.ArgMin, "argmax": np.ArgMax, "nanargmin": np.NanArgMin, "nanargmax": np.NanArgMax} {
				t.Run(name, func(t *testing.T) {
					out, e := fn(a, np.ArgOptions{Axis: np.Int(0), KeepDims: true})
					refcheck.Case(t, "behavior_reduce_"+name+"_"+d.String(), out, e)
				})
			}
		})
	}
}
func TestBehaviorAccumulatorPrecision(t *testing.T) {
	for _, sample := range []struct {
		name string
		d    np.DType
	}{
		{"f32_cancel", 0}, {"f64_as_f32", np.Float32}, {"f32_overflow", 0}, {"f16_square_sum", 0}, {"f32_product", 0}, {"explicit_int", np.Int64}, {"c64_cancel", 0}, {"nan_f32", 0},
	} {
		t.Run(sample.name, func(t *testing.T) {
			a := refcheck.Input(t, "behavior_precision_input_"+sample.name)
			for _, name := range []string{"sum", "prod", "mean", "nansum", "nanprod", "nanmean"} {
				t.Run(name, func(t *testing.T) {
					v, e := behaviorReductions[name](a, np.SumOptions{DType: sample.d})
					refcheck.Case(t, "behavior_precision_"+name+"_"+sample.name, v, e)
				})
			}
			for name, fn := range behaviorVariances {
				t.Run(name, func(t *testing.T) {
					v, e := fn(a, np.VarOptions{DType: sample.d})
					refcheck.Case(t, "behavior_precision_"+name+"_"+sample.name, v, e)
				})
			}
		})
	}
}
func TestBehaviorSignedZeros(t *testing.T) {
	m := refcheck.Require(t)
	a := refcheck.Input(t, "behavior_special_a")
	b := refcheck.Input(t, "behavior_special_b")
	check := func(t *testing.T, name string, v *np.Array) {
		refcheck.Check(t, "behavior_special_"+name, v)
		values := refcheck.RequireValue[[]float64](t)(np.Values[float64](v))
		signs := []bool{}
		for _, x := range values {
			if x == 0 {
				signs = append(signs, math.Signbit(x))
			}
		}
		refcheck.Check(t, "behavior_special_sign_"+name, m(np.WrapSlice(signs, len(signs))))
	}
	for _, name := range []string{"add", "subtract", "multiply", "divide", "floor_divide", "remainder", "minimum", "maximum", "fmin", "fmax", "copysign", "arctan2", "nextafter"} {
		t.Run(name, func(t *testing.T) { check(t, name, m(behaviorBinary[name](a, b))) })
	}
	for _, name := range []string{"negative", "absolute", "rint", "sqrt", "sign", "spacing"} {
		t.Run(name, func(t *testing.T) { check(t, name, m(behaviorUnary[name](a))) })
	}
}

func TestBehaviorDenseKernels(t *testing.T) {
	for _, d := range []np.DType{np.Float32, np.Float64} {
		m := refcheck.Require(t)
		a := refcheck.Input(t, "behavior_dense_a_"+d.String())
		b := refcheck.Input(t, "behavior_dense_b_"+d.String())
		scalar := refcheck.Input(t, "behavior_dense_scalar_"+d.String())
		for _, name := range []string{"add", "subtract", "multiply", "divide"} {
			for _, tc := range []struct {
				label string
				x, y  *np.Array
			}{{"array", a, b}, {"right", a, scalar}, {"left", scalar, b}, {"scalar", scalar, scalar}} {
				t.Run(d.String()+"/"+name+"/"+tc.label, func(t *testing.T) {
					refcheck.Check(t, "behavior_dense_"+name+"_"+d.String()+"_"+tc.label, m(behaviorBinary[name](tc.x, tc.y)))
				})
			}
		}
	}
}

func TestBehaviorUFuncOptions(t *testing.T) {
	m := refcheck.Require(t)
	lookup := refcheck.RequireValue[*np.UFunc](t)
	a := m(m(np.ArangeInt64(0, 6, 1)).Astype(np.Int16))
	a = m(a.Reshape(2, 3))
	b := m(np.WrapSlice([]float32{.5, 1.5, 2.5}, 3))
	mask := m(np.WrapSlice([]bool{true, false, true}, 3))
	add := lookup(np.LookupUFunc("add"))
	base := m(np.Full(np.Float64, -99., 2, 3))
	out := m(base.Slice(np.Slice{}, np.Reverse()))
	got := m(add.CallWithOptions(np.UFuncOptions{Out: out, Where: mask}, a, b))
	if got != out {
		t.Fatal("out was replaced instead of written")
	}
	refcheck.Check(t, "behavior_ufunc_mask_out", got)
	refcheck.Check(t, "behavior_ufunc_mask_base", base)
	overlap := m(np.ArangeInt64(1, 7, 1))
	m(add.CallWithOptions(np.UFuncOptions{Out: m(overlap.Slice(np.Range(1, 6)))}, m(overlap.Slice(np.Range(0, 5))), np.ScalarOf(int64(10))))
	refcheck.Check(t, "behavior_ufunc_overlap", overlap)
	pow := lookup(np.LookupUFunc("power"))
	refcheck.Check(t, "behavior_ufunc_mask_power", m(pow.CallWithOptions(np.UFuncOptions{Out: m(np.Full(np.Int16, int16(77), 3)), Where: m(np.WrapSlice([]bool{false, true, false}, 3))}, m(np.Full(np.Int16, int16(2), 3)), m(np.WrapSlice([]int16{-1, 3, -2}, 3)))))
	refcheck.Check(t, "behavior_ufunc_explicit_dtype", m(add.CallWithOptions(np.UFuncOptions{DType: np.Float32}, a, b)))
	refcheck.Check(t, "behavior_ufunc_cast_unsafe", m(add.CallWithOptions(np.UFuncOptions{DType: np.Int8, Casting: "unsafe"}, m(np.WrapSlice([]int16{127, 128, 300}, 3)), np.ScalarOf(int16(1)))))
	refcheck.Check(t, "behavior_ufunc_complex_sqrt", m(lookup(np.LookupUFunc("sqrt")).CallWithOptions(np.UFuncOptions{DType: np.Complex64}, m(np.WrapSlice([]float32{-1, 4}, 2)))))
	abs, e := lookup(np.LookupUFunc("absolute")).CallWithOptions(np.UFuncOptions{DType: np.Float32}, m(np.WrapSlice([]complex128{3 + 4i, 1 + 2i}, 2)))
	refcheck.Case(t, "behavior_ufunc_abs_complex_cast", abs, e)
	refcheck.Check(t, "behavior_ufunc_bool_compare", m(lookup(np.LookupUFunc("equal")).CallWithOptions(np.UFuncOptions{DType: np.Bool}, a, np.ScalarOf(int16(2)))))
	refcheck.Check(t, "behavior_ufunc_out_broadcast", m(add.CallWithOptions(np.UFuncOptions{Out: m(np.Zeros(np.Int64, 2, 3))}, m(np.WrapSlice([]int64{1, 2, 3}, 3)), np.ScalarOf(int64(5)))))
	refcheck.Check(t, "behavior_ufunc_where_broadcast", m(add.CallWithOptions(np.UFuncOptions{Out: m(np.Full(np.Float64, -1., 2, 3)), Where: m(np.WrapSlice([]bool{true, false}, 2, 1))}, m(np.WrapSlice([]float64{1, 2, 3}, 3)), np.Scalar(2))))
	alias := m(np.WrapSlice([]bool{true, false, true}, 3))
	m(lookup(np.LookupUFunc("logical_not")).CallWithOptions(np.UFuncOptions{Out: alias, Where: alias}, alias))
	refcheck.Check(t, "behavior_ufunc_where_alias", alias)
	// With no out, false positions are deliberately deterministic in Go.
	zeroed := m(add.CallWithOptions(np.UFuncOptions{Where: mask}, a, b))
	if v, e := zeroed.AtValue(1, 1); e != nil || v != float32(0) {
		t.Fatal(v, e)
	}
	for _, tc := range []struct {
		name string
		opt  np.UFuncOptions
	}{
		{"cast_no", np.UFuncOptions{Casting: "no"}},
		{"cast_safe", np.UFuncOptions{DType: np.Int8, Casting: "safe"}},
		{"cast_out", np.UFuncOptions{Out: m(np.Zeros(np.Int8, 2, 3))}},
		{"bad_mask_dtype", np.UFuncOptions{Where: m(np.WrapSlice([]int64{1, 0, 1}, 3))}},
	} {
		v, e := add.CallWithOptions(tc.opt, a, b)
		refcheck.Case(t, "behavior_ufunc_"+tc.name, v, e)
	}
	_, e = add.CallWithOptions(np.UFuncOptions{Out: m(np.Zeros(np.Float64, 3, 2))}, a, b)
	refcheck.Error(t, "behavior_ufunc_bad_out_shape", "ValueError", e, np.ErrShape)
	_, e = add.CallWithOptions(np.UFuncOptions{Out: m(np.Scalar(0).BroadcastTo(2, 3))}, a, b)
	refcheck.Error(t, "behavior_ufunc_readonly_out", "ValueError", e, np.ErrReadOnly)
}
