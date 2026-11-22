package np_test

import (
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/extended"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func extFunction(t *testing.T, name string, a *np.Array) (*np.Array, error) {
	t.Helper()
	if name == "sinc" {
		return np.Sinc(a)
	}
	u, e := np.LookupUFunc(name)
	if e != nil {
		return nil, e
	}
	return u.Call(a)
}
func TestExtendedTranscendentalsNumPy(t *testing.T) {
	realNames := "sqrt exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin arccos arctan sinh cosh tanh arcsinh arccosh arctanh cbrt degrees radians rint floor ceil trunc fabs sign spacing sinc"
	complexNames := "sqrt exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin arccos arctan sinh cosh tanh arcsinh arccosh arctanh rint sign sinc"
	for _, tc := range []struct {
		label, names string
		dtype        np.DType
		reverse      bool
		ulps         int64
	}{
		{"real", realNames, np.Float128, true, 8},
		{"complex", complexNames, np.Complex256, true, 32},
		{"far", "exp expm1 exp2 log log1p sin cos tan", np.Float128, false, 8},
		{"nonfinite", "sqrt exp log arcsin arctan arccosh sign", np.Complex256, false, 8},
	} {
		t.Run(tc.label, func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.ExtendedInput(t, "exmath_"+tc.label+"_input", tc.dtype)
			if tc.reverse {
				a = m(a.Slice(np.Reverse()))
			}
			for _, name := range strings.Fields(tc.names) {
				t.Run(name, func(t *testing.T) {
					v := m(extFunction(t, name, a))
					refcheck.CheckExtended(t, "exmath_"+tc.label+"_"+name, v, tc.dtype, tc.ulps)
				})
			}
		})
	}
	z := refcheck.ExtendedInput(t, "exmath_complex_input", np.Complex256)
	for _, name := range strings.Fields("floor ceil trunc cbrt degrees radians spacing fabs") {
		t.Run("reject_"+name, func(t *testing.T) {
			a, e := extFunction(t, name, z)
			refcheck.Case(t, "exmath_bad_complex_"+name, a, e)
		})
	}
}
func TestExtendedBinaryNumPy(t *testing.T) {
	for _, name := range strings.Fields("power float_power floor_divide remainder fmod hypot arctan2 copysign logaddexp logaddexp2 heaviside") {
		t.Run(name, func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.ExtendedInput(t, "exmath_binary_a", np.Float128)
			b := refcheck.ExtendedInput(t, "exmath_binary_b", np.Float128)
			u := refcheck.RequireValue[*np.UFunc](t)(np.LookupUFunc(name))
			refcheck.CheckExtended(t, "exmath_binary_"+name, m(u.Call(a, b)), np.Float128, 8)
		})
	}
	m := refcheck.Require(t)
	a := refcheck.ExtendedInput(t, "exmath_cpower_a", np.Complex256)
	b := refcheck.ExtendedInput(t, "exmath_cpower_b", np.Complex256)
	refcheck.CheckExtended(t, "exmath_cpower", m(np.Power(a, b)), np.Complex256, 32)
	refcheck.CheckExtended(t, "exmath_cfloat_power", m(np.FloatPower(a, b)), np.Complex256, 32)
}
func TestExtendedMathHelpersNumPy(t *testing.T) {
	m := refcheck.Require(t)
	a := refcheck.ExtendedInput(t, "exmath_real_input", np.Float128)
	z := refcheck.ExtendedInput(t, "exmath_complex_input", np.Complex256)
	for _, tc := range []struct {
		name string
		d    np.DType
		v    *np.Array
		ulps int64
	}{
		{"angle", np.Float128, m(np.Angle(z, false)), 8},
		{"angle_degree", np.Float128, m(np.Angle(z, true)), 8},
		{"nan_to_num", np.Float128, m(np.NanToNum(a)), 0},
		{"cnan_to_num", np.Complex256, m(np.NanToNum(refcheck.ExtendedInput(t, "exmath_nonfinite_input", np.Complex256))), 0},
		{"round", np.Float128, m(np.Round(a, 2)), 1},
		{"cround", np.Complex256, m(np.Round(z, 2)), 1},
	} {
		refcheck.CheckExtended(t, "exmath_"+tc.name, tc.v, tc.d, tc.ulps)
	}
	frac, integ, e := np.Modf(a)
	if e != nil {
		t.Fatal(e)
	}
	refcheck.CheckExtended(t, "exmath_modf_frac", frac, np.Float128, 0)
	refcheck.CheckExtended(t, "exmath_modf_integ", integ, np.Float128, 0)
	mant, exp, e := np.Frexp(a)
	if e != nil {
		t.Fatal(e)
	}
	refcheck.CheckExtended(t, "exmath_frexp", mant, np.Float128, 0)
	refcheck.Check(t, "exmath_frexp_exponent", exp)
	refcheck.CheckExtended(t, "exmath_ldexp", m(np.Ldexp(refcheck.ExtendedInput(t, "exmath_ldexp_input", np.Float128), refcheck.Input(t, "exmath_ldexp_exponents"))), np.Float128, 0)
	refcheck.Check(t, "exmath_isreal", m(np.IsReal(z)))
	refcheck.Check(t, "exmath_iscomplex", m(np.IsComplex(z)))
	refcheck.Check(t, "exmath_isposinf", m(np.IsPosInf(a)))
	refcheck.Check(t, "exmath_isneginf", m(np.IsNegInf(a)))
	parse := func(s ...string) *np.Array {
		v := make([]np.ExtendedFloat, len(s))
		for i, x := range s {
			v[i] = extended.MustParse(x)
		}
		return m(np.FromSlice(v, len(v)))
	}
	refcheck.Check(t, "exmath_isclose", m(np.IsClose(parse("1", "1e4000", "nan", "0"), parse("1.0000000000000000001", "1.01e4000", "nan", "1e-4000"), 0, 0, true)))
	// NumPy i0 uses a double-precision polynomial approximation.
	refcheck.CheckExtended(t, "exmath_i0", m(np.I0(refcheck.ExtendedInput(t, "exmath_i0_input", np.Float128))), np.Float128, 4096)
	near := refcheck.ExtendedInput(t, "exmath_near_real_input", np.Complex256)
	real := m(np.RealIfClose(near, 100))
	if !np.SharesMemory(real, near) {
		t.Fatal("real_if_close should retain the real component view")
	}
	refcheck.CheckExtended(t, "exmath_near_real", real, np.Float128, 0)
	refcheck.CheckExtended(t, "exmath_not_near_real", m(np.RealIfClose(z, 100)), np.Complex256, 0)
}
func TestExtendedFPNumPy(t *testing.T) {
	for _, tc := range []struct {
		label, op string
		args      []np.ExtendedFloat
	}{
		{"finite", "exp", []np.ExtendedFloat{extended.FromInt64(10000)}},
		{"overflow", "exp", []np.ExtendedFloat{extended.FromInt64(12000)}},
		{"underflow", "exp", []np.ExtendedFloat{extended.FromInt64(-12000)}},
		{"divide", "divide", []np.ExtendedFloat{extended.MustParse("1e4000"), extended.Zero(false)}},
		{"invalid", "divide", []np.ExtendedFloat{extended.Zero(false), extended.Zero(false)}},
		{"sqrt_invalid", "sqrt", []np.ExtendedFloat{extended.MustParse("-1e4000")}},
		{"exact_tiny", "multiply", []np.ExtendedFloat{extended.Info().Tiny, extended.MustParse("0.5")}},
		{"inexact_tiny", "multiply", []np.ExtendedFloat{extended.MustParse("1e-4950"), extended.MustParse("0.3")}},
	} {
		t.Run(tc.label, func(t *testing.T) {
			ctx := np.NewErrorContext()
			if _, e := ctx.SetErr(np.ErrorOptions{All: np.Call}); e != nil {
				t.Fatal(e)
			}
			flags := 0
			ctx.SetErrCall(&np.ErrorHandler{Callback: func(_ string, f int) { flags |= f }})
			args := make([]*np.Array, len(tc.args))
			for i, x := range tc.args {
				args[i] = np.ScalarOf(x)
			}
			if _, e := ctx.Call(tc.op, args...); e != nil {
				t.Fatal(e)
			}
			refcheck.Check(t, "exmath_fp_"+tc.label, np.ScalarOf(int64(flags)))
		})
	}
}
func TestExtendedUFuncNumPy(t *testing.T) {
	m := refcheck.Require(t)
	lookup := func(s string) *np.UFunc { return refcheck.RequireValue[*np.UFunc](t)(np.LookupUFunc(s)) }
	a := refcheck.ExtendedInput(t, "exmath_ufunc_input", np.Float128)
	mask := refcheck.Input(t, "exmath_ufunc_mask")
	r := m(a.Slice(np.Slice{}, np.Reverse()))
	out := func() *np.Array {
		return m(m(np.Full(np.Float128, extended.FromInt64(91), 2, 3)).Slice(np.Slice{}, np.Reverse()))
	}
	for _, name := range []string{"sin", "log", "exp"} {
		dest := out()
		v := m(lookup(name).CallWithOptions(np.UFuncOptions{DType: np.Float128, Where: mask, Out: dest}, r))
		if v != dest {
			t.Fatal("out identity lost")
		}
		refcheck.CheckExtended(t, "exmath_ufunc_"+name, v, np.Float128, 8)
	}
	dest := out()
	npow := m(np.FromSlice([]int64{2, 3, 4}, 3))
	refcheck.CheckExtended(t, "exmath_ufunc_power", m(lookup("power").CallWithOptions(np.UFuncOptions{Where: mask, Out: dest}, a, npow)), np.Float128, 0)
	refcheck.CheckExtended(t, "exmath_ufunc_reduce", m(lookup("add").ReduceWithOptions(a, np.ReduceOptions{Axes: []int{1}, KeepDims: true, Where: mask, Initial: extended.MustParse("0.0000000000000000001")})), np.Float128, 1)
	refcheck.CheckExtended(t, "exmath_ufunc_hypot", m(lookup("hypot").Reduce(a, np.SumOptions{Axes: []int{1}})), np.Float128, 8)
	refcheck.CheckExtended(t, "exmath_ufunc_accumulate", m(lookup("maximum").Accumulate(r, 1, 0)), np.Float128, 0)
	refcheck.CheckExtended(t, "exmath_ufunc_reduceat", m(lookup("subtract").ReduceAt(a, []int{0, 2, 1}, 1, 0, nil)), np.Float128, 0)
	x := m(np.FromSlice([]np.ExtendedFloat{extended.FromInt64(2), extended.FromInt64(3)}, 2))
	y := m(np.FromSlice([]np.ExtendedFloat{extended.MustParse("0.5"), extended.FromInt64(2)}, 2))
	refcheck.CheckExtended(t, "exmath_ufunc_outer", m(lookup("power").Outer(x, y)), np.Float128, 8)
	at := m(np.FromSlice([]np.ExtendedFloat{extended.FromInt64(1), extended.FromInt64(2), extended.FromInt64(3)}, 3))
	updates := m(np.FromSlice([]np.ExtendedFloat{extended.MustParse("0.0000000000000000001"), extended.FromInt64(1), extended.FromInt64(2)}, 3))
	if e := lookup("add").At(at, []int{1, 1, 0}, updates); e != nil {
		t.Fatal(e)
	}
	refcheck.CheckExtended(t, "exmath_ufunc_at", at, np.Float128, 0)
}
