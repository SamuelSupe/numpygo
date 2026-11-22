package np_test

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestGufuncAxesNumPy(t *testing.T) {
	m := refcheck.Require(t)
	get := refcheck.RequireValue[*np.UFunc](t)
	for _, name := range []string{"matmul", "matvec", "vecmat", "vecdot"} {
		u := get(np.LookupUFunc(name))
		lm, rm := name == "matmul" || name == "matvec", name == "matmul" || name == "vecmat"
		pa, pb := []int{2, 0, 1}, []int{0, 2, 1}
		axes := [][]int{{0}, {1}, {1}}
		if lm {
			pa, axes[0] = []int{2, 0, 3, 1}, []int{0, 2}
		}
		if rm {
			pb, axes[1] = []int{0, 3, 1, 2}, []int{3, 1}
		}
		if name == "matmul" {
			axes[2] = []int{0, 3}
		} else if name == "vecdot" {
			axes[2] = []int{}
		}
		for _, dtype := range []string{"bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "float128", "complex64", "complex128", "complex256"} {
			prefix := "ufopt_" + name + "_" + dtype
			a, b := refcheck.Input(t, prefix+"_a"), refcheck.Input(t, prefix+"_b")
			rb := -1
			if rm {
				rb = -2
			}
			a = m(m(np.Flip(a, -1)).Transpose(pa...))
			b = m(m(np.Flip(b, rb)).Transpose(pb...))
			for _, order := range []string{"K", "A", "C", "F"} {
				t.Run(name+"/"+dtype+"/"+order, func(t *testing.T) {
					got := refcheck.Require(t)(u.CallWithOptions(np.UFuncOptions{Axes: axes, Order: order}, a, b))
					refcheck.CheckLayout(t, prefix+"_"+order, got)
					if got.Base() != nil || !got.Writable() {
						t.Fatal("allocated result is not an independent writable array")
					}
				})
			}
		}
	}
}

func TestGufuncOptionsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	get := refcheck.RequireValue[*np.UFunc](t)
	ar := func(n int, shape ...int) *np.Array {
		return m(m(m(np.ArangeInt64(0, int64(n), 1)).Astype(np.Float64)).Reshape(shape...))
	}
	mm, mv, vm, vd := get(np.LookupUFunc("matmul")), get(np.LookupUFunc("matvec")), get(np.LookupUFunc("vecmat")), get(np.LookupUFunc("vecdot"))
	a, b, v := ar(24, 2, 3, 4), ar(20, 4, 5), ar(4, 4)
	keep, drop := true, false
	p := ar(24, 2, 4, 3)
	q := m(np.Flip(p, -1))
	check := func(name string, u *np.UFunc, a, b *np.Array, opt np.UFuncOptions) {
		t.Helper()
		res := m(u.CallWithOptions(opt, a, b))
		if opt.Out != nil && res != opt.Out {
			t.Fatal("out identity")
		}
		refcheck.CheckLayout(t, "ufopt_"+name, res)
	}
	check("mv_optional", mm, a, v, np.UFuncOptions{Axes: [][]int{{1, 2}, {0}, {0}}})
	check("vm_optional", mm, v, m(np.SwapAxes(a, -1, -2)), np.UFuncOptions{Axes: [][]int{{0}, {1, 2}, {0}}})
	check("vv_optional", mm, v, v, np.UFuncOptions{Axes: [][]int{{0}, {0}, {}}})
	check("vd_axis", vd, p, q, np.UFuncOptions{Axis: np.Int(1)})
	check("vd_axis_keep", vd, p, q, np.UFuncOptions{Axis: np.Int(1), KeepDims: &keep})
	check("vd_axes_keep", vd, p, q, np.UFuncOptions{Axes: [][]int{{1}, {1}}, KeepDims: &keep})
	check("vd_axes_keep_out", vd, p, q, np.UFuncOptions{Axes: [][]int{{1}, {1}, {0}}, KeepDims: &keep})
	check("vd_axis_neg", vd, p, q, np.UFuncOptions{Axis: np.Int(-2), KeepDims: &keep})
	check("vd_axis_false", vd, p, q, np.UFuncOptions{Axis: np.Int(1), KeepDims: &drop})
	check("vd_mixed_ranks", vd, p, m(v.Reshape(4, 1)), np.UFuncOptions{Axis: np.Int(-2), KeepDims: &keep})
	for _, tc := range []struct {
		name  string
		u     *np.UFunc
		a, b  *np.Array
		opt   np.UFuncOptions
		class string
		err   error
	}{
		{"vd_axis_both", vd, p, q, np.UFuncOptions{Axis: np.Int(1), Axes: [][]int{{1}, {1}}}, "TypeError", np.ErrValue},
		{"vd_axis_bad", vd, p, q, np.UFuncOptions{Axis: np.Int(3)}, "AxisError", np.ErrAxis},
		{"vd_keep_wrong_axes", vd, p, q, np.UFuncOptions{KeepDims: &keep, Axes: [][]int{{1}, {1}, {}}}, "AxisError", np.ErrAxis},
		{"mm_axis", mm, a, b, np.UFuncOptions{Axis: np.Int(-1)}, "TypeError", np.ErrValue},
		{"mm_keep", mm, a, b, np.UFuncOptions{KeepDims: &drop}, "TypeError", np.ErrValue},
		{"mm_axes_short", mm, a, b, np.UFuncOptions{Axes: [][]int{{1, 2}, {0, 1}}}, "ValueError", np.ErrValue},
		{"mm_axes_duplicate", mm, a, b, np.UFuncOptions{Axes: [][]int{{1, 1}, {0, 1}, {1, 2}}}, "ValueError", np.ErrAxis},
		{"mm_axes_wrong_rank", mm, a, b, np.UFuncOptions{Axes: [][]int{{1, 2}, {0}, {1, 2}}}, "AxisError", np.ErrAxis},
		{"mm_axes_bad_output", mm, a, b, np.UFuncOptions{Axes: [][]int{{1, 2}, {0, 1}, {0, 4}}}, "AxisError", np.ErrAxis},
		{"mm_where", mm, a, b, np.UFuncOptions{Where: np.ScalarOf(true)}, "TypeError", np.ErrValue},
		{"mm_bad_order", mm, a, b, np.UFuncOptions{Order: "Q"}, "ValueError", np.ErrValue},
		{"mm_core_mismatch", mm, a, m(np.Zeros(np.Float64, 1, 5)), np.UFuncOptions{}, "ValueError", np.ErrShape},
		{"mm_bad_core_out", mm, m(np.Ones(np.Float64, 2, 1)), m(np.Ones(np.Float64, 1, 1)), np.UFuncOptions{Out: m(np.Zeros(np.Float64, 2, 3))}, "ValueError", np.ErrShape},
		{"mm_bad_batch_out", mm, a, b, np.UFuncOptions{Out: m(np.Zeros(np.Float64, 1, 3, 5))}, "ValueError", np.ErrShape},
		{"vd_bad_keep_out", vd, p, q, np.UFuncOptions{Axis: np.Int(1), KeepDims: &keep, Out: m(np.Zeros(np.Float64, 2, 2, 3))}, "ValueError", np.ErrShape},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, e := tc.u.CallWithOptions(tc.opt, tc.a, tc.b)
			refcheck.Error(t, "ufopt_"+tc.name, tc.class, e, tc.err)
		})
	}
	for _, tc := range []struct {
		name  string
		u     *np.UFunc
		a, b  *np.Array
		shape []int
	}{
		{"mm", mm, a, b, []int{7, 2, 3, 5}},
		{"mv", mv, a, v, []int{7, 2, 3}},
		{"vm", vm, v, m(np.SwapAxes(a, -1, -2)), []int{7, 2, 3}},
		{"vd", vd, a, v, []int{7, 2, 3}},
	} {
		out := m(np.Flip(m(np.Full(np.Float32, float32(-99), tc.shape...)), -1))
		check(tc.name+"_broadcast_out", tc.u, tc.a, tc.b, np.UFuncOptions{Out: out})
	}
	out := m(np.Flip(m(np.Full(np.Float64, -99., 3, 7, 2, 5)), 2))
	check("mm_mapped_out", mm, a, b, np.UFuncOptions{Axes: [][]int{{1, 2}, {0, 1}, {0, 3}}, Out: out})
	out = m(np.Flip(m(np.Full(np.Float64, -99., 2, 1, 3)), -1))
	check("vd_keep_out", vd, p, q, np.UFuncOptions{Axis: np.Int(1), KeepDims: &keep, Out: out})
	x := ar(12, 2, 2, 3)
	twoEye := m(np.Multiply(m(np.Eye(3, 3, 0, np.Float64)), np.Scalar(2)))
	check("mm_alias_out", mm, x, twoEye, np.UFuncOptions{Out: x})
	x = m(np.SwapAxes(ar(120, 2, 3, 4, 5), 0, 1))
	y := ar(30, 5, 6)
	xf, yf := m(x.AstypeWithOptions(np.Float64, np.CastOptions{Order: "F"})), m(y.AstypeWithOptions(np.Float64, np.CastOptions{Order: "F"}))
	af := m(m(a.Transpose(1, 0, 2)).AstypeWithOptions(np.Float64, np.CastOptions{Order: "F"}))
	bf := m(b.AstypeWithOptions(np.Float64, np.CastOptions{Order: "F"}))
	for _, order := range []string{"K", "A", "C", "F"} {
		check("mm_batch_order_"+order, mm, x, y, np.UFuncOptions{Order: order})
		check("mm_f_order_"+order, mm, xf, yf, np.UFuncOptions{Order: order})
		check("mm_empty_order_"+order, mm, m(np.Zeros(np.Float64, 2, 0, 5)), y, np.UFuncOptions{Order: order})
		check("mm_empty_contract_"+order, mm, m(np.Zeros(np.Float64, 2, 3, 0)), m(np.Zeros(np.Float64, 0, 4)), np.UFuncOptions{Order: order})
		check("mm_mapped_f_order_"+order, mm, af, bf, np.UFuncOptions{Order: order, Axes: [][]int{{0, 2}, {0, 1}, {0, 2}}})
	}
	check("signature_matmul", mm, a, b, np.UFuncOptions{Signature: []np.DType{np.Float32, np.Float32, np.Float32}})
	refcheck.CheckLayout(t, "ufopt_mv_optional", m(mm.CallValues(np.UFuncOptions{Axes: [][]int{{1, 2}, {0}, {0}}}, a, v)))
}

func TestUFuncOrderAndSignatureNumPy(t *testing.T) {
	m := refcheck.Require(t)
	get := refcheck.RequireValue[*np.UFunc](t)
	multi := refcheck.RequireValue[[]*np.Array](t)
	add, modf, frexp := get(np.LookupUFunc("add")), get(np.LookupUFunc("modf")), get(np.LookupUFunc("frexp"))
	x := m(m(m(np.ArangeInt64(0, 24, 1)).Astype(np.Float64)).Reshape(2, 3, 4))
	x = m(m(x.Transpose(1, 0, 2)).Slice(np.Slice{}, np.Reverse(), np.Slice{Step: np.Int(2)}))
	y := m(m(m(np.ArangeInt64(0, 4, 1)).Astype(np.Float64)).Reshape(2, 2))
	xf := m(x.AstypeWithOptions(np.Float64, np.CastOptions{Order: "F"}))
	xc := m(x.Copy())
	div := get(np.LookupUFunc("divide"))
	frac := m(div.CallWithOptions(np.UFuncOptions{}, x, np.Scalar(3)))
	for _, order := range []string{"K", "A", "C", "F"} {
		opt := np.UFuncOptions{Order: order}
		for name, r := range map[string]*np.Array{
			"add_order_":      m(add.CallWithOptions(opt, x, y)),
			"add_f_order_":    m(add.CallWithOptions(opt, xf, np.Scalar(1))),
			"add_f_conflict_": m(add.CallWithOptions(opt, xf, xc)),
		} {
			refcheck.CheckLayout(t, "ufopt_"+name+order, r)
		}
		for i, r := range multi(modf.CallAllWithOptions(np.MultiUFuncOptions{Order: order}, frac)) {
			refcheck.CheckLayout(t, "ufopt_modf_order_"+order+"_"+strconv.Itoa(i), r)
		}
		for i, r := range multi(frexp.CallAllWithOptions(np.MultiUFuncOptions{Order: order, Signature: []np.DType{np.Float32, np.Float32, np.Int32}}, frac)) {
			refcheck.CheckLayout(t, "ufopt_frexp_signature_"+order+"_"+strconv.Itoa(i), r)
		}
	}
	for _, shape := range [][]int{{2, 1, 3}, {1, 2, 3}, {2, 3, 1}, {2, 1, 1, 3}} {
		a := m(m(m(m(np.ArangeInt64(0, 6, 1)).Astype(np.Float64)).Reshape(shape...)).AstypeWithOptions(np.Float64, np.CastOptions{Order: "F"}))
		row := m(np.Ones(np.Float64, shape[len(shape)-1]))
		cast := m(m(np.Ones(np.Float32, shape...)).AstypeWithOptions(np.Float32, np.CastOptions{Order: "F"}))
		parts := make([]string, len(shape))
		for i, d := range shape {
			parts[i] = strconv.Itoa(d)
		}
		prefix := "ufopt_singleton_" + strings.Join(parts, "_") + "_"
		frac := m(div.CallWithOptions(np.UFuncOptions{}, a, np.Scalar(3)))
		for _, order := range []string{"K", "A", "C", "F"} {
			opt := np.UFuncOptions{Order: order}
			for name, rhs := range map[string]*np.Array{"scalar": np.Scalar(1), "row": row, "cast": cast} {
				refcheck.CheckLayout(t, prefix+name+"_"+order, m(add.CallWithOptions(opt, a, rhs)))
			}
			refcheck.CheckLayout(t, prefix+"mask_"+order, m(add.CallWithOptions(np.UFuncOptions{Order: order, Where: np.ScalarOf(true)}, a, np.Scalar(1))))
			for i, r := range multi(modf.CallAllWithOptions(np.MultiUFuncOptions{Order: order}, frac)) {
				refcheck.CheckLayout(t, prefix+"modf_"+order+"_"+strconv.Itoa(i), r)
			}
		}
	}
	pt := m(div.CallWithOptions(np.UFuncOptions{}, m(m(m(m(np.ArangeInt64(0, 6, 1)).Astype(np.Float64)).Reshape(2, 3)).Transpose(1, 0)), np.Scalar(3)))
	for _, order := range []string{"K", "A", "C", "F"} {
		for _, arrOrder := range []string{"C", "F"} {
			out := m(m(np.Zeros(np.Float64, pt.Shape()...)).AstypeWithOptions(np.Float64, np.CastOptions{Order: arrOrder}))
			result := multi(modf.CallAllWithOptions(np.MultiUFuncOptions{Order: order, Out: []*np.Array{out, nil}}, pt))
			if result[0] != out {
				t.Fatal("partial out identity")
			}
			for i, r := range result {
				refcheck.CheckLayout(t, fmt.Sprintf("ufopt_partial_%s_%s_%d", arrOrder, order, i), r)
			}
		}
		refcheck.CheckLayout(t, "ufopt_lower_"+order, m(add.CallWithOptions(np.UFuncOptions{Order: strings.ToLower(order)}, pt, np.Scalar(1))))
	}
	x = m(np.FromSlice([]float64{1.1, 1.2, 1.0000000001}, 3))
	y = m(np.FromSlice([]float64{1.2, 1.1, 1.0000000002}, 3))
	z := m(np.FromSlice([]complex128{3 + 4i, -5 + 12i}, 2))
	less, abs := get(np.LookupUFunc("less")), get(np.LookupUFunc("absolute"))
	for _, tc := range []struct {
		name string
		u    *np.UFunc
		args []*np.Array
		opt  np.UFuncOptions
	}{
		{"less_dtype", less, []*np.Array{x, y}, np.UFuncOptions{DType: np.Bool}},
		{"less_signature", less, []*np.Array{x, y}, np.UFuncOptions{Signature: []np.DType{np.Float64, np.Float64, np.Bool}}},
		{"abs_dtype", abs, []*np.Array{z}, np.UFuncOptions{DType: np.Float64}},
		{"abs_dtype_unsafe", abs, []*np.Array{z}, np.UFuncOptions{DType: np.Float64, Casting: "unsafe"}},
		{"abs_dtype_widen", abs, []*np.Array{m(z.Astype(np.Complex64))}, np.UFuncOptions{DType: np.Float64}},
		{"abs_signature", abs, []*np.Array{z}, np.UFuncOptions{Signature: []np.DType{np.Complex64, np.Float32}}},
		{"signature_add", add, []*np.Array{x, y}, np.UFuncOptions{Signature: []np.DType{np.Float32, np.Float32, np.Float32}}},
		{"signature_ldexp", get(np.LookupUFunc("ldexp")), []*np.Array{m(np.FromSlice([]int16{1, 2}, 2)), m(np.FromSlice([]int8{2, 3}, 2))}, np.UFuncOptions{Signature: []np.DType{np.Float32, np.Int32, np.Float32}}},
		{"abs_narrow_dtype", abs, []*np.Array{z}, np.UFuncOptions{DType: np.Float32}},
		{"invalid_signature", add, []*np.Array{x, y}, np.UFuncOptions{Signature: []np.DType{np.Float64, np.Float32, np.Float64}}},
		{"signature_safe", add, []*np.Array{x, y}, np.UFuncOptions{Signature: []np.DType{np.Float32, np.Float32, np.Float32}, Casting: "safe"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v, e := tc.u.CallWithOptions(tc.opt, tc.args...)
			refcheck.Case(t, "ufopt_"+tc.name, v, e)
		})
	}
	_, e := add.CallWithOptions(np.UFuncOptions{DType: np.Float64, Signature: []np.DType{np.Float64, np.Float64, np.Float64}}, x, y)
	refcheck.Error(t, "ufopt_dtype_signature", "TypeError", e, np.ErrValue)
	_, e = add.CallWithOptions(np.UFuncOptions{Axes: [][]int{{}, {}, {}}}, x, y)
	refcheck.Error(t, "ufopt_elementwise_axes", "TypeError", e, np.ErrValue)
	_, e = add.CallWithOptions(np.UFuncOptions{Order: "Q"}, x, y)
	refcheck.Error(t, "ufopt_elementwise_order", "ValueError", e, np.ErrValue)
	_, e = frexp.CallAllWithOptions(np.MultiUFuncOptions{DType: np.Float64}, x)
	refcheck.Error(t, "ufopt_frexp_dtype", "TypeError", e, np.ErrDType)
	i8 := m(np.FromSlice([]int8{1, 2}, 2))
	refcheck.Check(t, "ufopt_signature_weak", m(add.CallValues(np.UFuncOptions{Signature: []np.DType{np.Int16, np.Int16, np.Int16}}, i8, np.Weak(300))))
	_, e = add.CallValues(np.UFuncOptions{Signature: []np.DType{np.Int8, np.Int8, np.Int8}}, i8, np.Weak(300))
	refcheck.Error(t, "ufopt_signature_weak_overflow", "OverflowError", e, np.ErrOverflow)
}
