package linalg_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
	"testing"
)

func TestAdvancedLinearAlgebraParameters(t *testing.T) {
	must := refcheck.Require(t)
	cut := must(np.FromSlice([]float32{.05, .5}, 2))
	abs := must(np.FromSlice([]float64{.01, 3}, 2, 1))
	for _, d := range []string{"float32", "float64", "complex64", "complex128"} {
		t.Run(d, func(t *testing.T) {
			must := refcheck.Require(t)
			p := "param_linalg_hermitian_" + d
			a := refcheck.Input(t, p+"_input")
			u, s, vh, e := linalg.SVD(a, linalg.SVDOptions{Hermitian: true})
			if e != nil {
				t.Fatal(e)
			}
			refcheck.Metadata(t, p+"_u", u)
			refcheck.Metadata(t, p+"_vh", vh)
			refcheck.Check(t, p+"_s", s)
			verifyComplexSVD(t, a, u, s, vh)
			u, s, vh, e = linalg.SVD(a, linalg.SVDOptions{Hermitian: true, ValuesOnly: true})
			if e != nil || u != nil || vh != nil {
				t.Fatalf("values-only: %v", e)
			}
			refcheck.Check(t, p+"_values", s)
			refcheck.Check(t, p+"_pinv", must(linalg.PinvWithOptions(a, linalg.PinvOptions{Hermitian: true, RCond: cut})))
			refcheck.Check(t, p+"_pinv_default", must(linalg.PinvWithOptions(a, linalg.PinvOptions{Hermitian: true, DefaultRTol: true})))
			refcheck.Check(t, p+"_pinv_relative", must(linalg.PinvWithOptions(a, linalg.PinvOptions{Hermitian: true, RTol: cut})))
			refcheck.Check(t, p+"_rank", must(linalg.MatrixRankWithOptions(a, linalg.RankOptions{Hermitian: true})))
			refcheck.Check(t, p+"_rank_relative", must(linalg.MatrixRankWithOptions(a, linalg.RankOptions{Hermitian: true, RTol: cut})))
			refcheck.Check(t, p+"_rank_expand", must(linalg.MatrixRankWithOptions(a, linalg.RankOptions{Hermitian: true, Tol: abs})))
			for _, kind := range []string{"tall", "wide", "batch", "empty"} {
				p := "param_linalg_raw_" + d + "_" + kind
				h, tau, e := linalg.QRRaw(refcheck.Input(t, p+"_input"))
				if e != nil {
					t.Fatal(e)
				}
				refcheck.Check(t, p+"_h", h)
				refcheck.Check(t, p+"_tau", tau)
			}
		})
	}
	a := must(np.Ones(np.Float64, 2, 2))
	_, e := linalg.PinvWithOptions(a, linalg.PinvOptions{RCond: cut, RTol: cut})
	refcheck.Error(t, "param_linalg_pinv_conflict", "ValueError", e, np.ErrValue)
	_, e = linalg.MatrixRankWithOptions(a, linalg.RankOptions{Tol: cut, RTol: cut})
	refcheck.Error(t, "param_linalg_rank_conflict", "ValueError", e, np.ErrValue)
	_, e = linalg.PinvWithOptions(must(np.Ones(np.Float64, 2, 2, 2)), linalg.PinvOptions{RCond: must(np.Ones(np.Float64, 3))})
	refcheck.Error(t, "param_linalg_pinv_shape", "ValueError", e, np.ErrShape)
	_, _, _, e = linalg.SVD(must(np.Ones(np.Float64, 2, 3)), linalg.SVDOptions{Hermitian: true})
	refcheck.Error(t, "param_linalg_hermitian_shape", "LinAlgError", e, np.ErrShape)
	_, _, e = linalg.QRRaw(must(np.Ones(np.Float16, 2, 2)))
	refcheck.Error(t, "param_linalg_raw_half", "TypeError", e, np.ErrDType)
	refcheck.Check(t, "param_linalg_pinv_empty", must(linalg.PinvWithOptions(must(np.Zeros(np.Int16, 0, 3)), linalg.PinvOptions{})))
}

func TestTypedMatrixProducts(t *testing.T) {
	for _, d := range []string{"float16", "float32", "float64", "complex64", "complex128", "int64", "bool", "nonfinite_float32", "nonfinite_float64", "nonfinite_complex64", "nonfinite_complex128"} {
		t.Run(d, func(t *testing.T) {
			must := refcheck.Require(t)
			p := "param_product_" + d
			a, b := refcheck.Input(t, p+"_a"), refcheck.Input(t, p+"_b")
			if a.NDim() == 4 {
				a = must(a.Slice(np.Ellipsis, np.Reverse()))
				b = must(b.Slice(np.Ellipsis, np.Reverse(), np.Slice{}))
			}
			refcheck.Check(t, p+"_matmul", must(linalg.MatMul(a, b)))
			if a.NDim() == 2 {
				refcheck.Check(t, p+"_tensordot", must(np.Tensordot(a, b, []int{1}, []int{0})))
				return
			}
			a = must(refcheck.Input(t, p+"_ta").Slice(np.Slice{}, np.Reverse(), np.Reverse()))
			b = must(refcheck.Input(t, p+"_tb").Slice(np.Reverse(), np.Reverse(), np.Slice{}))
			refcheck.Check(t, p+"_tensordot", must(np.Tensordot(a, b, []int{2, 1}, []int{0, 1})))
			refcheck.Check(t, p+"_einsum", must(np.EinsumWithPath("ijk,kjl->li", np.EinsumPlan{Steps: [][]int{{0, 1}}}, a, b)))
		})
	}
	must := refcheck.Require(t)
	a, b := refcheck.Input(t, "param_product_mixed_a"), refcheck.Input(t, "param_product_mixed_b")
	refcheck.Check(t, "param_product_mixed_matmul", must(linalg.MatMul(a, b)))
	refcheck.Check(t, "param_product_mixed_dot", must(np.Dot(a, b)))
	refcheck.Check(t, "param_product_mixed_inner", must(np.Inner(a, must(b.Transpose()))))
	plan := np.EinsumPlan{Steps: [][]int{{0, 1}}}
	refcheck.Check(t, "param_product_mixed_einsum", must(np.EinsumWithPath("ik,kj->ji", plan, a, b)))
	refcheck.Check(t, "param_product_zero_tensordot", must(np.Tensordot(must(np.Zeros(np.Float64, 3, 0)), must(np.Zeros(np.Float64, 0, 5)), []int{1}, []int{0})))
	refcheck.Check(t, "param_product_empty_tensordot", must(np.Tensordot(must(np.Zeros(np.Float64, 0, 3)), must(np.Zeros(np.Float64, 3, 5)), []int{1}, []int{0})))
	arr := must(a.Slice(np.NewAxis, np.Ellipsis))
	two := must(np.Multiply(b, np.ScalarOf(complex64(2))))
	batch := must(np.Stack([]*np.Array{b, two}, 0))
	refcheck.Check(t, "param_product_broadcast_einsum", must(np.EinsumWithPath("bij,bjk->bik", plan, arr, batch)))
}
