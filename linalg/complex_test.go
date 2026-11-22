package linalg_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
	"math/cmplx"
	"testing"
)

func TestComplexLinearAlgebra(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "complex_linalg_a")
	b := refcheck.Input(t, "complex_linalg_b")
	refcheck.Check(t, "complex_matmul", must(linalg.MatMul(a, b)))
	refcheck.Check(t, "complex_matmul_mixed", must(linalg.MatMul(a, must(np.FromSlice([]float32{1, 2, 3, 4}, 2, 2)))))
	refcheck.Check(t, "complex_solve", must(linalg.Solve(a, b)))
	refcheck.Check(t, "complex_solve_matrix", must(linalg.Solve(a, must(np.FromSlice([]complex64{1 + 1i, 3 - 1i, 2i, 5 + 2i}, 2, 2)))))
	refcheck.Check(t, "complex_solve_single", must(linalg.Solve(must(a.Astype(np.Complex64)), b)))
	refcheck.Check(t, "complex_inv", must(linalg.Inv(a)))
	refcheck.Check(t, "complex_det", must(linalg.Det(a)))
	sg, lg, err := linalg.Slogdet(a)
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "complex_slogdet_sign", sg)
	refcheck.Check(t, "complex_slogdet_log", lg)
	refcheck.Check(t, "integer_matmul", must(linalg.MatMul(must(np.FromSlice([]int8{120, 2, -3, 5}, 2, 2)), must(np.FromSlice([]int8{3, 2, 5, 6}, 2, 2)))))
	refcheck.Check(t, "bool_matmul", must(linalg.MatMul(must(np.FromSlice([]bool{true, false, true, true}, 2, 2)), must(np.FromSlice([]bool{false, true, true, false}, 2, 2)))))
	refcheck.Check(t, "real_float32_solve", must(linalg.Solve(must(np.FromSlice([]float32{3, 1, 2, 4}, 2, 2)), must(np.FromSlice([]float32{5, 6}, 2)))))
	refcheck.Check(t, "real_integer_solve", must(linalg.Solve(must(np.FromSlice([]int8{3, 1, 2, 4}, 2, 2)), must(np.FromSlice([]int8{5, 6}, 2)))))
	h := refcheck.Input(t, "complex_cholesky_input")
	refcheck.Check(t, "complex_cholesky_lower", must(linalg.Cholesky(h)))
	refcheck.Check(t, "complex_cholesky_upper", must(linalg.Cholesky(h, true)))
	_, err = linalg.Solve(must(np.FromComplex128([]complex128{1 + 1i, 2 + 2i, 2 + 2i, 4 + 4i}, 2, 2)), must(np.Ones(np.Float64, 2)))
	refcheck.Error(t, "complex_singular", "LinAlgError", err, linalg.ErrSingular)
	_, err = linalg.Cholesky(must(np.FromComplex128([]complex128{1, 2i, -2i, 1}, 2, 2)))
	refcheck.Error(t, "complex_not_pd", "LinAlgError", err, linalg.ErrNotPositiveDefinite)
	refcheck.Check(t, "complex_linalg_a", a)
	refcheck.Check(t, "complex_linalg_b", b)
}

func TestComplexSVDAndQR(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "complex_svd_input")
	cases := []struct {
		name    string
		input   *np.Array
		reduced bool
	}{
		{"complex_svd_tall", a, false}, {"complex_svd_wide", must(a.Transpose(0, 2, 1)), true},
		{"complex_svd_single", must(a.Astype(np.Complex64)), true}, {"complex_svd_empty", must(np.Zeros(np.Complex128, 0, 3)), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			u, s, vh, err := linalg.SVD(tc.input, linalg.SVDOptions{Reduced: tc.reduced})
			if err != nil {
				t.Fatal(err)
			}
			refcheck.Metadata(t, tc.name+"_u", u)
			refcheck.Check(t, tc.name+"_s", s)
			refcheck.Metadata(t, tc.name+"_vh", vh)
			verifyComplexSVD(t, tc.input, u, s, vh)
		})
	}
	_, s, _, err := linalg.SVD(a, linalg.SVDOptions{ValuesOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "complex_svd_values", s)
	refcheck.Check(t, "complex_pinv", must(linalg.Pinv(a, -1)))
	refcheck.Check(t, "complex_rank", must(linalg.MatrixRank(a, -1)))
	for _, mode := range []string{"reduced", "complete"} {
		q, r, err := linalg.QR(a, linalg.QROptions{Complete: mode == "complete"})
		if err != nil {
			t.Fatal(err)
		}
		refcheck.Metadata(t, "complex_qr_"+mode+"_q", q)
		refcheck.Metadata(t, "complex_qr_"+mode+"_r", r)
		got := must(linalg.MatMul(q, r))
		refcheck.Check(t, "complex_svd_input", got)
		qh := must(must(np.Conjugate(q)).Transpose(0, 2, 1))
		qq := must(linalg.MatMul(qh, q))
		vals, _ := qq.Complex128s()
		cols := q.Shape()[2]
		for i, z := range vals {
			var expected complex128
			if i%(cols*cols)/cols == i%cols {
				expected = 1
			}
			assertComplex(t, z, expected, np.Complex128)
		}
	}
	refcheck.Check(t, "complex_svd_input", a)
}
func assertComplex(t *testing.T, got, want complex128, d np.DType) {
	t.Helper()
	if !refcheck.CloseDType(real(got), real(want), d) || !refcheck.CloseDType(imag(got), imag(want), d) {
		t.Fatalf("got %v; want %v", got, want)
	}
}
func verifyComplexSVD(t *testing.T, a, u, s, vh *np.Array) {
	t.Helper()
	as, us, vs := a.Shape(), u.Shape(), vh.Shape()
	m, n := as[len(as)-2], as[len(as)-1]
	k := min(m, n)
	uc, vr := us[len(us)-1], vs[len(vs)-2]
	batches := 1
	for _, v := range as[:len(as)-2] {
		batches *= v
	}
	av, _ := a.Complex128s()
	uv, _ := u.Complex128s()
	sv, _ := s.Complex128s()
	vv, _ := vh.Complex128s()
	for b := 0; b < batches; b++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var z complex128
				for h := 0; h < k; h++ {
					z += uv[b*m*uc+i*uc+h] * sv[b*k+h] * vv[b*vr*n+h*n+j]
				}
				assertComplex(t, z, av[b*m*n+i*n+j], a.DType())
			}
		}
		for i := 0; i < uc; i++ {
			for j := 0; j < uc; j++ {
				var z, want complex128
				if i == j {
					want = 1
				}
				for h := 0; h < m; h++ {
					z += cmplx.Conj(uv[b*m*uc+h*uc+i]) * uv[b*m*uc+h*uc+j]
				}
				assertComplex(t, z, want, a.DType())
			}
		}
		for i := 0; i < vr; i++ {
			for j := 0; j < vr; j++ {
				var z, want complex128
				if i == j {
					want = 1
				}
				for h := 0; h < n; h++ {
					z += vv[b*vr*n+i*n+h] * cmplx.Conj(vv[b*vr*n+j*n+h])
				}
				assertComplex(t, z, want, a.DType())
			}
		}
	}
}
