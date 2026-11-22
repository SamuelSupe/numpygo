package linalg_test

import (
	"errors"
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
)

func TestMatMul(t *testing.T) {
	must := refcheck.Require(t)
	raw := refcheck.Input(t, "matmul_input_a")
	a := must(raw.Transpose(0, 1, 3, 2))
	b := refcheck.Input(t, "matmul_input_b")
	refcheck.Check(t, "matmul_batched", must(linalg.MatMul(a, b)))
	refcheck.Check(t, "matmul_vv", must(linalg.MatMul(
		must(np.FromFloat64([]float64{0, 1, 2}, 3)), must(np.FromFloat64([]float64{1, 3, -1}, 3)))))
	refcheck.Check(t, "matmul_vm", must(linalg.MatMul(must(np.FromFloat64([]float64{0, 1}, 2)), a)))
	refcheck.Check(t, "matmul_mv", must(linalg.MatMul(a, must(np.FromFloat64([]float64{0, 1, 2}, 3)))))
	refcheck.Check(t, "matmul_empty_contract", must(linalg.MatMul(must(np.Zeros(np.Float64, 2, 0)), must(np.Zeros(np.Float64, 0, 3)))))
	refcheck.Check(t, "matmul_empty_batch", must(linalg.MatMul(must(np.Zeros(np.Float64, 0, 2, 3)), must(np.Zeros(np.Float64, 3, 5)))))
	refcheck.Check(t, "matmul_input_a", raw)
	refcheck.Check(t, "matmul_input_b", b)
	_, err := linalg.MatMul(np.Scalar(2), must(np.Zeros(np.Float64, 2, 2)))
	refcheck.Error(t, "matmul_scalar", "ValueError", err, np.ErrShape)
	_, err = linalg.MatMul(must(np.Zeros(np.Float64, 2, 3)), must(np.Zeros(np.Float64, 2, 3)))
	refcheck.Error(t, "matmul_mismatch", "ValueError", err, np.ErrShape)
	_, err = linalg.MatMul(&np.Array{}, must(np.Zeros(np.Float64, 2, 2)))
	if !errors.Is(err, np.ErrDType) {
		t.Fatal("invalid matmul dtype:", err)
	}
}

func TestSolve(t *testing.T) {
	must := refcheck.Require(t)
	raw := refcheck.Input(t, "solve_input_a")
	a := must(raw.Transpose(0, 1, 3, 2))
	b := refcheck.Input(t, "solve_input_b")
	x := must(linalg.Solve(a, b))
	refcheck.Check(t, "solve_matrix", x)
	reconstructed := must(linalg.MatMul(a, x))
	expected := must(b.BroadcastTo(2, 3, 2, 2))
	gotData, _ := reconstructed.Float64s()
	wantData, _ := expected.Float64s()
	for i, v := range gotData {
		if !refcheck.Close(v, wantData[i]) {
			t.Fatalf("A @ X[%d] = %g; B = %g", i, v, wantData[i])
		}
	}
	refcheck.Check(t, "solve_vector", must(linalg.Solve(a, must(np.FromFloat64([]float64{7, 11}, 2)))))
	refcheck.Check(t, "solve_rank2_rhs", must(linalg.Solve(a, must(np.FromFloat64([]float64{1, 2, 3, 4}, 2, 2)))))
	refcheck.Check(t, "solve_empty", must(linalg.Solve(must(np.Zeros(np.Float64, 0, 0)), must(np.Zeros(np.Float64, 0)))))
	refcheck.Check(t, "solve_ill_conditioned", must(linalg.Solve(
		must(np.FromFloat64([]float64{1e-20, 0, 0, 1}, 2, 2)), must(np.FromFloat64([]float64{1e-20, 2}, 2)))))
	refcheck.Check(t, "solve_pivot", must(linalg.Solve(
		must(np.FromFloat64([]float64{0, 2, 3, 4}, 2, 2)), must(np.FromFloat64([]float64{4, 11}, 2)))))
	refcheck.Check(t, "solve_zero_rhs", must(linalg.Solve(
		must(np.FromFloat64([]float64{1, 0, 0, 1}, 2, 2)), must(np.Zeros(np.Float64, 2, 0)))))
	refcheck.Check(t, "solve_input_a", raw)
	refcheck.Check(t, "solve_input_b", b)

	singular := must(np.FromFloat64([]float64{2, 0, 0, 2, 1, 2, 2, 4}, 2, 2, 2))
	_, err := linalg.Solve(singular, must(np.FromFloat64([]float64{1, 1}, 2)))
	refcheck.Error(t, "solve_singular", "LinAlgError", err, linalg.ErrSingular)
	_, err = linalg.Solve(must(singular.Slice(np.AtIndex(1))), must(np.Zeros(np.Float64, 2, 0)))
	refcheck.Error(t, "solve_singular_zero_rhs", "LinAlgError", err, linalg.ErrSingular)
	if v, _ := singular.At(1, 1, 0); v != 2 {
		t.Fatal("singular solve mutated its input")
	}
	_, err = linalg.Solve(must(np.Zeros(np.Float64, 2, 3)), must(np.Zeros(np.Float64, 3)))
	refcheck.Error(t, "solve_bad_shape", "LinAlgError", err, np.ErrShape)
	_, err = linalg.Solve(must(np.Zeros(np.Float64, 2, 2)), must(np.Zeros(np.Float64, 3)))
	refcheck.Error(t, "solve_rhs_shape", "ValueError", err, np.ErrShape)
}

func TestSVD(t *testing.T) {
	must := refcheck.Require(t)
	raw := refcheck.Input(t, "svd_input")
	tests := []struct {
		name    string
		input   *np.Array
		reduced bool
	}{
		{"svd_full", must(raw.Transpose(0, 2, 1)), false},
		{"svd_reduced", raw, true},
		{"svd_empty_wide", must(np.Zeros(np.Float64, 0, 3)), false},
		{"svd_empty_tall", must(np.Zeros(np.Float64, 3, 0)), false},
		{"svd_empty_batch", must(np.Zeros(np.Float64, 0, 3, 2)), true},
		{"svd_degenerate", must(np.FromFloat64([]float64{1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3)), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			u, s, vh, err := linalg.SVD(tc.input, linalg.SVDOptions{Reduced: tc.reduced})
			if err != nil {
				t.Fatal(err)
			}
			refcheck.Metadata(t, tc.name+"_u", u)
			refcheck.Metadata(t, tc.name+"_vh", vh)
			refcheck.Check(t, tc.name+"_s", s)
			verifySVD(t, tc.input, u, s, vh)
		})
	}
	u, s, vh, err := linalg.SVD(raw, linalg.SVDOptions{ValuesOnly: true})
	if err != nil || u != nil || vh != nil {
		t.Fatalf("values only: %v %v %v", u, vh, err)
	}
	refcheck.Check(t, "svd_values_only", s)
	refcheck.Check(t, "svd_input", raw)
	_, _, _, err = linalg.SVD(must(np.Zeros(np.Float64, 3)))
	refcheck.Error(t, "svd_rank", "LinAlgError", err, np.ErrShape)
	_, _, _, err = linalg.SVD(must(np.FromFloat64([]float64{math.NaN(), 1, 1, 1}, 2, 2)))
	refcheck.Error(t, "svd_nan", "LinAlgError", err, linalg.ErrConvergence)
}

func verifySVD(t *testing.T, a, u, s, vh *np.Array) {
	t.Helper()
	av, _ := a.Float64s()
	uv, _ := u.Float64s()
	sv, _ := s.Float64s()
	vv, _ := vh.Float64s()
	as, us, vs := a.Shape(), u.Shape(), vh.Shape()
	m, n := as[len(as)-2], as[len(as)-1]
	k := min(m, n)
	uc, vr := us[len(us)-1], vs[len(vs)-2]
	batch := 1
	for _, d := range as[:len(as)-2] {
		batch *= d
	}
	for b := 0; b < batch; b++ {
		for i := 0; i < k; i++ {
			if sv[b*k+i] < 0 || (i > 0 && sv[b*k+i] > sv[b*k+i-1]) {
				t.Fatalf("singular values are not nonnegative and descending: %v", sv)
			}
		}
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var x float64
				for z := 0; z < k; z++ {
					x += uv[b*m*uc+i*uc+z] * sv[b*k+z] * vv[b*vr*n+z*n+j]
				}
				if !refcheck.Close(x, av[b*m*n+i*n+j]) {
					t.Fatalf("batch %d reconstruction at (%d,%d): %g != %g", b, i, j, x, av[b*m*n+i*n+j])
				}
			}
		}
		for i := 0; i < uc; i++ {
			for j := 0; j < uc; j++ {
				var dot, want float64
				if i == j {
					want = 1
				}
				for row := 0; row < m; row++ {
					dot += uv[b*m*uc+row*uc+i] * uv[b*m*uc+row*uc+j]
				}
				if !refcheck.Close(dot, want) {
					t.Fatalf("U columns (%d,%d) dot=%g", i, j, dot)
				}
			}
		}
		for i := 0; i < vr; i++ {
			for j := 0; j < vr; j++ {
				var dot, want float64
				if i == j {
					want = 1
				}
				for col := 0; col < n; col++ {
					dot += vv[b*vr*n+i*n+col] * vv[b*vr*n+j*n+col]
				}
				if !refcheck.Close(dot, want) {
					t.Fatalf("Vh rows (%d,%d) dot=%g", i, j, dot)
				}
			}
		}
	}
}
