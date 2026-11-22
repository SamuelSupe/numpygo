package linalg_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
	"math/cmplx"
	"testing"
)

func TestEigen(t *testing.T) {
	must := refcheck.Require(t)
	for _, name := range []string{"eigh_complex", "eigh_single", "eigh_repeated"} {
		t.Run(name, func(t *testing.T) {
			a := refcheck.Input(t, name+"_input")
			e, v, err := linalg.Eigh(a, name == "eigh_single")
			if err != nil {
				t.Fatal(err)
			}
			refcheck.Check(t, name+"_values", e)
			refcheck.Metadata(t, name+"_vectors", v)
			verifyEigen(t, a, e, v, true)
		})
	}
	refcheck.Check(t, "eigvalsh", must(linalg.Eigvalsh(refcheck.Input(t, "eigh_complex_input"))))
	for _, name := range []string{"eig_complex", "eig_real", "eig_repeated", "eig_complex_real_pair"} {
		t.Run(name, func(t *testing.T) {
			a := refcheck.Input(t, name+"_input")
			e, v, err := linalg.Eig(a)
			if err != nil {
				t.Fatal(err)
			}
			refcheck.Metadata(t, name+"_values", e)
			refcheck.Metadata(t, name+"_vectors", v)
			matchEigenvalues(t, name+"_values", e)
			verifyEigen(t, a, e, v, false)
		})
	}
	matchEigenvalues(t, "eigvals", must(linalg.Eigvals(refcheck.Input(t, "eig_complex_input"))))
}
func matchEigenvalues(t *testing.T, name string, got *np.Array) {
	t.Helper()
	refcheck.Metadata(t, name, got)
	want := refcheck.Input(t, name)
	wv, _ := want.Complex128s()
	gv, _ := got.Complex128s()
	n := got.Shape()[got.NDim()-1]
	for batch := 0; batch < len(gv)/n; batch++ {
		used := make([]bool, n)
		for i := 0; i < n; i++ {
			match := -1
			for j := 0; j < n; j++ {
				if !used[j] && refcheck.CloseDType(real(gv[batch*n+i]), real(wv[batch*n+j]), got.DType()) && refcheck.CloseDType(imag(gv[batch*n+i]), imag(wv[batch*n+j]), got.DType()) {
					match = j
					break
				}
			}
			if match < 0 {
				t.Fatalf("%s: unmatched eigenvalue %v, expected %v", name, gv[batch*n+i], wv[batch*n:(batch+1)*n])
			}
			used[match] = true
		}
	}
}
func verifyEigen(t *testing.T, a, e, v *np.Array, hermitian bool) {
	t.Helper()
	shape := a.Shape()
	n := shape[len(shape)-1]
	av, _ := a.Complex128s()
	ev, _ := e.Complex128s()
	vv, _ := v.Complex128s()
	for q := 0; q < len(ev)/n; q++ {
		for j := 0; j < n; j++ {
			for i := 0; i < n; i++ {
				var product complex128
				for h := 0; h < n; h++ {
					product += av[q*n*n+i*n+h] * vv[q*n*n+h*n+j]
				}
				assertComplex(t, product, ev[q*n+j]*vv[q*n*n+i*n+j], v.DType())
			}
			for k := 0; k < n; k++ {
				if !hermitian && k != j {
					continue
				}
				var dot, want complex128
				for h := 0; h < n; h++ {
					dot += cmplx.Conj(vv[q*n*n+h*n+j]) * vv[q*n*n+h*n+k]
				}
				if j == k {
					want = 1
				}
				assertComplex(t, dot, want, v.DType())
			}
		}
	}
}
func TestLeastSquaresAndMatrixFunctions(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "complex_linalg_a")
	for _, tc := range []struct {
		name string
		p    int
	}{{"matrix_power_pos", 3}, {"matrix_power_neg", -2}, {"matrix_power_zero", 0}} {
		refcheck.Check(t, tc.name, must(linalg.MatrixPower(a, tc.p)))
	}
	for _, name := range []string{"lstsq_full", "lstsq_rank_deficient"} {
		t.Run(name, func(t *testing.T) {
			x, res, rank, s, err := linalg.Lstsq(refcheck.Input(t, name+"_input"), refcheck.Input(t, name+"_rhs"), nil)
			if err != nil {
				t.Fatal(err)
			}
			refcheck.Check(t, name+"_x", x)
			refcheck.Check(t, name+"_residuals", res)
			refcheck.Check(t, name+"_rank", np.ScalarOf(rank))
			refcheck.Check(t, name+"_s", s)
		})
	}
	for _, ord := range []string{"default", "fro", "nuc", "2", "-2", "1", "-1", "inf", "-inf"} {
		value := ord
		if ord == "default" {
			value = ""
		}
		refcheck.Check(t, "norm_matrix_"+ord, must(linalg.Norm(a, linalg.NormOptions{Ord: value, Axes: []int{-2, -1}, KeepDims: true})))
	}
	rhs := refcheck.Input(t, "lstsq_full_rhs")
	for _, ord := range []string{"default", "0", "1", "2", "3", "-2", "inf", "-inf"} {
		value := ord
		if ord == "default" {
			value = ""
		}
		refcheck.Check(t, "norm_vector_"+ord, must(linalg.Norm(rhs, linalg.NormOptions{Ord: value})))
	}
	refcheck.Check(t, "norm_all", must(linalg.Norm(a)))
}
