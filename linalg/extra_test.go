package linalg_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
	"testing"
)

func TestRemainingLinalgRoutines(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "complex_linalg_a")
	for _, name := range []string{"default", "1", "-2", "fro"} {
		ord := name
		if name == "default" {
			ord = ""
		}
		refcheck.Check(t, "linalg_cond_"+name, must(linalg.Cond(a, ord)))
	}
	refcheck.Check(t, "linalg_svdvals", must(linalg.SVDVals(a)))
	refcheck.CheckLayout(t, "linalg_matrix_transpose", must(linalg.MatrixTranspose(a)))
	refcheck.Check(t, "linalg_matrix_norm", must(linalg.MatrixNorm(a, "1", true)))
	refcheck.Check(t, "linalg_vector_norm", must(linalg.VectorNorm(a, "3", []int{0, 2}, true)))
	v, w := refcheck.Input(t, "linalg_cross_input"), refcheck.Input(t, "linalg_cross_other")
	refcheck.Check(t, "linalg_cross", must(linalg.Cross(v, w, -1)))
	refcheck.Check(t, "linalg_vecdot", must(linalg.Vecdot(v, w, -1)))
	_, err := np.Cross(must(np.FromSlice([]int64{2, 3}, 2)), must(np.FromSlice([]int64{4, 5}, 2)), -1, -1, -1)
	refcheck.Error(t, "core_cross2", "ValueError", err, np.ErrShape)
	refcheck.Check(t, "linalg_multidot", must(linalg.MultiDot(must(np.FromFloat64([]float64{1, 2}, 2)), must(np.FromFloat64([]float64{0, 1, 2, 3, 4, 5}, 2, 3)), must(np.FromFloat64([]float64{2, 1, -1}, 3)))))
	tensor := must(must(np.Identity(6, np.Float64)).Reshape(2, 3, 3, 2))
	refcheck.Check(t, "linalg_tensorinv", must(linalg.TensorInv(tensor, 2)))
	refcheck.Check(t, "linalg_tensorsolve", must(linalg.TensorSolve(tensor, must(np.FromFloat64([]float64{0, 1, 2, 3, 4, 5}, 2, 3)), nil)))
}
