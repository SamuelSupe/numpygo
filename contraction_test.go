package np_test

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestContraction(t *testing.T) {
	must := refcheck.Require(t)
	a := refcheck.Input(t, "contract_a")
	b := refcheck.Input(t, "contract_b")
	refcheck.Check(t, "contract_dot", must(np.Dot(a, b)))
	refcheck.Check(t, "contract_tensordot", must(np.Tensordot(a, b, []int{2}, []int{0})))
	refcheck.Check(t, "contract_inner", must(np.Inner(a, a)))
	refcheck.Check(t, "contract_outer", must(np.Outer(must(a.Slice(np.AtIndex(0), np.AtIndex(0))), must(b.Slice(np.Slice{}, np.AtIndex(0))))))
	refcheck.Check(t, "contract_vdot", must(np.VDot(must(np.FromComplex128([]complex128{1 + 2i, 3 - 1i}, 2)), must(np.FromComplex128([]complex128{4, 1 + 2i}, 2)))))
	refcheck.Check(t, "contract_kron", must(np.Kron(must(np.FromSlice([]int8{1, 2, 3, 4}, 2, 2)), must(np.FromSlice([]int8{2, 3}, 2)))))
	refcheck.Check(t, "contract_einsum", must(np.Einsum("...ij,jk->...ik", a, b)))
	refcheck.Check(t, "contract_einsum_reduce", must(np.Einsum("ijk,ijk->j", a, a)))
	refcheck.CheckLayout(t, "contract_einsum_implicit", must(np.Einsum("ji", b)))
	diag := must(must(np.ArangeInt64(0, 9, 1)).Reshape(3, 3))
	view := must(np.Einsum("ii->i", diag))
	refcheck.CheckLayout(t, "contract_einsum_diagonal", view)
	refcheck.Check(t, "contract_einsum_trace", must(np.Einsum("ii", diag)))
	if err := view.SetValue(int64(55), 1); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "contract_einsum_writeback", diag)
	_, err := np.Einsum("ii", must(np.Ones(np.Float64, 2, 3)))
	refcheck.Error(t, "contract_bad_einsum", "ValueError", err, np.ErrShape)
	_, err = np.Tensordot(a, b, []int{1}, []int{0})
	refcheck.Error(t, "contract_bad_tensordot", "ValueError", err, np.ErrShape)
}
