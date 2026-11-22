package matrix

import (
	"encoding/json"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
	"testing"
)

func TestMatrixClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mm := refcheck.RequireValue[*Matrix](t)
	check := func(n string, a *np.Array) { t.Helper(); refcheck.Check(t, "numeric_class_matrix_"+n, a) }
	mi := mm(New(m(np.FromSlice([]int64{1, 2, 3, 4, 5, 6}, 2, 3)), 0, false))
	a := mm(New(m(np.FromSlice([]float64{2, 1, 1, 3}, 2, 2)), 0, false))
	z := mm(New(m(np.FromSlice([]complex128{1 + 2i, 3 - 4i, 5 + 6i, -1i}, 2, 2)), 0, false))
	check("A", mi.A())
	check("A1", m(mi.A1()))
	check("transpose", mm(mi.TransposeMatrix(1, 0)).Array)
	list := refcheck.RequireValue[any](t)(mi.ToList())
	b, e := json.Marshal(list)
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Text(t, "numeric_class_matrix_tolist", string(b))
	row := func(a *np.Array) *Matrix { return mm(New(a, 0, false)) }
	three := m(np.FromSlice([]int64{0, 1, 2}, 3))
	two := m(np.FromSlice([]int64{0, 1}, 2))
	for n, v := range map[string]*Matrix{
		"T": mm(mi.T()), "H": mm(z.H()), "I": mm(a.I()), "rect_I": mm(mi.I()),
		"reshape": mm(mi.ReshapeMatrix(3, 2)), "reshape1": mm(mi.ReshapeMatrix(-1)),
		"ravel": mm(mi.Ravel()), "flatten": mm(mi.Flatten()),
		"row_squeeze":    mm(row(m(mi.IndexMatrix(np.Range(0, 1)))).Squeeze()),
		"col_squeeze":    mm(row(m(mi.IndexMatrix(np.Slice{}, np.Range(0, 1)))).Squeeze()),
		"scalar_squeeze": mm(row(np.ScalarOf(int64(2))).Squeeze()),
		"take":           mm(mi.Take(m(np.FromSlice([]int64{0, 3}, 2)), nil)),
		"diagonal":       mm(mi.Diagonal(0)), "copy": mm(mi.CopyMatrix()), "astype": mm(mi.CastMatrix(np.Float32, np.CastOptions{})),
		"real": mm(z.Real()), "imag": mm(z.Imag()),
		"mul":   mm(mi.Mul(m(np.FromSlice([]int64{1, 2, 3, 4, 5, 6}, 3, 2)), false)),
		"scale": mm(mi.Mul(np.ScalarOf(int64(2)), false)), "rmul": mm(mi.Mul(two, true)),
		"matmul": mm(mi.MatMul(three, false)), "rmatmul": mm(mi.MatMul(two, true)),
		"add": mm(mi.Add(m(np.FromSlice([]int64{1, 2, 3}, 3)))), "sub": mm(mi.Sub(np.ScalarOf(int64(2)), false)),
		"rsub": mm(mi.Sub(np.ScalarOf(int64(2)), true)), "div": mm(mi.Divide(np.ScalarOf(int64(2)), false)),
		"rdiv": mm(mi.Divide(np.ScalarOf(int64(2)), true)),
		"pow0": mm(a.Pow(0)), "pow3": mm(a.Pow(3)), "powneg": mm(a.Pow(-2)),
		"parse_int": mm(Parse("1,2;3,4", 0)), "parse_complex": mm(Parse("1+2j 3;4 -2j", 0)),
		"block": mm(Block([]any{[]any{a.Array, a.Array}, []any{a.Array, a.Array}})),
	} {
		check(n, v.Array)
	}
	check("row", m(mi.IndexMatrix(np.AtIndex(0))))
	check("col", m(mi.IndexMatrix(np.Slice{}, np.AtIndex(1))))
	check("scalar", m(mi.IndexMatrix(np.AtIndex(1), np.AtIndex(1))))
	check("fancy", m(mi.IndexMatrix(two, m(np.FromSlice([]int64{2, 0}, 2)))))
	check("bool", m(mi.IndexMatrix(m(np.Greater(mi.Array, np.ScalarOf(int64(3)))))))
	check("rev", m(mi.IndexMatrix(np.Reverse(), np.Reverse())))
	for _, p := range []struct {
		label string
		axes  []int
	}{{"none", nil}, {"0", []int{0}}, {"1", []int{1}}, {"both", []int{0, 1}}} {
		for n, fn := range map[string]func([]int) (*np.Array, error){
			"sum":  func(s []int) (*np.Array, error) { return mi.Sum(s, 0) },
			"prod": func(s []int) (*np.Array, error) { return mi.Prod(s, 0) },
			"mean": func(s []int) (*np.Array, error) { return mi.Mean(s, 0) },
			"min":  mi.Min, "max": mi.Max, "ptp": mi.PTP, "all": mi.All, "any": mi.Any,
			"var": func(s []int) (*np.Array, error) {
				ddof := 0.
				if len(s) == 1 {
					ddof = 1
				}
				return mi.Var(s, 0, ddof)
			},
			"std": func(s []int) (*np.Array, error) {
				ddof := 0.
				if len(s) == 1 {
					ddof = 1
				}
				return mi.Std(s, 0, ddof)
			},
		} {
			if n == "ptp" && p.label == "both" {
				_, e := fn(p.axes)
				refcheck.Error(t, "numeric_class_matrix_ptp_both", "ValueError", e, np.ErrValue)
				continue
			}
			check(n+"_"+p.label, m(fn(p.axes)))
		}
	}
	for _, p := range []struct {
		n  string
		ax *int
	}{{"none", nil}, {"0", np.Int(0)}, {"1", np.Int(1)}} {
		check("argmin_"+p.n, m(mi.ArgMin(p.ax)))
		check("argmax_"+p.n, m(mi.ArgMax(p.ax)))
	}
	src := m(m(np.ArangeInt64(0, 6, 1)).Reshape(2, 3))
	v := mm(New(src, 0, false))
	if e := v.A().SetValue(int64(99), 0, 1); e != nil {
		t.Fatal(e)
	}
	check("shared", src)
	cp := mm(New(src, 0, true))
	if e := cp.A().SetValue(int64(77), 0, 1); e != nil {
		t.Fatal(e)
	}
	check("copy_original", src)
	if e := mi.AssignMatrix([]np.Index{np.Slice{}, np.AtIndex(1)}, m(np.FromSlice([]int64{90, 91}, 2, 1))); e != nil {
		t.Fatal(e)
	}
	check("assign", mi.Array)
	_, e = New(m(np.Zeros(np.Float64, 2, 2, 2)), 0, false)
	refcheck.Error(t, "numeric_class_matrix_bad_rank", "ValueError", e, np.ErrShape)
	_, e = mi.Mul(three, false)
	refcheck.Error(t, "numeric_class_matrix_bad_multiply", "ValueError", e, np.ErrShape)
	_, e = Parse("1 2;3", 0)
	refcheck.Error(t, "numeric_class_matrix_bad_parse", "ValueError", e, np.ErrShape)
	_, e = mm(New(m(np.FromSlice([]float64{1, 2, 2, 4}, 2, 2)), 0, false)).I()
	refcheck.Error(t, "numeric_class_matrix_singular", "LinAlgError", e, linalg.ErrSingular)
}
