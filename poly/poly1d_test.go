package poly_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/poly"
)

func TestPoly1DClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	pm := refcheck.RequireValue[*poly.Poly1D](t)
	value := refcheck.RequireValue[any](t)
	tm := refcheck.RequireValue[string](t)
	ok := func(e error) {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
	}
	a := m(np.FromSlice([]int16{0, 1, -2, 3}, 4))
	p := pm(poly.NewPoly1D(a, false, "z"))
	refcheck.Check(t, "class_poly_coeff", p.Coeffs())
	ints := []int64{int64(p.Order()), int64(p.Len())}
	for _, i := range []int{0, 2, -1, 8} {
		ints = append(ints, int64(value(p.Coefficient(i)).(int16)))
	}
	refcheck.Check(t, "class_poly_meta", m(np.FromSlice(ints, len(ints))))
	refcheck.Text(t, "class_poly_variable", p.Variable())
	refcheck.Text(t, "class_poly_repr", tm(p.Repr()))
	refcheck.Text(t, "class_poly_str", tm(p.Format()))
	ok(a.SetValue(int16(-4), 2))
	refcheck.Check(t, "class_poly_shared", p.Coefficients())
	ok(p.SetCoefficient(3, int16(2)))
	refcheck.Check(t, "class_poly_expand", p.Coefficients())
	ok(p.SetCoefficient(3, int16(0)))
	refcheck.Check(t, "class_poly_leading_zero", p.Coefficients())
	refcheck.Error(t, "class_poly_bad_power", "ValueError", p.SetCoefficient(-1, 2), np.ErrValue)
	p = pm(poly.NewPoly1D(m(np.FromSlice([]float64{1, -2, 3}, 3)), false, ""))
	q := pm(poly.NewPoly1D(m(np.FromSlice([]float64{2, 1}, 2)), false, ""))
	two := m(np.FromSlice([]float64{2}))
	polys := map[string]*poly.Poly1D{
		"add": pm(p.Add(q)), "sub": pm(p.Sub(q)), "mul": pm(p.Mul(q)), "pow": pm(p.Pow(3)), "compose": pm(p.Compose(q)),
		"negative": pm(p.Negative()), "positive": p.Positive(), "scale": pm(p.Scale(two)), "scalar_div": pm(p.DivideScalar(two, false)),
		"deriv": pm(p.Deriv(2)), "integ": pm(p.Integ(2, []complex128{3, 4})),
	}
	for name, v := range polys {
		refcheck.Check(t, "class_poly_"+name, v.Coefficients())
	}
	quot, rem, e := p.Divide(q)
	ok(e)
	refcheck.Check(t, "class_poly_quotient", quot.Coeffs())
	refcheck.Check(t, "class_poly_remainder", rem.Coeffs())
	refcheck.Check(t, "class_poly_eval", m(p.Eval(m(np.FromSlice([]float64{-2, 0, 3}, 3)))))
	refcheck.Check(t, "class_poly_fromroots", pm(poly.NewPoly1D(m(np.FromSlice([]float64{1, -1}, 2)), true, "")).Coefficients())
	r := m(pm(poly.NewPoly1D(m(np.FromSlice([]float64{1, 0, -1}, 3)), false, "")).Roots())
	refcheck.Check(t, "class_poly_roots", m(np.Sort(r, nil)))
	bm := refcheck.RequireValue[bool](t)
	eq := []int64{0, 0}
	if bm(p.Equal(p.Alias(""))) {
		eq[0] = 1
	}
	if bm(p.Equal(q)) {
		eq[1] = 1
	}
	refcheck.Check(t, "class_poly_equal", m(np.FromSlice(eq, 2)))
}
