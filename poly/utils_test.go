package poly_test

import (
	"math"
	"strconv"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/poly"
)

func TestPolyutilsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	check := func(n string, a *np.Array) { t.Helper(); refcheck.Check(t, "polyutils_"+n, a) }
	ok := func(e error) {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
	}
	am := refcheck.RequireValue[[]*np.Array](t)
	a := m(np.FromSlice([]int16{1, 2, 0, 0}, 4))
	v := m(poly.TrimSequence(a))
	check("trimseq", v)
	ok(v.SetValue(int16(9), 0))
	check("trimseq_shared", a)
	check("trimseq_all", m(poly.TrimSequence(m(np.Zeros(np.Float32, 4)))))
	check("trimseq_empty", m(poly.TrimSequence(m(np.Zeros(np.Int64, 0)))))
	check("trimseq_complex", m(poly.TrimSequence(m(np.FromSlice([]complex64{1 + 2i, 0, 0}, 3)))))
	in := []*np.Array{m(np.FromSlice([]int16{2}, 1)), m(np.FromSlice([]float32{1, 2, 0}, 3)), m(np.FromSlice([]complex64{1 + 2i, 0}, 2))}
	for i, trim := range []bool{false, true} {
		values := am(poly.AsSeries(in, trim))
		for j, v := range values {
			check("as_series_"+strconv.Itoa(i)+"_"+strconv.Itoa(j), v)
		}
		ok(values[1].SetValue(complex128(44), 0))
		check("as_series_original_"+strconv.Itoa(i), in[1])
	}
	check("as_series_half", am(poly.AsSeries([]*np.Array{m(m(np.FromSlice([]float64{1, 2}, 2)).Astype(np.Float16))}, true))[0])
	check("as_series_scalar", am(poly.AsSeries([]*np.Array{np.ScalarOf(float32(2.5))}, true))[0])
	check("trimcoef_complex", m(poly.TrimCoefficients(m(np.FromSlice([]complex64{1 + 1i, .2i, 0}, 3)), .5)))
	check("trimcoef_integer", m(poly.TrimCoefficients(m(np.FromSlice([]int32{1, 2, 0, 0}, 4)), 0)))
	check("trimcoef_nan", m(poly.TrimCoefficients(m(np.FromSlice([]float64{1, 2, math.NaN()}, 3)), 0)))
	check("trimcoef_nan_tol", m(poly.TrimCoefficients(m(np.FromSlice([]float64{1, 2}, 2)), math.NaN())))
	check("trimcoef_all", m(poly.TrimCoefficients(m(np.FromSlice([]float32{-.25, .125}, 2)), .5)))
	for _, d := range []np.DType{np.Int32, np.Float16, np.Float32, np.Float64, np.Complex64, np.Complex128} {
		x, old, new := []complex128{-2, .5, 3}, []complex128{-2, 2}, []complex128{1, 5}
		if d.IsComplex() {
			x[0] += 1i
			x[1] += 2i
			x[2] -= 3i
			old[0] -= 1i
			old[1] += 1i
			new[0] += 2i
			new[1] -= 2i
		}
		typed := func(v []complex128) *np.Array { return m(m(np.FromSlice(v, len(v))).Astype(d)) }
		aa, bb, xx := typed(old), typed(new), typed(x)
		check("getdomain_"+d.String(), m(poly.GetDomain(xx)))
		off, scale, e := poly.MapParameters(aa, bb)
		ok(e)
		check("mapparms_"+d.String()+"_off", off)
		check("mapparms_"+d.String()+"_scale", scale)
		check("mapdomain_"+d.String(), m(poly.MapDomain(xx, aa, bb)))
	}
	check("getdomain_nan", m(poly.GetDomain(m(np.FromSlice([]float64{1, math.NaN(), 3}, 3)))))
	x := m(m(m(np.ArangeInt64(0, 6, 1)).Astype(np.Float32)).Reshape(2, 3))
	check("mapdomain_2d", m(poly.MapDomain(x, m(np.FromSlice([]float64{0, 5}, 2)), m(np.FromSlice([]float64{10, -2}, 2)))))
	original := np.GetPrintOptions()
	defer np.SetPrintOptions(original)
	opt := original
	opt.Precision, opt.FloatMode, opt.Sign, opt.NaN, opt.Inf = 5, "fixed", "+", "NaN", "Infinity"
	ok(np.SetPrintOptions(opt))
	for i, v := range []float64{0, math.Copysign(0, -1), 1.25, -2.5, 1e-8, 3e9, math.NaN(), math.Inf(1), math.Inf(-1)} {
		refcheck.Text(t, "polyutils_format_float_"+strconv.Itoa(i), poly.FormatFloat(v, true))
	}
	_, e := poly.TrimCoefficients(a, -1)
	refcheck.Error(t, "polyutils_bad_trim", "ValueError", e, np.ErrValue)
	_, e = poly.AsSeries([]*np.Array{m(np.Zeros(np.Float64, 0))}, true)
	refcheck.Error(t, "polyutils_bad_empty", "ValueError", e, np.ErrShape)
	_, e = poly.AsSeries([]*np.Array{m(np.Zeros(np.Float64, 2, 2))}, true)
	refcheck.Error(t, "polyutils_bad_rank", "ValueError", e, np.ErrShape)
	_, e = poly.AsSeries([]*np.Array{m(np.FromSlice([]bool{true, false}, 2))}, true)
	refcheck.Error(t, "polyutils_bad_bool", "ValueError", e, np.ErrValue)
}
