package np_test

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/extended"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
	"github.com/SamuelSupe/numpygo/ma"
	"github.com/SamuelSupe/numpygo/poly"
	"github.com/SamuelSupe/numpygo/random"
	"github.com/SamuelSupe/numpygo/rec"
	"github.com/SamuelSupe/numpygo/records"
	"github.com/SamuelSupe/numpygo/scalar"
)

func TestAuditRecordOrdering(t *testing.T) {
	m := refcheck.Require(t)
	schema := refcheck.RequireValue[*records.Schema](t)
	arr := refcheck.RequireValue[*records.Array](t)
	d := schema(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Numeric(np.Int32, binary.BigEndian)}, {Name: "y", Type: records.Numeric(np.Float64, nil)}}, true, 0))
	a := arr(records.FromRows(d, [][]any{{int32(2), 1.}, {int32(1), 3.}, {int32(2), 1.}, {int32(-2), 5.}, {int32(0), math.Copysign(0, -1)}, {int32(0), 0.}}, 6)).NDArray()
	check := func(name string, v *np.Array) {
		t.Helper()
		for _, f := range []string{"x", "y"} {
			refcheck.Check(t, "audit_"+name+"_"+f, m(m(v.Field(f)).Astype(map[string]np.DType{"x": np.Int32, "y": np.Float64}[f])))
		}
	}
	for _, order := range []struct {
		name   string
		fields []string
	}{{"default", nil}, {"y", []string{"y"}}} {
		check("record_sort_"+order.name, m(np.SortWithOptions(a, np.Int(0), np.SortOptions{Order: order.fields})))
		refcheck.Check(t, "audit_record_argsort_"+order.name, m(np.ArgsortWithOptions(a, np.Int(0), np.SortOptions{Order: order.fields})))
	}
	u := refcheck.RequireValue[np.UniqueResult](t)(np.Unique(a))
	check("record_unique", u.Values)
	for name, v := range map[string]*np.Array{"indices": u.Indices, "inverse": u.Inverse, "counts": u.Counts} {
		refcheck.Check(t, "audit_record_unique_"+name, v)
	}
	b := m(np.Take(a, m(np.WrapSlice([]int64{0, 3}, 2)), np.Int(0)))
	refcheck.Check(t, "audit_record_isin", m(np.Isin(a, b, false)))
	check("record_intersect", m(np.Intersect1D(a, b)))
	refcheck.Check(t, "audit_record_search", m(np.SearchSorted(u.Values, a, false)))
	_, e := np.SortWithOptions(a, np.Int(0), np.SortOptions{Order: []string{"missing"}})
	refcheck.Error(t, "audit_record_sort_bad_order", "ValueError", e, np.ErrValue)
	sub := schema(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Numeric(np.Int16, nil)}, {Name: "y", Type: records.Numeric(np.Int16, nil)}}, false, 0))
	nested := schema(records.NewSchema([]records.FieldSpec{{Name: "p", Type: records.Subarray(records.Struct(sub), 2)}, {Name: "n", Type: records.Numeric(np.Int32, nil)}}, false, 0))
	v := arr(records.Zeros(nested, 3)).NDArray()
	for n, x := range map[string][]int16{"x": {1, 0, 1, -1, 1, 4}, "y": {0, 0, 1, 0, -1, 0}} {
		if e := m(v.Field("p", n)).Assign(nil, m(np.WrapSlice(x, 3, 2))); e != nil {
			t.Fatal(e)
		}
	}
	if e := m(v.Field("n")).Assign(nil, m(np.WrapSlice([]int32{10, 20, 30}, 3))); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_record_nested_sort", m(m(np.Sort(v, np.Int(0))).Field("n")))
}

func TestAuditAdditionalOptions(t *testing.T) {
	m := refcheck.Require(t)
	p := refcheck.RequireValue[*poly.Series](t)
	d := [2]complex128{1i, 3i}
	s := p(poly.NewSeries(poly.Power, m(np.WrapSlice([]complex128{1, 2, 3}, 3)), poly.SeriesOptions{ComplexDomain: &d}))
	refcheck.Check(t, "audit_poly_complex_domain", m(np.WrapSlice(d[:], 2)))
	o, k := s.MapParametersComplex()
	refcheck.Check(t, "audit_poly_complex_map", m(np.WrapSlice([]complex128{o, k}, 2)))
	refcheck.Check(t, "audit_poly_complex_eval", m(s.Eval(m(np.WrapSlice([]complex128{1i, 2i, 3i}, 3)))))
	refcheck.Check(t, "audit_poly_complex_deriv", m(p(s.Derivative(1)).Coefficients()))
	refcheck.Check(t, "audit_poly_complex_integ", m(p(s.IntegrateComplex(1, nil, nil)).Coefficients()))
	refcheck.Check(t, "audit_poly_complex_convert", m(p(s.Convert(poly.Power, poly.SeriesOptions{})).Coefficients()))
	refcheck.Check(t, "audit_poly_complex_roots", m(np.Sort(m(s.Roots()), nil)))
	x, y, e := s.Linspace(3, nil)
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_poly_complex_linspace_x", x)
	refcheck.Check(t, "audit_poly_complex_linspace_y", y)
	s = p(poly.NewSeries(poly.Power, m(np.WrapSlice([]float64{1, 2, 3}, 3)), poly.SeriesOptions{ComplexDomain: &d}))
	_, e = s.Derivative(1)
	refcheck.Error(t, "audit_poly_complex_real_deriv", "UFuncTypeError", e, np.ErrDType)
	_, e = s.Integrate(1, nil, nil)
	refcheck.Error(t, "audit_poly_complex_real_integ", "UFuncTypeError", e, np.ErrDType)
	q := m(m(np.WrapSlice([]int64{1, 2, 4}, 3)).CastTemporal(np.DateTime64, "D"))
	for name, spacing := range map[string]*np.Array{"uniform": np.Scalar(1), "nonuniform": m(np.WrapSlice([]float64{0, 1, 3}, 3))} {
		v := refcheck.RequireValue[[]*np.Array](t)(np.GradientAxes(q, nil, []*np.Array{spacing}, 2))
		refcheck.Check(t, "audit_gradient_temporal_"+name, m(v[0].Astype(np.Int64)))
	}
	out := m(np.WrapSlice([]int64{10, 20, 30, 40}, 4))
	mask := m(np.WrapSlice([]bool{true, false, true, false}, 4))
	it := refcheck.RequireValue[*np.NDIterator](t)(np.NewNDIterator([]*np.Array{out, mask}, np.NDIterOptions{Buffered: true, Access: []string{"readwrite", "readonly"}, DTypes: []np.DType{np.Float64, 0}, Casting: "unsafe", MaskOperand: np.Int(1), WriteMasked: []bool{true, false}, BufferSize: 2}))
	for row, e := range it.Iter() {
		if e != nil {
			t.Fatal(e)
		}
		if e = row[0].SetValue(int64(99)); e != nil {
			t.Fatal(e)
		}
	}
	if e = it.Close(); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_nditer_writemasked", out)
	_, e = np.FromString("1+2j", np.Complex256, " ", -1)
	refcheck.Error(t, "audit_extended_complex_fromstring", "ValueError", e, np.ErrValue)
	const text = "1.0000000000000000001+0.0000000000000000001j"
	refcheck.CheckExtended(t, "audit_extended_complex_loadtxt", m(np.LoadText(strings.NewReader(text), np.TextOptions{DType: np.Complex256})), np.Complex256, 0)
	z := refcheck.RequireValue[np.ExtendedComplex](t)(np.ParseExtendedComplex(text))
	refcheck.CheckExtended(t, "audit_extended_complex_explicit", np.ScalarOf(z), np.Complex256, 0)
	rs := random.NewRandomState(17)
	a := m(np.WrapSlice([]int64{10, 20, 30}, 3))
	refcheck.Check(t, "audit_legacy_choice_array", m(rs.ChoiceArray(a, 0, random.ChoiceOptions{}, 4)))
	_, e = rs.ChoiceArray(m(np.WrapSlice([]int64{10, 20, 30, 40}, 2, 2)), 0, random.ChoiceOptions{}, 2)
	refcheck.Error(t, "audit_legacy_choice_bad_dim", "ValueError", e, np.ErrShape)
	refcheck.Check(t, "audit_legacy_choice_next", m(rs.Randint(0, 100, np.Int64, 4)))
}

func TestAuditAxisAndReduction(t *testing.T) {
	m := refcheck.Require(t)
	i64 := func(x []int64, shape ...int) *np.Array { return m(np.WrapSlice(x, shape...)) }
	z := m(i64([]int64{1, 2, 1, 2, 3, 4, 0, 1}, 4, 2).Slice(np.Slice{}, np.Reverse()))
	for ax := 0; ax < 2; ax++ {
		u := refcheck.RequireValue[np.UniqueResult](t)(np.Unique(z, np.UniqueOptions{Axis: np.Int(ax)}))
		p := "audit_unique_axis" + strconv.Itoa(ax)
		refcheck.Check(t, p, u.Values)
		refcheck.Check(t, p+"_idx", u.Indices)
		refcheck.Check(t, p+"_inverse", u.Inverse)
		refcheck.Check(t, p+"_counts", u.Counts)
	}
	u := refcheck.RequireValue[np.UniqueResult](t)(np.Unique(m(np.Zeros(np.Int64, 3, 0)), np.UniqueOptions{Axis: np.Int(0)}))
	refcheck.Check(t, "audit_unique_empty_width", u.Values)
	refcheck.Check(t, "audit_search_sorter", m(np.SearchSortedWithSorter(i64([]int64{30, 10, 20}, 3), i64([]int64{15, 25}, 2), false, i64([]int64{1, 2, 0}, 3))))
	_, e := np.SearchSortedWithSorter(i64([]int64{1, 2}, 2), i64([]int64{1}, 1), false, i64([]int64{0}, 1))
	refcheck.Error(t, "audit_search_bad_sorter", "ValueError", e, np.ErrShape)
	add := refcheck.RequireValue[*np.UFunc](t)(np.LookupUFunc("add"))
	a := i64([]int64{1, 2, 3, 4}, 2, 2)
	refcheck.Check(t, "audit_reduce_default", m(add.ReduceWithOptions(a, np.ReduceOptions{})))
	refcheck.Check(t, "audit_reduce_default", m(add.Reduce(a)))
	refcheck.Check(t, "audit_reduce_all", m(add.ReduceWithOptions(a, np.ReduceOptions{AllAxes: true})))
	d := m(a.CastTemporal(np.TimeDelta64, "D"))
	refcheck.Check(t, "audit_reduce_temporal", m(m(add.ReduceWithOptions(d, np.ReduceOptions{})).Astype(np.Int64)))
	td := refcheck.RequireValue[np.TimeDelta](t)(np.NewTimeDelta(5, "D"))
	mask := m(np.WrapSlice([]bool{true, false, true, true}, 2, 2))
	refcheck.Check(t, "audit_reduce_temporal_where", m(m(add.ReduceWithOptions(d, np.ReduceOptions{Axes: []int{1}, Where: mask, Initial: td})).Astype(np.Int64)))
}

func TestAuditExtendedAndGradient(t *testing.T) {
	m := refcheck.Require(t)
	ex := func(name string, a *np.Array, ulps int64) {
		t.Helper()
		refcheck.CheckExtended(t, "audit_"+name, a, np.Float128, ulps)
	}
	ex("extended_text", m(np.FromString("1.0000000000000000001 1e-4000 -1e4000", np.Float128, " ", -1)), 0)
	a := refcheck.ExtendedInput(t, "audit_extended_grad_input", np.Float128)
	ex("extended_grad", m(np.Gradient(a, 0, 1, 1)), 0)
	for _, ord := range []string{"1", "-1", "0", "inf", "-inf", "3"} {
		ex("extended_norm_"+ord, m(linalg.Norm(a, linalg.NormOptions{Ord: ord, Axes: []int{0}})), 8)
	}
	mat := m(m(np.WrapSlice([]int64{-1, 2, 4, 3, -5, 1}, 2, 3)).Astype(np.Float128))
	for _, ord := range []string{"1", "-1", "inf", "-inf"} {
		ex("extended_matrix_norm_"+ord, m(linalg.Norm(mat, linalg.NormOptions{Ord: ord})), 0)
	}
	s, stop, step := extended.MustParse("1"), extended.MustParse("1.0000000000000000004"), extended.MustParse("0.0000000000000000001")
	v, h, e := np.LinspaceArray(np.ScalarOf(s), np.ScalarOf(stop), np.SpaceOptions{Num: 3})
	if e != nil {
		t.Fatal(e)
	}
	ex("extended_linspace", v, 0)
	ex("extended_linspace_step", h, 0)
	ex("extended_arange", m(np.ArangeTyped(s, stop, step, np.Float128)), 0)
	refcheck.Check(t, "audit_arange_uint64", m(np.ArangeTyped(uint64(math.MaxUint64-6), uint64(math.MaxUint64), uint64(2), np.Uint64)))
	g := m(np.WrapSlice([]float64{0, 1, 9}, 3))
	x := m(np.WrapSlice([]float64{0, 1, 3}, 3))
	gr := refcheck.RequireValue[[]*np.Array](t)(np.GradientAxes(g, nil, []*np.Array{x}, 2))
	refcheck.Check(t, "audit_gradient_nonuniform", gr[0])
	q := m(np.Arange(0, 12, 1, np.Float32))
	q = m(q.Reshape(3, 4))
	q = m(q.Slice(np.Slice{}, np.Reverse()))
	q = m(np.Square(q))
	gr = refcheck.RequireValue[[]*np.Array](t)(np.GradientAxes(q, nil, []*np.Array{x, np.Scalar(2)}, 2))
	for i, v := range gr {
		refcheck.Check(t, "audit_gradient_axes"+strconv.Itoa(i), v)
	}
	d := m(m(np.WrapSlice([]int64{0, 4, 10}, 3)).CastTemporal(np.DateTime64, "D"))
	v = m(np.Gradient(d, 0, 1, 1))
	if v.DType() != np.TimeDelta64 {
		t.Fatal(v.DType())
	}
	refcheck.Check(t, "audit_gradient_temporal", m(v.Astype(np.Int64)))
}

func TestAuditStatisticsOptions(t *testing.T) {
	m := refcheck.Require(t)
	a := m(np.Arange(0, 12, 1, np.Float64))
	a = m(a.Reshape(3, 4))
	a = m(a.Slice(np.Slice{}, np.Reverse()))
	mask := m(np.WrapSlice([]bool{true, false, true, true, false, true, false, true, true, true, true, true}, 3, 4))
	o := np.VarOptions{Axes: []int{1}, Where: mask}
	refcheck.Check(t, "audit_variance_where", m(np.Var(a, o)))
	center := m(np.WrapSlice([]float64{4. / 3, 5, 9.5}, 3, 1))
	corr := 1.
	o.Mean = center
	o.Correction = &corr
	refcheck.Check(t, "audit_variance_mean", m(np.Var(a, o)))
	dst := m(np.Zeros(np.Float64, 6))
	dst = m(dst.Slice(np.Strided(0, 6, 2)))
	o.Out = dst
	if got := m(np.Std(a, o)); got != dst {
		t.Fatal("out was not reused")
	}
	refcheck.Check(t, "audit_std_where", dst)
	refcheck.Check(t, "audit_variance_supplied", m(np.Var(m(np.WrapSlice([]float64{1, 2, 3}, 3)), np.VarOptions{Mean: m(np.WrapSlice([]float64{0}, 1))})))
	if e := a.SetValue(math.NaN(), 0, 0); e != nil {
		t.Fatal(e)
	}
	o = np.VarOptions{Axes: []int{1}, Where: mask}
	refcheck.Check(t, "audit_nanvar_where", m(np.NanVar(a, o)))
	refcheck.Check(t, "audit_nanstd_where", m(np.NanStd(a, o)))
	x := m(np.WrapSlice([]float64{1, 2, 3, 4, 7, 6}, 2, 3))
	w := m(np.WrapSlice([]float64{1, 2, 3}, 3))
	refcheck.Check(t, "audit_cov_weighted", m(np.CovWithOptions(x, nil, np.CovOptions{AWeights: w})))
	f := m(np.WrapSlice([]int64{1, 3, 2}, 3))
	xt := m(x.Transpose())
	refcheck.Check(t, "audit_cov_both_weights", m(np.CovWithOptions(xt, nil, np.CovOptions{ColumnVariables: true, FWeights: f, AWeights: w, DDOF: np.Int(1)})))
	x1 := m(x.Slice(np.AtIndex(0)))
	x2 := m(x.Slice(np.AtIndex(1)))
	refcheck.Check(t, "audit_cov_cross", m(np.CovWithOptions(x1, x2, np.CovOptions{Bias: true})))
	refcheck.Check(t, "audit_corr_cross", m(np.CorrcoefWithOptions(x1, x2, np.CovOptions{})))
	v := m(np.Arange(0, 24, 1, np.Float64))
	v = m(v.Reshape(2, 3, 4))
	w = m(np.Arange(1, 9, 1, np.Float64))
	w = m(w.Reshape(4, 2))
	avg, sw, e := np.AverageWithWeights(v, w, np.SumOptions{Axes: []int{2, 0}, KeepDims: true})
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_average_multi", avg)
	refcheck.Check(t, "audit_average_multi_weights", sw)
}

func TestAuditContractionSpaceAndIndex(t *testing.T) {
	m := refcheck.Require(t)
	a := m(np.WrapSlice([]int16{500, 500}, 2))
	refcheck.Check(t, "audit_einsum_dtype", m(np.EinsumWithOptions("i,i->", np.EinsumOptions{DType: np.Int64}, a, a)))
	x := m(m(np.Arange(0, 6, 1, np.Float64)).Reshape(2, 3))
	y := m(m(np.Arange(0, 12, 1, np.Float64)).Reshape(3, 4))
	v := m(np.EinsumWithOptions("ij,jk->ik", np.EinsumOptions{Order: "F"}, x, y))
	refcheck.Check(t, "audit_einsum_f", v)
	if !v.IsFContiguous() {
		t.Fatal("expected F order")
	}
	refcheck.Check(t, "audit_einsum_opt", m(np.EinsumWithOptions("ij,jk->ik", np.EinsumOptions{Optimize: "greedy"}, x, y)))
	out := m(np.Zeros(np.Float64, 2, 8))
	out = m(out.Slice(np.Slice{}, np.Strided(0, 8, 2)))
	if m(np.EinsumWithOptions("ij,jk->ik", np.EinsumOptions{Out: out}, x, y)) != out {
		t.Fatal("out not reused")
	}
	refcheck.Check(t, "audit_einsum_out", out)
	_, e := np.EinsumWithOptions("i,i->", np.EinsumOptions{DType: np.Int8}, a, a)
	refcheck.Error(t, "audit_einsum_cast", "TypeError", e, np.ErrDType)
	start := m(np.WrapSlice([]float32{0, 10}, 2))
	stop := m(np.WrapSlice([]float32{2, 20}, 2))
	v, h, e := np.LinspaceArray(start, stop, np.SpaceOptions{Num: 3, Axis: 1})
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_linspace_array", v)
	refcheck.Check(t, "audit_linspace_step", h)
	if !v.IsFContiguous() {
		t.Fatal("sample axis layout")
	}
	no := false
	v, h, e = np.LinspaceArray(start, stop, np.SpaceOptions{Num: 4, Endpoint: &no, DType: np.Int64})
	if e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_linspace_int", v)
	refcheck.Check(t, "audit_linspace_int_step", h)
	f64 := func(v ...float64) *np.Array { return m(np.WrapSlice(v, len(v))) }
	refcheck.Check(t, "audit_logspace_array", m(np.LogspaceArray(f64(0, 1), f64(2, 3), f64(2, 10), np.SpaceOptions{Num: 3, Axis: 1})))
	refcheck.Check(t, "audit_geomspace_array", m(np.GeomspaceArray(f64(1, -1), f64(100, -1000), np.SpaceOptions{Num: 4, Axis: 1})))
	refcheck.Check(t, "audit_geomspace_complex", m(np.GeomspaceArray(m(np.WrapSlice([]complex128{1 + 1i, -1 + 1i}, 2)), m(np.WrapSlice([]complex128{-1 - .5i, 1 - 2i}, 2)), np.SpaceOptions{Num: 5})))
	i64 := func(v ...int64) *np.Array { return m(np.WrapSlice(v, len(v))) }
	for _, mode := range []string{"wrap", "clip"} {
		refcheck.Check(t, "audit_take_"+mode, m(np.TakeWithOptions(i64(10, 20, 30), i64(-4, -1, 4), nil, np.TakeOptions{Mode: mode})))
		refcheck.Check(t, "audit_choose_"+mode, m(np.ChooseWithOptions(i64(-2, 0, 4), []*np.Array{i64(10, 20, 30), i64(40, 50, 60)}, np.TakeOptions{Mode: mode})))
	}
	refcheck.Check(t, "audit_interp_period", m(np.InterpPeriod(f64(-10, 370), f64(0, 180), f64(0, 1), 360)))
}

func TestAuditPadTextAndIterator(t *testing.T) {
	m := refcheck.Require(t)
	a := m(np.WrapSlice([]int64{1, 2, 3, 4, 5, 6}, 2, 3))
	for _, mode := range []string{"linear_ramp", "mean", "median", "minimum", "maximum"} {
		opt := np.PadOptions{Mode: mode}
		if mode == "linear_ramp" {
			opt.EndValues = [][2]any{{int64(0), int64(4)}, {int64(2), int64(1)}}
		} else {
			opt.StatLength = [][2]int{{1, 2}, {2, 1}}
		}
		refcheck.Check(t, "audit_pad_"+mode, m(np.PadWithOptions(a, [][2]int{{2, 1}, {1, 2}}, opt)))
	}
	for _, mode := range []string{"reflect", "symmetric"} {
		refcheck.Check(t, "audit_pad_odd_"+mode, m(np.PadWithOptions(m(np.WrapSlice([]int64{1, 2, 4}, 3)), [][2]int{{5, 4}}, np.PadOptions{Mode: mode, ReflectType: "odd"})))
	}
	refcheck.Check(t, "audit_pad_constants", m(np.PadWithOptions(a, [][2]int{{1, 2}, {2, 1}}, np.PadOptions{ConstantValues: [][2]any{{int64(3), int64(4)}, {int64(5), int64(6)}}})))
	refcheck.Check(t, "audit_pad_callback", m(np.PadWithOptions(a, [][2]int{{1, 1}, {2, 1}}, np.PadOptions{Callback: func(v *np.Array, pad [2]int, axis int) error {
		if e := m(v.Slice(np.Range(0, pad[0]))).Fill(int64(10 + axis)); e != nil {
			return e
		}
		return m(v.Slice(np.Range(v.Size()-pad[1], v.Size()))).Fill(int64(20 + axis))
	}})))
	refcheck.Check(t, "audit_text_quotes", m(np.LoadText(strings.NewReader("1,\"2\",3\n4,\"5\",6 #comment"), np.TextOptions{Delimiter: ",", QuoteChar: '"'})))
	refcheck.Check(t, "audit_text_converter", m(np.LoadText(strings.NewReader("1,0x10\n2,0x20"), np.TextOptions{Delimiter: ",", Converters: map[int]func(string) (any, error){1: func(s string) (any, error) { return strconv.ParseInt(s, 0, 64) }}})))
	s := refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Numeric(np.Int32, nil)}, {Name: "y", Type: records.Numeric(np.Float64, nil)}}, false, 0))
	d := refcheck.RequireValue[*records.Array](t)(records.LoadText(strings.NewReader("1,2.5\n3,4.5"), s, np.TextOptions{Delimiter: ","}))
	for _, name := range []string{"x", "y"} {
		refcheck.Check(t, "audit_text_structured_"+name, m(d.NDArray().Field(name)))
	}
	d = refcheck.RequireValue[*records.Array](t)(records.GenFromText(strings.NewReader("x,y\n1,2.5\n3,4.5"), records.NamedTextOptions{Text: np.TextOptions{Delimiter: ","}, Header: true}))
	for _, name := range []string{"x", "y"} {
		refcheck.Check(t, "audit_text_inferred_"+name, m(d.NDArray().Field(name)))
	}
	iter := refcheck.RequireValue[*np.NDIterator](t)
	it := iter(np.NewNDIterator([]*np.Array{m(np.WrapSlice([]int64{0, 1}, 2)), m(np.WrapSlice([]int64{0, 1, 2}, 3))}, np.NDIterOptions{OpAxes: [][]int{{0, -1}, {-1, 0}}, MultiIndex: true, Order: "C"}))
	shape := make([]int64, len(it.Shape()))
	for i, n := range it.Shape() {
		shape[i] = int64(n)
	}
	refcheck.Check(t, "audit_nditer_axes_shape", m(np.WrapSlice(shape, len(shape))))
	val := refcheck.RequireValue[any](t)
	var pairs []int64
	for row, e := range it.Iter() {
		if e != nil {
			t.Fatal(e)
		}
		for _, v := range row {
			pairs = append(pairs, val(v.AtValue()).(int64))
		}
	}
	if e := it.Close(); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_nditer_axes_values", m(np.WrapSlice(pairs, 6, 2)))
	it = iter(np.NewNDIterator([]*np.Array{m(np.WrapSlice([]int16{0, 1, 2}, 3)), m(np.WrapSlice([]float32{0, 1, 2}, 3))}, np.NDIterOptions{CommonDType: true, Buffered: true, Order: "C"}))
	var floats []float32
	for row, e := range it.Iter() {
		if e != nil {
			t.Fatal(e)
		}
		for _, v := range row {
			floats = append(floats, val(v.AtValue()).(float32))
		}
	}
	if e := it.Close(); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_nditer_common", m(np.WrapSlice(floats, 3, 2)))
	c := np.NewErrorContext()
	_, e := c.SetErr(np.ErrorOptions{Divide: np.Raise})
	if e != nil {
		t.Fatal(e)
	}
	x := m(np.WrapSlice([]float64{1, 2}, 2))
	y := m(np.WrapSlice([]float64{0, 1}, 2))
	out := m(np.WrapSlice([]float64{9, 9}, 2))
	refcheck.Check(t, "audit_fp_where", m(c.CallWithOptions("divide", np.UFuncOptions{Where: m(np.WrapSlice([]bool{false, true}, 2)), Out: out}, x, y)))
	_, e = c.CallWithOptions("divide", np.UFuncOptions{Where: m(np.WrapSlice([]bool{true, true}, 2))}, x, y)
	refcheck.Error(t, "audit_fp_raise_option", "FloatingPointError", e, np.ErrFloatingPoint)
}

func TestAuditScalarAndMRecords(t *testing.T) {
	m := refcheck.Require(t)
	s := scalar.Of(3.25)
	for name, f := range map[string]func() (*np.Array, error){
		"all": func() (*np.Array, error) { return s.All() }, "any": func() (*np.Array, error) { return s.Any() }, "argmax": func() (*np.Array, error) { return s.ArgMax() }, "argmin": func() (*np.Array, error) { return s.ArgMin() }, "argsort": s.Argsort,
		"cumprod": func() (*np.Array, error) { return s.Cumprod(nil, 0) }, "cumsum": func() (*np.Array, error) { return s.Cumsum(nil, 0) }, "max": func() (*np.Array, error) { return s.Max() }, "min": func() (*np.Array, error) { return s.Min() },
		"mean": func() (*np.Array, error) { return s.Mean() }, "prod": func() (*np.Array, error) { return s.Prod() }, "std": func() (*np.Array, error) { return s.Std() }, "sum": func() (*np.Array, error) { return s.Sum() }, "var": func() (*np.Array, error) { return s.Var() },
		"choose": func() (*np.Array, error) {
			return scalar.Of(int64(2)).Choose([]*np.Array{np.ScalarOf(int64(10)), np.ScalarOf(int64(20)), np.ScalarOf(int64(30))}, np.TakeOptions{})
		},
		"clip": func() (*np.Array, error) { return s.Clip(np.Scalar(0), np.Scalar(3)) }, "compress": func() (*np.Array, error) { return s.Compress(m(np.WrapSlice([]bool{true}, 1)), nil) },
		"repeat": func() (*np.Array, error) { return s.Repeat(3, nil) }, "round": func() (*np.Array, error) { return s.Round(1) }, "take": func() (*np.Array, error) { return s.Take(m(np.WrapSlice([]int64{0, 0}, 2)), nil, np.TakeOptions{}) },
	} {
		t.Run(name, func(t *testing.T) { refcheck.Check(t, "audit_scalar_"+name, m(f())) })
	}
	dev := refcheck.RequireValue[scalar.Value](t)(s.ToDevice("cpu"))
	refcheck.Check(t, "audit_scalar_to_device", m(dev.Array()))
	v := refcheck.RequireValue[any](t)(s.ToList())
	refcheck.Check(t, "audit_scalar_tolist", np.ScalarOf(v.(float64)))
	refcheck.Check(t, "audit_scalar_base", np.ScalarOf(s.Base() == nil))
	b := refcheck.RequireValue[[]byte](t)(s.Data())
	refcheck.Bytes(t, "audit_scalar_data", b)
	f := s.Flags()
	refcheck.Check(t, "audit_scalar_flags", m(np.WrapSlice([]bool{f.CContiguous, f.FContiguous, f.OwnData, f.Writable, f.Aligned, f.Writeback}, 6)))
	fi := refcheck.RequireValue[*np.FlatIterator](t)(s.Flat())
	refcheck.Check(t, "audit_scalar_flat", m(fi.Get(np.Slice{})))
	var buf bytes.Buffer
	if e := s.Dump(&buf); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_scalar_dump", m(np.LoadNPY(&buf)))
	refcheck.Check(t, "audit_scalar_dumps", m(np.LoadNPY(bytes.NewReader(refcheck.RequireValue[[]byte](t)(s.Dumps())))))
	buf.Reset()
	if e := s.ToFile(&buf); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "audit_scalar_tofile", buf.Bytes())
	for name, fn := range map[string]func() error{"fill": func() error { return s.Fill(7.) }, "put": func() error { return s.Put(m(np.WrapSlice([]int64{0}, 1)), np.Scalar(9), "raise") }, "resize": func() error { return s.Resize(2) }, "setflags": func() error { yes := true; return s.SetFlags(&yes, nil, nil) }} {
		if e := fn(); e != nil {
			t.Fatal(e)
		}
		refcheck.Check(t, "audit_scalar_"+name, m(s.Array()))
	}
	_, e := s.Diagonal(0, 0, 1)
	refcheck.Error(t, "audit_scalar_diagonal", "ValueError", e, np.ErrShape)
	_, e = s.Nonzero()
	refcheck.Error(t, "audit_scalar_nonzero", "ValueError", e, np.ErrShape)
	_, e = s.SearchSorted(np.Scalar(1), false)
	refcheck.Error(t, "audit_scalar_searchsorted", "ValueError", e, np.ErrShape)
	refcheck.Error(t, "audit_scalar_setfield", "TypeError", s.SetField(np.Scalar(1), np.Float64, 0), np.ErrDType)
	refcheck.Error(t, "audit_scalar_sort", "AxisError", s.Sort(-1), np.ErrAxis)
	_, e = s.SwapAxes(0, 0)
	refcheck.Error(t, "audit_scalar_swapaxes", "AxisError", e, np.ErrAxis)
	_, e = s.Trace(0, 0, 1, 0)
	refcheck.Error(t, "audit_scalar_trace", "ValueError", e, np.ErrShape)
	o := np.VoidScalar([]byte{1, 2, 3, 4})
	field := refcheck.RequireValue[any](t)(o.GetField(np.Uint8, 1, nil))
	refcheck.Check(t, "audit_void_getfield", np.ScalarOf(field.(uint8)))
	if e := o.SetField(uint8(5), np.Uint8, 1, nil); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "audit_void_setfield", o)
	refcheck.Text(t, "audit_void_dtype", o.DType().String())
	refcheck.Check(t, "audit_void_base", np.ScalarOf(o.Base() == nil))
	of := o.Flags()
	refcheck.Check(t, "audit_void_flags", m(np.WrapSlice([]bool{of.CContiguous, of.FContiguous, of.OwnData, of.Writable, of.Aligned, of.Writeback}, 6)))
	schema := refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Numeric(np.Int32, nil)}, {Name: "y", Type: records.Numeric(np.Float64, nil)}}, false, 0))
	r := refcheck.RequireValue[*records.Array](t)(records.FromRows(schema, [][]any{{int32(4), 1.5}}, 1))
	record := refcheck.RequireValue[*records.Record](t)(r.At(0))
	field = refcheck.RequireValue[any](t)(record.GetField(np.Float64, 4, nil))
	refcheck.Check(t, "audit_void_record_getfield", np.ScalarOf(field.(float64)))
	if e := record.SetField(int32(9), np.Int32, 0, nil); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "audit_void_record_writeback", m(r.NDArray().Field("x")))
	refcheck.Check(t, "audit_void_record_base", np.ScalarOf(record.Base() == r.NDArray()))
	maarr := refcheck.RequireValue[*ma.Array](t)
	a := maarr(ma.New(m(np.WrapSlice([]int64{1, 2}, 2)), m(np.WrapSlice([]bool{false, true}, 2))))
	c := maarr(ma.New(m(np.WrapSlice([]float64{3, 4}, 2)), m(np.WrapSlice([]bool{true, true}, 2))))
	mr := refcheck.RequireValue[*ma.StructuredArray](t)
	q := mr(ma.FromMaskedArrays([]*ma.Array{a, c}, rec.Options{Names: []string{"a", "b"}}))
	for _, name := range []string{"a", "b"} {
		refcheck.Check(t, "audit_mrecords_data_"+name, m(q.Data.NDArray().Field(name)))
		refcheck.Check(t, "audit_mrecords_mask_"+name, m(q.Mask.NDArray().Field(name)))
	}
	refcheck.Check(t, "audit_mrecords_recordmask", m(q.RecordMask()))
	js := refcheck.RequireValue[[]byte](t)(json.Marshal(refcheck.RequireValue[any](t)(q.ToList())))
	refcheck.Text(t, "audit_mrecords_list", string(js))
	refcheck.Check(t, "audit_mrecords_view", m(mr(q.View()).Mask.NDArray().Field("a")))
	refcheck.Check(t, "audit_mrecords_copy", m(mr(q.Copy()).Mask.NDArray().Field("b")))
	q.HardenMask()
	refcheck.Check(t, "audit_mrecords_harden", np.ScalarOf(q.HardMask()))
	q.SoftenMask()
	refcheck.Check(t, "audit_mrecords_soften", np.ScalarOf(q.HardMask()))
	add := maarr(ma.New(m(np.WrapSlice([]int64{5, 6}, 2)), m(np.WrapSlice([]bool{true, false}, 2))))
	q = mr(ma.AddField(q, "c", add))
	refcheck.Check(t, "audit_mrecords_add_data", m(q.Data.NDArray().Field("c")))
	refcheck.Check(t, "audit_mrecords_add_mask", m(q.Mask.NDArray().Field("c")))
	schema = refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{{Name: "a", Type: records.Numeric(np.Int64, nil)}, {Name: "b", Type: records.Numeric(np.Float64, nil)}}, false, 0))
	q = mr(ma.FromMaskedRecords(schema, [][]any{{int64(1), 3.}, {int64(2), 4.}}, 2))
	refcheck.Check(t, "audit_mrecords_fromrecords", m(q.Data.NDArray().Field("a")))
	refcheck.Check(t, "audit_mrecords_fromrecords_mask", m(q.Mask.NDArray().Field("a")))
	q = mr(ma.FromMaskedText(strings.NewReader("a,b\n1,2.5\n3,4.5\n"), records.NamedTextOptions{Text: np.TextOptions{Delimiter: ","}, Header: true}))
	for _, name := range []string{"a", "b"} {
		refcheck.Check(t, "audit_mrecords_fromtextfile_"+name, m(q.Data.NDArray().Field(name)))
	}
	refcheck.Check(t, "audit_mrecords_fromtextfile_mask", m(q.Mask.NDArray().Field("a")))
}

func TestAuditExtendedPolynomial(t *testing.T) {
	m := refcheck.Require(t)
	series := refcheck.RequireValue[*poly.Series](t)
	as := func(s ...string) *np.Array {
		v := make([]np.ExtendedFloat, len(s))
		for i, x := range s {
			v[i] = extended.MustParse(x)
		}
		return m(np.WrapSlice(v, len(v)))
	}
	for b := poly.Power; b <= poly.Laguerre; b++ {
		t.Run(b.String(), func(t *testing.T) {
			p := series(poly.NewSeries(b, as("1.0000000000000000001", "0.25", "0.5")))
			q := series(poly.NewSeries(b, as("0.5", "-0.125")))
			check := func(name string, v *np.Array) {
				t.Helper()
				refcheck.CheckExtended(t, "audit_poly_ext_"+strconv.Itoa(int(b))+"_"+name, v, np.Float128, 64)
			}
			for name, r := range map[string]*poly.Series{
				"add": series(p.Add(q)), "sub": series(p.Sub(q)), "mul": series(p.Mul(q)), "pow": series(p.Pow(2)),
				"deriv": series(p.Derivative(1)), "integ": series(p.Integrate(1, []complex128{.25}, nil)),
				"div": series(p.DivideScalar(np.Scalar(2))), "convert": series(p.Convert(poly.Power, poly.SeriesOptions{})),
			} {
				check(name, m(r.Coefficients()))
			}
			check("eval", m(p.Eval(as("0.125", "0.5"))))
			check("linear_roots", m(q.Roots()))
			check("vander", m(b.Vander(as("0.125", "0.5"), 3)))
			refcheck.CheckExtended(t, "audit_poly_ext_"+strconv.Itoa(int(b))+"_companion", m(b.Companion(m(p.Coefficients()))), np.Float128, 4096)
			a := m(m(np.Arange(1, 13, 1, np.Float64)).Reshape(3, 4))
			a = m(a.Slice(np.Slice{}, np.Reverse()))
			refcheck.Check(t, "audit_poly_ext_"+strconv.Itoa(int(b))+"_deriv_axis", m(b.DerivativeAxis(a, 2, 1, 1)))
			refcheck.Check(t, "audit_poly_ext_"+strconv.Itoa(int(b))+"_integ_axis", m(b.IntegralAxis(a, 1, []complex128{.25}, 0, 1, 1)))
		})
	}
}
