package np_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/extended"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
)

func TestOptimizationSetSearch(t *testing.T) {
	m := refcheck.Require(t)
	r := func(name string, a *np.Array) { refcheck.Check(t, "optimization_"+name, a) }
	unique := func(t *testing.T, name string, a *np.Array) {
		t.Helper()
		u := refcheck.RequireValue[np.UniqueResult](t)(np.Unique(a))
		for label, v := range map[string]*np.Array{"values": u.Values, "indices": u.Indices, "inverse": u.Inverse, "counts": u.Counts} {
			refcheck.Check(t, "optimization_unique_"+name+"_"+label, v)
		}
	}
	for _, dtype := range []string{"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"} {
		t.Run(dtype, func(t *testing.T) {
			a := refcheck.Input(t, "optimization_unique_input_"+dtype)
			unique(t, dtype, refcheck.Require(t)(a.Slice(np.Reverse(), np.Slice{Step: np.Int(2)})))
		})
	}
	buffer := func(values []int32) *np.Array {
		raw := make([]byte, len(values)*4)
		for i, v := range values {
			binary.BigEndian.PutUint32(raw[4*i:], uint32(v))
		}
		return m(np.FromBuffer(raw, np.Int32, len(values), 0, binary.BigEndian))
	}
	unique(t, "empty", m(np.Zeros(np.Int64, 2, 0)))
	unique(t, "scalar", np.ScalarOf(int64(42)))
	unique(t, "endian", m(buffer([]int32{3, -5, -5, 7, 1, 3}).Slice(np.Reverse())))
	grid := m(m(buffer([]int32{8, 99, 4, 98, 4, 97, 1, 96, -2, 95}).Slice(np.Slice{Step: np.Int(2)})).Slice(np.Reverse()))
	q := m(np.WrapSlice([]int32{-4, 4, 8, 1, 9, -2}, 2, 3))
	for side, right := range map[string]bool{"left": false, "right": true} {
		r("search_strided_"+side, m(np.SearchSorted(grid, q, right)))
	}
	temporal := func(d np.DType, u string, vs []int64) *np.Array {
		return m(m(np.WrapSlice(vs, len(vs))).CastTemporal(d, u))
	}
	var dt, td *np.Array
	for name, d := range map[string]np.DType{"datetime64": np.DateTime64, "timedelta64": np.TimeDelta64} {
		a := temporal(d, "D", []int64{0, 1, 2, np.NaT})
		b := temporal(d, "h", []int64{12, 24, 25, 48, 72, np.NaT})
		for side, right := range map[string]bool{"left": false, "right": true} {
			r("search_"+name+"_"+side, m(np.SearchSorted(a, b, right)))
		}
		r("isin_"+name, m(np.Isin(a, b, false)))
		if d == np.DateTime64 {
			dt = a
		} else {
			td = a
		}
	}
	qi := m(np.WrapSlice([]int64{0, 2, 3}, 3))
	r("search_delta_integer", m(np.SearchSorted(td, qi, false)))
	r("isin_delta_integer", m(np.Isin(td, qi, false)))
	r("isin_incompatible", m(np.Isin(dt, td, false)))
	_, err := np.SearchSorted(td, dt, false)
	refcheck.Error(t, "optimization_search_incompatible", "TypeError", err, np.ErrDType)
	parse := func(ss ...string) []np.ExtendedFloat {
		v := make([]np.ExtendedFloat, len(ss))
		for i, s := range ss {
			v[i] = extended.MustParse(s)
		}
		return v
	}
	x, y := parse("-0", "1.0000000000000000001", "1", "inf", "nan"), parse("0", "1.0000000000000000001", "inf", "nan")
	r("isin_extended", m(np.Isin(m(np.WrapSlice(x, len(x))), m(np.WrapSlice(y, len(y))), false)))
	complexValues := func(v []np.ExtendedFloat) *np.Array {
		z := make([]np.ExtendedComplex, len(v))
		for i, x := range v {
			z[i] = np.ExtendedComplex{Real: x, Imag: extended.FromInt64(1)}
		}
		return m(np.WrapSlice(z, len(z)))
	}
	r("isin_extended_complex", m(np.Isin(complexValues(x), complexValues(y), false)))
	a := m(np.FromObjects([]any{[]int{1, 2}, []int{2, 3}, "x", int64(2)}, 4))
	b := m(np.FromObjects([]any{[]int{1, 2}, "x", 2.}, 3))
	r("isin_object", m(np.Isin(a, b, false)))
	r("isin_object_invert", m(np.Isin(a, b, true)))
}

func TestOptimizationRecordMembership(t *testing.T) {
	m := refcheck.Require(t)
	schema := refcheck.RequireValue[*records.Schema](t)
	arr := refcheck.RequireValue[*records.Array](t)
	d := schema(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Numeric(np.Int32, binary.BigEndian)}, {Name: "y", Type: records.Numeric(np.Float64, nil)}}, true, 0))
	rows := make([][]any, 24)
	for i := range rows {
		rows[i] = []any{int32(i % 5), float64(i % 7)}
	}
	rows[0][1] = math.Copysign(0, -1)
	rows[7][1] = math.NaN()
	a := arr(records.FromRows(d, rows, 24)).NDArray()
	b := m(m(a.Slice(np.Slice{Step: np.Int(2)})).Copy())
	if err := m(b.Field("y")).SetValue(0., 0); err != nil {
		t.Fatal(err)
	}
	a = m(a.Slice(np.Reverse()))
	refcheck.Check(t, "optimization_record_isin", m(np.Isin(a, b, false)))
	refcheck.Check(t, "optimization_record_isin_invert", m(np.Isin(a, b, true)))
	d = schema(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Subarray(records.Numeric(np.Float64, nil), 2)}}, false, 0))
	a, b = arr(records.Zeros(d, 3)).NDArray(), arr(records.Zeros(d, 10)).NDArray()
	if err := m(a.Field("x")).Assign(nil, m(np.WrapSlice([]float64{math.Copysign(0, -1), 1, 1, 2, math.NaN(), 3}, 3, 2))); err != nil {
		t.Fatal(err)
	}
	if err := m(m(b.Field("x")).Slice(np.Slice{}, np.AtIndex(1))).Assign(nil, np.Scalar(1)); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "optimization_record_isin_subarray", m(np.Isin(a, b, false)))
}

func TestOptimizationSamplingStatistics(t *testing.T) {
	m := refcheck.Require(t)
	r := func(name string, a *np.Array) { refcheck.Check(t, "optimization_"+name, a) }
	gradient := refcheck.RequireValue[[]*np.Array](t)
	a := refcheck.Input(t, "optimization_gradient_input")
	a = m(m(a.Slice(np.Reverse(), np.Slice{Step: np.Int(2)})).Transpose())
	for _, axis := range []int{0, 1} {
		for _, edge := range []int{1, 2} {
			name := string(rune('0'+axis)) + "_" + string(rune('0'+edge))
			r("gradient_"+name, gradient(np.GradientAxes(a, []int{axis}, []*np.Array{np.Scalar(-.5)}, edge))[0])
		}
	}
	a = m(np.WrapFloat64([]float64{0, 1, 4, 9}, 4))
	for name, h := range map[string]float64{"tiny": 1e-200, "huge": 1e200, "zero": 0} {
		r("gradient_"+name, gradient(np.GradientAxes(a, nil, []*np.Array{np.Scalar(h)}, 2))[0])
	}
	r("gradient_nonfinite", gradient(np.GradientAxes(m(np.WrapFloat64([]float64{math.NaN(), math.Inf(1), math.Inf(-1), 5}, 4)), nil, nil, 2))[0])
	end := false
	for _, d := range []string{"float16", "float32", "float64", "complex64", "complex128"} {
		x, y := refcheck.Input(t, "optimization_geom_start_"+d), refcheck.Input(t, "optimization_geom_stop_"+d)
		r("geom_"+d, m(np.GeomspaceArray(x, y, np.SpaceOptions{Num: 7, Axis: 1, Endpoint: &end})))
	}
	a = refcheck.Input(t, "optimization_cov_input")
	a = m(m(a.Slice(np.Reverse(), np.Slice{Step: np.Int(2)})).Transpose())
	r("cov_strided", m(np.CovWithOptions(a, nil, np.CovOptions{})))
	r("cov_float32", m(np.CovWithOptions(a, nil, np.CovOptions{DType: np.Float32})))
	rev := m(a.Slice(np.Slice{}, np.Reverse()))
	z := m(np.Add(a, m(np.Multiply(rev, np.ComplexScalar(1i)))))
	w := m(np.WrapFloat64([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 8))
	r("cov_complex", m(np.CovWithOptions(z, nil, np.CovOptions{AWeights: w})))
	r("cov_nonfinite", m(np.CovWithOptions(m(np.WrapFloat64([]float64{1, 2, math.Inf(1), 4, 5, 6}, 2, 3)), nil, np.CovOptions{})))
}
