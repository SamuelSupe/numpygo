package np_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"slices"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/datetime"
	"github.com/SamuelSupe/numpygo/extended"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/linalg"
	"github.com/SamuelSupe/numpygo/objects"
	"github.com/SamuelSupe/numpygo/records"
)

func TestUnifiedExtendedNumPy(t *testing.T) {
	m := refcheck.Require(t)
	me := refcheck.RequireValue[*extended.Array](t)
	check := func(name string, a *np.Array, d np.DType, shape ...int) {
		t.Helper()
		if a.DType() != d || !slices.Equal(a.Shape(), shape) {
			t.Fatalf("%s: wrong dtype or shape: %s %v", name, a.DType(), a.Shape())
		}
		b := refcheck.RequireValue[[]byte](t)(a.ToBytes("C"))
		raw := make([]uint8, 0, len(b)/16*10)
		for i := 0; i < len(b); i += 16 {
			raw = append(raw, b[i:i+10]...)
		}
		refcheck.Check(t, "unified_ext_"+name, m(np.WrapSlice(raw, len(b)/16, 10)))
	}
	a := me(extended.FromStrings([]string{"1.0000000000000000001", "2.25", "-3.5", "4", "1e4000", "9223372036854775809"}, 2, 3)).NDArray()
	rev := m(a.Slice(np.Slice{}, np.Reverse()))
	check("broadcast", m(np.Add(rev, m(np.FromSlice([]int64{1, 2, 3}, 3)))), np.Float128, 2, 3)
	i := m(np.FromSlice([]int64{1, 0}, 2))
	j := m(np.FromSlice([]int64{2, 0}, 2))
	check("index", m(a.Index(i, j)), np.Float128, 2)
	check("transpose_reshape", m(m(a.Transpose()).Reshape(6)), np.Float128, 6)
	mask := m(np.FromSlice([]bool{true, false}, 2, 1))
	check("where", m(np.Where(mask, a, np.ScalarOf(extended.MustParse("3.0000000000000000002")))), np.Float128, 2, 3)
	check("sum", m(np.Sum(a, np.SumOptions{Axes: []int{0}})), np.Float128, 3)
	b := m(m(np.FromSlice([]int64{1, 2, 3, 4}, 2, 2)).Astype(np.Float128))
	check("dot", m(linalg.MatMul(m(a.Slice(np.Slice{}, np.Range(0, 2))), b)), np.Float128, 2, 2)
	check("integer", m(m(np.FromSlice([]uint64{1<<63 + 1, ^uint64(0)}, 2)).Astype(np.Float128)), np.Float128, 2)
	v := me(extended.FromNDArray(m(a.Transpose())))
	if e := v.Set(extended.Complex{Real: extended.MustParse("1.0000000000000000002")}, 1, 0); e != nil {
		t.Fatal(e)
	}
	if !np.SharesMemory(v.NDArray(), a) {
		t.Fatal("extended adapter lost shared storage")
	}
	check("writeback", a, np.Float128, 2, 3)
	z := m(np.FromSlice([]complex128{1 - 1i, 2, 3 + 1i, 4 + 2i}, 2, 2))
	z = m(z.Astype(np.Complex256))
	imag := m(np.Imag(z))
	it := m(imag.Transpose())
	if !np.SharesMemory(imag, z) {
		t.Fatal("imaginary part must be a view")
	}
	if e := it.SetValue(extended.MustParse("0.0000000000000000001"), 1, 0); e != nil {
		t.Fatal(e)
	}
	check("complex_writeback", z, np.Complex256, 2, 2)
	check("complex_sum", m(np.Sum(z, np.SumOptions{Axes: []int{0}})), np.Complex256, 2)
	check("complex_dot", m(linalg.MatMul(z, z)), np.Complex256, 2, 2)
	refcheck.Check(t, "unified_ext_compare", m(np.Greater(m(np.Real(z)), np.ScalarOf(extended.FromInt64(2)))))
	stat := me(extended.FromStrings([]string{"1.0000000000000000001", "2", "3", "4"}, 2, 2)).NDArray()
	check("cumsum", m(np.Cumsum(stat, np.Int(1), 0)), np.Float128, 2, 2)
	check("cumprod", m(np.Cumprod(stat, np.Int(0), 0)), np.Float128, 2, 2)
	check("mean", m(np.Mean(stat, np.SumOptions{Axes: []int{0}})), np.Float128, 2)
	check("var", m(np.Var(stat, np.VarOptions{Axes: []int{0}})), np.Float128, 2)
	check("std", m(np.Std(stat)), np.Float128)
	cstat := m(m(np.FromSlice([]complex128{1 + 2i, 3 + 4i}, 2)).Astype(np.Complex256))
	check("complex_var", m(np.Var(cstat)), np.Float128)
	next := m(m(np.FromSlice([]int64{0, 1, -1}, 3)).Astype(np.Float128))
	check("nextafter", m(np.Nextafter(next, np.ScalarOf(extended.FromInt64(2)))), np.Float128, 3)
	info := np.ExtendedFInfo()
	check("info", m(np.FromSlice([]np.ExtendedFloat{info.Eps, info.Tiny, info.Max}, 3)), np.Float128, 3)
	check("norm", m(linalg.Norm(stat)), np.Float128)
	check("complex_norm", m(linalg.Norm(z, linalg.NormOptions{Axes: []int{1}})), np.Float128, 2)
}

func TestUnifiedRecordNumPy(t *testing.T) {
	m := refcheck.Require(t)
	ms := refcheck.RequireValue[*records.Schema](t)
	nested := ms(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Numeric(np.Float32, binary.LittleEndian)}, {Name: "tag", Type: records.FixedBytes(3)}}, false, 0))
	dt := ms(records.NewSchema([]records.FieldSpec{
		{Name: "id", Type: records.Numeric(np.Int8, nil)}, {Name: "value", Type: records.Numeric(np.Float64, binary.BigEndian)},
		{Name: "vec", Type: records.Subarray(records.Numeric(np.Int16, binary.LittleEndian), 2)},
		{Name: "nested", Type: records.Struct(nested)},
	}, false, 0))
	rows := make([][]any, 6)
	for i := range rows {
		rows[i] = []any{i, float64(i) + .5, []int{2 * i, 2*i + 1}, []any{2.5, "abc"}}
	}
	r := refcheck.RequireValue[*records.Array](t)(records.FromRows(dt, rows, 2, 3))
	a := r.NDArray()
	if a.ItemSize() != dt.ItemSize() || a.DType() != np.RecordDType {
		t.Fatal("record descriptor lost")
	}
	field := m(a.Field("value"))
	if !np.SharesMemory(field, a) {
		t.Fatal("packed field did not share storage")
	}
	strides := field.Strides()
	refcheck.Check(t, "unified_record_stride", m(np.FromSlice([]int64{int64(strides[0]), int64(strides[1])}, 2)))
	refcheck.Check(t, "unified_record_nested", m(a.Field("nested", "x")))
	v := m(m(m(a.Transpose()).Slice(np.Reverse())).Field("value"))
	if e := v.SetValue(90.25, 0, 1); e != nil {
		t.Fatal(e)
	}
	if e := m(a.Field("vec")).SetValue(int16(-11), 0, 1, 1); e != nil {
		t.Fatal(e)
	}
	if e := m(a.Field("nested", "x")).SetValue(float32(7.5), 1, 0); e != nil {
		t.Fatal(e)
	}
	check := func(name string, a *np.Array, shape ...int) {
		t.Helper()
		want := np.StructuredSchema(dt)
		if name == "concat" || name == "where" {
			want = refcheck.RequireValue[np.StructuredSchema](t)(dt.NativeSchema())
		}
		if a.DType() != np.RecordDType || !slices.Equal(a.Shape(), shape) || !want.EqualSchema(a.StructuredSchema()) {
			t.Fatalf("record metadata: %v", a.Shape())
		}
		refcheck.Bytes(t, "unified_record_"+name, refcheck.RequireValue[[]byte](t)(a.ToBytes("C")))
	}
	check("writeback", a, 2, 3)
	i := m(np.FromSlice([]int64{1, 0}, 2))
	j := m(np.FromSlice([]int64{2, 1}, 2))
	index := m(a.Index(i, j))
	check("index", index, 2)
	if np.SharesMemory(a, index) {
		t.Fatal("advanced indexing must copy records")
	}
	check("reshape", m(m(a.Transpose()).Reshape(6)), 6)
	check("concat", m(np.Concatenate([]*np.Array{a, m(a.Slice(np.Reverse()))}, 0)), 4, 3)
	mask := m(np.FromSlice([]bool{true, false}, 2, 1))
	check("where", m(np.Where(mask, a, m(a.Slice(np.Slice{}, np.Reverse())))), 2, 3)
	b := m(a.Copy())
	if e := m(b.Field("id")).SetValue(int8(99), 1, 1); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "unified_record_equal", m(np.Equal(a, b)))
	readonly := m(m(a.BroadcastTo(2, 2, 3)).Field("id"))
	refcheck.Error(t, "unified_record_readonly", "ValueError", readonly.SetValue(int8(1), 0, 0, 0), np.ErrReadOnly)
}

func TestUnifiedDateNumPy(t *testing.T) {
	m := refcheck.Require(t)
	a := m(np.FromDateStrings([]string{"1969-12-31", "1970-01-01", "2000-02-29", "NaT"}, "D", 2, 2))
	ticks := func(v []int64, d np.DType, u string) *np.Array {
		return m(np.FromDateTicks(m(np.FromSlice(v, len(v))), d, u))
	}
	check := func(name string, a *np.Array, d np.DType, u string) {
		t.Helper()
		meta, e := a.DateUnit()
		if e != nil || a.DType() != d || meta.String() != u {
			t.Fatalf("%s temporal dtype: %s %v %v", name, a.DType(), meta, e)
		}
		refcheck.Check(t, "unified_date_"+name, m(a.DateTicks()))
	}
	b := ticks([]int64{3, -2}, np.TimeDelta64, "h")
	check("plus", m(np.Add(a, b)), np.DateTime64, "h")
	origin := m(m(np.FromDateStrings([]string{"1970-01-01T03"}, "h", 1)).Reshape())
	check("minus", m(np.Subtract(a, origin)), np.TimeDelta64, "h")
	check("cast", m(a.CastTemporal(np.DateTime64, "M")), np.DateTime64, "M")
	check("sort", m(np.Sort(a, nil)), np.DateTime64, "D")
	check("index", m(a.Index(m(np.FromSlice([]int64{1, 0}, 2)), m(np.FromSlice([]int64{0, 1}, 2)))), np.DateTime64, "D")
	h := m(a.CastTemporal(np.DateTime64, "h"))
	check("concat", m(np.Concatenate([]*np.Array{a, h}, 1)), np.DateTime64, "h")
	check("where", m(np.Where(m(np.FromSlice([]bool{true, false}, 2, 1)), a, h)), np.DateTime64, "h")
	refcheck.Check(t, "unified_date_nat", m(np.IsNaT(a)))
	refcheck.Check(t, "unified_date_eq", m(np.Equal(a, h)))
	view := refcheck.RequireValue[*datetime.Array](t)(datetime.FromNDArray(m(a.Transpose())))
	if !np.SharesMemory(view.Ticks, a) {
		t.Fatal("date tick adapter lost storage")
	}
	day := refcheck.RequireValue[np.DateTime](t)(np.ParseDateTime("2001-01-01", "D"))
	if e := m(view.NDArray()).SetValue(day, 0, 1); e != nil {
		t.Fatal(e)
	}
	check("writeback", a, np.DateTime64, "D")
	d := ticks([]int64{-5, 6, 9}, np.TimeDelta64, "3h")
	e := ticks([]int64{1, 2, 3}, np.TimeDelta64, "2h")
	check("duration_add", m(np.Add(d, e)), np.TimeDelta64, "h")
	check("duration_mod", m(np.Remainder(d, e)), np.TimeDelta64, "h")
	check("duration_scale", m(np.Multiply(d, np.ScalarOf(int64(2)))), np.TimeDelta64, "3h")
	check("duration_sum", m(np.Sum(d)), np.TimeDelta64, "3h")
	refcheck.Check(t, "unified_date_duration_ratio", m(np.Divide(d, e)))
	check("duration_cumsum", m(np.Cumsum(d, nil, 0)), np.TimeDelta64, "3h")
}

func TestUnifiedNPYNumPy(t *testing.T) {
	m := refcheck.Require(t)
	for _, name := range []string{"extended", "complex", "datetime", "timedelta"} {
		t.Run(name, func(t *testing.T) {
			b := refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "unified_npy_"+name)))
			a := m(np.LoadNPY(bytes.NewReader(b)))
			d := map[string]np.DType{"extended": np.Float128, "complex": np.Complex256, "datetime": np.DateTime64, "timedelta": np.TimeDelta64}[name]
			if a.DType() != d {
				t.Fatalf("dtype: %s", a.DType())
			}
			var out bytes.Buffer
			if e := np.SaveNPY(&out, a); e != nil {
				t.Fatal(e)
			}
			// Header whitespace is not a contract; compare decoded metadata
			// and the payload from NumPy's file, in its declared order.
			x := out.Bytes()
			offset := func(b []byte) int { return 10 + int(binary.LittleEndian.Uint16(b[8:10])) }
			if !bytes.Equal(b[offset(b):], x[offset(x):]) {
				t.Fatal("NPY payload does not match NumPy")
			}
			reloaded := m(np.LoadNPY(bytes.NewReader(x)))
			if !slices.Equal(a.Shape(), reloaded.Shape()) || !slices.Equal(a.Strides(), reloaded.Strides()) || reloaded.DType() != d {
				t.Fatal("NPY metadata not preserved")
			}
			if d.IsTemporal() {
				u, _ := a.DateUnit()
				v, _ := reloaded.DateUnit()
				if u != v {
					t.Fatal("NPY temporal unit not preserved")
				}
			}
		})
	}
	if e := np.SaveNPY(&bytes.Buffer{}, np.ObjectScalar("no pickle")); !errors.Is(e, np.ErrDType) {
		t.Fatal(e)
	}
}

func TestUnifiedObjectNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mo := refcheck.RequireValue[*objects.Array](t)
	a := mo(objects.FromSlice([]any{int64(1), int64(2), int64(3), int64(4), int64(5), int64(6)}, 2, 3)).NDArray()
	check := func(name string, v *np.Array) {
		t.Helper()
		if v.DType() != np.Object {
			t.Fatal("object dtype lost")
		}
		refcheck.Check(t, "unified_obj_"+name, m(v.Astype(np.Int64)))
	}
	reverse := m(a.Slice(np.Slice{}, np.Reverse()))
	check("add", m(np.Add(reverse, m(np.FromSlice([]int64{0, 1, 2}, 3)))))
	check("index", m(a.Index(m(np.FromSlice([]int64{1, 0}, 2)), m(np.FromSlice([]int64{2, 1}, 2)))))
	check("sum", m(np.Sum(a, np.SumOptions{Axes: []int{0}})))
	refcheck.Check(t, "unified_obj_compare", m(np.Greater(a, np.ScalarOf(int64(3)))))
	v := mo(objects.FromNDArray(m(a.Transpose())))
	if e := v.Set(int64(90), 1, 0); e != nil {
		t.Fatal(e)
	}
	check("writeback", a)
	check("where", m(np.Where(m(np.FromSlice([]bool{true, false}, 2, 1)), a, reverse)))
	mixed := m(np.FromObjects([]any{nil, "", "abc", int64(0), int64(2)}, 5))
	refcheck.Check(t, "unified_obj_truth", m(mixed.Astype(np.Bool)))
	value := map[string]int{"x": 1}
	obj := np.ObjectScalar(value)
	copyObj := m(obj.Copy())
	loaded := refcheck.RequireValue[any](t)(copyObj.AtValue()).(map[string]int)
	loaded["x"] = 2
	if value["x"] != 2 || np.SharesMemory(obj, copyObj) {
		t.Fatal("object copy should copy slots, not pointed-to values")
	}
	if _, e := obj.ToBytes("C"); !errors.Is(e, np.ErrDType) {
		t.Fatalf("object raw memory exposed: %v", e)
	}
}
