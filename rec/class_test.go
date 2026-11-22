package rec

import (
	"encoding/binary"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
)

func TestRecordArrayClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	rm := refcheck.RequireValue[*records.Array](t)
	sm := refcheck.RequireValue[*records.Schema](t)
	cm := refcheck.RequireValue[*records.Column](t)
	vm := refcheck.RequireValue[any](t)
	ok := func(e error) {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
	}
	check := func(n string, v *np.Array) {
		t.Helper()
		if !v.NativeByteOrder() {
			d := refcheck.RequireValue[dtypes.Descriptor](t)(dtypes.FromArray(v))
			refcheck.Text(t, "subclass_rec_"+n+"_dtype", d.String())
		}
		refcheck.Check(t, "subclass_rec_"+n, v)
	}
	nested := sm(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Numeric(np.Float32, nil)}, {Name: "flag", Type: records.Numeric(np.Bool, nil)}}, false, 0))
	vec := records.Numeric(np.Int16, nil)
	vec.Shape = []int{2}
	schema := sm(records.NewSchema([]records.FieldSpec{
		{Name: "id", Title: "ident", Type: records.Numeric(np.Int16, binary.BigEndian)},
		{Name: "shape", Type: records.Numeric(np.Float64, nil)}, {Name: "vec", Type: vec}, {Name: "sub", Type: records.Struct(nested)},
	}, false, 0))
	a := rm(RecArray(nil, Options{Schema: schema, Shape: []int{2, 3}}))
	field := func(a *records.Array, paths ...string) *np.Array { return m(a.NDArray().Field(paths...)) }
	ids := m(m(np.ArangeInt64(0, 6, 1)).Reshape(2, 3))
	ok(field(a, "id").Assign(nil, ids))
	ok(field(a, "shape").Assign(nil, m(np.Add(ids, np.Scalar(.5)))))
	ok(field(a, "vec").Assign(nil, m(m(np.ArangeInt64(0, 12, 1)).Reshape(2, 3, 2))))
	ok(field(a, "sub", "x").Assign(nil, m(np.Add(ids, np.Scalar(.25)))))
	ok(field(a, "sub", "flag").Assign(nil, m(np.FromSlice([]bool{true, false, true, false, true, false}, 2, 3))))
	check("field_name", m(cm(a.Field("id")).NDArray()))
	check("field_title", m(cm(a.Field("ident")).NDArray()))
	check("field_index", m(cm(a.FieldAt(0)).NDArray()))
	check("field_collision", m(cm(a.Field("shape")).NDArray()))
	check("field_vector", m(cm(a.Field("vec")).NDArray()))
	check("nested_x", field(a, "sub", "x"))
	check("nested_flag", field(a, "sub", "flag"))
	r := refcheck.RequireValue[*records.Record](t)(a.At(1, 0))
	check("scalar_index", np.ScalarOf(vm(r.FieldAt(0)).(int16)))
	check("scalar_title", np.ScalarOf(vm(r.Field("ident")).(int16)))
	sub := vm(r.FieldAt(-1)).(*records.Record)
	check("scalar_last", np.ScalarOf(vm(sub.Field("x")).(float32)))
	ok(r.SetAt(0, int16(91)))
	ok(r.Set("shape", -7.25))
	ok(sub.Set("x", float32(99.5)))
	check("scalar_write_id", field(a, "id"))
	check("scalar_write_shape", field(a, "shape"))
	check("scalar_write_nested", field(a, "sub", "x"))
	view := rm(a.Slice(np.Reverse(), np.Reverse()))
	check("reverse", field(view, "id"))
	ints := func(n string, v []int) {
		t.Helper()
		x := make([]int64, len(v))
		for i, k := range v {
			x[i] = int64(k)
		}
		check(n, m(np.FromSlice(x, len(x))))
	}
	ints("reverse_strides", view.Strides())
	v := rm(view.SelectFields("shape", "id"))
	layout := []int{v.Schema().ItemSize()}
	for _, f := range v.Schema().Fields() {
		layout = append(layout, f.Offset)
	}
	ints("multifield_layout", append(layout, v.Strides()...))
	refcheck.Text(t, "subclass_rec_multifield_names", strings.Join(v.Schema().Names(), "|"))
	ok(field(v, "id").SetValue(int16(123), 0, 0))
	check("multifield_write", field(a, "id"))
	i0 := m(np.FromSlice([]int64{0, 1}, 2))
	i1 := m(np.FromSlice([]int64{1, 0}, 2))
	v = rm(a.Index(i0, i1))
	check("gather_id", field(v, "id"))
	ok(field(v, "id").SetValue(int16(42), 0))
	check("gather_original", field(a, "id"))
	ok(a.Assign([]np.Index{i0, i0}, v))
	check("assign_id", field(a, "id"))
	check("assign_shape", field(a, "shape"))
	check("assign_vec", field(a, "vec"))
	_, e := a.FieldAt(99)
	refcheck.Error(t, "subclass_rec_field_oob", "IndexError", e, np.ErrIndex)
	_, e = a.Field("unknown")
	refcheck.Error(t, "subclass_rec_field_unknown", "KeyError", e, np.ErrIndex)
	_, e = a.SelectFields("id", "id")
	refcheck.Error(t, "subclass_rec_multifield_duplicate", "ValueError", e, np.ErrValue)
	plain := sm(records.NewSchema([]records.FieldSpec{{Name: "id", Type: records.Numeric(np.Int16, nil)}, {Name: "value", Type: records.Numeric(np.Float64, nil)}, {Name: "flag", Type: records.Numeric(np.Bool, nil)}}, false, 0))
	for _, p := range []struct {
		n   string
		row []any
	}{
		{"pprint", []any{int16(7), 1.25, true}}, {"pprint_integral", []any{int16(3), 2., false}},
	} {
		r := refcheck.RequireValue[*records.Record](t)(records.NewRecord(plain, p.row...))
		refcheck.Text(t, "subclass_rec_"+p.n, refcheck.RequireValue[string](t)(r.PPrint()))
	}
}
