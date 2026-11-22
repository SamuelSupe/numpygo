package records

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func managedSchema(t *testing.T) (*Schema, *Schema) {
	t.Helper()
	sm := refcheck.RequireValue[*Schema](t)
	inner := sm(NewSchema([]FieldSpec{{Name: "value", Type: Numeric(np.Float32, nil)}, {Name: "obj", Type: Object()}}, false, 0))
	a := sm(NewSchema([]FieldSpec{{Name: "id", Type: Numeric(np.Int16, binary.BigEndian)}, {Name: "payload", Type: Object()},
		{Name: "tags", Type: Subarray(Object(), 2)}, {Name: "inner", Type: Struct(inner)}}, false, 0))
	node := sm(NewSchema([]FieldSpec{{Name: "v", Type: Numeric(np.Float64, nil)}, {Name: "o", Type: Object()}}, false, 0))
	b := sm(NewSchema([]FieldSpec{{Name: "code", Type: Numeric(np.Int32, nil)}, {Name: "data", Type: Object()},
		{Name: "names", Type: Subarray(Object(), 2)}, {Name: "node", Type: Struct(node)}}, false, 0))
	return a, b
}

func TestManagedRecordsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	sm := refcheck.RequireValue[*Schema](t)
	rm := refcheck.RequireValue[*Array](t)
	s, target := managedSchema(t)
	a := rm(Zeros(s, 2, 3))
	check := func(e error) {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
	}
	dump := func(name string, x *np.Array) {
		t.Helper()
		v := refcheck.RequireValue[any](t)(x.ToList())
		b := refcheck.RequireValue[[]byte](t)(json.Marshal(v))
		refcheck.Text(t, "closure_"+name, string(b))
	}
	strides := func(name string, x *np.Array) {
		v := []int64{}
		for _, n := range x.Strides() {
			v = append(v, int64(n))
		}
		refcheck.Check(t, "closure_"+name, m(np.FromSlice(v, len(v))))
	}
	col := func(x *np.Array, n ...string) *np.Array { return m(x.Field(n...)) }
	nd := a.NDArray()
	if !nd.HasObject() || !s.HasObject() {
		t.Fatal("object schema lost")
	}
	dump("managed_zero_payload", col(nd, "payload"))
	dump("managed_zero_tags", col(nd, "tags"))
	ids := m(m(np.ArangeInt64(0, 6, 1)).Reshape(2, 3))
	check(col(nd, "id").Assign(nil, ids))
	shared := &[]any{"seed"}
	p := col(nd, "payload")
	for i, v := range []any{shared, "hello", nil, int64(7), "猫", false} {
		check(p.SetValue(v, i/3, i%3))
	}
	check(col(nd, "tags").Assign(nil, m(np.FromObjects([]any{"L", "R"}, 1, 2))))
	check(col(nd, "inner", "value").Assign(nil, m(np.Add(ids, np.Scalar(.25)))))
	check(col(nd, "inner", "obj").Assign(nil, np.ObjectScalar("inner")))
	check(col(nd, "inner", "obj").SetValue(nil, 1, 2))
	checkManaged := func(name string, x *np.Array) {
		t.Helper()
		refcheck.Check(t, "closure_"+name+"_id", m(col(x, "id").Astype(np.Int16)))
		dump(name+"_payload", col(x, "payload"))
		dump(name+"_tags", col(x, "tags"))
		refcheck.Check(t, "closure_"+name+"_inner_value", col(x, "inner", "value"))
		dump(name+"_inner_obj", col(x, "inner", "obj"))
		strides(name+"_strides", x)
	}
	checkManaged("managed_initial", nd)
	strides("managed_field_strides", col(nd, "tags"))
	v := m(m(nd.Transpose()).Slice(np.Reverse()))
	check(col(v, "payload").SetValue("view", 0, 1))
	check(col(v, "tags").SetValue("changed", 1, 0, 1))
	checkManaged("managed_view_write", nd)
	b := m(nd.CopyWithOrder("F"))
	got := refcheck.RequireValue[any](t)(col(b, "payload").AtValue(0, 0))
	if got != shared {
		t.Fatal("object copy did not retain reference")
	}
	*shared = append(*shared, "shared")
	checkManaged("managed_copy_shared", nd)
	check(col(b, "payload").SetValue("replacement", 0, 0))
	checkManaged("managed_copy_detached", b)
	ix := m(np.FromSlice([]int64{2, 0}, 2))
	fancy := m(nd.Index(np.Slice{}, ix))
	checkManaged("managed_fancy", fancy)
	if np.SharesMemory(nd, fancy) {
		t.Fatal("managed gather aliased slots")
	}
	projection := rm(a.SelectFields("id", "payload"))
	check(col(projection.NDArray(), "payload").SetValue("projection", 0, 2))
	dump("managed_projection", col(nd, "payload"))
	strides("managed_projection_strides", projection.NDArray())
	c := m(np.Concatenate([]*np.Array{m(nd.Slice(np.Range(0, 1))), m(nd.Slice(np.Range(1, 2)))}, 0))
	checkManaged("managed_concat", c)
	refcheck.Check(t, "closure_managed_equal", m(np.Equal(nd, c)))
	check(col(c, "payload").SetValue("different", 1, 2))
	refcheck.Check(t, "closure_managed_neq", m(np.NotEqual(nd, c)))
	x := m(m(nd.Slice(np.Reverse())).CastRecords(target))
	refcheck.Check(t, "closure_managed_cast_code", col(x, "code"))
	dump("managed_cast_data", col(x, "data"))
	dump("managed_cast_names", col(x, "names"))
	refcheck.Check(t, "closure_managed_cast_value", col(x, "node", "v"))
	dump("managed_cast_object", col(x, "node", "o"))
	p0 := sm(NewSchema([]FieldSpec{{Name: "x", Type: Numeric(np.Int16, nil)}}, false, 0))
	p1 := sm(NewSchema([]FieldSpec{{Name: "y", Type: Object()}}, false, 0))
	p2 := sm(NewSchema([]FieldSpec{{Name: "q", Type: Numeric(np.Int32, nil)}}, false, 0))
	plain := rm(FromRows(p0, [][]any{{int16(2)}, {int16(7)}}, 2))
	promoted := m(plain.NDArray().CastRecords(p1))
	refcheck.Check(t, "closure_managed_promote_numeric", m(col(promoted, "y").Astype(np.Int64)))
	refcheck.Check(t, "closure_managed_cast_numeric", col(m(promoted.CastRecords(p2)), "q"))
	// The scalar record and its object subarray remain shared with nd.
	row := refcheck.RequireValue[*Record](t)(a.At(0, 0))
	tag := refcheck.RequireValue[any](t)(row.Field("tags")).(*np.Array)
	check(tag.SetValue("scalar-view", 0))
	got = refcheck.RequireValue[any](t)(col(nd, "tags").AtValue(0, 0, 0))
	if got != "scalar-view" {
		t.Fatal("scalar object subarray did not write through")
	}
	column := refcheck.RequireValue[*Column](t)(a.Field("payload"))
	check(column.Set("column", 0, 1))
	got = refcheck.RequireValue[any](t)(row.Field("payload"))
	if got != shared {
		t.Fatal("unrelated object slot changed")
	}
	got = refcheck.RequireValue[any](t)(col(nd, "payload").AtValue(0, 1))
	if got != "column" {
		t.Fatal("column did not write through")
	}
	_, e := NewSchema([]FieldSpec{{Name: "o", Type: Object(), Offset: np.Int(0)}, {Name: "x", Type: Numeric(np.Int32, nil), Offset: np.Int(2)}}, false, 8)
	refcheck.Error(t, "closure_managed_overlap", "TypeError", e, np.ErrValue)
	var buf bytes.Buffer
	refcheck.Error(t, "closure_managed_pickle", "ValueError", np.SaveNPY(&buf, nd), np.ErrDType)
	if buf.Len() != 0 {
		t.Fatal("pointer-bearing payload emitted")
	}
	if _, e = nd.ToBytes("C"); !errors.Is(e, np.ErrDType) {
		t.Fatal(e)
	}
	if _, e = nd.View(np.Uint8); !errors.Is(e, np.ErrDType) {
		t.Fatal(e)
	}
	if row.Bytes() != nil {
		t.Fatal("pointer-bearing scalar exposed raw bytes")
	}
}
