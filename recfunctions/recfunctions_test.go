package recfunctions

import (
	"bytes"
	"encoding/binary"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
	"strconv"
	"strings"
	"testing"
)

func checkRecord(t *testing.T, name string, a *np.Array) {
	t.Helper()
	m := refcheck.Require(t)
	h := refcheck.RequireValue[np.NPYHeader](t)(np.NPYHeaderFromArray(a))
	var b bytes.Buffer
	if e := np.WriteNPYHeader(&b, h, np.NPYVersion{Major: 1}); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, name+"_header", b.Bytes())
	if a.DType() == np.RecordDType {
		s := a.StructuredSchema().(*records.Schema)
		aligned := int64(0)
		if s.Aligned() {
			aligned = 1
		}
		refcheck.Check(t, name+"_meta", m(np.FromSlice([]int64{int64(s.ItemSize()), int64(s.Alignment()), aligned}, 3)))
	}
	var visit func(string, *np.Array)
	visit = func(p string, v *np.Array) {
		if v.DType() == np.RecordDType {
			for _, n := range v.StructuredSchema().(*records.Schema).Names() {
				visit(p+"_"+n, m(v.Field(n)))
			}
		} else {
			refcheck.Bytes(t, p, refcheck.RequireValue[[]byte](t)(v.ToBytes("C")))
		}
	}
	visit(name, a)
}
func checkMasked(t *testing.T, name string, a Array) {
	t.Helper()
	if a.Mask == nil {
		t.Fatal("missing structured mask")
	}
	checkRecord(t, name, a.Data)
	checkRecord(t, name+"_mask", a.Mask)
}
func schema(t *testing.T, fs ...records.FieldSpec) *records.Schema {
	t.Helper()
	return refcheck.RequireValue[*records.Schema](t)(records.NewSchema(fs, false, 0))
}
func field(name string, d np.DType) records.FieldSpec {
	return records.FieldSpec{Name: name, Type: records.Numeric(d, nil)}
}
func fromRows(t *testing.T, s *records.Schema, rows ...[]any) *np.Array {
	t.Helper()
	return refcheck.RequireValue[*records.Array](t)(records.FromRows(s, rows, len(rows))).NDArray()
}
func namesText(n []Name) string {
	parts := []string{}
	for _, x := range n {
		v := "'" + x.Name + "'"
		if x.Children != nil {
			v = "(" + v + ", " + namesText(x.Children) + ")"
		}
		parts = append(parts, v)
	}
	s := strings.Join(parts, ", ")
	if len(parts) == 1 {
		s += ","
	}
	return "(" + s + ")"
}

func TestRecordFieldUtilitiesNumPy(t *testing.T) {
	m := refcheck.Require(t)
	deep := schema(t, field("y", np.Int32))
	nested := schema(t, records.FieldSpec{Name: "x", Type: records.Numeric(np.Float32, binary.BigEndian)}, records.FieldSpec{Name: "deep", Type: records.Struct(deep)})
	s := schema(t, field("id", np.Int16), records.FieldSpec{Name: "nested", Type: records.Struct(nested)}, records.FieldSpec{Name: "vec", Type: records.Subarray(records.Numeric(np.Int16, nil), 2)})
	a := fromRows(t, s, []any{1, []any{2.5, []any{3}}, []any{4, 5}}, []any{6, []any{7.5, []any{8}}, []any{9, 10}})
	refcheck.Text(t, "rf_get_names", namesText(GetNames(s)))
	refcheck.Text(t, "rf_get_names_flat", strings.Join(GetNamesFlat(s), "|"))
	paths := GetFieldStructure(s)
	p := []string{}
	for _, name := range GetNamesFlat(s) {
		p = append(p, name+":"+strings.Join(paths[name], "/"))
	}
	refcheck.Text(t, "rf_get_fieldstructure", strings.Join(p, "|"))
	checkRecord(t, "rf_flatten_descr", m(np.ZerosRecords(schema(t, FlattenDescr(s)...), 1)))
	b := m(RenameFields(a, map[string]string{"id": "ID", "x": "X", "y": "Y"}))
	checkRecord(t, "rf_rename_fields", b)
	refcheck.Check(t, "rf_rename_shares", m(np.FromSlice([]bool{np.SharesMemory(a, b)})))
	if e := m(b.Field("nested", "deep", "Y")).SetValue(int32(91), 0); e != nil {
		t.Fatal(e)
	}
	checkRecord(t, "rf_rename_writeback", a)
	checkRecord(t, "rf_drop_fields", m(DropFields(a, "x", "vec")))
	checkRecord(t, "rf_rec_drop_fields", refcheck.RequireValue[*records.Array](t)(RecDropFields(a, "nested")).NDArray())
	checkRecord(t, "rf_drop_all", m(DropFields(a, "id", "nested", "vec")))
	dstSchema := schema(t, records.FieldSpec{Name: "vec", Type: records.Subarray(records.Numeric(np.Int32, nil), 2)},
		records.FieldSpec{Name: "nested", Type: records.Struct(schema(t, field("x", np.Float64), field("z", np.Int16)))}, field("missing", np.Int32))
	dst := m(np.ZerosRecords(dstSchema, 2))
	if e := fillTree(dst, 1); e != nil {
		t.Fatal(e)
	}
	if e := AssignFieldsByName(dst, a, false); e != nil {
		t.Fatal(e)
	}
	checkRecord(t, "rf_assign_keep", dst)
	if e := AssignFieldsByName(dst, a, true); e != nil {
		t.Fatal(e)
	}
	checkRecord(t, "rf_assign_zero", dst)
	checkRecord(t, "rf_require_fields", m(RequireFields(a, dstSchema)))
	dst = m(np.ZerosRecords(dstSchema, 3))
	if e := fillTree(dst, 1); e != nil {
		t.Fatal(e)
	}
	checkRecord(t, "rf_recursive_fill", m(RecursiveFillFields(a, dst)))
	point := schema(t, field("x", np.Int64), field("y", np.Float32))
	padded := fromRows(t, schema(t, field("tag", np.Uint8), records.FieldSpec{Name: "point", Type: records.Struct(point)}, field("value", np.Float64)),
		[]any{2, []any{9, 4.5}, 7.25}, []any{3, []any{13, -2.5}, 8.5})
	for _, tc := range []struct {
		align, recurse bool
		name           string
	}{{true, false, "1_0"}, {true, true, "1_1"}, {false, true, "0_1"}} {
		checkRecord(t, "rf_repack_"+tc.name, m(RepackFields(padded, tc.align, tc.recurse)))
	}
	rep := m(RepackFields(padded, false, false))
	checkRecord(t, "rf_repack_already", rep)
	refcheck.Check(t, "rf_repack_shares", m(np.FromSlice([]bool{np.SharesMemory(rep, padded)})))
}

func TestRecordConversionsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	desc := refcheck.RequireValue[dtypes.Descriptor](t)
	s := schema(t, field("a", np.Int16), records.FieldSpec{Name: "sub", Type: records.Struct(schema(t, field("x", np.Float32), field("y", np.Int32)))}, records.FieldSpec{Name: "v", Type: records.Subarray(records.Numeric(np.Int16, nil), 2)})
	a := fromRows(t, s, []any{1, []any{2.5, 3}, []any{4, 5}}, []any{6, []any{7.5, 8}, []any{9, 10}})
	refcheck.Check(t, "rf_struct_to_unstruct", m(StructuredToUnstructured(m(a.Slice(np.Reverse())), nil, false, "")))
	refcheck.Check(t, "rf_apply_along_fields", m(ApplyAlongFields(a, func(a *np.Array, ax int) (*np.Array, error) { return np.Mean(a, np.SumOptions{Axes: []int{ax}}) })))
	d := desc(dtypes.Parse("i2"))
	_, e := StructuredToUnstructured(a, &d, false, "safe")
	refcheck.Error(t, "rf_struct_bad_cast", "TypeError", e, np.ErrDType)
	_, e = StructuredToUnstructured(m(np.ZerosRecords(schema(t), 2)), nil, false, "")
	refcheck.Error(t, "rf_struct_no_fields", "ValueError", e, np.ErrValue)
	for _, tc := range []struct {
		name    string
		offsets []int
	}{
		{"padded", []int{1, 9, 17}}, {"reversed", []int{17, 9, 1}}, {"overlap", []int{1, 1, 1}}, {"irregular", []int{1, 9, 22}}} {
		t.Run(tc.name, func(t *testing.T) {
			fs := []records.FieldSpec{field("a", np.Int32), field("b", np.Int32), field("c", np.Int32)}
			for i := range fs {
				fs[i].Offset = &tc.offsets[i]
			}
			s := refcheck.RequireValue[*records.Schema](t)(records.NewSchema(fs, false, 28))
			r := m(np.ZerosRecords(s, 2))
			for i, n := range s.Names() {
				if e := m(r.Field(n)).Assign(nil, m(np.FromSlice([]int32{int32(i + 1), int32(i + 4)}, 2))); e != nil {
					t.Fatal(e)
				}
			}
			out := m(StructuredToUnstructured(r, nil, false, ""))
			refcheck.CheckLayout(t, "rf_uniform_"+tc.name, out)
			refcheck.Check(t, "rf_uniform_share_"+tc.name, m(np.FromSlice([]bool{np.SharesMemory(r, out)})))
			if e := out.SetValue(int32(77), 1, 0); e != nil {
				t.Fatal(e)
			}
			refcheck.Check(t, "rf_uniform_write_"+tc.name, m(r.Field("a")))
			refcheck.Check(t, "rf_uniform_copy_"+tc.name, m(StructuredToUnstructured(r, nil, true, "")))
		})
	}
	src := m(np.FromSlice([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 3, 5))
	b := m(UnstructuredToStructured(src, nil, []string{"a", "b", "c", "d", "e"}, false, false, ""))
	checkRecord(t, "rf_unstruct_to_struct", b)
	refcheck.Check(t, "rf_unstruct_shares", m(np.FromSlice([]bool{np.SharesMemory(src, b)})))
	if e := m(b.Field("d")).SetValue(int32(99), 2); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "rf_unstruct_writeback", src)
	checkRecord(t, "rf_unstruct_nested", m(UnstructuredToStructured(src, s, nil, false, false, "")))
	checkRecord(t, "rf_unstruct_strided", m(UnstructuredToStructured(m(src.Slice(np.Slice{}, np.Reverse())), nil, nil, false, false, "")))
	_, e = UnstructuredToStructured(src, schema(t, field("a", np.Int32)), nil, false, false, "")
	refcheck.Error(t, "rf_unstruct_bad_size", "ValueError", e, np.ErrShape)
	_, e = UnstructuredToStructured(src, s, []string{"a"}, false, false, "")
	refcheck.Error(t, "rf_unstruct_both", "ValueError", e, np.ErrValue)
}

func TestRecordCombineNumPy(t *testing.T) {
	m := refcheck.Require(t)
	ma := refcheck.RequireValue[Array](t)
	desc := refcheck.RequireValue[dtypes.Descriptor](t)
	s := schema(t, field("id", np.Int32), field("v", np.Int16))
	base := fromRows(t, s, []any{2, 20}, []any{1, 10})
	b := Array{Data: base}
	i := func(v ...int32) *np.Array { return m(np.FromSlice(v, len(v))) }
	c := Array{Data: i(9, 8), Mask: m(np.FromSlice([]bool{false, true}, 2))}
	checkMasked(t, "rf_merge_flat", ma(MergeArrays([]Array{b, c}, true, Options{UseMask: true})))
	_, e := MergeArrays([]Array{b, {Data: i(9)}}, true, Options{UseMask: true})
	refcheck.Error(t, "rf_merge_flat_unequal", "ValueError", e, np.ErrValue)
	checkRecord(t, "rf_merge_nested", ma(MergeArrays([]Array{b, {Data: i(9, 8, 7)}}, false, Options{})).Data)
	f := m(np.FromFloat64([]float64{2.5, 4.5, 6.5}, 3))
	checkMasked(t, "rf_append_fields", ma(AppendFields(b, []string{"c", "d"}, []Array{{Data: i(3)}, {Data: f}},
		[]dtypes.Descriptor{desc(dtypes.Parse("i2")), desc(dtypes.Parse("f4"))}, Options{UseMask: true})))
	checkRecord(t, "rf_rec_append_fields", refcheck.RequireValue[*records.Array](t)(RecAppendFields(base, []string{"c"}, []*np.Array{i(3)}, []dtypes.Descriptor{desc(dtypes.Parse("i4"))})).NDArray())
	_, e = AppendFields(b, []string{"c", "d"}, []Array{{Data: i(1)}}, nil, Options{})
	refcheck.Error(t, "rf_append_bad_names", "ValueError", e, np.ErrShape)
	os := schema(t, field("id", np.Int32), field("v", np.Float32), field("new", np.Int16))
	other := fromRows(t, os, []any{3, 30.5, 4}, []any{4, 40.5, 5})
	defaults := map[string]any{"new": -7}
	checkMasked(t, "rf_stack_arrays", ma(StackArrays([]Array{b, {Data: other}}, true, Options{UseMask: true, Defaults: defaults})))
	checkRecord(t, "rf_stack_filled", ma(StackArrays([]Array{b, {Data: other}}, true, Options{Defaults: defaults})).Data)
	_, e = StackArrays([]Array{b, {Data: other}}, false, Options{})
	refcheck.Error(t, "rf_stack_bad_dtype", "TypeError", e, np.ErrDType)
	dup := fromRows(t, s, []any{3, 30}, []any{1, 10}, []any{1, 11}, []any{2, 20}, []any{3, 31}, []any{5, 50})
	mask := fromRows(t, refcheck.RequireValue[*records.Schema](t)(s.MaskSchema()), []any{true, false}, []any{false, false}, []any{false, false}, []any{false, false}, []any{true, false}, []any{false, false})
	for _, ignore := range []bool{true, false} {
		n := 0
		if ignore {
			n = 1
		}
		out, idx, e := FindDuplicates(Array{dup, mask}, []string{"id"}, ignore)
		if e != nil {
			t.Fatal(e)
		}
		// Like NumPy .filled(), expose the default fill for masked duplicates.
		if e = fillMissing(out, nil, nil); e != nil {
			t.Fatal(e)
		}
		checkMasked(t, "rf_dup_"+strconv.Itoa(n), out)
		refcheck.Check(t, "rf_dup_index_"+strconv.Itoa(n), idx)
	}
	right := fromRows(t, os, []any{3, 30.5, 4}, []any{2, 40.5, 5})
	defaults = map[string]any{"v1": -7, "v2": -8, "new": -9}
	for _, kind := range []string{"inner", "outer", "leftouter"} {
		checkMasked(t, "rf_join_"+kind, ma(JoinBy([]string{"id"}, b, Array{Data: right}, kind, "1", "2", Options{UseMask: true, Defaults: defaults})))
	}
	checkRecord(t, "rf_rec_join", refcheck.RequireValue[*records.Array](t)(RecJoin([]string{"id"}, base, right, "outer", "1", "2", defaults)).NDArray())
	for _, tc := range []struct{ name, key, mode, p1, p2 string }{
		{"missing", "not_here", "inner", "1", "2"}, {"bad_mode", "id", "full", "1", "2"}, {"collision", "id", "inner", "", ""}} {
		_, e := JoinBy([]string{tc.key}, b, Array{Data: right}, tc.mode, tc.p1, tc.p2, Options{})
		refcheck.Error(t, "rf_join_"+tc.name, "ValueError", e, np.ErrValue)
	}
	l2 := fromRows(t, schema(t, field("j", np.Int16), field("i", np.Int32), field("x", np.Int16)), []any{2, 1, 8}, []any{1, 2, 9}, []any{1, 1, 7})
	r2 := fromRows(t, schema(t, field("i", np.Int32), field("j", np.Int16), field("y", np.Int16)), []any{1, 2, 15}, []any{1, 1, 16}, []any{2, 2, 17})
	checkMasked(t, "rf_join_multiple", ma(JoinBy([]string{"i", "j"}, Array{Data: l2}, Array{Data: r2}, "outer", "1", "2", Options{UseMask: true, Defaults: map[string]any{"x": -1, "y": -2}})))
}
