package dtypes

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
)

func TestNestedSubarrayAndLayoutNumPy(t *testing.T) {
	m := refcheck.Require(t)
	dm := refcheck.RequireValue[Descriptor](t)
	var describe func(Descriptor) any
	describe = func(d Descriptor) any {
		if base, sh, ok := d.Subdtype(); ok {
			return []any{describe(base), sh, d.ItemSize(), d.Flags(), d.Metadata().HasObject}
		}
		return []any{d.String(), d.ItemSize(), d.Flags(), d.Metadata().HasObject}
	}
	dump := func(n string, v any) {
		t.Helper()
		var b bytes.Buffer
		enc := json.NewEncoder(&b)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(v); err != nil {
			t.Fatal(err)
		}
		refcheck.Text(t, "closure_"+n, strings.TrimSuffix(b.String(), "\n"))
	}
	ints := func(n string, v []int) {
		x := make([]int64, len(v))
		for i, a := range v {
			x[i] = int64(a)
		}
		refcheck.Check(t, "closure_"+n, m(np.FromSlice(x, len(x))))
	}
	raw := func(n string, a *np.Array, logical ...bool) {
		t.Helper()
		name := n + "_bytes"
		if len(logical) > 0 && logical[0] {
			name = n + "_logical_bytes"
		}
		refcheck.Bytes(t, "closure_"+name, refcheck.RequireValue[[]byte](t)(a.ToBytes("C")))
		ints(n+"_shape", a.Shape())
		ints(n+"_strides", a.Strides())
		refcheck.Text(t, "closure_"+n+"_dtype", dm(FromArray(a)).String())
	}
	sub := func(base Descriptor, sh ...int) Descriptor { return dm(Subarray(base, sh...)) }
	d := sub(sub(dm(Parse(">i2")), 2), 3)
	dump("nested_dtype", describe(d))
	raw("nested_zero", m(d.Zeros(2, 3)))
	ar := func(n int, shape ...int) *np.Array {
		return m(m(m(np.ArangeInt64(0, int64(n), 1)).Astype(np.Int16)).Reshape(shape...))
	}
	a := m(ar(6, 2, 3).Transpose())
	fLike := func(a *np.Array, o string) bool {
		return o == "F" || (o == "K" || o == "A") && a.IsFContiguous() && !a.IsCContiguous()
	}
	for _, o := range []string{"K", "A", "C", "F"} {
		raw("nested_cast_"+o, m(d.CastWithOptions(a, np.CastOptions{Order: o})), fLike(a, o))
	}
	e := sub(sub(dm(Parse("f4")), 2), 3)
	promoted := dm(Promote(d, e))
	var semantic func(Descriptor) any
	semantic = func(d Descriptor) any {
		if b, sh, ok := d.Subdtype(); ok {
			return []any{semantic(b), sh, d.Flags(), d.Metadata().HasObject}
		}
		return describe(d)
	}
	dump("nested_promote_semantic", semantic(promoted))
	if promoted.ItemSize() != 24 || promoted.Base().ItemSize() != 8 {
		t.Fatal("promotion must allocate for widened base")
	}
	flags := []bool{}
	for _, rule := range []string{"no", "equiv", "safe", "same_kind", "unsafe"} {
		flags = append(flags, refcheck.RequireValue[bool](t)(CanCast(d, e, rule)))
	}
	refcheck.Check(t, "closure_nested_cast_predicate", m(np.FromSlice(flags, len(flags))))
	sm := refcheck.RequireValue[*records.Schema](t)
	rt := refcheck.RequireValue[records.Type](t)(d.RecordType())
	rd := dm(New("VoidDType", Options{Schema: sm(records.NewSchema([]records.FieldSpec{{Name: "x", Type: rt}}, false, 0))}))
	flat := dm(New("VoidDType", Options{Schema: sm(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Subarray(records.Numeric(np.Int16, binary.BigEndian), 3, 2)}}, false, 0))}))
	dump("nested_record_descr", refcheck.RequireValue[[]any](t)(rd.Descr()))
	dump("nested_record_swapped", refcheck.RequireValue[[]any](t)(dm(rd.NewByteOrder("S")).Descr()))
	f := refcheck.RequireValue[FieldDescription](t)(rd.Field("x"))
	dump("nested_record_field", describe(f.DType))
	flags = nil
	for _, rule := range []string{"no", "equiv", "safe", "same_kind", "unsafe"} {
		flags = append(flags, refcheck.RequireValue[bool](t)(CanCast(rd, flat, rule)))
	}
	refcheck.Check(t, "closure_nested_record_predicate", m(np.FromSlice(flags, len(flags))))
	rows := m(rd.Zeros(2))
	field := m(rows.Field("x"))
	if err := field.Assign(nil, ar(12, 2, 3, 2)); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "closure_nested_record_values", m(field.Astype(np.Int16)))
	var npy bytes.Buffer
	if err := np.SaveNPY(&npy, rows); err != nil {
		t.Fatal(err)
	}
	if dir := os.Getenv("NUMPYGO_INTERCHANGE_DIR"); dir != "" {
		if err := os.WriteFile(filepath.Join(dir, "closure_nested.npy"), npy.Bytes(), 0600); err != nil {
			t.Fatal(err)
		}
	}
	back := m(records.LoadNPY(bytes.NewReader(npy.Bytes())))
	if !dm(FromArray(back)).Equal(rd) {
		t.Fatal("nested record dtype lost on NPY roundtrip")
	}
	refcheck.Check(t, "closure_nested_record_values", m(m(back.Field("x")).Astype(np.Int16)))
	od := sub(dm(Parse("O")), 2)
	dump("object_subdtype", describe(od))
	refcheck.Check(t, "closure_object_sub_zero", m(m(od.Zeros(2, 3)).Astype(np.Int64)))
	refcheck.Check(t, "closure_object_sub_cast", m(m(od.Cast(ar(3, 3), "unsafe")).Astype(np.Int64)))
	td := sub(dm(Parse("T")), 2)
	dump("string_sub_dtype", describe(td))
	dump("string_sub_zero", refcheck.RequireValue[any](t)(m(td.Zeros(2)).ToList()))
	dump("string_sub_cast", refcheck.RequireValue[any](t)(m(td.Cast(ar(3, 3), "unsafe")).ToList()))
	_, err := dm(Parse("T")).RecordType()
	refcheck.Error(t, "closure_string_field", "TypeError", err, np.ErrDType)
	a = m(dm(Parse(">i2")).Cast(ar(24, 2, 3, 4), "unsafe"))
	raw("copy_default", m(np.Copy(m(a.Transpose(2, 0, 1)), "")))
	for _, tc := range []struct {
		name string
		x    *np.Array
	}{
		{"c", a}, {"f", m(a.CopyWithOrder("F"))}, {"perm", m(a.Transpose(2, 0, 1))},
		{"rev", m(a.Slice(np.Reverse(), np.Slice{}, np.Slice{Step: np.Int(2)}))},
	} {
		for _, o := range []string{"K", "A", "C", "F"} {
			t.Run(tc.name+"/"+o, func(t *testing.T) {
				x := m(tc.x.CopyWithOrder(o))
				raw("copy_"+tc.name+"_"+o, x)
				if x.Base() != nil || np.SharesMemory(x, tc.x) {
					t.Fatal("copy not independent")
				}
				raw("cast_"+tc.name+"_"+o, m(dm(Parse(">i4")).CastWithOptions(tc.x, np.CastOptions{Order: o})))
				raw("subcast_"+tc.name+"_"+o, m(sub(dm(Parse(">i4")), 2, 3).CastWithOptions(tc.x, np.CastOptions{Order: o})), fLike(tc.x, o))
				y := m(dm(FromArray(tc.x)).CastWithOptions(tc.x, np.CastOptions{Order: o, NoCopy: true}))
				refcheck.Check(t, "closure_nocopy_"+tc.name+"_"+o, np.ScalarOf(y == tc.x))
			})
		}
	}
}

func TestManagedDescriptorNumPy(t *testing.T) {
	m := refcheck.Require(t)
	sm := refcheck.RequireValue[*records.Schema](t)
	dm := refcheck.RequireValue[Descriptor](t)
	inner := sm(records.NewSchema([]records.FieldSpec{
		{Name: "value", Type: records.Numeric(np.Float32, nil)}, {Name: "obj", Type: records.Object()},
	}, false, 0))
	s := sm(records.NewSchema([]records.FieldSpec{
		{Name: "id", Type: records.Numeric(np.Int16, binary.BigEndian)}, {Name: "payload", Type: records.Object()},
		{Name: "tags", Type: records.Subarray(records.Object(), 2)}, {Name: "inner", Type: records.Struct(inner)},
	}, false, 0))
	d := dm(New("VoidDType", Options{Schema: s}))
	meta := d.Metadata()
	has := 0
	if meta.HasObject {
		has = 1
	}
	values := []int64{int64(meta.ItemSize), int64(meta.Alignment), int64(d.Flags()), int64(has)}
	for _, f := range s.Fields() {
		values = append(values, int64(f.Offset))
	}
	refcheck.Check(t, "closure_managed_meta", m(np.FromSlice(values, len(values))))
	descr := refcheck.RequireValue[[]any](t)(d.Descr())
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if e := enc.Encode(descr); e != nil {
		t.Fatal(e)
	}
	refcheck.Text(t, "closure_managed_descr", strings.TrimSuffix(b.String(), "\n"))
}
