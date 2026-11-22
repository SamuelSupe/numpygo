package records

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func recordLayout(t *testing.T, name string, a *np.Array) {
	t.Helper()
	s := a.StructuredSchema().(*Schema)
	vals := []int64{int64(s.ItemSize())}
	for _, f := range s.Fields() {
		vals = append(vals, int64(f.Offset))
	}
	refcheck.Check(t, "recwide_"+name+"_layout", refcheck.Require(t)(np.WrapSlice(vals, len(vals))))
}
func TestPromotedRecordsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	ms := refcheck.RequireValue[*Schema](t)
	rows := func(s *Schema, v [][]any) *np.Array {
		return refcheck.RequireValue[*Array](t)(FromRows(s, v, len(v))).NDArray()
	}
	schema := func(fields ...FieldSpec) *Schema { return ms(NewSchema(fields, false, 0)) }
	subA := schema(FieldSpec{Name: "x", Type: Numeric(np.Float32, binary.LittleEndian)})
	subB := schema(FieldSpec{Name: "x", Type: Numeric(np.Float64, binary.LittleEndian)})
	a := rows(schema(FieldSpec{Name: "id", Type: Numeric(np.Int16, binary.BigEndian)}, FieldSpec{Name: "tag", Type: FixedBytes(3)}, FieldSpec{Name: "v", Type: Subarray(Numeric(np.Int16, binary.LittleEndian), 2)}, FieldSpec{Name: "sub", Type: Struct(subA)}),
		[][]any{{-4, "abc", []int{2, 3}, []any{1.5}}, {9, "def", []int{4, 5}, []any{2.}}})
	b := rows(schema(FieldSpec{Name: "id", Type: Numeric(np.Int64, binary.LittleEndian)}, FieldSpec{Name: "tag", Type: FixedBytes(5)}, FieldSpec{Name: "v", Type: Subarray(Numeric(np.Int16, binary.LittleEndian), 2)}, FieldSpec{Name: "sub", Type: Struct(subB)}),
		[][]any{{-4, "abc", []int{2, 3}, []any{1.5}}, {int64(1) << 40, "hello", []int{-2, -3}, []any{-5.25}}})
	mask := m(np.FromSlice([]bool{true, false}, 2))
	cases := func(name string, a, b *np.Array) {
		t.Helper()
		s := refcheck.RequireValue[np.StructuredSchema](t)(np.PromoteRecordSchemas(a.StructuredSchema(), b.StructuredSchema()))
		for _, v := range []struct {
			label string
			value *np.Array
		}{
			{"concat", m(np.Concatenate([]*np.Array{a, b}, 0))},
			{"where", m(np.Where(mask, a, b))},
			{"cast", m(a.CastRecords(s))},
		} {
			n := name + "_" + v.label
			recordLayout(t, n, v.value)
			refcheck.Bytes(t, "recwide_"+n, refcheck.RequireValue[[]byte](t)(v.value.ToBytes("C")))
		}
		refcheck.Check(t, "recwide_"+name+"_equal", m(np.Equal(a, b)))
	}
	cases("nested", a, b)
	a = rows(schema(FieldSpec{Name: "id", Type: Numeric(np.Int8, nil)}, FieldSpec{Name: "tag", Type: FixedBytes(3)}), [][]any{{1, "abc"}, {2, "def"}})
	b = rows(schema(FieldSpec{Name: "id", Type: Numeric(np.Int32, nil)}, FieldSpec{Name: "tag", Type: FixedUnicode(2, nil)}), [][]any{{1, "abc"}, {70000, "中文"}})
	cases("text", a, b)
	a = rows(schema(FieldSpec{Name: "id", Type: Numeric(np.Int8, nil)}, FieldSpec{Name: "x", Type: Numeric(np.Float32, nil)}), [][]any{{1, 3.5}, {2, 4.5}})
	b = rows(ms(NewSchema([]FieldSpec{{Name: "id", Type: Numeric(np.Int32, nil)}, {Name: "x", Type: Numeric(np.Float64, nil)}}, true, 0)),
		[][]any{{1, 3.5}, {1000, 6.5}})
	s := refcheck.RequireValue[np.StructuredSchema](t)(np.PromoteRecordSchemas(a.StructuredSchema(), b.StructuredSchema()))
	for _, v := range []struct {
		label string
		value *np.Array
	}{
		{"concat", m(np.Concatenate([]*np.Array{a, b}, 0))}, {"where", m(np.Where(mask, a, b))}, {"cast", m(a.CastRecords(s))},
	} {
		name := "aligned_" + v.label
		recordLayout(t, name, v.value)
		refcheck.Check(t, "recwide_"+name+"_id", m(v.value.Field("id")))
		refcheck.Check(t, "recwide_"+name+"_x", m(v.value.Field("x")))
	}
	for _, tc := range []struct {
		label string
		a, b  FieldSpec
	}{
		{"names", FieldSpec{Name: "a", Type: Numeric(np.Int32, nil)}, FieldSpec{Name: "b", Type: Numeric(np.Int32, nil)}},
		{"shape", FieldSpec{Name: "a", Type: Subarray(Numeric(np.Int16, nil), 2)}, FieldSpec{Name: "a", Type: Subarray(Numeric(np.Int16, nil), 3)}},
		{"void", FieldSpec{Name: "a", Type: Opaque(2)}, FieldSpec{Name: "a", Type: Opaque(3)}},
	} {
		x := m(np.ZerosRecords(schema(tc.a), 1))
		y := m(np.ZerosRecords(schema(tc.b), 1))
		_, e := np.Concatenate([]*np.Array{x, y}, 0)
		refcheck.Error(t, "recwide_bad_"+tc.label, "DTypePromotionError", e, np.ErrDType)
	}
}
func TestStructuredNPYNumPy(t *testing.T) {
	m := refcheck.Require(t)
	for _, name := range []string{"structured_packed", "structured_fortran"} {
		t.Run(name, func(t *testing.T) {
			data := refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "recwide_"+name+"_file")))
			a := m(LoadNPY(bytes.NewReader(data)))
			recordLayout(t, name, a)
			raw := refcheck.RequireValue[[]byte](t)(a.ToBytes("C"))
			refcheck.Bytes(t, "recwide_"+name, raw)
			strides := a.Strides()
			v := make([]int64, len(strides))
			for i, s := range strides {
				v[i] = int64(s)
			}
			refcheck.Check(t, "recwide_"+name+"_stride", m(np.WrapSlice(v, len(v))))
			if name == "structured_packed" {
				if _, e := a.Field("温度'"); e != nil {
					t.Fatal(e)
				}
			} else if !a.IsFContiguous() || a.IsCContiguous() {
				t.Fatal("Fortran order was not retained")
			}
			var buf bytes.Buffer
			if e := np.SaveNPY(&buf, a); e != nil {
				t.Fatal(e)
			}
			b := m(LoadNPY(bytes.NewReader(buf.Bytes())))
			if !a.StructuredSchema().EqualSchema(b.StructuredSchema()) {
				t.Fatal("descriptor changed")
			}
			refcheck.Bytes(t, "recwide_"+name, refcheck.RequireValue[[]byte](t)(b.ToBytes("C")))
			if _, e := LoadNPY(bytes.NewReader(data[:len(data)-1])); e != io.ErrUnexpectedEOF {
				t.Fatalf("truncated NPY: %v", e)
			}
			if path := os.Getenv("NUMPYGO_INTERCHANGE_DIR"); path != "" {
				if e := os.WriteFile(filepath.Join(path, name+".npy"), buf.Bytes(), 0600); e != nil {
					t.Fatal(e)
				}
			}
		})
	}
	o1, o2 := 0, 1
	s := refcheck.RequireValue[*Schema](t)(NewSchema([]FieldSpec{{Name: "a", Type: Numeric(np.Int32, nil), Offset: &o1}, {Name: "b", Type: Numeric(np.Int16, nil), Offset: &o2}}, false, 4))
	e := np.SaveNPY(io.Discard, m(np.ZerosRecords(s, 1)))
	refcheck.Error(t, "recwide_overlap_npy", "ValueError", e, np.ErrDType)
}
