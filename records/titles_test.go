package records

import (
	"bytes"
	"encoding/binary"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTitledStructuredNPYNumPy(t *testing.T) {
	m := refcheck.Require(t)
	ms := refcheck.RequireValue[*Schema](t)
	spec := []FieldSpec{
		{Name: "speed", Title: "速度", Type: Numeric(np.Int16, binary.BigEndian)},
		{Name: "v", Title: 7, Type: Subarray(Numeric(np.Float64, binary.LittleEndian), 2)},
		{Name: "tag", Title: "other", Type: FixedBytes(3)},
	}
	schema := ms(NewSchema(spec, false, 0))
	refcheck.Check(t, "titles_itemsize", np.ScalarOf(int64(schema.ItemSize())))
	refcheck.Text(t, "titles_names", strings.Join(schema.Names(), "|"))
	a := m(np.ZerosRecords(schema, 2, 2))
	speed := m(a.Field("速度"))
	tag := m(a.Field("other"))
	vec := m(a.Field("v"))
	if !np.SharesMemory(speed, m(a.Field("speed"))) {
		t.Fatal("title not an alias")
	}
	for i := range 4 {
		if e := speed.SetValue(int16(i+1), i/2, i%2); e != nil {
			t.Fatal(e)
		}
		if e := tag.SetValue(np.BytesScalar([]string{"x", "ab", "abc", "z"}[i]), i/2, i%2); e != nil {
			t.Fatal(e)
		}
		for j := range 2 {
			if e := vec.SetValue(float64(i*2+j)/2, i/2, i%2, j); e != nil {
				t.Fatal(e)
			}
		}
	}
	rev := m(speed.Slice(np.Slice{}, np.Reverse()))
	if e := rev.SetValue(int16(-9), 0, 0); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "titles_speed", m(a.Field("speed")))
	refcheck.Check(t, "titles_v", vec)
	refcheck.CheckFlexible(t, "titles_tag", tag, false)
	refcheck.Bytes(t, "titles_bytes", refcheck.RequireValue[[]byte](t)(a.ToBytes("C")))
	// Read NumPy's independently written titled NPY, then write the Go
	// equivalent for the reciprocal Python reader.
	data := refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "titles_npy_file")))
	loaded := m(LoadNPY(bytes.NewReader(data)))
	if !loaded.StructuredSchema().EqualSchema(schema) {
		t.Fatal("NumPy title descriptor lost")
	}
	refcheck.Bytes(t, "titles_bytes", refcheck.RequireValue[[]byte](t)(loaded.ToBytes("C")))
	var buf bytes.Buffer
	if e := np.SaveNPY(&buf, a); e != nil {
		t.Fatal(e)
	}
	round := m(LoadNPY(bytes.NewReader(buf.Bytes())))
	if !round.StructuredSchema().EqualSchema(schema) {
		t.Fatal("Go titled descriptor lost")
	}
	if dir := os.Getenv("NUMPYGO_INTERCHANGE_DIR"); dir != "" {
		if e := os.WriteFile(filepath.Join(dir, "structured_titles.npy"), buf.Bytes(), 0600); e != nil {
			t.Fatal(e)
		}
	}
	x := append([]FieldSpec(nil), spec...)
	x[0].Type = Numeric(np.Int32, nil)
	x[2].Type = FixedBytes(5)
	promote := refcheck.RequireValue[np.StructuredSchema](t)(np.PromoteRecordSchemas(schema, ms(NewSchema(x, false, 0))))
	refcheck.Check(t, "titles_promote_itemsize", np.ScalarOf(int64(promote.ItemSize())))
	if f, e := promote.(*Schema).Field("速度"); e != nil || f.Title != "速度" {
		t.Fatal("promoted title lost")
	}
	x[0].Title = "different"
	_, e := np.PromoteRecordSchemas(schema, ms(NewSchema(x, false, 0)))
	refcheck.Error(t, "titles_mismatch", "DTypePromotionError", e, np.ErrDType)
	_, e = NewSchema([]FieldSpec{{Name: "a", Title: "b", Type: Numeric(np.Int16, nil)}, {Name: "b", Type: Numeric(np.Int16, nil)}}, false, 0)
	refcheck.Error(t, "titles_duplicate", "ValueError", e, np.ErrValue)
	_, e = NewSchema([]FieldSpec{{Name: "a", Title: "a", Type: Numeric(np.Int16, nil)}}, false, 0)
	refcheck.Error(t, "titles_self", "ValueError", e, np.ErrValue)
	mask := ms(schema.MaskSchema())
	if f, e := mask.Field("速度"); e != nil || f.Type.DType != np.Bool || f.Title != "速度" {
		t.Fatal("mask title not retained")
	}
	// A nested descriptor with a non-string title and a zero-length alias
	// retains metadata without adding physical fields.
	nestedFile := refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "titles_nested_file")))
	nested := m(LoadNPY(bytes.NewReader(nestedFile)))
	refcheck.Check(t, "titles_nested_ok", m(nested.Field("data", "ok")))
	refcheck.Check(t, "titles_nested_val", m(nested.Field("meta", "")))
	refcheck.CheckFlexible(t, "titles_nested_label", m(nested.Field("标签")), false)
	ns := nested.StructuredSchema().(*Schema)
	field := refcheck.RequireValue[Field](t)(ns.Field("meta"))
	if f, e := field.Type.Schema.Field("ok"); e != nil || f.Title != true {
		t.Fatal("boolean title lost")
	}
	if f, e := field.Type.Schema.Field(""); e != nil || f.Title != "" || f.Name != "val" {
		t.Fatal("empty title lost")
	}
	refcheck.Check(t, "titles_nested_mask_itemsize", np.ScalarOf(int64(ms(ns.MaskSchema()).ItemSize())))
	buf.Reset()
	if e := np.SaveNPY(&buf, nested); e != nil {
		t.Fatal(e)
	}
	round = m(LoadNPY(bytes.NewReader(buf.Bytes())))
	if !ns.EqualSchema(round.StructuredSchema()) {
		t.Fatal("nested title descriptor changed")
	}
	refcheck.Bytes(t, "titles_nested_bytes", refcheck.RequireValue[[]byte](t)(round.ToBytes("C")))
	if dir := os.Getenv("NUMPYGO_INTERCHANGE_DIR"); dir != "" {
		if e := os.WriteFile(filepath.Join(dir, "structured_titles_nested.npy"), buf.Bytes(), 0600); e != nil {
			t.Fatal(e)
		}
	}
}
