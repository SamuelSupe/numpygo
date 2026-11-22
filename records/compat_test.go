package records

import (
	"encoding/binary"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"regexp"
	"strings"
	"testing"
)

func TestFlexibleScalars(t *testing.T) {
	m := refcheck.Require(t)
	b := ByteString([]byte("a\x00b\x00"))
	s := UnicodeString("中é\x00")
	refcheck.Bytes(t, "final_record_bytes", []byte(b))
	refcheck.Check(t, "final_record_bytes_len", np.ScalarOf(int64(b.Len())))
	refcheck.Text(t, "final_record_unicode", s.String())
	refcheck.Check(t, "final_record_unicode_len", np.ScalarOf(int64(s.Len())))
	refcheck.Bytes(t, "final_record_void", NewVoid([]byte{1, 255, 0}))
	var flags []bool
	for _, v := range []Type{FixedBytes(4), FixedUnicode(3, nil), Opaque(3), Numeric(np.Int64, nil)} {
		flags = append(flags, v.IsCharacter(), v.IsFlexible())
	}
	refcheck.Check(t, "final_record_hierarchy", m(np.WrapSlice(flags, len(flags))))
}
func TestStructuredRecords(t *testing.T) {
	m := refcheck.Require(t)
	schema := refcheck.RequireValue[*Schema](t)
	arr := refcheck.RequireValue[*Array](t)
	col := refcheck.RequireValue[*Column](t)
	r := refcheck.RequireValue[*Record](t)
	value := refcheck.RequireValue[any](t)
	dt := schema(NewSchema([]FieldSpec{{Name: "tag", Type: FixedBytes(3)}, {Name: "n", Type: Numeric(np.Int32, binary.BigEndian)}, {Name: "vec", Type: Subarray(Numeric(np.Float64, binary.LittleEndian), 2)}}, false, 0))
	a := arr(FromRows(dt, [][]any{{"one", int32(10), []float64{1.5, 2.5}}, {"two", int32(-20), []float64{3.5, 4.5}}}, 2))
	view := arr(a.Slice(np.Reverse()))
	if e := col(view.Field("n")).Set(int32(70), 0); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "final_record_layout_bytes", a.Bytes())
	offs := []int64{}
	for _, f := range dt.Fields() {
		offs = append(offs, int64(f.Offset))
	}
	offs = append(offs, int64(dt.ItemSize()), int64(dt.Alignment()))
	refcheck.Check(t, "final_record_layout_offsets", m(np.WrapSlice(offs, len(offs))))
	r0 := r(a.At(0))
	r1 := r(a.At(1))
	refcheck.Check(t, "final_record_column", value(r0.Field("vec")).(*np.Array))
	refcheck.Check(t, "final_record_scalar", np.ScalarOf(value(r1.Field("n")).(int32)))
	refcheck.Text(t, "final_record_scalar_tag", string(value(r1.Field("tag")).(Bytes)))
	refcheck.Check(t, "final_record_is_record", np.ScalarOf(r0.Schema() == dt))
	nested := schema(NewSchema([]FieldSpec{{Name: "x", Type: Numeric(np.Int16, nil)}, {Name: "tag", Type: FixedBytes(2)}}, true, 0))
	aligned := schema(NewSchema([]FieldSpec{{Name: "id", Type: Numeric(np.Int32, nil)}, {Name: "pos", Type: Subarray(Numeric(np.Float64, nil), 2)}, {Name: "name", Type: FixedUnicode(3, nil)}, {Name: "nested", Type: Struct(nested)}}, true, 0))
	offs = nil
	for _, f := range aligned.Fields() {
		offs = append(offs, int64(f.Offset))
	}
	offs = append(offs, int64(aligned.ItemSize()), int64(aligned.Alignment()))
	refcheck.Check(t, "final_record_aligned_offsets", m(np.WrapSlice(offs, len(offs))))
	mask := schema(aligned.MaskSchema())
	offs = nil
	for _, f := range mask.Fields() {
		offs = append(offs, int64(f.Offset))
	}
	offs = append(offs, int64(mask.ItemSize()))
	refcheck.Check(t, "final_record_mask_offsets", m(np.WrapSlice(offs, len(offs))))
	re := schema(NewSchema([]FieldSpec{{Name: "name", Type: FixedBytes(5)}, {Name: "count", Type: Numeric(np.Int32, nil)}}, false, 0))
	result := arr(FromRegex(strings.NewReader("item=alpha count=17\nignored\nitem=beta count=-3\n"), regexp.MustCompile(`item=(\w+) count=(-?\d+)`), re))
	refcheck.Bytes(t, "final_record_fromregex", result.Bytes())
}
