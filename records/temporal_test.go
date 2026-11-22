package records

import (
	"bytes"
	"encoding/binary"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"os"
	"path/filepath"
	"testing"
)

func TestStructuredTemporalNumPy(t *testing.T) {
	m := refcheck.Require(t)
	ms := refcheck.RequireValue[*Schema](t)
	mt := refcheck.RequireValue[Type](t)
	sub := ms(NewSchema([]FieldSpec{{Name: "stamp", Type: mt(Temporal(np.DateTime64, "ns", binary.BigEndian))}}, false, 0))
	schema := ms(NewSchema([]FieldSpec{
		{Name: "tag", Type: Numeric(np.Int8, nil)},
		{Name: "when", Type: mt(Temporal(np.DateTime64, "D", binary.BigEndian))},
		{Name: "delta", Type: Subarray(mt(Temporal(np.TimeDelta64, "3h", nil)), 2)},
		{Name: "sub", Type: Struct(sub)},
	}, false, 0))
	a := m(np.ZerosRecords(schema, 2, 2))
	tag := m(a.Field("tag"))
	when := m(a.Field("when"))
	delta := m(a.Field("delta"))
	stamp := m(a.Field("sub", "stamp"))
	dates := m(when.DateTicks())
	deltas := m(delta.DateTicks())
	stamps := m(stamp.DateTicks())
	for _, p := range []struct{ dst, src *np.Array }{
		{tag, m(np.FromSlice([]int8{1, 2, 3, 4}, 2, 2))},
		{dates, m(np.FromSlice([]int64{0, 2, -4, np.NaT}, 2, 2))},
		{deltas, m(np.FromSlice([]int64{-3, -2, -1, 0, 1, 2, 3, 4}, 2, 2, 2))},
		{stamps, m(np.FromSlice([]int64{1234, 5678, -9000, 0}, 2, 2))},
	} {
		if e := p.dst.Assign(nil, p.src); e != nil {
			t.Fatal(e)
		}
	}
	rev := m(when.Slice(np.Slice{}, np.Reverse()))
	if e := rev.SetValue("1970-01-04", 0, 0); e != nil {
		t.Fatal(e)
	}
	r := refcheck.RequireValue[*Array](t)(FromNDArray(a))
	c := refcheck.RequireValue[*Column](t)(r.Field("delta"))
	if e := c.Set([]any{"NaT", "2"}, 1, 0); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "rectime_when", dates)
	refcheck.Check(t, "rectime_delta", deltas)
	refcheck.Check(t, "rectime_stamp", stamps)
	for _, p := range []struct {
		n string
		a *np.Array
	}{{"rectime_stride", when}, {"rectime_delta_stride", delta}} {
		s := p.a.Strides()
		v := make([]int64, len(s))
		for i, x := range s {
			v[i] = int64(x)
		}
		refcheck.Check(t, p.n, m(np.FromSlice(v, len(v))))
	}
	refcheck.Bytes(t, "rectime_bytes", refcheck.RequireValue[[]byte](t)(a.ToBytes("C")))
	file := refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "rectime_file")))
	loaded := m(LoadNPY(bytes.NewReader(file)))
	if !loaded.StructuredSchema().EqualSchema(schema) {
		t.Fatal("temporal descriptor changed in NPY")
	}
	refcheck.Bytes(t, "rectime_bytes", refcheck.RequireValue[[]byte](t)(loaded.ToBytes("C")))
	var b bytes.Buffer
	if e := np.SaveNPY(&b, a); e != nil {
		t.Fatal(e)
	}
	back := m(LoadNPY(bytes.NewReader(b.Bytes())))
	if !back.StructuredSchema().EqualSchema(schema) {
		t.Fatal("Go NPY descriptor changed")
	}
	if dir := os.Getenv("NUMPYGO_INTERCHANGE_DIR"); dir != "" {
		if e := os.WriteFile(filepath.Join(dir, "structured_temporal.npy"), b.Bytes(), 0600); e != nil {
			t.Fatal(e)
		}
	}
	create := func(dateUnit, deltaUnit string, order binary.ByteOrder, rows [][]any) *np.Array {
		s := ms(NewSchema([]FieldSpec{{Name: "when", Type: mt(Temporal(np.DateTime64, dateUnit, order))}, {Name: "delta", Type: mt(Temporal(np.TimeDelta64, deltaUnit, nil))}}, false, 0))
		return refcheck.RequireValue[*Array](t)(FromRows(s, rows, len(rows))).NDArray()
	}
	x := create("D", "3h", binary.BigEndian, [][]any{{"1970-01-02", 2}, {"1970-01-03", -1}})
	y := create("3h", "h", nil, [][]any{{"1970-01-02T00", 6}, {"1970-01-03T00", -3}})
	for _, p := range []struct {
		n string
		a *np.Array
	}{
		{"concat", m(np.Concatenate([]*np.Array{x, y}, 0))},
		{"where", m(np.Where(m(np.FromSlice([]bool{true, false}, 2)), x, y))},
	} {
		refcheck.Bytes(t, "rectime_"+p.n, refcheck.RequireValue[[]byte](t)(p.a.ToBytes("C")))
		for _, f := range []string{"when", "delta"} {
			refcheck.Check(t, "rectime_"+p.n+"_"+f, m(m(p.a.Field(f)).DateTicks()))
		}
	}
	refcheck.Check(t, "rectime_equal", m(np.Equal(x, y)))
	py := ms(NewSchema([]FieldSpec{{Name: "x", Type: mt(Temporal(np.TimeDelta64, "Y", nil))}}, false, 0))
	pd := ms(NewSchema([]FieldSpec{{Name: "x", Type: mt(Temporal(np.TimeDelta64, "D", nil))}}, false, 0))
	_, e := np.PromoteRecordSchemas(py, pd)
	refcheck.Error(t, "rectime_bad_promotion", "TypeError", e, np.ErrDType)
}
