package np_test

import (
	"encoding/binary"
	"errors"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
	"os"
	"path/filepath"
	"testing"
)

func TestDescriptorMemMapNumPy(t *testing.T) {
	m := refcheck.Require(t)
	open := refcheck.RequireValue[*np.MappedArray](t)
	for _, p := range []struct {
		name   string
		desc   np.FlexibleDType
		values []any
	}{
		{"unicode", np.FixedUnicode(4, binary.BigEndian), []any{"x", "é", "中", "", "a\x00b", "1234"}},
		{"bytes", np.FixedBytes(3), []any{np.BytesScalar("x"), np.BytesScalar("\xff"), np.BytesScalar("ab"), np.BytesScalar(""), np.BytesScalar("xy"), np.BytesScalar("z")}},
		{"void", np.FixedVoid(3), []any{np.VoidScalar("abc"), np.VoidScalar("123"), np.VoidScalar("\xffxx"), np.VoidScalar("\x00\x00\x00"), np.VoidScalar("xyz"), np.VoidScalar("z\x00\x00")}},
	} {
		t.Run(p.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "map.bin")
			opt := np.MemMapOptions{DType: p.desc.Kind, Flexible: &p.desc, Shape: []int{2, 3}, Mode: "w+", Offset: 7, Order: "F"}
			a := open(np.MemMap(path, opt))
			defer a.Close()
			for i, v := range p.values {
				if e := a.SetValue(v, i/3, i%3); e != nil {
					t.Fatal(e)
				}
			}
			rev := m(a.Slice(np.Slice{}, np.Reverse()))
			if e := rev.SetValue(p.values[0], 0, 0); e != nil {
				t.Fatal(e)
			}
			if e := a.Flush(); e != nil {
				t.Fatal(e)
			}
			refcheck.CheckFlexible(t, "descriptor_map_"+p.name, a.Array, true)
			refcheck.Bytes(t, "descriptor_map_"+p.name+"_file", refcheck.RequireValue[[]byte](t)(os.ReadFile(path)))
			if e := a.Close(); e != nil {
				t.Fatal(e)
			}
			if _, e := rev.AtValue(0, 0); !errors.Is(e, np.ErrClosedMapping) {
				t.Fatalf("closed view: %v", e)
			}
			if _, e := rev.ToBytes("C"); !errors.Is(e, np.ErrClosedMapping) {
				t.Fatalf("closed serialization: %v", e)
			}
			if _, e := rev.CastStrings(false); !errors.Is(e, np.ErrDType) {
				t.Fatalf("closed string conversion: %v", e)
			}
		})
	}
	for _, p := range []struct {
		name  string
		dtype np.DType
		unit  string
	}{
		{"date", np.DateTime64, "D"}, {"delta", np.TimeDelta64, "3h"},
	} {
		t.Run(p.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "map.bin")
			a := open(np.MemMap(path, np.MemMapOptions{DType: p.dtype, DateUnit: p.unit, Mode: "w+", Offset: 5, Order: "F", Shape: []int{2, 3}, ByteOrder: binary.BigEndian}))
			defer a.Close()
			x := m(a.DateTicks())
			for i := range 6 {
				if e := x.SetValue(int64(i+1), i/3, i%3); e != nil {
					t.Fatal(e)
				}
			}
			rev := m(x.Slice(np.Slice{}, np.Reverse()))
			if e := rev.SetValue(int64(-9), 0, 0); e != nil {
				t.Fatal(e)
			}
			if e := a.Flush(); e != nil {
				t.Fatal(e)
			}
			refcheck.Check(t, "descriptor_map_"+p.name, m(x.Copy()))
			refcheck.Bytes(t, "descriptor_map_"+p.name+"_file", refcheck.RequireValue[[]byte](t)(os.ReadFile(path)))
			if e := a.Close(); e != nil {
				t.Fatal(e)
			}
			if _, e := rev.AtValue(0, 0); !errors.Is(e, np.ErrClosedMapping) {
				t.Fatalf("closed temporal view: %v", e)
			}
		})
	}
	t.Run("record", func(t *testing.T) {
		schema := refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{
			{Name: "id", Type: records.Numeric(np.Int16, binary.BigEndian)},
			{Name: "tag", Type: records.FixedBytes(3)},
			{Name: "text", Type: records.FixedUnicode(2, binary.BigEndian)},
		}, false, 0))
		path := filepath.Join(t.TempDir(), "map.bin")
		opt := np.MemMapOptions{DType: np.RecordDType, Schema: schema, Shape: []int{2, 2}, Mode: "w+", Offset: 3, Order: "F"}
		a := open(np.MemMap(path, opt))
		defer a.Close()
		id, tag, txt := m(a.Field("id")), m(a.Field("tag")), m(a.Field("text"))
		for i := range 4 {
			for _, p := range []struct {
				a *np.Array
				v any
			}{
				{id, int16((i + 1) * 10)}, {tag, np.BytesScalar([]string{"a", "bc", "abc", "z"}[i])}, {txt, []string{"é", "中", "xy", ""}[i]},
			} {
				if e := p.a.SetValue(p.v, i/2, i%2); e != nil {
					t.Fatal(e)
				}
			}
		}
		rev := m(id.Slice(np.Slice{}, np.Reverse()))
		if e := rev.SetValue(int16(-7), 0, 0); e != nil {
			t.Fatal(e)
		}
		if e := a.Flush(); e != nil {
			t.Fatal(e)
		}
		refcheck.Check(t, "descriptor_map_record_id", id)
		s := id.Strides()
		refcheck.Check(t, "descriptor_map_record_id_strides", m(np.WrapSlice([]int64{int64(s[0]), int64(s[1])}, 2)))
		refcheck.CheckFlexible(t, "descriptor_map_record_tag", tag, true)
		refcheck.CheckFlexible(t, "descriptor_map_record_text", txt, true)
		refcheck.Bytes(t, "descriptor_map_record_file", refcheck.RequireValue[[]byte](t)(os.ReadFile(path)))
		if e := a.Close(); e != nil {
			t.Fatal(e)
		}
		if _, e := tag.AtValue(0, 0); !errors.Is(e, np.ErrClosedMapping) {
			t.Fatalf("packed view lifetime: %v", e)
		}
		opt.Mode = "r"
		opt.Shape = nil
		opt.Order = "C"
		r := open(np.MemMap(path, opt))
		defer r.Close()
		refcheck.Error(t, "descriptor_map_readonly", "ValueError", m(r.Field("id")).SetValue(int16(100), 0), np.ErrReadOnly)
		opt.Mode = "c"
		c := open(np.MemMap(path, opt))
		defer c.Close()
		ct := m(c.Field("tag"))
		if e := ct.SetValue(np.BytesScalar("new"), 0); e != nil {
			t.Fatal(e)
		}
		refcheck.CheckFlexible(t, "descriptor_map_private_tag", ct, false)
		if e := c.Flush(); e != nil {
			t.Fatal(e)
		}
		refcheck.Bytes(t, "descriptor_map_private_file", refcheck.RequireValue[[]byte](t)(os.ReadFile(path)))
	})
	// Pointer-managed array slots and missing/invalid descriptors cannot
	// become mapped binary data.
	if _, e := np.MemMap(filepath.Join(t.TempDir(), "not-created"), np.MemMapOptions{DType: np.StringDType, Shape: []int{1}, Mode: "w+"}); !errors.Is(e, np.ErrDType) {
		t.Fatal(e)
	}
}
