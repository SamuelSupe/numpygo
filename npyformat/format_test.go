package npyformat

import (
	"bytes"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestNPYFormatNumPy(t *testing.T) {
	m := refcheck.Require(t)
	md := refcheck.RequireValue[dtypes.Descriptor](t)
	mh := refcheck.RequireValue[Header](t)
	mb := refcheck.RequireValue[[]byte](t)
	for _, tc := range []struct {
		name         string
		major, minor int
	}{{"1", 1, 0}, {"2", 2, 0}, {"other", 42, 255}} {
		raw := mb(Magic(tc.major, tc.minor))
		refcheck.Bytes(t, "format_magic_"+tc.name, raw)
		v := refcheck.RequireValue[Version](t)(ReadMagic(bytes.NewReader(raw)))
		refcheck.Check(t, "format_read_magic_"+tc.name, m(np.FromSlice([]int64{int64(v.Major), int64(v.Minor)}, 2)))
	}
	_, e := Magic(256, 0)
	refcheck.Error(t, "format_magic_invalid", "ValueError", e, np.ErrValue)
	_, e = ReadMagic(bytes.NewBufferString("not_npy!"))
	refcheck.Error(t, "format_read_magic_invalid", "ValueError", e, np.ErrValue)

	d := md(dtypes.Parse(">i4"))
	endian := m(m(d.Cast(m(np.FromSlice([]int32{1, 7, -2, 9, 3, -8}, 3, 2)), "unsafe")).Transpose())
	h := mh(HeaderDataFromArray(endian))
	for i := 1; i <= 2; i++ {
		var b bytes.Buffer
		write, read := WriteArrayHeader1, ReadArrayHeader1
		if i == 2 {
			write, read = WriteArrayHeader2, ReadArrayHeader2
		}
		if e := write(&b, h); e != nil {
			t.Fatal(e)
		}
		name := strconv.Itoa(i)
		refcheck.Bytes(t, "format_header_"+name, b.Bytes())
		p := mh(read(bytes.NewReader(b.Bytes()[8:]), 0))
		shape := []int64{}
		for _, x := range p.Shape {
			shape = append(shape, int64(x))
		}
		refcheck.Check(t, "format_header_shape_"+name, m(np.FromSlice(shape, len(shape))))
		refcheck.Check(t, "format_header_order_"+name, m(np.FromSlice([]bool{p.FortranOrder})))
		refcheck.Text(t, "format_header_dtype_"+name, md(DescrToDType(p.Descriptor)).String())
		_, e := read(bytes.NewReader(b.Bytes()[8:]), 10)
		refcheck.Error(t, "format_header_limit_"+name, "ValueError", e, np.ErrShape)
	}
	for name, spec := range map[string]string{"plain": ">f8", "object": "O", "variable": "T"} {
		v := refcheck.RequireValue[any](t)(DTypeToDescr(md(dtypes.Parse(spec))))
		refcheck.Text(t, "format_descr_"+name, v.(string))
	}
	s := refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{
		{Name: "a", Title: "title", Type: records.Numeric(np.Int16, nil)},
		{Name: "b", Type: records.Subarray(records.Numeric(np.Float64, nil), 2)},
	}, true, 0))
	desc := md(dtypes.New("VoidDType", dtypes.Options{Schema: s}))
	v := refcheck.RequireValue[any](t)(DTypeToDescr(desc))
	var b bytes.Buffer
	if e = WriteArrayHeader1(&b, Header{Descriptor: v, Shape: []int{2}}); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "format_descr_record", b.Bytes())
	rt := md(DescrToDType(v))
	if rt.Schema().Aligned() {
		t.Fatal("NPY does not retain aligned flag")
	}
	back := m(rt.Zeros(2))
	b.Reset()
	if e = WriteArrayHeader1(&b, mh(HeaderDataFromArray(back))); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "format_descr_roundtrip_header", b.Bytes())
	for _, name := range []string{"a", "b"} {
		refcheck.Bytes(t, "format_descr_roundtrip_"+name, mb(m(back.Field(name)).ToBytes("C")))
	}
	refcheck.Check(t, "format_descr_roundtrip_meta", m(np.FromSlice([]int64{int64(rt.ItemSize()), int64(rt.Schema().Alignment()), 0}, 3)))
	refcheck.Text(t, "format_drop_metadata", DropMetadata(md(dtypes.Parse("f4"))).String())
	b.Reset()
	if e = WriteArrayHeader1(&b, h); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "format_header_data_header", b.Bytes())
	refcheck.Bytes(t, "format_header_data", mb(endian.ToBytes("C")))
	for i := 1; i <= 3; i++ {
		b.Reset()
		if e = WriteArray(&b, endian, Version{Major: i}); e != nil {
			t.Fatal(e)
		}
		name := strconv.Itoa(i)
		refcheck.Bytes(t, "format_write_"+name, b.Bytes())
		if i == 2 {
			writeInterchange(t, "format_endian.npy", b.Bytes())
		}
		a := m(ReadArray(bytes.NewReader(b.Bytes()), 0))
		refcheck.Check(t, "format_read_"+name, m(a.Astype(np.Int32)))
		refcheck.Bytes(t, "format_read_raw_"+name, mb(a.ToBytes("F")))
		if a.NativeByteOrder() || !a.IsFContiguous() {
			t.Fatal("file endian or Fortran layout lost")
		}
	}
	s = refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{
		{Name: "时间", Type: records.Numeric(np.Int32, nil)}, {Name: "café", Type: records.Numeric(np.Int16, nil)},
	}, false, 0))
	uni := refcheck.RequireValue[*records.Array](t)(records.FromRows(s, [][]any{{1, -2}, {3, 7}}, 2)).NDArray()
	b.Reset()
	if e = WriteArray(&b, uni, Version{Major: 3}); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "format_unicode", b.Bytes())
	writeInterchange(t, "format_unicode.npy", b.Bytes())
	a := m(ReadArray(bytes.NewReader(b.Bytes()), 0))
	if !a.StructuredSchema().EqualSchema(s) {
		t.Fatal("UTF-8 schema changed")
	}
	b.Reset()
	e = WriteArray(&b, uni, Version{Major: 1})
	refcheck.Error(t, "format_unicode_v1", "UnicodeEncodeError", e, np.ErrValue)
	if e = WriteArray(&b, uni, Version{}); e != nil {
		t.Fatal(e)
	}
	x := refcheck.RequireValue[Version](t)(ReadMagic(bytes.NewReader(b.Bytes())))
	if x.Major != 3 {
		t.Fatal("auto version must handle non-Latin-1")
	}
	long := Header{Descriptor: []any{[]any{strings.Repeat("a", 66000), "|i1"}}, Shape: []int{1}}
	e = WriteArrayHeader1(io.Discard, long)
	refcheck.Error(t, "format_long_v1", "ValueError", e, np.ErrShape)
	b.Reset()
	if e = WriteArrayHeader2(&b, long); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "format_long_v2", m(np.FromSlice([]int64{int64(b.Len())})))
	_, e = ReadArrayHeader2(bytes.NewReader(b.Bytes()[8:]), 0)
	refcheck.Error(t, "format_long_default_limit", "ValueError", e, np.ErrShape)
	rt = md(DescrToDType(mh(ReadArrayHeader2(bytes.NewReader(b.Bytes()[8:]), 100000)).Descriptor))
	refcheck.Check(t, "format_long_read", m(np.FromSlice([]int64{int64(len(rt.Schema().Names()[0]))})))
	obj := m(np.FromObjects([]any{int64(1)}, 1))
	e = WriteArray(io.Discard, obj, Version{})
	refcheck.Error(t, "format_write_object", "ValueError", e, np.ErrDType)
	_, e = ReadArray(bytes.NewReader(mb(np.Values[uint8](refcheck.Input(t, "format_object_header")))), 0)
	refcheck.Error(t, "format_read_object", "ValueError", e, np.ErrDType)
}

func writeInterchange(t *testing.T, name string, data []byte) {
	t.Helper()
	if dir := os.Getenv("NUMPYGO_INTERCHANGE_DIR"); dir != "" {
		if e := os.WriteFile(filepath.Join(dir, name), data, 0600); e != nil {
			t.Fatal(e)
		}
	}
}

func TestNPYOpenMemMapNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mm := refcheck.RequireValue[*np.MappedArray](t)
	md := refcheck.RequireValue[dtypes.Descriptor](t)
	path := filepath.Join(t.TempDir(), "mapped.npy")
	d := md(dtypes.Parse(">i4"))
	a := mm(OpenMemMap(path, MemMapOptions{Mode: "w+", DType: &d, Shape: []int{2, 3}, FortranOrder: true, Version: Version{Major: 2}}))
	closeMap := func(a *np.MappedArray) {
		t.Helper()
		if e := a.Close(); e != nil {
			t.Fatal(e)
		}
	}
	if e := a.Assign(nil, m(np.FromSlice([]int32{1, -2, 3, 7, 9, -8}, 2, 3))); e != nil {
		t.Fatal(e)
	}
	if e := a.Flush(); e != nil {
		t.Fatal(e)
	}
	ints := func(v []int) *np.Array {
		x := make([]int64, len(v))
		for i, n := range v {
			x[i] = int64(n)
		}
		return m(np.FromSlice(x, len(x)))
	}
	refcheck.Check(t, "format_mmap_shape", ints(a.Shape()))
	refcheck.Check(t, "format_mmap_strides", ints(a.Strides()))
	refcheck.Check(t, "format_mmap_offset", m(np.FromSlice([]int64{a.Offset})))
	closeMap(a)
	a = mm(OpenMemMap(path, MemMapOptions{Mode: "r+"}))
	if e := a.SetValue(int32(67), 1, 2); e != nil {
		t.Fatal(e)
	}
	closeMap(a)
	a = mm(OpenMemMap(path, MemMapOptions{Mode: "c"}))
	if e := a.SetValue(int32(88), 0, 0); e != nil {
		t.Fatal(e)
	}
	closeMap(a)
	a = mm(OpenMemMap(path, MemMapOptions{Mode: "r"}))
	refcheck.Check(t, "format_mmap_value", m(a.Astype(np.Int32)))
	refcheck.Error(t, "format_mmap_readonly", "ValueError", a.SetValue(int32(1), 0, 0), np.ErrReadOnly)
	closeMap(a)
	raw := refcheck.RequireValue[[]byte](t)(os.ReadFile(path))
	refcheck.Bytes(t, "format_mmap_file", raw)
	writeInterchange(t, "format_memmap.npy", raw)
	f := refcheck.RequireValue[*os.File](t)(os.Open(path))
	refcheck.Check(t, "format_isfileobj", m(np.FromSlice([]bool{IsFileObj(f), IsFileObj(bytes.NewReader(nil))}, 2)))
	if e := f.Close(); e != nil {
		t.Fatal(e)
	}
	d = md(dtypes.Parse("O"))
	_, e := OpenMemMap(filepath.Join(t.TempDir(), "obj.npy"), MemMapOptions{Mode: "w+", DType: &d, Shape: []int{1}})
	refcheck.Error(t, "format_mmap_object", "ValueError", e, np.ErrDType)
}
