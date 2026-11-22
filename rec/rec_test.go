package rec

import (
	"bytes"
	"encoding/binary"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
	"strconv"
	"strings"
	"testing"
)

func tuple(s []int) string {
	v := make([]string, len(s))
	for i, n := range s {
		v[i] = strconv.Itoa(n)
	}
	x := strings.Join(v, ", ")
	if len(v) == 1 {
		x += ","
	}
	return "(" + x + ")"
}
func checkLayout(t *testing.T, name string, a *records.Array) {
	t.Helper()
	m := refcheck.Require(t)
	raw := refcheck.RequireValue[[]byte](t)(a.NDArray().ToBytes("C"))
	refcheck.Bytes(t, name, raw)
	checkInts := func(n string, v []int) {
		x := make([]int64, len(v))
		for i, y := range v {
			x[i] = int64(y)
		}
		refcheck.Check(t, name+"_"+n, m(np.FromSlice(x, len(x))))
	}
	checkInts("shape", a.Shape())
	checkInts("strides", a.Strides())
	s := a.Schema()
	aligned := 0
	if s.Aligned() {
		aligned = 1
	}
	layout := []int{s.ItemSize(), s.Alignment(), aligned}
	desc := []string{}
	for _, f := range s.Fields() {
		layout = append(layout, f.Offset)
		desc = append(desc, f.Type.Descriptor()+":"+tuple(f.Type.Shape))
	}
	checkInts("layout", layout)
	refcheck.Text(t, name+"_names", strings.Join(s.Names(), "|"))
	refcheck.Text(t, name+"_descr", strings.Join(desc, "|"))
}

func TestRecNamespaceNumPy(t *testing.T) {
	m := refcheck.Require(t)
	r := refcheck.RequireValue[*records.Array](t)
	sc := refcheck.RequireValue[*records.Schema](t)
	cols := func(a *records.Array, name string) *np.Array { return m(a.NDArray().Field(name)) }
	parser := sc(FormatParser([]string{"i2", "U3", "(2,)f8", "M8[D]"}, []string{" id ", "label", "vec", "date"}, []any{"ident", "标签", "vector", "time"}, true, binary.BigEndian))
	checkLayout(t, "rec_parser", r(records.Zeros(parser, 2)))
	_, e := FormatParser([]string{"i2", "i2"}, []string{"id", "id"}, nil, false, nil)
	refcheck.Error(t, "rec_parser_duplicate", "ValueError", e, np.ErrValue)
	ids := m(np.FromSlice([]int16{0, 1, 2, 3, 4, 5}, 2, 3))
	dates := m(np.FromDateTicks(m(np.FromSlice([]int64{0, 1, 2, 3, 4, 5}, 2, 3)), np.DateTime64, "D"))
	labels := m(np.FromUnicodeStrings([]string{"a", "bc", "中文", "d", "e", "f"}, 3, 2, 3))
	a := r(FromArrays([]*np.Array{ids, labels, dates}, Options{Names: []string{"id", "label", "date"}}))
	checkLayout(t, "rec_fromarrays", a)
	v := make([]float64, 12)
	for i := range v {
		v[i] = float64(i)
	}
	s := r(FromArrays([]*np.Array{m(np.FromSlice(v, 2, 3, 2)), ids}, Options{Formats: []string{"2f8", "i2"}, Names: []string{"vec", "id"}}))
	checkLayout(t, "rec_subarrays", s)
	_, e = FromArrays([]*np.Array{m(np.Zeros(np.Int64, 2)), m(np.Zeros(np.Int64, 3))}, Options{})
	refcheck.Error(t, "rec_bad_arrays", "ValueError", e, np.ErrShape)
	row := r(FromRecords([][]any{{1, "中", 1.25}, {-7, "ab", -3.5}}, Options{Names: []string{"id", "tag", "value"}}))
	checkLayout(t, "rec_fromrecords", row)
	record := refcheck.RequireValue[*records.Record](t)(NewRecord(row.Schema(), 1, "中", 1.25))
	refcheck.Bytes(t, "rec_record", record.Bytes())
	array := r(RecArray(nil, Options{Shape: []int{2, 3}, Formats: []string{"i2", ">f8"}, Names: []string{"id", "value"}, Order: "F"}))
	if e = cols(array, "id").Assign(nil, ids); e != nil {
		t.Fatal(e)
	}
	f := make([]float64, 6)
	for i := range f {
		f[i] = float64(i) / 4
	}
	if e = cols(array, "value").Assign(nil, m(np.FromSlice(f, 2, 3))); e != nil {
		t.Fatal(e)
	}
	checkLayout(t, "rec_recarray", array)
	buf := make([]byte, 40)
	for i := range buf {
		buf[i] = byte(i)
	}
	strided := r(RecArray(buf, Options{Shape: []int{2, 2}, Formats: []string{"<i2", ">i2"}, Names: []string{"x", "y"}, Offset: 9, Strides: []int{11, -4}}))
	if e = cols(strided, "x").SetValue(int16(-55), 1, 0); e != nil {
		t.Fatal(e)
	}
	checkLayout(t, "rec_strided", strided)
	refcheck.Bytes(t, "rec_strided_buffer", buf)
	_, e = RecArray(make([]byte, 8), Options{Shape: []int{2, 2}, Formats: []string{"i2"}, Offset: 0, Strides: []int{8, 8}})
	refcheck.Error(t, "rec_bad_strides", "ValueError", e, np.ErrShape)
	data := make([]byte, 23)
	for i := range data {
		data[i] = byte(i)
	}
	opt := Options{Formats: []string{"i2", "i2"}, Offset: 4}
	str := r(FromString(data, opt))
	checkLayout(t, "rec_fromstring", str)
	refcheck.Check(t, "rec_fromstring_writable", np.ScalarOf(str.Writable()))
	file := r(FromFile(bytes.NewReader(data), opt))
	checkLayout(t, "rec_fromfile", file)
	refcheck.Check(t, "rec_fromfile_writable", np.ScalarOf(file.Writable()))
	opt.Offset = 0
	opt.Shape = []int{2}
	_, e = FromString(make([]byte, 7), opt)
	refcheck.Error(t, "rec_buffer_short", "TypeError", e, np.ErrShape)
	_, e = FromFile(bytes.NewReader(make([]byte, 7)), opt)
	refcheck.Error(t, "rec_file_short", "ValueError", e, np.ErrShape)
	for _, copyData := range []bool{false, true} {
		label := "0"
		if copyData {
			label = "1"
		}
		q := r(Array(row, Options{NoCopy: !copyData}))
		checkLayout(t, "rec_array_"+label, q)
		refcheck.Check(t, "rec_share_"+label, np.ScalarOf(np.SharesMemory(row.NDArray(), q.NDArray())))
	}
	refcheck.Text(t, "rec_duplicate", strings.Join(FindDuplicate([]string{"z", "a", "b", "a", "z", "b"}), "|"))
}
