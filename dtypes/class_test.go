package dtypes

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
	"strconv"
	"strings"
	"testing"
)

func TestDescriptorClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	dm := refcheck.RequireValue[Descriptor](t)
	sm := refcheck.RequireValue[*records.Schema](t)
	prefix := "numeric_class_"
	yes := func(v bool) int64 {
		if v {
			return 1
		}
		return 0
	}
	text := func(n, s string) { t.Helper(); refcheck.Text(t, prefix+n, s) }
	check := func(n string, a *np.Array) { t.Helper(); refcheck.Check(t, prefix+n, a) }
	ints := func(n string, v []int64) { t.Helper(); check(n, m(np.FromSlice(v, len(v)))) }
	shape := func(n string, sh []int) {
		v := make([]int64, len(sh))
		for i, x := range sh {
			v[i] = int64(x)
		}
		ints(n, v)
	}
	descrText := func(d Descriptor) string {
		v, e := d.Descr()
		if e != nil {
			t.Fatal(e)
		}
		var b bytes.Buffer
		enc := json.NewEncoder(&b)
		enc.SetEscapeHTML(false)
		if e = enc.Encode(v); e != nil {
			t.Fatal(e)
		}
		return strings.TrimSuffix(b.String(), "\n")
	}
	desc := func(n string, d Descriptor) {
		t.Helper()
		md := d.Metadata()
		text(n+"_meta", fmt.Sprintf("%s|%s|%c|%c|%c|%s", md.Name, md.Str, md.Char, md.Kind, md.ByteOrder, d.ScalarTypeName()))
		ints(n+"_flags", []int64{int64(md.ItemSize), int64(md.Alignment), int64(md.Num), int64(d.Flags()), int64(d.IsBuiltin()), yes(d.IsNative()), yes(d.IsAlignedStruct()), yes(md.HasObject), int64(d.NDim())})
		shape(n+"_shape", d.Shape())
		text(n+"_base", d.Base().String())
		text(n+"_descr", descrText(d))
	}
	specs := strings.Split(referenceText(t, prefix+"dtype_specs"), "\n")
	orders := []string{"S", "<", ">", "=", "|"}
	for i, s := range specs {
		d := dm(Parse(s))
		n := "dtype_" + strconv.Itoa(i)
		desc(n, d)
		if s == "T" {
			_, e := d.NewByteOrder("S")
			refcheck.Error(t, prefix+"dtype_string_order", "TypeError", e, np.ErrDType)
			continue
		}
		for j, o := range orders {
			desc("dtype_order_"+strconv.Itoa(i)+"_"+strconv.Itoa(j), dm(d.NewByteOrder(o)))
		}
	}
	nested := sm(records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Numeric(np.Int32, binary.BigEndian)}, {Name: "flag", Type: records.Numeric(np.Bool, nil)}}, true, 0))
	structure := sm(records.NewSchema([]records.FieldSpec{
		{Name: "id", Title: "ident", Type: records.Numeric(np.Int16, binary.BigEndian)},
		{Name: "name", Type: records.FixedUnicode(2, nil)},
		{Name: "vec", Type: records.Subarray(records.Numeric(np.Float32, nil), 2)},
		{Name: "nested", Type: records.Struct(nested)},
	}, true, 0))
	d := dm(New("VoidDType", Options{Schema: structure}))
	desc("dtype_struct", d)
	text("dtype_names", strings.Join(d.Names(), "|"))
	fm := refcheck.RequireValue[FieldDescription](t)
	for i, n := range d.Names() {
		f := fm(d.Field(n))
		desc("dtype_field_"+strconv.Itoa(i), f.DType)
		ints("dtype_field_offset_"+strconv.Itoa(i), []int64{int64(f.Offset)})
		title := ""
		if f.Title != nil {
			title = fmt.Sprint(f.Title)
		}
		text("dtype_field_title_"+strconv.Itoa(i), title)
	}
	desc("dtype_title", fm(d.Field("ident")).DType)
	desc("dtype_last", fm(d.FieldAt(-1)).DType)
	for i, o := range orders {
		q := dm(d.NewByteOrder(o))
		desc("dtype_struct_order_"+strconv.Itoa(i), q)
		for j, n := range q.Names() {
			text("dtype_struct_order_field_"+strconv.Itoa(i)+"_"+strconv.Itoa(j), descrText(fm(q.Field(n)).DType))
		}
	}
	renamed := dm(d.WithNames([]string{"code", "label", "samples", "sub"}))
	desc("dtype_renamed", renamed)
	text("dtype_renamed_names", strings.Join(renamed.Names(), "|"))
	if d.Names()[0] != "id" {
		t.Fatal("renamed original descriptor")
	}
	for i, p := range []struct {
		s  string
		sh []int
	}{{"i2", []int{2, 3}}, {">f4", []int{2}}, {"U2", []int{3}}, {"i2", []int{0}}, {"i2", nil}} {
		v := dm(Subarray(dm(Parse(p.s)), p.sh...))
		desc("dtype_sub_"+strconv.Itoa(i), v)
		if len(p.sh) > 0 {
			b, sh, ok := v.Subdtype()
			if !ok || !b.Equal(dm(Parse(p.s))) || len(sh) != len(p.sh) {
				t.Fatal("subdtype")
			}
		}
		z := m(v.Zeros(2))
		shape("dtype_sub_zeros_shape_"+strconv.Itoa(i), z.Shape())
		text("dtype_sub_zeros_type_"+strconv.Itoa(i), dm(FromArray(z)).String())
	}
	meta := dm(Parse("i4")).WithMetadata(map[string]any{"unit": "ms", "scale": 2})
	desc("dtype_metadata", meta)
	b, e := json.Marshal(meta.UserMetadata())
	if e != nil {
		t.Fatal(e)
	}
	text("dtype_metadata_value", string(b))
	nullable := dm(New("StringDType", Options{String: np.StringDescriptor{HasNA: true, NA: nil, Strict: true}}))
	desc("dtype_nullable", nullable)
	coerce := refcheck.RequireValue[bool](t)(nullable.Coerce())
	na, ok := nullable.NAObject()
	_, has := dm(Parse("T")).NAObject()
	ints("dtype_nullable_options", []int64{yes(coerce), yes(ok && na == nil), yes(has)})
	for i, p := range [][2]string{{"S0", "S3"}, {"l", "q"}, {"i4", ">i4"}, {"i4", "i8"}, {"f8", "i8"}, {"U2", "U3"}, {"M8[D]", "M8[h]"}, {"V2", "V3"}} {
		a, b := dm(Parse(p[0])), dm(Parse(p[1]))
		v := []bool{}
		for _, op := range []string{"==", "!=", "<", "<=", ">", ">="} {
			v = append(v, refcheck.RequireValue[bool](t)(a.Compare(b, op)))
		}
		check("dtype_compare_"+strconv.Itoa(i), m(np.FromSlice(v, 6)))
	}
	_, e = dm(Parse("i4")).NewByteOrder("unknown")
	refcheck.Error(t, prefix+"dtype_bad_order", "ValueError", e, np.ErrValue)
	_, e = d.Field("absent")
	refcheck.Error(t, prefix+"dtype_bad_field", "KeyError", e, np.ErrIndex)
	_, e = Subarray(dm(Parse("i2")), -1)
	refcheck.Error(t, prefix+"dtype_bad_shape", "ValueError", e, np.ErrShape)
}
