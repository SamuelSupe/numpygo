package np_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestNumericUFunc(t *testing.T) {
	m := refcheck.Require(t)
	u := refcheck.RequireValue[*np.UFunc](t)(np.LookupUFunc("add"))
	a := m(np.WrapSlice([]int64{1, 2, 3}, 3))
	refcheck.Check(t, "final_ufunc_meta", m(np.WrapSlice([]int64{int64(u.NIn), int64(u.NOut), u.Identity.(int64)}, 3)))
	refcheck.Check(t, "final_ufunc_call", m(u.Call(a, np.ScalarOf(int64(10)))))
	refcheck.Check(t, "final_ufunc_reduce", m(u.Reduce(a)))
	refcheck.Check(t, "final_ufunc_accumulate", m(u.Accumulate(a, 0, np.Int64)))
	refcheck.Check(t, "final_ufunc_outer", m(u.Outer(a, m(np.WrapSlice([]int64{4, 5}, 2)))))
	if e := u.At(a, []int{0, 0, 2}, m(np.WrapSlice([]int64{4, 5, 6}, 3))); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "final_ufunc_at", a)
}
func TestFloatingErrorPolicy(t *testing.T) {
	m := refcheck.Require(t)
	p := refcheck.RequireValue[np.ErrorPolicy](t)
	n := refcheck.RequireValue[int](t)
	c := np.NewErrorContext()
	var uninitialized np.ErrorContext
	if _, e := uninitialized.Call("add", np.Scalar(1), np.Scalar(2)); !errors.Is(e, np.ErrValue) {
		t.Fatal(e)
	}
	policy := func(name string, c *np.ErrorContext) {
		v := c.GetErr()
		refcheck.Text(t, "final_fp_"+name, fmt.Sprintf("%s|%s|%s|%s", v.Divide, v.Over, v.Under, v.Invalid))
	}
	policy("default", c)
	refcheck.Check(t, "final_fp_buf_default", np.ScalarOf(int64(c.GetBufSize())))
	var events []string
	cb := &np.ErrorHandler{Callback: func(kind string, flag int) { events = append(events, fmt.Sprintf("%s:%d", kind, flag)) }}
	previous := p(c.SetErr(np.ErrorOptions{All: np.Call}))
	refcheck.Text(t, "final_fp_seterr_old", fmt.Sprintf("%s|%s|%s|%s", previous.Divide, previous.Over, previous.Under, previous.Invalid))
	old := c.SetErrCall(cb)
	refcheck.Check(t, "final_fp_seterrcall", np.ScalarOf(old == nil && c.GetErrCall() == cb))
	refcheck.Check(t, "final_fp_divide", m(c.Call("divide", m(np.WrapSlice([]float64{1, 0}, 2)), m(np.WrapSlice([]float64{0, 0}, 2)))))
	m(c.Call("exp", np.Scalar(1000)))
	m(c.Call("multiply", np.Scalar(1e-300), np.Scalar(1e-300)))
	m(c.Call("sqrt", np.Scalar(-1)))
	m(c.Call("multiply", np.Scalar(math.Ldexp(1, -1022)), np.Scalar(0.5)))
	m(c.Call("floor_divide", m(np.WrapSlice([]int64{1}, 1)), m(np.WrapSlice([]int64{0}, 1))))
	refcheck.Text(t, "final_fp_callback", strings.Join(events, "|"))
	events = nil
	m(c.Call("multiply", np.ScalarOf(float32(1e30)), np.ScalarOf(float32(1e30))))
	m(c.Call("remainder", np.Scalar(1), np.Scalar(0)))
	refcheck.Text(t, "final_fp_callback_extra", strings.Join(events, "|"))
	e := c.ErrState(np.ErrorOptions{All: np.Ignore, Divide: np.Raise}, func(child *np.ErrorContext) error {
		policy("scope", child)
		_, e := child.Call("divide", np.Scalar(1), np.Scalar(0))
		refcheck.Error(t, "final_fp_raises", "FloatingPointError", e, np.ErrFloatingPoint)
		if e == nil {
			t.Fatal("expected floating point error")
		}
		var f *np.FloatingPointError
		if !errors.As(e, &f) || f.Kind != np.FPDivide {
			t.Fatal(e)
		}
		return nil
	})
	if e != nil {
		t.Fatal(e)
	}
	policy("restore", c)
	refcheck.Check(t, "final_fp_buf_old", np.ScalarOf(int64(n(c.SetBufSize(64)))))
	refcheck.Check(t, "final_fp_buf_new", np.ScalarOf(int64(c.GetBufSize())))
	_, e = c.SetBufSize(17)
	refcheck.Error(t, "final_fp_invalid_buf", "ValueError", e, np.ErrValue)
	var b bytes.Buffer
	p(c.SetErr(np.ErrorOptions{All: np.Ignore, Invalid: np.LogError}))
	c.SetErrCall(&np.ErrorHandler{Writer: &b})
	m(c.Call("sqrt", np.Scalar(-1)))
	refcheck.Text(t, "final_fp_log", b.String())
	b.Reset()
	p(c.SetErr(np.ErrorOptions{All: np.Ignore, Invalid: np.Warn}))
	c.SetWarningWriter(&b)
	m(c.Call("sqrt", np.Scalar(-1)))
	refcheck.Text(t, "final_fp_warn", strings.TrimSpace(strings.TrimPrefix(b.String(), "RuntimeWarning: ")))
}
func TestArrayFormattingParity(t *testing.T) {
	m := refcheck.Require(t)
	str := refcheck.RequireValue[string](t)
	p := np.GetPrintOptions()
	refcheck.Text(t, "final_format_default", fmt.Sprintf("%d|%d|%d|%d|%s|%s|%s|%s", p.Precision, p.Threshold, p.EdgeItems, p.LineWidth, p.FloatMode, p.Sign, p.NaN, p.Inf))
	matrix := m(np.Arange(0, 14, 1, np.Int64))
	cases := []struct {
		name string
		a    *np.Array
	}{
		{"int", m(np.WrapSlice([]int16{1, -20, 300}, 3))}, {"bool", m(np.WrapSlice([]bool{true, false}, 2))},
		{"float", m(np.WrapSlice([]float64{1, 2.345, -12.3, math.NaN()}, 4))}, {"sci", m(np.WrapSlice([]float64{1e-10, 12345.4}, 2))},
		{"complex", m(np.WrapSlice([]complex128{1 + 2i, -3 - 4.5i}, 2))},
		{"matrix", m(matrix.Reshape(2, 7))}, {"scalar", np.Scalar(1.25)}, {"empty", m(np.Zeros(np.Float64, 0, 2))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			refcheck.Text(t, "final_format_"+tc.name+"_2string", str(np.Array2String(tc.a)))
			refcheck.Text(t, "final_format_"+tc.name+"_repr", str(np.ArrayRepr(tc.a)))
			refcheck.Text(t, "final_format_"+tc.name+"_str", str(np.ArrayStr(tc.a)))
		})
	}
	opt := p
	opt.Precision = 2
	opt.FloatMode = "fixed"
	opt.Sign = "+"
	opt.NaN = "NaN"
	opt.Inf = "Inf"
	e := np.WithPrintOptions(opt, func(f *np.Formatter) error {
		refcheck.Text(t, "final_format_scope", str(f.Array2String(m(np.WrapSlice([]float64{1, -2.345, math.NaN(), math.Inf(1)}, 4)))))
		v := f.GetPrintOptions()
		refcheck.Text(t, "final_format_scope_options", fmt.Sprintf("%d|%s|%s|%s|%s", v.Precision, v.FloatMode, v.Sign, v.NaN, v.Inf))
		return nil
	})
	if e != nil {
		t.Fatal(e)
	}
	opt = p
	opt.Threshold = 5
	opt.EdgeItems = 2
	f := refcheck.RequireValue[*np.Formatter](t)(np.NewFormatter(opt))
	refcheck.Text(t, "final_format_summary", str(f.Array2String(m(np.Arange(0, 10, 1, np.Int64)))))
	opt = p
	opt.LineWidth = 18
	f = refcheck.RequireValue[*np.Formatter](t)(np.NewFormatter(opt))
	refcheck.Text(t, "final_format_line", str(f.Array2String(matrix)))
	opt = p
	opt.Formatter = map[string]func(any) string{"int": func(v any) string { return fmt.Sprintf("<%v>", v) }}
	f = refcheck.RequireValue[*np.Formatter](t)(np.NewFormatter(opt))
	refcheck.Text(t, "final_format_custom", str(f.Array2String(m(np.Arange(0, 3, 1, np.Int64)), np.ArrayStringOptions{Separator: ", "})))
	opt = p
	opt.Precision = 3
	opt.FloatMode = "maxprec_equal"
	defer np.SetPrintOptions(p)
	if e = np.SetPrintOptions(opt); e != nil {
		t.Fatal(e)
	}
	refcheck.Text(t, "final_format_set", str(np.ArrayStr(m(np.WrapSlice([]float64{1, 2.34567}, 2)))))
}
func TestMemMapParity(t *testing.T) {
	m := refcheck.Require(t)
	open := refcheck.RequireValue[*np.MappedArray](t)
	data := refcheck.RequireValue[[]byte](t)
	path := filepath.Join(t.TempDir(), "mapping.bin")
	opt := np.MemMapOptions{DType: np.Int32, Shape: []int{2, 3}, Mode: "w+", Offset: 7, Order: "F", ByteOrder: binary.LittleEndian}
	mm := open(np.MemMap(path, opt))
	defer mm.Close()
	for i := range 2 {
		for j := range 3 {
			if e := mm.SetValue(int32(i*3+j+1), i, j); e != nil {
				t.Fatal(e)
			}
		}
	}
	v := m(mm.Slice(np.Slice{}, np.Reverse()))
	// Reinterpreted views have a separate storage header but share map lifetime.
	reinterpreted := m(mm.View(np.Uint32))
	if e := v.SetValue(int32(99), 0, 0); e != nil {
		t.Fatal(e)
	}
	if e := mm.Flush(); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "final_mmap_layout", mm.Array)
	strides := mm.Strides()
	refcheck.Check(t, "final_mmap_strides", m(np.WrapSlice([]int64{int64(strides[0]), int64(strides[1])}, 2)))
	refcheck.Bytes(t, "final_mmap_file", data(os.ReadFile(path)))
	if _, e := mm.DataBytes(); e == nil {
		t.Fatal("raw mapped slice can escape Close")
	}
	if e := mm.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e := v.AtValue(0, 0); e == nil {
		t.Fatal("view remains accessible after unmap")
	}
	if _, e := reinterpreted.AtValue(0, 0); e == nil {
		t.Fatal("reinterpreted view remains accessible after unmap")
	}
	opt.Shape = nil
	opt.Mode = "r"
	opt.Order = "C"
	ro := open(np.MemMap(path, opt))
	defer ro.Close()
	refcheck.Check(t, "final_mmap_read", ro.Array)
	refcheck.Check(t, "final_mmap_readonly", np.ScalarOf(!ro.Writable()))
	refcheck.Error(t, "final_mmap_write_readonly", "ValueError", ro.SetValue(int32(0), 0), np.ErrReadOnly)
	if e := ro.SetWritable(true); !errors.Is(e, np.ErrReadOnly) {
		t.Fatal(e)
	}
	if e := ro.Close(); e != nil {
		t.Fatal(e)
	}
	opt.Mode = "c"
	c := open(np.MemMap(path, opt))
	defer c.Close()
	if e := c.SetValue(int32(42), 0); e != nil {
		t.Fatal(e)
	}
	if e := c.Flush(); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "final_mmap_private", c.Array)
	refcheck.Bytes(t, "final_mmap_private_file", data(os.ReadFile(path)))
	if e := c.Close(); e != nil {
		t.Fatal(e)
	}
	opt.Mode = "r+"
	rw := open(np.MemMap(path, opt))
	defer rw.Close()
	if e := rw.SetValue(int32(-12), 0); e != nil {
		t.Fatal(e)
	}
	if e := rw.Close(); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "final_mmap_update_file", data(os.ReadFile(path)))
}
func TestInteropEquivalents(t *testing.T) {
	m := refcheck.Require(t)
	handle := refcheck.RequireValue[*np.TensorHandle](t)
	a := m(m(np.Arange(0, 6, 1, np.Int32)).Reshape(2, 3))
	v := m(a.Transpose())
	h := handle(v.DLPack())
	imported := m(np.FromDLPack(h))
	refcheck.Check(t, "final_abi_dlpack_meta", m(np.WrapSlice([]bool{np.SharesMemory(v, imported), !imported.Writable()}, 2)))
	strides := imported.Strides()
	refcheck.Check(t, "final_abi_dlpack_strides", m(np.WrapSlice([]int64{int64(strides[0]), int64(strides[1])}, 2)))
	if e := a.SetValue(int32(77), 1, 2); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "final_abi_dlpack_data", imported)
	if e := imported.SetValue(int32(123), 1, 0); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "final_abi_dlpack_writeback", a)
	ptr := handle(a.CTypes())
	refcheck.Check(t, "final_abi_ctypes_meta", m(np.WrapSlice([]bool{ptr.DataIdentity() == a.DataIdentity(), slices.Equal(ptr.Shape(), a.Shape()), slices.Equal(ptr.Strides(), a.Strides())}, 3)))
	if e := ptr.Set(int32(-13), 0, 1); e != nil {
		t.Fatal(e)
	}
	owned := m(np.Arange(0, 3, 1, np.Int32))
	retained := handle(owned.CTypes())
	if e := owned.ResizeInPlace(4); !errors.Is(e, np.ErrShape) {
		t.Fatal("replaced storage retained by handle", e)
	}
	if got, e := retained.At(2); e != nil || got != int32(2) {
		t.Fatal(got, e)
	}
	refcheck.Check(t, "final_abi_ctypes_data", a)
	raw := refcheck.RequireValue[[]byte](t)(a.Dumps())
	refcheck.Check(t, "final_abi_dumps", m(np.LoadNPY(bytes.NewReader(raw))))
	var b bytes.Buffer
	if e := a.Dump(&b); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "final_abi_dump", m(np.LoadNPY(&b)))
	ns := np.ArrayNamespaceInfo()
	caps := ns.Capabilities()
	bit := func(v bool) int64 {
		if v {
			return 1
		}
		return 0
	}
	refcheck.Check(t, "final_abi_namespace_capabilities", m(np.WrapSlice([]int64{bit(caps.BooleanIndexing), bit(caps.DataDependentShapes), int64(caps.MaxDimensions)}, 3)))
	refcheck.Text(t, "final_abi_namespace_devices", strings.Join(ns.Devices(), "|"))
	d := ns.DefaultDTypes()
	refcheck.Text(t, "final_abi_namespace_default", fmt.Sprintf("%s|%s|%s|%s", d["real floating"], d["complex floating"], d["integral"], d["indexing"]))
	dtypes := func(kind ...string) string {
		v := refcheck.RequireValue[map[string]np.DType](t)(ns.DTypes(kind...))
		keys := []string{}
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		return strings.Join(keys, "|")
	}
	refcheck.Text(t, "final_abi_namespace_dtypes", dtypes())
	refcheck.Text(t, "final_abi_namespace_real", dtypes("real floating"))
	// The Go build needs no headers; source availability is optional in a static deployment.
	if p, e := np.GetInclude(); e == nil {
		if _, e = os.Stat(filepath.Join(p, "array.go")); e != nil {
			t.Fatal(e)
		}
	}
	refcheck.Check(t, "final_abi_include", np.ScalarOf(np.ModuleImportPath == "github.com/SamuelSupe/numpygo"))
	b.Reset()
	if e := np.Info(&b, a); e != nil {
		t.Fatal(e)
	}
	info := b.String()
	flags := []bool{}
	for _, s := range []string{"shape: [2 3]", "strides: [12 4]", "itemsize: 4", "dtype: int32"} {
		flags = append(flags, strings.Contains(info, s))
	}
	refcheck.Check(t, "final_abi_info", m(np.WrapSlice(flags, 4)))
	c := np.ShowConfig()
	refcheck.Check(t, "final_abi_config", np.ScalarOf(c.Module != "" && c.GoVersion != "" && c.Dependencies != nil))
	r := np.ShowRuntime()
	refcheck.Check(t, "final_abi_runtime", np.ScalarOf(r.OS != "" && r.Architecture != "" && r.Version != "" && r.LogicalCPUs > 0))
	refcheck.Check(t, "final_abi_test", np.ScalarOf(np.SelfTest() == nil))
}
