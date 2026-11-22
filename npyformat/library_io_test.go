package npyformat

import (
	"archive/zip"
	"bytes"
	"errors"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func inputBytes(t *testing.T, n string) []byte {
	t.Helper()
	return refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, n)))
}
func TestLazyNPZNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mb := refcheck.RequireValue[[]byte](t)
	mf := refcheck.RequireValue[*NpzFile](t)
	mv := refcheck.RequireValue[any](t)
	raw := inputBytes(t, "lib_npz_archive")
	f := mf(OpenNPZ(bytes.NewReader(raw), int64(len(raw)), 0))
	defer f.Close()
	refcheck.Text(t, "lib_npz_keys", strings.Join(f.Keys(), "|"))
	refcheck.Text(t, "lib_npz_repr", f.String())
	refcheck.Check(t, "lib_npz_len", np.ScalarOf(int64(f.Len())))
	keys := []string{"z", "z.npy", "readme.txt", "raw", "missing", "nested/record"}
	has := make([]bool, len(keys))
	for i, k := range keys {
		has[i] = f.Contains(k)
	}
	refcheck.Check(t, "lib_npz_contains", m(np.FromSlice(has, len(has))))
	a := mv(f.At("z")).(*np.Array)
	refcheck.Check(t, "lib_npz_z", m(a.Astype(np.Int32)))
	if a.NativeByteOrder() || !a.IsFContiguous() {
		t.Fatal("lazy loader lost original endian or Fortran order")
	}
	refcheck.Bytes(t, "lib_npz_z_raw", mb(mv(f.At("z.npy")).(*np.Array).ToBytes("F")))
	refcheck.Check(t, "lib_npz_a", mv(f.Get("a", nil)).(*np.Array))
	refcheck.Check(t, "lib_npz_attribute", mv(f.At("a")).(*np.Array))
	refcheck.Bytes(t, "lib_npz_readme", mv(f.At("readme.txt")).([]byte))
	refcheck.Bytes(t, "lib_npz_raw", mv(f.At("raw.npy")).([]byte))
	var b bytes.Buffer
	if e := WriteArray(&b, mv(f.At("nested/record")).(*np.Array), Version{}); e != nil {
		t.Fatal(e)
	}
	refcheck.Bytes(t, "lib_npz_struct", b.Bytes())
	refcheck.Check(t, "lib_npz_get_default", np.ScalarOf(mv(f.Get("missing", int64(71))).(int64)))
	tag := func(v any) string {
		if _, ok := v.(*np.Array); ok {
			return "array"
		}
		return "bytes"
	}
	items, values := []string{}, []string{}
	for x, e := range f.Items() {
		if e != nil {
			t.Fatal(e)
		}
		items = append(items, x.Key+":"+tag(x.Value))
	}
	for v, e := range f.Values() {
		if e != nil {
			t.Fatal(e)
		}
		values = append(values, tag(v))
	}
	refcheck.Text(t, "lib_npz_items", strings.Join(items, "|"))
	refcheck.Text(t, "lib_npz_values", strings.Join(values, "|"))
	zipNames := []string{}
	meta := []int64{}
	for _, z := range f.Zip() {
		zipNames = append(zipNames, z.Name)
		meta = append(meta, int64(z.Size), int64(z.CompressedSize), int64(z.Method))
	}
	refcheck.Text(t, "lib_npz_zip_names", strings.Join(zipNames, "|"))
	refcheck.Check(t, "lib_npz_zip_meta", m(np.FromSlice(meta, len(zipNames), 3)))
	_, e := f.At("missing")
	refcheck.Error(t, "lib_npz_missing", "KeyError", e, np.ErrIndex)
	if e = f.Close(); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "lib_npz_closed", m(np.FromSlice([]bool{f.Zip() == nil, f.FID() == nil}, 2)))
	_, e = f.At("a")
	if !errors.Is(e, os.ErrClosed) {
		t.Errorf("read after close: %v", e)
	}
	path := filepath.Join(t.TempDir(), "archive.npz")
	if e = os.WriteFile(path, raw, 0600); e != nil {
		t.Fatal(e)
	}
	owned := mf(OpenNPZPath(path, 0))
	original := owned.FID()
	refcheck.Check(t, "lib_npz_owned", np.ScalarOf(original != nil))
	if original == nil {
		t.Fatal("path archive should own a file")
	}
	refcheck.Check(t, "lib_npz_a", mv(owned.At("a")).(*np.Array))
	if e = owned.Close(); e != nil {
		t.Fatal(e)
	}
	_, e = original.Stat()
	refcheck.Check(t, "lib_npz_owned_close", np.ScalarOf(errors.Is(e, os.ErrClosed)))
	if !errors.Is(e, os.ErrClosed) {
		t.Errorf("owned file not closed: %v", e)
	}
	raw = inputBytes(t, "lib_npz_lazy_archive")
	lazy := mf(OpenNPZ(bytes.NewReader(raw), int64(len(raw)), 0))
	defer lazy.Close()
	refcheck.Check(t, "lib_npz_lazy_good", mv(lazy.At("good")).(*np.Array))
	_, e = lazy.At("bad")
	refcheck.Error(t, "lib_npz_lazy_bad", "ValueError", e, np.ErrDType)
	// The directory remains readable even if an unrelated member body is
	// corrupt. Only accessing that member must fail CRC verification.
	b.Reset()
	w := zip.NewWriter(&b)
	part := refcheck.RequireValue[io.Writer](t)(w.CreateHeader(&zip.FileHeader{Name: "bad.txt", Method: zip.Store}))
	data := []byte("crc-check-sample")
	if _, e = part.Write(data); e != nil {
		t.Fatal(e)
	}
	if e = w.Close(); e != nil {
		t.Fatal(e)
	}
	i := bytes.Index(b.Bytes(), data)
	if i < 0 {
		t.Fatal("stored member not found")
	}
	b.Bytes()[i] ^= 1
	bad := mf(OpenNPZ(bytes.NewReader(b.Bytes()), int64(b.Len()), 0))
	defer bad.Close()
	if _, e = bad.At("bad.txt"); !errors.Is(e, zip.ErrChecksum) {
		t.Errorf("CRC error: %v", e)
	}
}

func TestDataSourceNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mb := refcheck.RequireValue[[]byte](t)
	md := refcheck.RequireValue[*DataSource](t)
	mf := refcheck.RequireValue[io.ReadWriteCloser](t)
	dir := t.TempDir()
	cache := filepath.Join(dir, "cache")
	if e := os.Mkdir(cache, 0700); e != nil {
		t.Fatal(e)
	}
	d := md(NewDataSource(cache))
	defer d.Close()
	plain := filepath.Join(dir, "plain.txt")
	gz := filepath.Join(dir, "gzip.txt.gz")
	bz := filepath.Join(dir, "bzip.txt.bz2")
	for p, n := range map[string]string{plain: "lib_data_open_plain", gz: "lib_data_gzip", bz: "lib_data_bzip2", filepath.Join(cache, "cached.dat"): "lib_data_open_cache"} {
		if e := os.WriteFile(p, inputBytes(t, n), 0600); e != nil {
			t.Fatal(e)
		}
	}
	read := func(p string) []byte {
		t.Helper()
		r := mf(d.Open(p, "rb"))
		v := mb(io.ReadAll(r))
		if e := r.Close(); e != nil {
			t.Fatal(e)
		}
		return v
	}
	for n, p := range map[string]string{"plain": plain, "gzip": filepath.Join(dir, "gzip.txt"), "bzip": filepath.Join(dir, "bzip.txt"), "cache": filepath.Join(cache, "cached.dat")} {
		refcheck.Bytes(t, "lib_data_open_"+n, read(p))
	}
	paths := []string{plain, gz, filepath.Join(dir, "gzip.txt"), "cached.dat", filepath.Join(dir, "missing")}
	x := make([]bool, len(paths))
	for i, p := range paths {
		x[i] = refcheck.RequireValue[bool](t)(d.Exists(p))
	}
	refcheck.Check(t, "lib_data_exists", m(np.FromSlice(x, len(x))))
	for n, p := range map[string]string{"relative": "folder/file.dat", "absolute": "/folder/file.dat", "up": "../../folder/file.dat", "space": "my file.txt"} {
		v := refcheck.RequireValue[string](t)(d.AbsPath(p))
		rel := refcheck.RequireValue[string](t)(filepath.Rel(cache, v))
		refcheck.Text(t, "lib_data_abspath_"+n, filepath.ToSlash(rel))
	}
	for _, tc := range []struct{ path, mode, text string }{{plain, "ab", "5,6\n"}, {gz, "wb", "7,8\n"}} {
		f := mf(d.Open(tc.path, tc.mode))
		if _, e := io.WriteString(f, tc.text); e != nil {
			t.Fatal(e)
		}
		if e := f.Close(); e != nil {
			t.Fatal(e)
		}
	}
	refcheck.Bytes(t, "lib_data_append", read(plain))
	refcheck.Bytes(t, "lib_data_gzip_write", read(gz))
	_, e := d.Open(filepath.Join(dir, "missing"), "rb")
	refcheck.Error(t, "lib_data_missing", "FileNotFoundError", e, os.ErrNotExist)
	_, e = d.Open(filepath.Join(dir, "missing"), "wb")
	refcheck.Error(t, "lib_data_missing_write", "FileNotFoundError", e, os.ErrNotExist)
	sample := inputBytes(t, "lib_data_http")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/numbers.txt" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(sample)
	}))
	defer server.Close()
	address := server.URL + "/numbers.txt"
	refcheck.Check(t, "lib_data_http_exists", np.ScalarOf(refcheck.RequireValue[bool](t)(d.Exists(address))))
	refcheck.Bytes(t, "lib_data_http", read(address))
	p := refcheck.RequireValue[string](t)(d.AbsPath(address))
	refcheck.Text(t, "lib_data_http_path", filepath.Base(p))
	refcheck.Bytes(t, "lib_data_http", mb(os.ReadFile(p)))
	_, e = d.Open(address, "wb")
	refcheck.Error(t, "lib_data_http_write", "ValueError", e, np.ErrValue)
	server.Close()
	refcheck.Check(t, "lib_data_http_cached_exists", np.ScalarOf(refcheck.RequireValue[bool](t)(d.Exists(address))))
	refcheck.Bytes(t, "lib_data_http_cached", read(address))
	tmp := md(NewDataSource(""))
	tmpPath := tmp.Destination()
	if e = tmp.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e = os.Stat(tmpPath); !errors.Is(e, os.ErrNotExist) {
		t.Errorf("private cache not removed: %v", e)
	}
}
