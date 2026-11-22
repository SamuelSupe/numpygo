package np_test

import (
	"bytes"
	"embed"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

//go:embed testdata/numpy/*
var numpyFiles embed.FS

func TestNumPyInterchange(t *testing.T) {
	names := []string{"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128", "fortran", "bigendian", "scalar", "empty", "version2", "version3"}
	dir := os.Getenv("NUMPYGO_INTERCHANGE_DIR")
	write := func(name string, data []byte) {
		t.Helper()
		if dir != "" {
			if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			data, err := numpyFiles.ReadFile("testdata/numpy/" + name + ".npy")
			if err != nil {
				t.Fatal(err)
			}
			a, err := np.LoadNPY(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			refcheck.CheckLayout(t, "io_npy_"+name, a)
			var buf bytes.Buffer
			if err = np.SaveNPY(&buf, a); err != nil {
				t.Fatal(err)
			}
			b, err := np.LoadNPY(bytes.NewReader(buf.Bytes()))
			if err != nil {
				t.Fatal(err)
			}
			refcheck.CheckLayout(t, "io_npy_"+name, b)
			write(name+".npy", buf.Bytes())
		})
	}
	for _, name := range []string{"stored", "compressed"} {
		t.Run(name, func(t *testing.T) {
			data, err := numpyFiles.ReadFile("testdata/numpy/" + name + ".npz")
			if err != nil {
				t.Fatal(err)
			}
			a, err := np.LoadNPZ(bytes.NewReader(data), int64(len(data)))
			if err != nil {
				t.Fatal(err)
			}
			refcheck.Check(t, "io_npy_uint64", a["a"])
			refcheck.CheckLayout(t, "io_npy_fortran", a["b"])
			var buf bytes.Buffer
			if err = np.SaveNPZ(&buf, a, name == "compressed"); err != nil {
				t.Fatal(err)
			}
			b, err := np.LoadNPZ(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
			if err != nil {
				t.Fatal(err)
			}
			refcheck.Check(t, "io_npy_uint64", b["a"])
			refcheck.CheckLayout(t, "io_npy_fortran", b["b"])
			write(name+".npz", buf.Bytes())
		})
	}
	structured, err := numpyFiles.ReadFile("testdata/numpy/structured.npy")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = np.LoadNPY(bytes.NewReader(structured)); err == nil {
		t.Fatal("structured dtype unexpectedly read")
	}
	if _, err = np.LoadNPY(bytes.NewReader([]byte("not-npy"))); err == nil {
		t.Fatal("bad signature unexpectedly read")
	}
	data, err := numpyFiles.ReadFile("testdata/numpy/int64.npy")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = np.LoadNPY(bytes.NewReader(data[:len(data)-1])); err == nil || errors.Is(err, np.ErrValue) {
		t.Fatal("truncated file not detected", err)
	}
}
