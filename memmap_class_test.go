package np_test

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestMemMapClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	path := filepath.Join(t.TempDir(), "mapped.bin")
	a := refcheck.RequireValue[*np.MappedArray](t)(np.MemMap(path, np.MemMapOptions{DType: np.Int32, Mode: "w+", Offset: 3, Shape: []int{2, 3}, Order: "F", ByteOrder: binary.BigEndian}))
	defer a.Close()
	classOK(t, a.Assign(nil, m(m(np.ArangeInt64(0, 6, 1)).Reshape(2, 3))))
	v := m(m(a.Slice(np.Slice{}, np.Slice{Step: np.Int(-1)})).Transpose())
	info, ok := a.MappingInfo()
	if !ok {
		t.Fatal("missing mapping metadata")
	}
	vi, ok := v.MappingInfo()
	if !ok {
		t.Fatal("missing view mapping metadata")
	}
	refcheck.Text(t, "class_memmap_filename", filepath.Base(info.Filename))
	refcheck.Text(t, "class_memmap_mode", info.Mode)
	refcheck.Text(t, "class_memmap_view_filename", filepath.Base(vi.Filename))
	refcheck.Text(t, "class_memmap_view_mode", vi.Mode)
	if info.Filename != path || vi.Filename != path {
		t.Fatal("incorrect original file identity")
	}
	data := append([]int{int(info.Offset)}, a.Shape()...)
	data = append(data, a.Strides()...)
	data = append(data, int(vi.Offset))
	data = append(data, v.Shape()...)
	data = append(data, v.Strides()...)
	classIntSlice(t, "memmap_metadata", data)
	cp := m(a.Copy())
	empty, ok := cp.MappingInfo()
	classInts(t, "memmap_copy", []int64{classBool(!ok && empty.Filename == ""), classBool(!ok), classBool(!ok && empty.Mode == "")}, 3)
	classOK(t, v.SetValue(int32(123), 0, 1))
	classOK(t, v.FlushMapping())
	refcheck.Check(t, "class_memmap_view", v)
	raw := refcheck.RequireValue[[]byte](t)(os.ReadFile(path))
	refcheck.Bytes(t, "class_memmap_bytes", raw)
}
