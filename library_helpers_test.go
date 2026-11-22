package np_test

import (
	"encoding/binary"
	"errors"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"github.com/SamuelSupe/numpygo/records"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestArrayUtilsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	a := m(m(m(np.ArangeInt64(0, 24, 1)).Astype(np.Int32)).Reshape(2, 3, 4))
	s := refcheck.RequireValue[*records.Schema](t)(records.NewSchema([]records.FieldSpec{
		{Name: "tag", Type: records.Numeric(np.Uint8, nil)},
		{Name: "x", Type: records.Numeric(np.Int32, binary.BigEndian)},
	}, false, 0))
	re := m(np.ZerosRecords(s, 3))
	c := m(np.FromComplex128([]complex128{1 + 2i, 3 + 4i}, 2))
	u := m(m(np.ArangeInt64(0, 16, 1)).Astype(np.Uint8))
	v := m(np.AsStrided(m(u.View(np.Int16)), []int{4}, []int{1}, false))
	arrays := map[string]*np.Array{
		"c": a, "f": m(a.Transpose(2, 1, 0)), "reverse": m(a.Slice(np.Reverse(), np.Range(1, 3), np.Slice{Step: np.Int(-2)})),
		"gapped":    m(a.Slice(np.Slice{}, np.Strided(0, 3, 2), np.Strided(0, 4, 2))),
		"broadcast": m(m(a.Slice(np.AtIndex(0), np.AtIndex(0), np.Range(0, 1))).BroadcastTo(5, 2)),
		"unaligned": v, "scalar": np.ScalarOf(int64(4)), "empty": m(np.Zeros(np.Float64, 2, 0, 3)),
		"record": re, "field": m(m(re.Field("x")).Slice(np.Reverse())), "real": m(np.Real(c)), "imag": m(m(np.Imag(c)).Slice(np.Reverse())),
		"unicode": m(np.FromFlexible([]any{"a", "中文"}, np.FixedUnicode(3, nil), 2)),
		"void0":   m(np.ZerosFlexible(np.FixedVoid(0), 4)),
		"object":  m(m(np.FromObjects([]any{int64(1), int64(2), int64(3)}, 3)).Slice(np.Reverse())),
	}
	for name, v := range arrays {
		t.Run("bounds_"+name, func(t *testing.T) {
			lo, hi, e := np.ByteBounds(v)
			if e != nil {
				t.Fatal(e)
			}
			start := v.DataIdentity()
			low, high := int64(lo-start), int64(hi-start)
			if name == "object" {
				w := int64(reflect.TypeFor[any]().Size())
				low /= w
				high /= w
			}
			refcheck.Check(t, "lib_bounds_"+name, m(np.FromSlice([]int64{low, high}, 2)))
		})
	}
	axes := refcheck.RequireValue[[]int64](t)(np.Values[int64](refcheck.Input(t, "lib_axis_indices")))
	want := refcheck.RequireValue[[]int64](t)(np.Values[int64](refcheck.Input(t, "lib_axis_matrix")))
	for n := 0; n < 8; n++ {
		for i, x := range axes {
			k, e := np.NormalizeAxisIndex(int(x), n, "")
			w := want[n*len(axes)+i]
			if w == -999 {
				if !errors.Is(e, np.ErrAxis) {
					t.Errorf("axis %d rank %d: %v", x, n, e)
				}
			} else if e != nil || int64(k) != w {
				t.Errorf("axis %d rank %d: %d %v, want %d", x, n, k, e, w)
			}
		}
	}
	_, e := np.NormalizeAxisIndex(9, 3, "window")
	refcheck.Error(t, "lib_axis_name", "AxisError", e, np.ErrAxis)
	if !strings.Contains(e.Error(), "window:") {
		t.Error("axis name missing")
	}
	for _, tc := range []struct {
		name string
		axes []int
		rank int
		dup  bool
	}{
		{"negative", []int{-3, -1}, 3, false}, {"order", []int{2, 0, -2}, 3, false}, {"dup", []int{0, -3, 1}, 3, true},
		{"empty", nil, 0, false}, {"scalar", []int{1}, 3, false},
	} {
		arr := refcheck.RequireValue[[]int](t)(np.NormalizeAxisTuple(tc.axes, tc.rank, tc.dup))
		x := make([]int64, len(arr))
		for i, k := range arr {
			x[i] = int64(k)
		}
		refcheck.Check(t, "lib_axis_tuple_"+tc.name, m(np.FromSlice(x, len(x))))
	}
	_, e = np.NormalizeAxisTuple([]int{0, -3}, 3, false)
	refcheck.Error(t, "lib_axis_tuple_duplicate", "ValueError", e, np.ErrValue)
	_, e = np.NormalizeAxisTuple([]int{0, 3}, 3, false)
	refcheck.Error(t, "lib_axis_tuple_bad", "AxisError", e, np.ErrAxis)
}

func TestArrayteratorNumPy(t *testing.T) {
	m := refcheck.Require(t)
	it := refcheck.RequireValue[*np.Arrayterator](t)
	b := m(m(np.ArangeInt64(0, 60, 1)).Reshape(3, 4, 5))
	ints := func(v []int) []int64 {
		x := make([]int64, len(v))
		for i, k := range v {
			x[i] = int64(k)
		}
		return x
	}
	check := func(t *testing.T, n string, a *np.Arrayterator) {
		t.Helper()
		m := refcheck.Require(t)
		s := a.Shape()
		ix := append(append(ints(a.Start()), ints(a.Stop())...), ints(a.Step())...)
		refcheck.Check(t, n+"_indices", m(np.FromSlice(ix, 3, len(s))))
		refcheck.Check(t, n+"_buffer", np.ScalarOf(int64(a.BufferSize())))
		refcheck.Check(t, n+"_shape", m(np.FromSlice(ints(s), len(s))))
		refcheck.CheckLayout(t, n+"_array", m(a.Array()))
		refcheck.CheckLayout(t, n+"_asarray", a.Var())
		shapes, sizes, edges := []int64{}, []int64{}, []int64{}
		flat := []int64{}
		for v, e := range a.Blocks() {
			if e != nil {
				t.Fatal(e)
			}
			if a.BufferSize() > 0 && v.Size() > a.BufferSize() {
				t.Fatal("block exceeds buffer")
			}
			if !np.SharesMemory(v, a.Var()) {
				t.Fatal("block must share input")
			}
			shapes = append(shapes, ints(v.Shape())...)
			sizes = append(sizes, int64(v.Size()))
			data := refcheck.RequireValue[[]int64](t)(np.Values[int64](v))
			edges = append(edges, data[0], data[len(data)-1])
			flat = append(flat, data...)
		}
		refcheck.Check(t, n+"_blocks", m(np.FromSlice(shapes, len(sizes), len(s))))
		refcheck.Check(t, n+"_sizes", m(np.FromSlice(sizes, len(sizes))))
		refcheck.Check(t, n+"_edges", m(np.FromSlice(edges, len(sizes), 2)))
		refcheck.Check(t, n+"_flat", m(np.FromSlice(flat, len(flat))))
		flat = flat[:0]
		for v, e := range a.Flat() {
			if e != nil {
				t.Fatal(e)
			}
			flat = append(flat, v.(int64))
		}
		refcheck.Check(t, n+"_flat", m(np.FromSlice(flat, len(flat))))
	}
	for _, n := range []int{0, 1, 2, 5, 7, 20, 21, 59, 100} {
		t.Run("buffer_"+strconv.Itoa(n), func(t *testing.T) { check(t, "lib_iter_"+strconv.Itoa(n), it(np.NewArrayterator(b, n))) })
	}
	a := it(np.NewArrayterator(b, 4))
	check(t, "lib_iter_slice", it(a.Slice(np.Strided(1, 3, 2), np.Ellipsis, np.Strided(1, 5, 2))))
	a = it(np.NewArrayterator(b, 3))
	check(t, "lib_iter_index", it(a.Slice(np.AtIndex(1), np.Strided(0, 4, 2))))
	a = it(a.Slice(np.Range(1, 3), np.Slice{Step: np.Int(2)}, np.Strided(1, 5, 2)))
	check(t, "lib_iter_nested", it(a.Slice(np.Slice{}, np.Range(1, 3), np.Range(1, 3))))
	check(t, "lib_iter_noncontiguous", it(np.NewArrayterator(m(m(b.Slice(np.Reverse())).Transpose(2, 0, 1)), 5)))
	check(t, "lib_iter_empty", it(np.NewArrayterator(m(np.Zeros(np.Int64, 2, 0, 3)), 3)))
	mapped := refcheck.RequireValue[*np.MappedArray](t)(np.MemMap(filepath.Join(t.TempDir(), "blocks.bin"), np.MemMapOptions{DType: np.Int64, Shape: []int{3, 4, 5}, Mode: "w+"}))
	if e := mapped.Assign(nil, b); e != nil {
		t.Fatal(e)
	}
	mi := it(np.NewArrayterator(mapped.Array, 7))
	check(t, "lib_iter_mmap", it(mi.Slice(np.Slice{}, np.Strided(1, 4, 2), np.Range(1, 5))))
	if e := mapped.Close(); e != nil {
		t.Fatal(e)
	}
	a = it(np.NewArrayterator(b, 2))
	for v, e := range a.Blocks() {
		if e != nil {
			t.Fatal(e)
		}
		refcheck.Check(t, "lib_iter_shares", np.ScalarOf(np.SharesMemory(b, v)))
		if e = v.SetValue(int64(99), 0, 0, 0); e != nil {
			t.Fatal(e)
		}
		break
	}
	refcheck.Check(t, "lib_iter_writeback", b)
	a = it(np.NewArrayterator(np.ScalarOf(int64(5)), 2))
	refcheck.Check(t, "lib_iter_scalar", m(a.Array()))
	count := 0
	for v, e := range a.Blocks() {
		refcheck.Check(t, "lib_iter_scalar_first", m(v, e))
		count++
	}
	if count != 1 {
		t.Fatal("scalar iterator must terminate")
	}
	if _, e := a.Slice(np.Reverse()); e == nil {
		t.Error("out-of-rank iterator slice accepted")
	}
	bad := it(np.NewArrayterator(b, 2))
	if _, e := bad.Slice(np.Reverse()); !errors.Is(e, np.ErrValue) {
		t.Errorf("negative iteration step: %v", e)
	}
}
