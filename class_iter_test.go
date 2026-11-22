package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func classInts(t *testing.T, name string, v []int64, shape ...int) {
	t.Helper()
	refcheck.Check(t, "class_"+name, refcheck.Require(t)(np.FromSlice(v, shape...)))
}
func classIntSlice(t *testing.T, name string, v []int) {
	t.Helper()
	x := make([]int64, len(v))
	for i, k := range v {
		x[i] = int64(k)
	}
	classInts(t, name, x, len(x))
}
func classBool(v bool) int64 {
	if v {
		return 1
	}
	return 0
}
func classOK(t *testing.T, e error) {
	t.Helper()
	if e != nil {
		t.Fatal(e)
	}
}

func TestFlatIteratorClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	fm := refcheck.RequireValue[*np.FlatIterator](t)
	a := m(m(np.ArangeInt64(0, 6, 1)).Reshape(2, 3))
	a = m(a.Transpose())
	f := fm(a.FlatIterator())
	state, values := []int64{}, []int64{}
	for i := 0; i <= a.Size(); i++ {
		state = append(state, int64(f.Index()))
		for _, x := range f.Coords() {
			state = append(state, int64(x))
		}
		v, ok, e := f.Next()
		classOK(t, e)
		if ok {
			values = append(values, v.(int64))
		} else if i != a.Size() {
			t.Fatal("premature exhaustion")
		}
	}
	classInts(t, "flat_state", state, 7, 3)
	classInts(t, "flat_values", values, 6)
	classIntSlice(t, "flat_base", f.Base().Shape())
	f = fm(a.FlatIterator())
	_, _, e := f.Next()
	classOK(t, e)
	refcheck.Check(t, "class_flat_copy", m(f.Copy()))
	refcheck.Check(t, "class_flat_array", m(f.Array()))
	classInts(t, "flat_copy_cursor", []int64{int64(f.Index()), int64(f.Len())}, 2)
	refcheck.Check(t, "class_flat_slice", m(f.Get(np.Slice{Start: np.Int(1), Stop: np.Int(5), Step: np.Int(2)})))
	classInts(t, "flat_get_cursor", []int64{int64(f.Index())}, 1)
	refcheck.Check(t, "class_flat_scalar", m(f.Get(np.AtIndex(-1))))
	refcheck.Check(t, "class_flat_gather", m(f.Get(m(np.FromSlice([]int64{0, -1, 2, 2}, 2, 2)))))
	classOK(t, f.Assign(np.Range(1, 5), m(np.FromSlice([]int64{10, 20}, 2))))
	refcheck.Check(t, "class_flat_assign", a)
	classOK(t, f.Assign(m(np.FromSlice([]int64{0, 0, 5}, 3)), m(np.FromSlice([]int64{9, 8, 7}, 3))))
	refcheck.Check(t, "class_flat_repeated", a)
	classOK(t, f.Assign(m(np.FromSlice([]bool{true, false, true, false, true, false}, 6)), m(np.FromSlice([]int64{99, 88}, 2))))
	refcheck.Check(t, "class_flat_mask", a)
	classOK(t, f.Assign(np.Slice{}, m(np.Zeros(np.Float64, 0))))
	refcheck.Check(t, "class_flat_empty_rhs", a)
	_, e = f.Get(np.AtIndex(6))
	refcheck.Error(t, "class_flat_oob", "IndexError", e, np.ErrIndex)
	_, e = f.Get(m(np.FromSlice([]bool{true, false}, 2)))
	refcheck.Error(t, "class_flat_bad_mask", "IndexError", e, np.ErrIndex)
	c := m(np.ArangeInt64(0, 4, 1))
	view := m(fm(c.FlatIterator()).Array())
	classOK(t, view.SetValue(int64(99), 0))
	refcheck.Check(t, "class_flat_shared", c)
}

func TestBroadcastIteratorClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	bm := refcheck.RequireValue[*np.BroadcastIterator](t)
	a := m(m(np.ArangeInt64(0, 3, 1)).Reshape(1, 3))
	b := m(m(np.ArangeInt64(0, 2, 1)).Reshape(2, 1))
	it := bm(np.NewBroadcast(a, b))
	classIntSlice(t, "broadcast_meta", append([]int{it.NDim(), it.NDim(), it.NumIter(), it.Size()}, it.Shape()...))
	state, values := []int64{}, []int64{}
	for i := 0; i < it.Size(); i++ {
		state = append(state, int64(it.Index()))
		for _, f := range it.Iterators() {
			state = append(state, int64(f.Index()))
		}
		row, ok, e := it.Next()
		classOK(t, e)
		if !ok {
			t.Fatal("premature exhaustion")
		}
		for _, x := range row {
			values = append(values, x.(int64))
		}
	}
	classInts(t, "broadcast_state", state, 6, 3)
	classInts(t, "broadcast_values", values, 6, 2)
	it.Reset()
	r := []int64{int64(it.Index())}
	for _, f := range it.Iterators() {
		r = append(r, int64(f.Index()))
	}
	row, ok, e := it.Next()
	classOK(t, e)
	if !ok {
		t.Fatal("reset exhausted")
	}
	for _, v := range row {
		r = append(r, v.(int64))
	}
	classInts(t, "broadcast_reset", r, 5)
	it = bm(np.NewBroadcast())
	row, ok, e = it.Next()
	classOK(t, e)
	if !ok {
		t.Fatal("scalar broadcast exhausted")
	}
	classInts(t, "broadcast_zero", []int64{int64(it.NDim()), int64(it.Size()), int64(it.NumIter()), int64(len(row))}, 4)
	_, e = np.NewBroadcast(m(np.Zeros(np.Float64, 2)), m(np.Zeros(np.Float64, 3)))
	refcheck.Error(t, "class_broadcast_shape", "ValueError", e, np.ErrShape)
	im := refcheck.RequireValue[*np.IndexIterator](t)
	ix := im(np.NewNDIndex(2, 3))
	values = nil
	for p, ok := ix.Next(); ok; p, ok = ix.Next() {
		for _, x := range p {
			values = append(values, int64(x))
		}
	}
	classInts(t, "ndindex", values, 6, 2)
	counts := []int64{}
	for _, ix := range []*np.IndexIterator{im(np.NewNDIndex(2, 0)), im(np.NewNDIndex())} {
		var n int64
		for _, ok := ix.Next(); ok; _, ok = ix.Next() {
			n++
		}
		counts = append(counts, n)
	}
	classInts(t, "ndindex_empty", counts, 2)
	en := refcheck.RequireValue[*np.EnumerateIterator](t)(np.NewNDEnumerate(m(m(m(np.ArangeInt64(0, 6, 1)).Reshape(2, 3)).Transpose())))
	values = nil
	first := true
	for {
		p, v, ok, e := en.Next()
		classOK(t, e)
		if !ok {
			break
		}
		if first {
			classIntSlice(t, "ndenumerate_iter", append([]int{en.Iterator().Index()}, en.Iterator().Coords()...))
			first = false
		}
		for _, x := range p {
			values = append(values, int64(x))
		}
		values = append(values, v.(int64))
	}
	classInts(t, "ndenumerate", values, 6, 3)
}
