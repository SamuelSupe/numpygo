package np_test

import (
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestNDIteratorClassNumPy(t *testing.T) {
	m := refcheck.Require(t)
	im := refcheck.RequireValue[*np.NDIterator](t)
	vm := refcheck.RequireValue[[]*np.Array](t)
	intm := refcheck.RequireValue[int](t)
	coordm := refcheck.RequireValue[[]int](t)
	a := m(m(np.ArangeInt64(0, 12, 1)).Reshape(3, 4))
	tr := m(a.Transpose())
	rev := m(a.Slice(np.Slice{Step: np.Int(-1)}, np.Slice{Step: np.Int(-1)}))
	gap := m(a.Slice(np.Slice{}, np.Slice{Step: np.Int(2)}))
	for _, tc := range []struct {
		name string
		args []*np.Array
		opt  np.NDIterOptions
	}{
		{"c", []*np.Array{a}, np.NDIterOptions{Order: "C"}},
		{"f", []*np.Array{a}, np.NDIterOptions{Order: "F"}},
		{"transpose", []*np.Array{tr}, np.NDIterOptions{Order: "K"}},
		{"reverse", []*np.Array{rev}, np.NDIterOptions{Order: "K"}},
		{"reverse_c", []*np.Array{rev}, np.NDIterOptions{Order: "C", MultiIndex: true}},
		{"multi", []*np.Array{tr}, np.NDIterOptions{MultiIndex: true}},
		{"c_index", []*np.Array{tr}, np.NDIterOptions{IndexOrder: "C"}},
		{"f_index", []*np.Array{a}, np.NDIterOptions{Order: "C", IndexOrder: "F"}},
		{"external", []*np.Array{a}, np.NDIterOptions{Order: "F", ExternalLoop: true}},
		{"gap", []*np.Array{gap}, np.NDIterOptions{Order: "C"}},
		{"broadcast", []*np.Array{a, m(np.ArangeInt64(0, 4, 1))}, np.NDIterOptions{Order: "C", ExternalLoop: true}},
		{"scalar", []*np.Array{m(np.FromSlice([]int64{9}))}, np.NDIterOptions{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			it := im(np.NewNDIterator(tc.args, tc.opt))
			defer it.Close()
			p := "nditer_" + tc.name
			classIntSlice(t, p+"_shape", it.Shape())
			classInts(t, p+"_meta", []int64{int64(it.NDim()), int64(it.NOp()), int64(it.IterSize()), classBool(it.HasIndex()), classBool(it.HasMultiIndex()), classBool(it.IterationNeedsAPI())}, 6)
			ds := []string{}
			for _, d := range it.DTypes() {
				ds = append(ds, d.String())
			}
			refcheck.Text(t, "class_"+p+"_dtypes", strings.Join(ds, ","))
			for j, v := range vm(it.ItViews()) {
				classIntSlice(t, p+"_view"+string(rune('0'+j)), append(v.Shape(), v.Strides()...))
			}
			rows, steps, track := []int64{}, []int64{}, []int64{}
			for {
				v, ok, e := it.Next()
				classOK(t, e)
				if !ok {
					break
				}
				data := make([][]int64, len(v))
				for j, x := range v {
					data[j] = refcheck.RequireValue[[]int64](t)(np.Values[int64](x))
				}
				for i := 0; i < v[0].Size(); i++ {
					for j := range v {
						rows = append(rows, data[j][i])
					}
				}
				steps = append(steps, int64(intm(it.IterIndex())), int64(v[0].Size()))
				if it.HasMultiIndex() {
					for _, k := range coordm(it.MultiIndex()) {
						track = append(track, int64(k))
					}
				} else if it.HasIndex() {
					track = append(track, int64(intm(it.Index())))
				}
			}
			classInts(t, p+"_values", rows, len(rows)/it.NOp(), it.NOp())
			classInts(t, p+"_steps", steps, len(steps)/2, 2)
			if len(track) > 0 {
				n := 1
				if it.HasMultiIndex() {
					n = it.NDim()
				}
				classInts(t, p+"_tracked", track, len(track)/n, n)
			}
		})
	}
	scalar := func(a *np.Array) int64 { return refcheck.RequireValue[any](t)(a.AtValue()).(int64) }
	it := im(np.NewNDIterator([]*np.Array{a}, np.NDIterOptions{MultiIndex: true, IndexOrder: "C", Ranged: true}))
	classOK(t, it.SetMultiIndex([]int{1, 2}))
	p := []int64{int64(intm(it.Index())), int64(intm(it.IterIndex())), scalar(vm(it.Value())[0])}
	for _, k := range coordm(it.MultiIndex()) {
		p = append(p, int64(k))
	}
	classInts(t, "nditer_seek", p, 5)
	cp := im(it.Copy())
	classOK(t, it.SetIndex(2))
	classInts(t, "nditer_copy", []int64{int64(intm(cp.IterIndex())), scalar(vm(cp.Value())[0]), int64(intm(it.IterIndex())), scalar(vm(it.Value())[0])}, 4)
	classOK(t, cp.Close())
	classOK(t, it.Close())
	it = im(np.NewNDIterator([]*np.Array{a}, np.NDIterOptions{Ranged: true}))
	classOK(t, it.SetIterRange(3, 7))
	begin, end := it.IterRange()
	classIntSlice(t, "nditer_range_meta", []int{begin, end})
	values := []int64{}
	for v, e := range it.Iter() {
		classOK(t, e)
		values = append(values, scalar(v[0]))
	}
	classInts(t, "nditer_range", values, 4)
	classOK(t, it.Reset())
	classInts(t, "nditer_reset", []int64{int64(intm(it.IterIndex())), scalar(vm(it.Value())[0])}, 2)
	classOK(t, it.SetIterIndex(5))
	classInts(t, "nditer_seek_iterindex", []int64{int64(intm(it.IterIndex())), scalar(vm(it.Value())[0])}, 2)
	classOK(t, it.Close())
	it = im(np.NewNDIterator([]*np.Array{a}, np.NDIterOptions{MultiIndex: true}))
	classOK(t, it.RemoveAxis(1))
	values = nil
	for v, e := range it.Iter() {
		classOK(t, e)
		values = append(values, scalar(v[0]))
	}
	classInts(t, "nditer_remove_axis", values, 3)
	classOK(t, it.Close())
	it = im(np.NewNDIterator([]*np.Array{a}, np.NDIterOptions{MultiIndex: true}))
	classOK(t, it.RemoveMultiIndex())
	classOK(t, it.EnableExternalLoop())
	values = nil
	blocks := 0
	for v, e := range it.Iter() {
		classOK(t, e)
		values = append(values, refcheck.RequireValue[[]int64](t)(np.Values[int64](v[0]))...)
		blocks++
	}
	classInts(t, "nditer_external_enable", values, blocks, len(values)/blocks)
	classOK(t, it.Close())
	it = im(np.NewNDIterator([]*np.Array{a}, np.NDIterOptions{IndexOrder: "C"}))
	values = []int64{scalar(vm(it.Value())[0])}
	for {
		ok, e := it.IterNext()
		classOK(t, e)
		if !ok {
			break
		}
		values = append(values, scalar(m(it.Get(0))))
	}
	classInts(t, "nditer_iternext", values, 12)
	classOK(t, it.Close())
	it = im(np.NewNDIterator([]*np.Array{a, nil}, np.NDIterOptions{Access: []string{"readonly", "writeonly"}, DTypes: []np.DType{np.Int64, np.Float64}}))
	two := m(np.FromSlice([]int64{2}))
	for v, e := range it.Iter() {
		classOK(t, e)
		classOK(t, it.Set(1, m(np.Multiply(v[0], two))))
	}
	refcheck.Check(t, "class_nditer_allocate", it.Operands()[1])
	classOK(t, it.Close())
	it = im(np.NewNDIterator([]*np.Array{a}, np.NDIterOptions{ExternalLoop: true, Buffered: true, DelayBufAlloc: true, Access: []string{"readwrite"}, DTypes: []np.DType{np.Float64}, Casting: "unsafe", BufferSize: 3}))
	classInts(t, "nditer_delayed", []int64{classBool(it.HasDelayedBufAlloc())}, 1)
	classOK(t, it.Reset())
	one := m(np.FromSlice([]float64{1}))
	for v, e := range it.Iter() {
		classOK(t, e)
		classOK(t, it.Set(0, m(np.Add(m(np.Multiply(v[0], two)), one))))
	}
	classOK(t, it.Close())
	refcheck.Check(t, "class_nditer_writeback", a)
	it = im(np.NewNDIterator([]*np.Array{m(np.Zeros(np.Float64, 0, 2))}, np.NDIterOptions{ZerosizeOK: true}))
	classInts(t, "nditer_empty", []int64{int64(it.IterSize()), classBool(it.Finished())}, 2)
	var debug strings.Builder
	classOK(t, it.DebugPrint(&debug))
	if debug.Len() == 0 {
		t.Fatal("missing diagnostics")
	}
	classOK(t, it.Close())
	_, e := np.NewNDIterator([]*np.Array{m(np.Zeros(np.Float64, 0))}, np.NDIterOptions{})
	refcheck.Error(t, "class_nditer_zero_error", "ValueError", e, np.ErrValue)
	_, e = np.NewNDIterator([]*np.Array{a}, np.NDIterOptions{ExternalLoop: true, MultiIndex: true})
	refcheck.Error(t, "class_nditer_flags_error", "ValueError", e, np.ErrValue)
}
