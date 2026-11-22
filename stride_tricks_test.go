package np_test

import (
	"errors"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"math"
	"testing"
)

func TestStrideTricksNumPy(t *testing.T) {
	m := refcheck.Require(t)
	create := func() *np.Array {
		return m(np.FromSlice([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 4))
	}
	a := create()
	v := m(np.AsStrided(a, []int{3, 2}, []int{4, 8}, true))
	refcheck.CheckLayout(t, "stride_overlap", v)
	if !np.SharesMemory(a, v) {
		t.Fatal("as_strided must share storage")
	}
	if e := v.SetValue(int32(88), 1, 0); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "stride_writeback", a)
	rev := m(a.Slice(np.AtIndex(2), np.Reverse()))
	refcheck.CheckLayout(t, "stride_reverse", m(np.AsStrided(rev, []int{3, 4}, []int{-16, -4}, false)))
	refcheck.CheckLayout(t, "stride_zero", m(np.AsStrided(a, []int{2, 3}, []int{0, 4}, true)))
	bytes := m(np.FromSlice([]uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 12))
	v = m(np.AsStrided(m(bytes.View(np.Int16)), []int{5}, []int{1}, false))
	refcheck.CheckLayout(t, "stride_unaligned", v)
	refcheck.Error(t, "stride_readonly", "ValueError", v.SetValue(int16(1), 0), np.ErrReadOnly)

	a = create()
	for _, tc := range []struct {
		name         string
		a            *np.Array
		window, axes []int
	}{
		{"all", a, []int{2, 3}, nil},
		{"negative", m(a.Slice(np.Reverse(), np.Reverse())), []int{2}, []int{-1}},
		{"repeated", a, []int{2, 2}, []int{1, 1}},
		{"transposed", m(a.Transpose()), []int{2}, []int{0}},
		{"zero", a, []int{0}, []int{1}},
		{"scalar", m(np.FromSlice([]int32{9})), []int{}, nil},
		{"empty_axis", m(np.Zeros(np.Int32, 0, 3)), []int{2}, []int{1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := refcheck.Require(t)(np.SlidingWindowView(tc.a, tc.window, tc.axes, false))
			refcheck.CheckLayout(t, "window_"+tc.name, out)
			if out.Writable() {
				t.Error("default window must be read-only")
			}
		})
	}
	v = m(np.SlidingWindowView(a, []int{2}, []int{0}, true))
	if e := v.SetValue(int32(99), 1, 2, 1); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "window_writeback", a)
	o := m(np.FromObjects([]any{int64(1), int64(2), int64(3), int64(4)}, 4))
	refcheck.Check(t, "window_object", m(m(np.SlidingWindowView(o, []int{2}, nil, false)).Astype(np.Int64)))
	for _, tc := range []struct {
		n, cls       string
		window, axes []int
		err          error
	}{
		{"bad_axis", "AxisError", []int{2}, []int{2}, np.ErrAxis},
		{"bad_size", "ValueError", []int{5}, []int{1}, np.ErrValue},
		{"negative_size", "ValueError", []int{-1}, []int{1}, np.ErrValue},
		{"bad_count", "ValueError", []int{2}, []int{0, 1}, np.ErrShape},
	} {
		_, e := np.SlidingWindowView(a, tc.window, tc.axes, false)
		refcheck.Error(t, "window_"+tc.n, tc.cls, e, tc.err)
	}
	ro := m(np.SlidingWindowView(a, []int{2}, []int{1}, false))
	refcheck.Error(t, "window_readonly", "ValueError", ro.SetValue(int32(1), 0, 0, 0), np.ErrReadOnly)
	// NumPy permits unchecked addresses; Go intentionally rejects them.
	for _, s := range []int{-4, 48, math.MaxInt, math.MinInt} {
		if _, e := np.AsStrided(a, []int{2}, []int{s}, true); !errors.Is(e, np.ErrShape) {
			t.Errorf("unsafe stride %d: %v", s, e)
		}
	}
	if _, e := np.AsStrided(o, []int{2}, []int{1}, true); !errors.Is(e, np.ErrShape) {
		t.Errorf("unaligned object slot: %v", e)
	}
}
