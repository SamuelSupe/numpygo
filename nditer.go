package np

import (
	"fmt"
	"io"
	"iter"
	"slices"
)

// NDIterOptions describes explicit iterator flags. Nil operands may be
// allocated by providing their dtype and write access. Axis order K uses
// stride-based traversal; C/F use logical order and A chooses contiguous order.
type NDIterOptions struct {
	Order, IndexOrder, Casting                                     string
	MultiIndex, ExternalLoop, Ranged, ZerosizeOK, RefsOK, ReduceOK bool
	Buffered, DelayBufAlloc                                        bool
	BufferSize                                                     int
	Access                                                         []string // readonly, readwrite, writeonly
	DTypes                                                         []DType  // per-operand loop dtype; 0 retains dtype
	Shape                                                          []int    // optional explicit broadcast iteration shape
	OpAxes                                                         [][]int  // each iteration axis maps to an operand axis, or -1
	NoBroadcast                                                    []bool
	CommonDType                                                    bool
	MaskOperand                                                    *int
	WriteMasked                                                    []bool
}

// NDIterator is a retained, stateful multidimensional iterator. Current
// values are scalar views, or one-dimensional chunks with ExternalLoop.
// Buffered writable operands are written back when the buffer is exhausted,
// reset, repositioned, or closed. Close before releasing any mapped storage.
type NDIterator struct {
	opt                      NDIterOptions
	operands, logical, views []*Array
	access                   []string
	dtypes                   []DType
	shape, axes              []int
	flips                    []bool
	size, begin, end, pos    int
	started, closed, delayed bool
	bufferStart, bufferEnd   int
	buffers                  []*Array
}

func NewNDIterator(arrays []*Array, opt NDIterOptions) (*NDIterator, error) {
	if len(arrays) == 0 || len(opt.Access) != 0 && len(opt.Access) != len(arrays) ||
		len(opt.DTypes) != 0 && len(opt.DTypes) != len(arrays) || opt.BufferSize < 0 ||
		len(opt.OpAxes) != 0 && len(opt.OpAxes) != len(arrays) || len(opt.NoBroadcast) != 0 && len(opt.NoBroadcast) != len(arrays) {
		return nil, ErrValue
	}
	if len(opt.WriteMasked) > 0 && len(opt.WriteMasked) != len(arrays) {
		return nil, ErrShape
	}
	if opt.MaskOperand != nil {
		i := *opt.MaskOperand
		if i < 0 || i >= len(arrays) || arrays[i].DType() != Bool || !opt.Buffered || opt.CommonDType {
			return nil, ErrValue
		}
		if len(opt.DTypes) > 0 && opt.DTypes[i] != 0 && opt.DTypes[i] != Bool {
			return nil, ErrDType
		}
		opt.MaskOperand = Int(i)
		opt.WriteMasked = slices.Clone(opt.WriteMasked)
	} else {
		for _, v := range opt.WriteMasked {
			if v {
				return nil, ErrValue
			}
		}
	}
	switch opt.Order {
	case "", "K":
		opt.Order = "K"
	case "C", "F", "A":
	default:
		return nil, ErrValue
	}
	if opt.IndexOrder != "" && opt.IndexOrder != "C" && opt.IndexOrder != "F" {
		return nil, ErrValue
	}
	if opt.ExternalLoop && (opt.MultiIndex || opt.IndexOrder != "" || opt.Ranged && !opt.Buffered) || opt.DelayBufAlloc && !opt.Buffered {
		return nil, ErrValue
	}
	if opt.Casting == "" {
		opt.Casting = "safe"
	}
	if _, e := CanCast(Bool, Bool, opt.Casting); e != nil {
		return nil, e
	}
	shapes := [][]int{}
	mapped := slices.Clone(arrays)
	rank := -1
	for _, m := range opt.OpAxes {
		if m == nil {
			continue
		}
		if rank >= 0 && rank != len(m) {
			return nil, ErrShape
		}
		rank = len(m)
	}
	if rank >= 0 && opt.Shape != nil && rank != len(opt.Shape) {
		return nil, ErrShape
	}
	for i, a := range arrays {
		if a != nil {
			if a.DType() == 0 {
				return nil, ErrDType
			}
			if rank >= 0 {
				if a.NDim() > rank {
					return nil, ErrShape
				}
				m := opt.OpAxes[i]
				if m != nil {
					var e error
					mapped[i], e = iteratorMappedOperand(a, m)
					if e != nil {
						return nil, e
					}
				}
			}
			shapes = append(shapes, mapped[i].shape)
		}
	}
	if opt.Shape != nil {
		shapes = append(shapes, opt.Shape)
	}
	if rank >= 0 {
		ones := make([]int, rank)
		for i := range ones {
			ones[i] = 1
		}
		shapes = append(shapes, ones)
	}
	if len(shapes) == 0 {
		return nil, ErrShape
	}
	sh, e := BroadcastShapes(shapes...)
	if e != nil {
		return nil, e
	}
	if opt.Shape != nil && !slices.Equal(sh, opt.Shape) {
		return nil, ErrShape
	}
	if rank >= 0 && rank != len(sh) {
		return nil, ErrShape
	}
	size, e := shapeSize(Bool, sh)
	if e != nil {
		return nil, e
	}
	if size == 0 && !opt.ZerosizeOK {
		return nil, ErrValue
	}
	it := &NDIterator{opt: opt, operands: slices.Clone(arrays), shape: sh, size: size, end: size, delayed: opt.DelayBufAlloc,
		access: make([]string, len(arrays)), dtypes: make([]DType, len(arrays)), logical: make([]*Array, len(arrays))}
	common := DType(0)
	if opt.CommonDType {
		for i, a := range arrays {
			d := a.DType()
			if len(opt.DTypes) > 0 && opt.DTypes[i] != 0 {
				d = opt.DTypes[i]
			}
			if d == 0 {
				continue
			}
			if common == 0 {
				common = d
			} else {
				common, e = PromoteTypes(common, d)
				if e != nil {
					return nil, e
				}
			}
		}
		if common == 0 {
			return nil, ErrDType
		}
	}
	for i, a := range arrays {
		mode := "readonly"
		if a == nil {
			mode = "writeonly"
		}
		if len(opt.Access) != 0 && opt.Access[i] != "" {
			mode = opt.Access[i]
		}
		if mode != "readonly" && mode != "readwrite" && mode != "writeonly" {
			return nil, ErrValue
		}
		it.access[i] = mode
		if opt.MaskOperand != nil && i == *opt.MaskOperand && mode != "readonly" {
			return nil, ErrValue
		}
		if len(opt.WriteMasked) > 0 && opt.WriteMasked[i] && mode == "readonly" {
			return nil, ErrValue
		}
		d := DType(0)
		if len(opt.DTypes) != 0 {
			d = opt.DTypes[i]
		}
		if common != 0 {
			d = common
		}
		if a == nil {
			if mode == "readonly" || d == 0 {
				return nil, ErrDType
			}
			shape := sh
			if rank >= 0 && opt.OpAxes[i] != nil {
				m := opt.OpAxes[i]
				r := 0
				for _, ax := range m {
					if ax < -1 || ax >= rank {
						return nil, ErrAxis
					}
					r = max(r, ax+1)
				}
				shape = make([]int, r)
				used := make([]bool, r)
				for j, ax := range m {
					if ax >= 0 {
						if used[ax] {
							return nil, ErrAxis
						}
						used[ax] = true
						shape[ax] = sh[j]
					}
				}
				for _, v := range used {
					if !v {
						return nil, ErrAxis
					}
				}
			}
			a, e = Zeros(d, shape...)
			if e != nil {
				return nil, e
			}
			it.operands[i] = a
			mapped[i] = a
			if rank >= 0 && opt.OpAxes[i] != nil {
				mapped[i], e = iteratorMappedOperand(a, opt.OpAxes[i])
				if e != nil {
					return nil, e
				}
			}
		}
		if a.HasObject() {
			if !opt.RefsOK {
				return nil, ErrDType
			}
		}
		if d == 0 {
			d = a.DType()
		}
		it.dtypes[i] = d
		if d != a.DType() {
			if !opt.Buffered || d.IsSpecial() && !d.IsExtended() || a.DType().IsSpecial() && !a.DType().IsExtended() {
				return nil, ErrDType
			}
			if mode != "writeonly" {
				ok, e := CanCast(a.DType(), d, opt.Casting)
				if e != nil || !ok {
					return nil, ErrDType
				}
			}
			if mode != "readonly" {
				ok, e := CanCast(d, a.DType(), opt.Casting)
				if e != nil || !ok {
					return nil, ErrDType
				}
			}
		}
		v, e := mapped[i].BroadcastTo(sh...)
		if e != nil {
			return nil, e
		}
		if len(opt.NoBroadcast) > 0 && opt.NoBroadcast[i] {
			if !slices.Equal(mapped[i].shape, sh) {
				return nil, ErrShape
			}
		}
		if mode != "readonly" {
			if !a.Writable() {
				return nil, ErrReadOnly
			}
			for k, s := range v.strides {
				if sh[k] > 1 && s == 0 && !opt.ReduceOK {
					return nil, ErrValue
				}
			}
			v.readOnly = false
		}
		it.logical[i] = v
	}
	if e = it.buildViews(); e != nil {
		return nil, e
	}
	return it, nil
}

func iteratorMappedOperand(a *Array, m []int) (*Array, error) {
	shape, strides := make([]int, len(m)), make([]int, len(m))
	used := make([]bool, a.NDim())
	for i, ax := range m {
		if ax == -1 {
			shape[i] = 1
			continue
		}
		if ax < 0 || ax >= a.NDim() || used[ax] {
			return nil, ErrAxis
		}
		used[ax] = true
		shape[i], strides[i] = a.shape[ax], a.strides[ax]
	}
	for i, yes := range used {
		if !yes && a.shape[i] != 1 {
			return nil, ErrShape
		}
	}
	return &Array{data: a.data, shape: shape, strides: strides, size: a.size, offset: a.offset, readOnly: a.readOnly, base: a.viewBase()}, nil
}
func absStride(x int) uint64 {
	if x < 0 {
		return uint64(-(x + 1)) + 1
	}
	return uint64(x)
}

func (it *NDIterator) buildViews() error {
	rank := len(it.shape)
	it.axes = make([]int, rank)
	it.flips = make([]bool, rank)
	for i := range it.axes {
		it.axes[i] = i
	}
	order := it.opt.Order
	if order == "A" {
		order = "F"
		for _, a := range it.logical {
			if !a.IsFContiguous() {
				order = "C"
				break
			}
		}
	}
	if order == "F" {
		slices.Reverse(it.axes)
	}
	if order == "K" {
		// Stable insertion ordering: conflicting operand strides keep
		// the existing order; zero broadcast strides do not decide it.
		for i := 1; i < rank; i++ {
			for j := i; j > 0; j-- {
				x, y := it.axes[j-1], it.axes[j]
				less, greater := false, false
				for _, a := range it.logical {
					s, t := absStride(a.strides[x]), absStride(a.strides[y])
					if s == 0 || t == 0 {
						continue
					}
					less = less || s < t
					greater = greater || s > t
				}
				if !less || greater {
					break
				}
				it.axes[j-1], it.axes[j] = y, x
			}
		}
		for ax := range it.shape {
			reverse, positive := false, false
			for _, a := range it.logical {
				reverse = reverse || a.strides[ax] < 0
				positive = positive || a.strides[ax] > 0
			}
			it.flips[ax] = reverse && !positive
		}
	}
	it.views = make([]*Array, len(it.logical))
	for i, a := range it.logical {
		v, e := a.Transpose(it.axes...)
		if e != nil {
			return e
		}
		ix := make([]Index, rank)
		for j, ax := range it.axes {
			ix[j] = Slice{}
			if it.flips[ax] {
				ix[j] = Reverse()
			}
		}
		v, e = v.Slice(ix...)
		if e != nil {
			return e
		}
		it.views[i] = v
	}
	if it.opt.MultiIndex || rank < 2 || it.size == 0 {
		return nil
	}
	sh := it.views[0].Shape()
	ss := make([][]int, len(it.views))
	for i, v := range it.views {
		ss[i] = slices.Clone(v.strides)
	}
	var idxStrides []int
	if it.opt.IndexOrder != "" {
		base := contiguousStrides(it.shape, it.size)
		if it.opt.IndexOrder == "F" {
			s := 1
			for i, d := range it.shape {
				base[i] = s
				s *= d
			}
		}
		idxStrides = make([]int, rank)
		for i, ax := range it.axes {
			idxStrides[i] = base[ax]
			if it.flips[ax] {
				idxStrides[i] = -idxStrides[i]
			}
		}
	}
	for j := len(sh) - 2; j >= 0; j-- {
		n, m := sh[j], sh[j+1]
		can := true
		mergeable := func(s []int) bool {
			if n == 1 || m == 1 {
				return true
			}
			stride, ok := checkedStride(s[j+1], m, 1)
			return ok && s[j] == stride
		}
		for _, s := range ss {
			if !mergeable(s) {
				can = false
				break
			}
		}
		if idxStrides != nil && !mergeable(idxStrides) {
			can = false
		}
		if !can {
			continue
		}
		merge := func(s []int) []int {
			if m != 1 {
				s[j] = s[j+1]
			}
			return slices.Delete(s, j+1, j+2)
		}
		for i := range ss {
			ss[i] = merge(ss[i])
		}
		if idxStrides != nil {
			idxStrides = merge(idxStrides)
		}
		sh[j] = n * m
		sh = slices.Delete(sh, j+1, j+2)
	}
	for i, v := range it.views {
		it.views[i] = &Array{data: v.data, shape: slices.Clone(sh), strides: ss[i], offset: v.offset, size: it.size, readOnly: v.readOnly, base: v.viewBase()}
	}
	return nil
}
func (it *NDIterator) DTypes() []DType          { return slices.Clone(it.dtypes) }
func (it *NDIterator) Operands() []*Array       { return slices.Clone(it.operands) }
func (it *NDIterator) NOp() int                 { return len(it.operands) }
func (it *NDIterator) IterSize() int            { return it.size }
func (it *NDIterator) HasIndex() bool           { return it.opt.IndexOrder != "" }
func (it *NDIterator) HasMultiIndex() bool      { return it.opt.MultiIndex }
func (it *NDIterator) HasDelayedBufAlloc() bool { return it.delayed }
func (it *NDIterator) IterationNeedsAPI() bool {
	for _, d := range it.dtypes {
		if d == Object || d == StringDType {
			return true
		}
	}
	return false
}
func (it *NDIterator) Finished() bool { return it.closed || it.pos >= it.end }
func (it *NDIterator) Shape() []int {
	if it.opt.MultiIndex {
		return slices.Clone(it.shape)
	}
	s := it.views[0].Shape()
	slices.Reverse(s)
	return s
}
func (it *NDIterator) NDim() int             { return len(it.Shape()) }
func (it *NDIterator) IterRange() (int, int) { return it.begin, it.end }
func (it *NDIterator) IterIndex() (int, error) {
	if it.Finished() {
		return 0, ErrIndex
	}
	return it.pos, nil
}
func (it *NDIterator) ItViews() ([]*Array, error) {
	if it.opt.Buffered || it.closed {
		return nil, ErrValue
	}
	return slices.Clone(it.views), nil
}
func (it *NDIterator) coord(pos int) []int {
	sh := make([]int, len(it.axes))
	for i, ax := range it.axes {
		sh[i] = it.shape[ax]
	}
	ordered := coordinates(pos, sh)
	out := make([]int, len(sh))
	for i, ax := range it.axes {
		val := ordered[i]
		if it.flips[ax] {
			val = it.shape[ax] - 1 - val
		}
		out[ax] = val
	}
	return out
}
func (it *NDIterator) MultiIndex() ([]int, error) {
	if !it.HasMultiIndex() || it.Finished() {
		return nil, ErrValue
	}
	return it.coord(it.pos), nil
}
func (it *NDIterator) Index() (int, error) {
	if !it.HasIndex() || it.Finished() {
		return 0, ErrValue
	}
	coord := it.coord(it.pos)
	r := 0
	if it.opt.IndexOrder == "C" {
		for i, d := range it.shape {
			r = r*d + coord[i]
		}
	} else {
		for i := len(coord) - 1; i >= 0; i-- {
			r = r*it.shape[i] + coord[i]
		}
	}
	return r, nil
}
func (it *NDIterator) seek(pos int) error {
	if it.closed || pos < it.begin || pos >= it.end {
		return ErrIndex
	}
	if e := it.flushBuffer(); e != nil {
		return e
	}
	it.pos = pos
	it.started = false
	return nil
}
func (it *NDIterator) SetIterIndex(pos int) error { return it.seek(pos) }
func (it *NDIterator) seekCoord(coord []int) error {
	if len(coord) != len(it.shape) {
		return ErrIndex
	}
	pos := 0
	for _, ax := range it.axes {
		v := coord[ax]
		if v < 0 || v >= it.shape[ax] {
			return ErrIndex
		}
		if it.flips[ax] {
			v = it.shape[ax] - 1 - v
		}
		pos = pos*it.shape[ax] + v
	}
	return it.seek(pos)
}
func (it *NDIterator) SetMultiIndex(coord []int) error {
	if !it.HasMultiIndex() {
		return ErrValue
	}
	return it.seekCoord(coord)
}
func (it *NDIterator) SetIndex(index int) error {
	if !it.HasIndex() {
		return ErrValue
	}
	if index < 0 || index >= it.size {
		return ErrIndex
	}
	if it.opt.IndexOrder == "C" {
		return it.seekCoord(coordinates(index, it.shape))
	}
	coord := make([]int, len(it.shape))
	for i, d := range it.shape {
		coord[i] = index % d
		index /= d
	}
	return it.seekCoord(coord)
}
func (it *NDIterator) SetIterRange(begin, end int) error {
	if !it.opt.Ranged || it.closed {
		return ErrValue
	}
	if begin < 0 || end < begin || end > it.size {
		return ErrIndex
	}
	if e := it.flushBuffer(); e != nil {
		return e
	}
	it.begin, it.end, it.pos = begin, end, begin
	it.started = false
	return nil
}
func (it *NDIterator) Reset() error {
	if it.closed {
		return ErrValue
	}
	if e := it.flushBuffer(); e != nil {
		return e
	}
	it.pos, it.started, it.delayed = it.begin, false, false
	return nil
}

// blockLength describes the contiguous inner traversal chunk, or one scalar.
func (it *NDIterator) blockLength() int {
	if !it.opt.ExternalLoop {
		return 1
	}
	if it.opt.Buffered {
		if it.bufferEnd > it.pos {
			return it.bufferEnd - it.pos
		}
		return min(it.bufferCapacity(), it.end-it.pos)
	}
	sh := it.views[0].shape
	if len(sh) == 0 {
		return 1
	}
	n := sh[len(sh)-1]
	return min(n-it.pos%n, it.end-it.pos)
}
func (it *NDIterator) viewAt(v *Array, pos, n int, external bool) *Array {
	off := v.offset + flatOffset(pos, v.shape, v.strides)
	var shape, strides []int
	if external {
		step := v.contiguousStep()
		if len(v.strides) > 0 {
			step = v.strides[len(v.strides)-1]
		}
		shape, strides = []int{n}, []int{step}
	} else {
		n = 1
	}
	return &Array{data: v.data, shape: shape, strides: strides, offset: off, size: n, readOnly: v.readOnly, base: v.viewBase()}
}
func (it *NDIterator) Value() ([]*Array, error) {
	if it.Finished() || it.delayed {
		return nil, ErrValue
	}
	for _, v := range it.views {
		if v.DType() == 0 {
			return nil, ErrDType
		}
	}
	if it.opt.Buffered {
		if e := it.ensureBuffer(); e != nil {
			return nil, e
		}
	}
	n := it.blockLength()
	out := make([]*Array, len(it.views))
	for i, v := range it.views {
		pos := it.pos
		if it.opt.Buffered {
			v = it.buffers[i]
			pos -= it.bufferStart
		}
		out[i] = it.viewAt(v, pos, n, it.opt.ExternalLoop)
		out[i].readOnly = it.access[i] == "readonly"
	}
	return out, nil
}
func (it *NDIterator) Get(operand int) (*Array, error) {
	i, e := normalizeIndex(operand, it.NOp())
	if e != nil {
		return nil, e
	}
	v, e := it.Value()
	if e != nil {
		return nil, e
	}
	return v[i], nil
}
func (it *NDIterator) Set(operand int, value *Array) error {
	v, e := it.Get(operand)
	if e != nil {
		return e
	}
	return v.Assign(nil, value)
}
func (it *NDIterator) IterNext() (bool, error) {
	if it.Finished() {
		return false, nil
	}
	if it.delayed {
		return false, ErrValue
	}
	it.pos += it.blockLength()
	if it.pos >= it.end || it.opt.Buffered && it.pos >= it.bufferEnd {
		if e := it.flushBuffer(); e != nil {
			return false, e
		}
	}
	return !it.Finished(), nil
}
func (it *NDIterator) Next() ([]*Array, bool, error) {
	if it.Finished() {
		return nil, false, nil
	}
	if it.started {
		ok, e := it.IterNext()
		if e != nil || !ok {
			return nil, false, e
		}
	}
	v, e := it.Value()
	if e != nil {
		return nil, false, e
	}
	it.started = true
	return v, true, nil
}
func (it *NDIterator) Iter() iter.Seq2[[]*Array, error] {
	return func(yield func([]*Array, error) bool) {
		for {
			v, ok, e := it.Next()
			if e != nil {
				yield(nil, e)
				return
			}
			if !ok || !yield(v, nil) {
				return
			}
		}
	}
}
func (it *NDIterator) Copy() (*NDIterator, error) {
	if it.closed {
		return nil, ErrValue
	}
	out := *it
	out.logical = slices.Clone(it.logical)
	out.views = slices.Clone(it.views)
	out.buffers = make([]*Array, len(it.buffers))
	for i, v := range it.buffers {
		b, e := v.Copy()
		if e != nil {
			return nil, e
		}
		out.buffers[i] = b
	}
	return &out, nil
}
func (it *NDIterator) RemoveAxis(axis int) error {
	if it.closed || !it.HasMultiIndex() || it.opt.Buffered {
		return ErrValue
	}
	ax, e := NormalizeAxis(axis, len(it.shape))
	if e != nil {
		return e
	}
	if it.shape[ax] == 0 {
		return ErrValue
	}
	if e = it.flushBuffer(); e != nil {
		return e
	}
	ix := make([]Index, len(it.shape))
	for i := range ix {
		ix[i] = Slice{}
	}
	ix[ax] = AtIndex(0)
	for i, v := range it.logical {
		it.logical[i], e = v.Slice(ix...)
		if e != nil {
			return e
		}
	}
	it.shape = it.logical[0].Shape()
	it.size = it.logical[0].Size()
	it.begin, it.end, it.pos, it.started = 0, it.size, 0, false
	return it.buildViews()
}
func (it *NDIterator) RemoveMultiIndex() error {
	if it.closed {
		return ErrValue
	}
	if e := it.Reset(); e != nil {
		return e
	}
	it.opt.MultiIndex = false
	return it.buildViews()
}
func (it *NDIterator) EnableExternalLoop() error {
	if it.closed || it.HasIndex() || it.HasMultiIndex() || it.opt.Ranged && !it.opt.Buffered {
		return ErrValue
	}
	if e := it.Reset(); e != nil {
		return e
	}
	it.opt.ExternalLoop = true
	return nil
}
func (it *NDIterator) DebugPrint(w io.Writer) error {
	if w == nil {
		return ErrValue
	}
	_, e := fmt.Fprintf(w, "NDIterator order=%s shape=%v size=%d operands=%d index=%d range=[%d,%d) external=%t buffered=%t finished=%t\n", it.opt.Order, it.Shape(), it.size, it.NOp(), it.pos, it.begin, it.end, it.opt.ExternalLoop, it.opt.Buffered, it.Finished())
	return e
}
func (it *NDIterator) Close() error {
	if it.closed {
		return nil
	}
	if e := it.flushBuffer(); e != nil {
		return e
	}
	it.closed = true
	return nil
}
