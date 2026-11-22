package np

import (
	"iter"
	"reflect"
	"slices"
)

// NDIndex yields independent C-order index tuples, including the sole empty
// tuple for a scalar. Any zero-length axis yields no indices.
func NDIndex(shape ...int) (iter.Seq[[]int], error) {
	n, e := shapeSize(Bool, shape)
	if e != nil {
		return nil, e
	}
	dims := slices.Clone(shape)
	return func(yield func([]int) bool) {
		idx := make([]int, len(dims))
		for i := 0; i < n; i++ {
			if !yield(slices.Clone(idx)) {
				return
			}
			for axis := len(idx) - 1; axis >= 0; axis-- {
				idx[axis]++
				if idx[axis] < dims[axis] {
					break
				}
				idx[axis] = 0
			}
		}
	}, nil
}
func NDEnumerate(a *Array) (iter.Seq2[[]int, any], error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	indices, e := NDIndex(a.shape...)
	if e != nil {
		return nil, e
	}
	return func(yield func([]int, any) bool) {
		for idx := range indices {
			v, _ := a.AtValue(idx...)
			if !yield(idx, v) {
				return
			}
		}
	}, nil
}
func NDIter(a *Array, order string) (iter.Seq[any], error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if order == "A" || order == "" {
		order = "C"
		if a.IsFContiguous() && !a.IsCContiguous() {
			order = "F"
		}
	}
	if order == "F" {
		var e error
		a, e = a.Transpose()
		if e != nil {
			return nil, e
		}
	} else if order != "C" {
		return nil, ErrValue
	}
	seq, e := NDEnumerate(a)
	if e != nil {
		return nil, e
	}
	return func(yield func(any) bool) {
		for _, v := range seq {
			if !yield(v) {
				return
			}
		}
	}, nil
}
func (a *Array) Flat() (iter.Seq[any], error) { return NDIter(a, "C") }

// NestedIters supplies two-level iteration: the first axis group selects
// coordinates, and each yielded array view traverses the second group.
func NestedIters(a *Array, groups [2][]int) (iter.Seq2[[]int, *Array], error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	seen := make([]bool, a.NDim())
	norm := [2][]int{}
	for k, g := range groups {
		for _, axis := range g {
			ax, e := NormalizeAxis(axis, a.NDim())
			if e != nil || seen[ax] {
				return nil, ErrAxis
			}
			seen[ax] = true
			norm[k] = append(norm[k], ax)
		}
	}
	for _, v := range seen {
		if !v {
			return nil, ErrAxis
		}
	}
	outer := make([]int, len(norm[0]))
	for i, ax := range norm[0] {
		outer[i] = a.shape[ax]
	}
	seq, e := NDIndex(outer...)
	if e != nil {
		return nil, e
	}
	var originalInner []int
	for ax := range seen {
		if !slices.Contains(norm[0], ax) {
			originalInner = append(originalInner, ax)
		}
	}
	permutation := make([]int, len(norm[1]))
	for i, ax := range norm[1] {
		permutation[i] = slices.Index(originalInner, ax)
	}
	return func(yield func([]int, *Array) bool) {
		for idx := range seq {
			index := make([]Index, a.NDim())
			for ax := range index {
				index[ax] = Slice{}
			}
			for i, ax := range norm[0] {
				index[ax] = AtIndex(idx[i])
			}
			v, _ := a.Slice(index...)
			v, _ = v.Transpose(permutation...)
			if !yield(idx, v) {
				return
			}
		}
	}, nil
}

type BroadcastIterator struct {
	arrays []*Array
	shape  []int
	size   int
	cursor int
	iters  []*FlatIterator
}

func NewBroadcast(arrays ...*Array) (*BroadcastIterator, error) {
	if len(arrays) == 0 {
		return &BroadcastIterator{size: 1}, nil
	}
	v, e := BroadcastArrays(arrays...)
	if e != nil {
		return nil, e
	}
	it := make([]*FlatIterator, len(v))
	for i, a := range v {
		it[i] = &FlatIterator{base: arrays[i], view: a}
	}
	return &BroadcastIterator{arrays: v, shape: v[0].Shape(), size: v[0].Size(), iters: it}, nil
}
func (b *BroadcastIterator) Shape() []int               { return slices.Clone(b.shape) }
func (b *BroadcastIterator) NDim() int                  { return len(b.shape) }
func (b *BroadcastIterator) Size() int                  { return b.size }
func (b *BroadcastIterator) Index() int                 { return b.cursor }
func (b *BroadcastIterator) NumIter() int               { return len(b.arrays) }
func (b *BroadcastIterator) Iterators() []*FlatIterator { return slices.Clone(b.iters) }
func (b *BroadcastIterator) Reset() {
	b.cursor = 0
	for _, it := range b.iters {
		it.Reset()
	}
}
func (b *BroadcastIterator) Next() ([]any, bool, error) {
	if b.cursor >= b.size {
		return nil, false, nil
	}
	row := make([]any, len(b.iters))
	for i, it := range b.iters {
		v, ok, e := it.Next()
		if e != nil {
			return nil, false, e
		}
		if !ok {
			return nil, false, ErrIndex
		}
		row[i] = v
	}
	b.cursor++
	return row, true, nil
}
func (b *BroadcastIterator) Iter() iter.Seq[[]any] {
	indices, _ := NDIndex(b.shape...)
	return func(yield func([]any) bool) {
		for idx := range indices {
			row := make([]any, len(b.arrays))
			for i, a := range b.arrays {
				row[i], _ = a.AtValue(idx...)
			}
			if !yield(row) {
				return
			}
		}
	}
}

func ApplyAlongAxis(a *Array, axis int, fn func(*Array) (*Array, error)) (*Array, error) {
	if a.DType() == 0 || fn == nil {
		return nil, ErrValue
	}
	ax, e := NormalizeAxis(axis, a.NDim())
	if e != nil {
		return nil, e
	}
	outer := append(slices.Clone(a.shape[:ax]), a.shape[ax+1:]...)
	indices, e := NDIndex(outer...)
	if e != nil {
		return nil, e
	}
	var out *Array
	var resultRank int
	for idx := range indices {
		in := make([]Index, a.NDim())
		for i := range in {
			if i < ax {
				in[i] = AtIndex(idx[i])
			} else if i == ax {
				in[i] = Slice{}
			} else {
				in[i] = AtIndex(idx[i-1])
			}
		}
		v, e := a.Slice(in...)
		if e != nil {
			return nil, e
		}
		r, e := fn(v)
		if e != nil {
			return nil, e
		}
		if r.DType() == 0 {
			return nil, ErrDType
		}
		if out == nil {
			resultRank = r.NDim()
			sh := append(slices.Clone(a.shape[:ax]), r.shape...)
			sh = append(sh, a.shape[ax+1:]...)
			out, e = Zeros(r.DType(), sh...)
			if e != nil {
				return nil, e
			}
		}
		target := make([]Index, out.NDim())
		for i := range target {
			if i < ax {
				target[i] = AtIndex(idx[i])
			} else if i < ax+resultRank {
				target[i] = Slice{}
			} else {
				target[i] = AtIndex(idx[i-resultRank])
			}
		}
		if e = out.Assign(target, r); e != nil {
			return nil, e
		}
	}
	if out == nil {
		return nil, ErrValue
	}
	return out, nil
}
func ApplyOverAxes(a *Array, axes []int, fn func(*Array, int) (*Array, error)) (*Array, error) {
	if a.DType() == 0 || fn == nil {
		return nil, ErrValue
	}
	rank := a.NDim()
	for _, axis := range axes {
		ax, e := NormalizeAxis(axis, rank)
		if e != nil {
			return nil, e
		}
		v, e := fn(a, ax)
		if e != nil {
			return nil, e
		}
		if v.DType() == 0 {
			return nil, ErrDType
		}
		if v.NDim() == rank-1 {
			v, e = ExpandDims(v, ax)
			if e != nil {
				return nil, e
			}
		} else if v.NDim() != rank {
			return nil, ErrShape
		}
		a = v
	}
	return a, nil
}

func FromFunction(shape []int, indexDType DType, fn func(...*Array) (*Array, error)) (*Array, error) {
	if fn == nil {
		return nil, ErrValue
	}
	if indexDType == 0 {
		indexDType = Float64
	}
	grid, e := Indices(shape, false)
	if e != nil {
		return nil, e
	}
	for i, a := range grid {
		grid[i], e = a.Astype(indexDType)
		if e != nil {
			return nil, e
		}
	}
	return fn(grid...)
}

// FromIter consumes count elements (or all for count=-1) of a typed Go iterator.
func FromIter[T Number](seq iter.Seq[T], count int) (*Array, error) {
	if count < -1 || seq == nil {
		return nil, ErrValue
	}
	var sample T
	if count >= 0 {
		if _, e := shapeSize(dtypeOf(sample), []int{count}); e != nil {
			return nil, e
		}
	}
	out := make([]T, 0)
	if count != 0 {
		for x := range seq {
			out = append(out, x)
			if count >= 0 && len(out) == count {
				break
			}
		}
	}
	if count >= 0 && len(out) != count {
		return nil, ErrValue
	}
	return WrapSlice(out, len(out))
}

// Block joins nested lists of arrays. The list nesting depth must be uniform;
// lower-rank operands are promoted with leading length-one axes.
func Block(items any) (*Array, error) {
	var depthOf func(any) (depth, rank int, err error)
	depthOf = func(item any) (int, int, error) {
		switch x := item.(type) {
		case *Array:
			if x.DType() == 0 {
				return 0, 0, ErrDType
			}
			return 0, x.NDim(), nil
		case []*Array:
			v := make([]any, len(x))
			for i, a := range x {
				v[i] = a
			}
			return depthOf(v)
		case []any:
			if len(x) == 0 {
				return 0, 0, ErrShape
			}
			depth, rank := -1, 0
			for _, child := range x {
				d, r, e := depthOf(child)
				if e != nil {
					return 0, 0, e
				}
				if depth == -1 {
					depth = d
				} else if d != depth {
					return 0, 0, ErrShape
				}
				rank = max(rank, r)
			}
			return depth + 1, rank, nil
		default:
			return 0, 0, ErrDType
		}
	}
	depth, rank, e := depthOf(items)
	if e != nil {
		return nil, e
	}
	rank = max(rank, depth)
	var build func(any, int) (*Array, error)
	build = func(item any, level int) (*Array, error) {
		switch x := item.(type) {
		case *Array:
			a := x
			for a.NDim() < rank {
				var e error
				a, e = ExpandDims(a, 0)
				if e != nil {
					return nil, e
				}
			}
			return a, nil
		case []*Array:
			v := make([]any, len(x))
			for i, a := range x {
				v[i] = a
			}
			return build(v, level)
		case []any:
			children := make([]*Array, len(x))
			for i, child := range x {
				var e error
				children[i], e = build(child, level-1)
				if e != nil {
					return nil, e
				}
			}
			return Concatenate(children, rank-level)
		default:
			return nil, ErrDType
		}
	}
	out, e := build(items, depth)
	if e != nil {
		return nil, e
	}
	if depth == 0 {
		return out.Copy()
	}
	return out, nil
}

// Piecewise evaluates each callback only on selected elements. Later true
// conditions overwrite earlier branches, and missing defaults are zero.
func Piecewise(x *Array, conditions []*Array, branches []func(*Array) (*Array, error), otherwise func(*Array) (*Array, error)) (*Array, error) {
	if x.DType() == 0 || len(conditions) != len(branches) {
		return nil, ErrShape
	}
	out, e := x.zerosLike(x.shape)
	if e != nil {
		return nil, e
	}
	covered, e := Zeros(Bool, x.shape...)
	if e != nil {
		return nil, e
	}
	apply := func(c *Array, fn func(*Array) (*Array, error)) error {
		if fn == nil {
			return ErrValue
		}
		c, e := c.BroadcastTo(x.shape...)
		if e != nil {
			return e
		}
		c, e = c.Astype(Bool)
		if e != nil {
			return e
		}
		v, e := x.Index(c)
		if e != nil {
			return e
		}
		if v.size == 0 {
			return nil
		}
		y, e := fn(v)
		if e != nil {
			return e
		}
		return out.Assign([]Index{c}, y)
	}
	for i, c := range conditions {
		if c.DType() == 0 {
			return nil, ErrDType
		}
		covered, e = LogicalOr(covered, c)
		if e != nil {
			return nil, e
		}
		if e = apply(c, branches[i]); e != nil {
			return nil, e
		}
	}
	if otherwise != nil {
		c, e := LogicalNot(covered)
		if e != nil {
			return nil, e
		}
		if e = apply(c, otherwise); e != nil {
			return nil, e
		}
	}
	return out, nil
}

// Vectorize evaluates a numeric Go callback on broadcast scalar arguments.
// The output dtype is explicit; Python object inference is not used.
func Vectorize(dtype DType, fn func(...any) (any, error), args ...*Array) (*Array, error) {
	if fn == nil {
		return nil, ErrValue
	}
	b, e := NewBroadcast(args...)
	if e != nil {
		return nil, e
	}
	out, e := Zeros(dtype, b.shape...)
	if e != nil {
		return nil, e
	}
	i := 0
	for v := range b.Iter() {
		x, e := fn(v...)
		if e != nil {
			return nil, e
		}
		z, e := out.convertValue(x)
		if e != nil {
			return nil, e
		}
		out.setElement(i, z)
		i++
	}
	return out, nil
}

func MaskIndices(n, k int, mask func(*Array, int) (*Array, error)) ([]*Array, error) {
	if mask == nil {
		return nil, ErrValue
	}
	a, e := Ones(Bool, n, n)
	if e != nil {
		return nil, e
	}
	m, e := mask(a, k)
	if e != nil {
		return nil, e
	}
	return Nonzero(m)
}

// Empty provides allocated writable storage. Go safely zero-initializes it;
// clients should not depend on contents they have not written.
func Empty(dtype DType, shape ...int) (*Array, error) {
	if dtype == Object {
		n, e := shapeSize(dtype, shape)
		if e != nil {
			return nil, e
		}
		return owned(&storage{dtype: Object, raw: make([]any, n)}, shape, n), nil
	}
	return Zeros(dtype, shape...)
}
func EmptyLike(a *Array, dtype DType, order string) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	return EmptyLikeWithShape(a, dtype, order, a.shape)
}

// EmptyLikeWithShape allocates the requested shape and dtype using the
// prototype's layout. K preserves its packed axis order when rank is unchanged.
func EmptyLikeWithShape(a *Array, dtype DType, order string, shape []int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if dtype == 0 {
		dtype = a.DType()
	}
	switch order {
	case "", "K":
		order = "K"
	case "A":
		order = "C"
		if a.IsFContiguous() && !a.IsCContiguous() {
			order = "F"
		}
	case "C", "F":
	default:
		return nil, ErrValue
	}
	if len(shape) != a.NDim() && order == "K" {
		order = "C"
	}
	var out *Array
	var e error
	if dtype == a.DType() && dtype != Object {
		out, e = a.zerosLike(shape)
	} else {
		out, e = Empty(dtype, shape...)
	}
	if e != nil {
		return nil, e
	}
	if out.size == 0 || order == "C" || len(shape) < 2 || order == "K" && a.IsCContiguous() {
		return out, nil
	}
	// The allocation itself owns its storage, including F and permuted
	// packed layouts; it is not a view of a hidden temporary array.
	stride := 1
	if order == "F" {
		for ax, d := range shape {
			out.strides[ax] = stride
			stride *= d
		}
	} else {
		axes, _ := layoutAxes(a, "K")
		for i := len(axes) - 1; i >= 0; i-- {
			ax := axes[i]
			out.strides[ax] = stride
			stride *= shape[ax]
		}
	}
	return out, nil
}
func AsAnyArray(a *Array, dtype DType) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if dtype == 0 || dtype == a.DType() {
		return a, nil
	}
	return a.Astype(dtype)
}
func AsArrayChkFinite(a *Array, dtype DType) (*Array, error) {
	out, e := AsAnyArray(a, dtype)
	if e != nil {
		return nil, e
	}
	finite, e := IsFinite(out)
	if e != nil {
		return nil, e
	}
	all, e := All(finite)
	if e != nil {
		return nil, e
	}
	v, _ := all.AtValue()
	if !v.(bool) {
		return nil, ErrValue
	}
	return out, nil
}

// Require returns an array matching the dtype and requested contiguous layout
// and writability. The owning flag requests a detached copy.
func Require(a *Array, dtype DType, order string, writable, owning bool) (*Array, error) {
	out, e := AsAnyArray(a, dtype)
	if e != nil {
		return nil, e
	}
	switch order {
	case "C":
		out, e = AsContiguousArray(out)
	case "F":
		out, e = AsFortranArray(out)
	case "A", "":
	default:
		return nil, ErrValue
	}
	if e != nil {
		return nil, e
	}
	if owning || writable && !out.Writable() {
		if order == "F" {
			r, e := out.Transpose()
			if e != nil {
				return nil, e
			}
			r, e = r.Copy()
			if e != nil {
				return nil, e
			}
			return r.Transpose()
		}
		return out.Copy()
	}
	return out, nil
}

func IsScalar(value any) bool { return dtypeOf(value) != 0 }
func Iterable(value any) bool {
	if a, ok := value.(*Array); ok {
		return a.DType() != 0 && a.NDim() > 0
	}
	if value == nil {
		return false
	}
	switch reflect.TypeOf(value).Kind() {
	case reflect.Array, reflect.Slice, reflect.String, reflect.Map:
		return true
	}
	return false
}

// AsMatrix provides the numeric 2D construction subset of numpy.asmatrix.
// Matrix multiplication is explicit through Go's linalg.MatMul.
func AsMatrix(a *Array, dtype DType, copyData bool) (*Array, error) {
	out, e := AsAnyArray(a, dtype)
	if e != nil {
		return nil, e
	}
	if out.NDim() > 2 {
		return nil, ErrShape
	}
	out, e = Atleast2D(out)
	if e != nil {
		return nil, e
	}
	if copyData {
		return out.Copy()
	}
	return out, nil
}
func BMat(items any) (*Array, error) {
	out, e := Block(items)
	if e != nil {
		return nil, e
	}
	return AsMatrix(out, 0, false)
}
func (a *Array) Device() string { return "cpu" }
func (a *Array) ToDevice(device string) (*Array, error) {
	if device != "cpu" {
		return nil, ErrValue
	}
	if a.DType() == 0 {
		return nil, ErrDType
	}
	return a, nil
}
func (a *Array) ToList() (any, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	var rec func(int, int) any
	rec = func(axis, off int) any {
		if axis == a.NDim() {
			return a.rawAt(off)
		}
		out := make([]any, a.shape[axis])
		for i := range out {
			out[i] = rec(axis+1, off+i*a.strides[axis])
		}
		return out
	}
	return rec(0, a.offset), nil
}

// ResizeInPlace replaces the owning array's header with new C-order storage,
// retaining the leading logical sequence and zero-filling new elements.
// As a conservative Go equivalent of NumPy's reference check, any previously
// exposed array view or byte span prevents in-place resizing; copy first.
func (a *Array) ResizeInPlace(shape ...int) error {
	if a.DType() == 0 {
		return ErrDType
	}
	if !a.Writable() {
		return ErrReadOnly
	}
	if !a.owner || a.data.shared.Load() || !a.IsCContiguous() || a.offset != 0 {
		return ErrShape
	}
	if a.DType() == Object {
		if len(a.data.raw.([]any)) != a.size {
			return ErrShape
		}
	} else if a.DType() == StringDType {
		if len(a.data.raw.([]any)) != a.size {
			return ErrShape
		}
	} else if len(rawStorageBytes(a.data)) != a.NBytes() {
		return ErrShape
	}
	out, e := a.zerosLike(shape)
	if e != nil {
		return e
	}
	n := min(a.size, out.size)
	for i := 0; i < n; i++ {
		out.setElement(i, a.elementAt(i))
	}
	*a = *out
	return nil
}
