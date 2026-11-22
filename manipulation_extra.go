package np

import (
	"slices"
	"sort"
)

func Append(a, values *Array, axis *int) (*Array, error) {
	if axis == nil {
		var err error
		a, err = Ravel(a)
		if err != nil {
			return nil, err
		}
		values, err = Ravel(values)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	return Concatenate([]*Array{a, values}, *axis)
}
func HStack(arrays ...*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, ErrShape
	}
	v := make([]*Array, len(arrays))
	for i, a := range arrays {
		var err error
		v[i], err = Atleast1D(a)
		if err != nil {
			return nil, err
		}
	}
	axis := 0
	if v[0].NDim() > 1 {
		axis = 1
	}
	return Concatenate(v, axis)
}
func VStack(arrays ...*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, ErrShape
	}
	v := make([]*Array, len(arrays))
	for i, a := range arrays {
		var err error
		v[i], err = Atleast2D(a)
		if err != nil {
			return nil, err
		}
	}
	return Concatenate(v, 0)
}
func DStack(arrays ...*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, ErrShape
	}
	v := make([]*Array, len(arrays))
	for i, a := range arrays {
		var err error
		v[i], err = Atleast3D(a)
		if err != nil {
			return nil, err
		}
	}
	return Concatenate(v, 2)
}
func ColumnStack(arrays ...*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, ErrShape
	}
	v := make([]*Array, len(arrays))
	for i, a := range arrays {
		if a.DType() == 0 {
			return nil, ErrDType
		}
		if a.NDim() < 2 {
			var err error
			a, err = a.Reshape(a.size, 1)
			if err != nil {
				return nil, err
			}
		}
		v[i] = a
	}
	return Concatenate(v, 1)
}

// SplitAt returns shared slices delimited by explicit indices, preserving
// empty pieces and Python's clipped/negative slice bounds.
func SplitAt(a *Array, indices []int, axis int) ([]*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	ax, err := NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	out := make([]*Array, len(indices)+1)
	start := 0
	for i := range out {
		stop := a.shape[ax]
		if i < len(indices) {
			stop = indices[i]
		}
		spec := make([]Index, ax+1)
		for j := 0; j < ax; j++ {
			spec[j] = Slice{}
		}
		spec[ax] = Range(start, stop)
		out[i], err = a.Slice(spec...)
		if err != nil {
			return nil, err
		}
		start = stop
	}
	return out, nil
}
func Split(a *Array, sections, axis int) ([]*Array, error) {
	return splitCount(a, sections, axis, false)
}
func ArraySplit(a *Array, sections, axis int) ([]*Array, error) {
	return splitCount(a, sections, axis, true)
}
func splitCount(a *Array, sections, axis int, uneven bool) ([]*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	ax, err := NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	if sections <= 0 {
		return nil, ErrValue
	}
	n := a.shape[ax]
	q, r := n/sections, n%sections
	if !uneven && r != 0 {
		return nil, ErrShape
	}
	s := make([]int, sections-1)
	sum := 0
	for i := range s {
		sum += q
		if i < r {
			sum++
		}
		s[i] = sum
	}
	return SplitAt(a, s, ax)
}
func HSplit(a *Array, sections int) ([]*Array, error) {
	if a.DType() == 0 || a.NDim() < 1 {
		return nil, ErrShape
	}
	axis := 0
	if a.NDim() > 1 {
		axis = 1
	}
	return Split(a, sections, axis)
}
func VSplit(a *Array, sections int) ([]*Array, error) {
	if a.DType() == 0 || a.NDim() < 2 {
		return nil, ErrShape
	}
	return Split(a, sections, 0)
}
func DSplit(a *Array, sections int) ([]*Array, error) {
	if a.DType() == 0 || a.NDim() < 3 {
		return nil, ErrShape
	}
	return Split(a, sections, 2)
}
func Unstack(a *Array, axis int) ([]*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	ax, err := NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	out := make([]*Array, a.shape[ax])
	for i := range out {
		spec := make([]Index, ax+1)
		for j := 0; j < ax; j++ {
			spec[j] = Slice{}
		}
		spec[ax] = AtIndex(i)
		out[i], err = a.Slice(spec...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
func Rot90(a *Array, k, axis1, axis2 int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	x, err := NormalizeAxis(axis1, a.NDim())
	if err != nil {
		return nil, err
	}
	y, err := NormalizeAxis(axis2, a.NDim())
	if err != nil || x == y {
		return nil, ErrAxis
	}
	k %= 4
	if k < 0 {
		k += 4
	}
	switch k {
	case 0:
		return a.Slice()
	case 2:
		return Flip(a, x, y)
	case 1:
		v, err := Flip(a, y)
		if err != nil {
			return nil, err
		}
		return SwapAxes(v, x, y)
	default:
		v, err := SwapAxes(a, x, y)
		if err != nil {
			return nil, err
		}
		return Flip(v, y)
	}
}
func RollAxis(a *Array, axis, start int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	ax, err := NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	if start < 0 {
		start += a.NDim()
	}
	if start < 0 || start > a.NDim() {
		return nil, ErrAxis
	}
	if ax < start {
		start--
	}
	return MoveAxis(a, ax, start)
}

// Resize repeats flattened data to fill the new shape (numpy.resize).
func Resize(a *Array, shape ...int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	out, err := a.zerosLike(shape)
	if err != nil {
		return nil, err
	}
	if a.size == 0 {
		return out, nil
	}
	flat, err := a.Copy()
	if err != nil {
		return nil, err
	}
	for i := 0; i < out.size; i++ {
		out.setElement(i, flat.elementAt(i%flat.size))
	}
	return out, nil
}
func Compress(condition, a *Array, axis *int) (*Array, error) {
	if condition.DType() == 0 || a.DType() == 0 || condition.NDim() != 1 {
		return nil, ErrShape
	}
	if axis == nil {
		var err error
		a, err = Ravel(a)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	ax, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, err
	}
	var indices []int64
	for i := 0; i < condition.size; i++ {
		if condition.elementAt(condition.offset + i*condition.strides[0]).truth(condition.DType()) {
			if i >= a.shape[ax] {
				return nil, ErrIndex
			}
			indices = append(indices, int64(i))
		}
	}
	idx, err := WrapSlice(indices, len(indices))
	if err != nil {
		return nil, err
	}
	return Take(a, idx, &ax)
}
func Extract(condition, a *Array) (*Array, error) {
	if condition.DType() == 0 || a.DType() == 0 {
		return nil, ErrDType
	}
	c, err := Ravel(condition)
	if err != nil {
		return nil, err
	}
	return Compress(c, a, nil)
}
func Argwhere(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 0 {
		n := 0
		if a.elementAt(a.offset).truth(a.DType()) {
			n = 1
		}
		return Zeros(Int64, n, 0)
	}
	coords, err := Nonzero(a)
	if err != nil {
		return nil, err
	}
	return Stack(coords, 1)
}
func Delete(a, indices *Array, axis *int) (*Array, error) {
	if a.DType() == 0 || indices.DType() == 0 {
		return nil, ErrDType
	}
	if axis == nil {
		var err error
		a, err = Ravel(a)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	ax, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, err
	}
	n := a.shape[ax]
	removed := make([]bool, n)
	if indices.DType() == Bool {
		if indices.NDim() != 1 || indices.size != n {
			return nil, ErrShape
		}
		indices.walk(func(i, off int) { removed[i] = indices.elementAt(off).i != 0 })
	} else {
		if !indices.DType().IsInteger() {
			return nil, ErrDType
		}
		var failure error
		indices.walk(func(_, off int) {
			if failure == nil {
				var i int
				i, failure = indexInteger(indices, off, n)
				if failure == nil {
					removed[i] = true
				}
			}
		})
		if failure != nil {
			return nil, failure
		}
	}
	keep := make([]int64, 0, n)
	for i, r := range removed {
		if !r {
			keep = append(keep, int64(i))
		}
	}
	index, _ := WrapSlice(keep, len(keep))
	return Take(a, index, &ax)
}

// Choose selects one of a broadcastable list of arrays at each integer index.
func Choose(indices *Array, choices []*Array) (*Array, error) {
	if !indices.DType().IsInteger() || len(choices) == 0 {
		return nil, ErrIndex
	}
	shapes := [][]int{indices.shape}
	d := choices[0].DType()
	for _, c := range choices {
		if c.DType() == 0 {
			return nil, ErrDType
		}
		var err error
		d, err = PromoteTypes(d, c.DType())
		if err != nil {
			return nil, err
		}
		shapes = append(shapes, c.shape)
	}
	shape, err := BroadcastShapes(shapes...)
	if err != nil {
		return nil, err
	}
	out, err := allocationFor(choices, d, shape)
	if err != nil {
		return nil, err
	}
	index, err := indices.BroadcastTo(shape...)
	if err != nil {
		return nil, err
	}
	data := make([]*Array, len(choices))
	for i, v := range choices {
		data[i], err = v.BroadcastTo(shape...)
		if err != nil {
			return nil, err
		}
		if d.IsSpecial() {
			data[i], err = out.castLike(data[i])
			if err != nil {
				return nil, err
			}
		}
	}
	for i := 0; i < out.size; i++ {
		v := index.elementAt(index.offset + flatOffset(i, shape, index.strides))
		if indices.DType().IsSigned() && int64(v.i) < 0 || v.i >= uint64(len(data)) {
			return nil, ErrIndex
		}
		a := data[v.i]
		out.setElement(i, castElement(a.elementAt(a.offset+flatOffset(i, shape, a.strides)), a.DType(), d))
	}
	return out, nil
}

func ChooseWithOptions(indices *Array, choices []*Array, opt TakeOptions) (*Array, error) {
	if len(choices) == 0 {
		return nil, ErrValue
	}
	ix, e := indexMode(indices, len(choices), opt.Mode)
	if e != nil {
		return nil, e
	}
	result, e := Choose(ix, choices)
	if e != nil {
		return nil, e
	}
	return reductionOutput(opt.Out, result)
}
func Select(conditions, choices []*Array, defaultValue *Array) (*Array, error) {
	if len(conditions) == 0 || len(conditions) != len(choices) || defaultValue.DType() == 0 {
		return nil, ErrShape
	}
	result := defaultValue
	for i := len(conditions) - 1; i >= 0; i-- {
		if conditions[i].DType() != Bool {
			return nil, ErrDType
		}
		var err error
		result, err = Where(conditions[i], choices[i], result)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

func indexForPut(e element, d DType, n int, mode string) (int, error) {
	if n == 0 {
		return 0, ErrIndex
	}
	switch mode {
	case "", "raise":
		if d.IsSigned() {
			i := int64(e.i)
			if i < 0 {
				i += int64(n)
			}
			if i < 0 || i >= int64(n) {
				return 0, ErrIndex
			}
			return int(i), nil
		}
		if e.i >= uint64(n) {
			return 0, ErrIndex
		}
		return int(e.i), nil
	case "wrap":
		if d.IsSigned() {
			i := int64(e.i) % int64(n)
			if i < 0 {
				i += int64(n)
			}
			return int(i), nil
		}
		return int(e.i % uint64(n)), nil
	case "clip":
		if d.IsSigned() && int64(e.i) < 0 {
			return 0, nil
		}
		return int(min(e.i, uint64(n-1))), nil
	default:
		return 0, ErrValue
	}
}
func Put(a, indices, values *Array, mode string) error {
	if !a.Writable() {
		return ErrReadOnly
	}
	if !indices.DType().IsInteger() || values.DType() == 0 {
		return ErrDType
	}
	if values.size == 0 || indices.size == 0 {
		return nil
	}
	positions := make([]int, indices.size)
	var failure error
	indices.walk(func(i, off int) {
		if failure == nil {
			positions[i], failure = indexForPut(indices.elementAt(off), indices.DType(), a.size, mode)
		}
	})
	if failure != nil {
		return failure
	}
	vals, err := a.castLike(values)
	if err != nil {
		return err
	}
	strides := a.strides
	for i, p := range positions {
		off := a.offset + flatOffset(p, a.shape, strides)
		a.setElement(off, vals.elementAt(i%vals.size))
	}
	return nil
}
func Putmask(a, mask, values *Array) error { return maskedPut(a, mask, values, false) }
func Place(a, mask, values *Array) error   { return maskedPut(a, mask, values, true) }
func maskedPut(a, mask, values *Array, place bool) error {
	if !a.Writable() {
		return ErrReadOnly
	}
	if mask.DType() == 0 || values.DType() == 0 || mask.size != a.size {
		return ErrShape
	}
	m, err := mask.Astype(Bool)
	if err != nil {
		return err
	}
	v, err := a.castLike(values)
	if err != nil {
		return err
	}
	if v.size == 0 {
		if place {
			for _, z := range m.data.raw.([]bool) {
				if z {
					return ErrValue
				}
			}
		}
		return nil
	}
	j := 0
	a.walk(func(i, off int) {
		if m.elementAt(i).i != 0 {
			idx := i
			if place {
				idx = j
				j++
			}
			a.setElement(off, v.elementAt(idx%v.size))
		}
	})
	return nil
}

// Partition and ArgPartition use a stable full sort, which satisfies the
// partition ordering contract. NumPy does not specify order within partitions.
func Partition(a *Array, kth []int, axis *int) (*Array, error) { return partition(a, kth, axis, false) }
func ArgPartition(a *Array, kth []int, axis *int) (*Array, error) {
	return partition(a, kth, axis, true)
}
func partition(a *Array, kth []int, axis *int, indices bool) (*Array, error) {
	return partitionWithOrder(a, kth, axis, indices, nil)
}
func PartitionWithOrder(a *Array, kth []int, axis *int, fields ...string) (*Array, error) {
	return partitionWithOrder(a, kth, axis, false, fields)
}
func ArgPartitionWithOrder(a *Array, kth []int, axis *int, fields ...string) (*Array, error) {
	return partitionWithOrder(a, kth, axis, true, fields)
}
func partitionWithOrder(a *Array, kth []int, axis *int, indices bool, fields []string) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if axis == nil {
		var err error
		a, err = Ravel(a)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	ax, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, err
	}
	for _, k := range kth {
		if _, err := normalizeIndex(k, a.shape[ax]); err != nil {
			return nil, ErrIndex
		}
	}
	if indices {
		return ArgsortWithOptions(a, &ax, SortOptions{Order: fields})
	}
	return SortWithOptions(a, &ax, SortOptions{Order: fields})
}
func Lexsort(keys []*Array, axis int) (*Array, error) {
	if len(keys) == 0 || keys[0].DType() == 0 {
		return nil, ErrShape
	}
	shape := keys[0].shape
	ax, err := NormalizeAxis(axis, len(shape))
	if err != nil {
		return nil, err
	}
	for _, k := range keys {
		if k.DType() == 0 || !slices.Equal(k.shape, shape) {
			return nil, ErrShape
		}
	}
	out, err := Zeros(Int64, shape...)
	if err != nil {
		return nil, err
	}
	if out.size == 0 {
		return out, nil
	}
	n := shape[ax]
	r, _, _ := reduction(out, []SumOptions{{Axes: []int{ax}}})
	strides := make([][]int, len(keys))
	for i, k := range keys {
		strides[i] = append(append([]int(nil), k.strides[:ax]...), k.strides[ax+1:]...)
	}
	bases := make([]int, len(keys))
	order := make([]int, n)
	for i := 0; i < out.size/n; i++ {
		ob := r.base(out, i)
		for j, k := range keys {
			bases[j] = k.offset + flatOffset(i, r.shape, strides[j])
		}
		for j := range order {
			order[j] = j
		}
		sort.SliceStable(order, func(p, q int) bool {
			for j := len(keys) - 1; j >= 0; j-- {
				k := keys[j]
				x, y := k.elementAt(bases[j]+order[p]*k.strides[ax]), k.elementAt(bases[j]+order[q]*k.strides[ax])
				if sortLess(x, y, k.DType()) {
					return true
				}
				if sortLess(y, x, k.DType()) {
					return false
				}
			}
			return false
		})
		for j, k := range order {
			out.setElement(ob+j*out.strides[ax], element{i: uint64(k)})
		}
	}
	return out, nil
}
