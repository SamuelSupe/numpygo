package np

import (
	"math"
	"slices"
	"sort"
	"strings"
)

func sortLess(x, y element, d DType) bool {
	if d.IsExtended() || d.IsTemporal() || d == Object {
		if nanElement(x, d) {
			return false
		}
		if nanElement(y, d) {
			return true
		}
		return lessElement(x, y, d)
	}
	if d.IsComplex() {
		category := func(z complex128) int {
			n := 0
			if math.IsNaN(imag(z)) {
				n++
			}
			if math.IsNaN(real(z)) {
				n += 2
			}
			return n
		}
		a, b := category(x.z), category(y.z)
		if a != b {
			return a < b
		}
		if a < 2 && real(x.z) != real(y.z) {
			return real(x.z) < real(y.z)
		}
		if a%2 == 0 {
			return imag(x.z) < imag(y.z)
		}
		return false
	}
	if nanElement(x, d) {
		return false
	}
	if nanElement(y, d) {
		return true
	}
	return lessElement(x, y, d)
}

type UniqueOptions struct {
	EqualNaN *bool
	Axis     *int
	Sorted   *bool // NumPy permits sorted output even when false.
}
type UniqueResult struct{ Values, Indices, Inverse, Counts *Array }

// Unique returns sorted flattened values, first indices, inverse indices with
// the original shape, and counts. EqualNaN defaults to true.
func Unique(a *Array, options ...UniqueOptions) (UniqueResult, error) {
	var result UniqueResult
	if a.DType() == 0 {
		return result, ErrDType
	}
	if len(options) > 1 {
		return result, ErrValue
	}
	equalNaN := true
	if len(options) == 1 && options[0].EqualNaN != nil {
		equalNaN = *options[0].EqualNaN
	}
	if len(options) == 1 && options[0].Axis != nil {
		return uniqueAxis(a, *options[0].Axis)
	}
	if a.DType().IsInteger() || a.DType() == Bool {
		return uniqueIntegers(a)
	}
	flat, err := a.Copy()
	if err != nil {
		return result, err
	}
	order := make([]int, a.size)
	for i := range order {
		if a.DType() == StringDType && a.size > 1 && stringBadOrder(flat.elementAt(i)) {
			return result, ErrValue
		}
		order[i] = i
	}
	cmp, err := arrayOrder(flat, nil)
	if err != nil {
		return result, err
	}
	sort.SliceStable(order, func(i, j int) bool { return cmp.less(order[i], order[j]) })
	var values []element
	var indices, counts []int64
	inverse := make([]int64, a.size)
	prev := -1
	for _, idx := range order {
		e := flat.elementAt(idx)
		same := prev >= 0 && (cmp.equal(prev, idx) || equalNaN && nanElement(flat.elementAt(prev), a.DType()) && nanElement(e, a.DType()))
		if !same {
			values = append(values, e)
			indices = append(indices, int64(idx))
			counts = append(counts, 0)
		}
		j := len(values) - 1
		counts[j]++
		inverse[idx] = int64(j)
		prev = idx
	}
	result.Values, err = a.zerosLike([]int{len(values)})
	if err != nil {
		return result, err
	}
	for i, e := range values {
		result.Values.setElement(i, e)
	}
	result.Indices, _ = WrapSlice(indices, len(indices))
	result.Counts, _ = WrapSlice(counts, len(counts))
	result.Inverse, err = WrapSlice(inverse, a.shape...)
	return result, err
}

// Only the distinct integer keys need sorting. The first pass records
// indices and counts in logical C traversal order, including strided views.
func uniqueIntegers(a *Array) (UniqueResult, error) {
	var out UniqueResult
	type group struct {
		key          uint64
		id           int
		first, count int64
	}
	hint := min(a.size, 4096)
	index := make(map[uint64]int, hint)
	groups := make([]group, 0, hint)
	inverse := make([]int64, a.size)
	a.walk(func(i, off int) {
		k := a.elementAt(off).i
		j, ok := index[k]
		if !ok {
			j = len(groups)
			index[k] = j
			groups = append(groups, group{key: k, id: j, first: int64(i)})
		}
		groups[j].count++
		inverse[i] = int64(j)
	})
	signed := a.DType().IsSigned()
	slices.SortFunc(groups, func(x, y group) int {
		if x.key == y.key {
			return 0
		}
		if signed && int64(x.key) < int64(y.key) || !signed && x.key < y.key {
			return -1
		}
		return 1
	})
	n := len(groups)
	values, e := Zeros(a.DType(), n)
	if e != nil {
		return out, e
	}
	indices, counts, remap := make([]int64, n), make([]int64, n), make([]int64, n)
	for i, g := range groups {
		values.setElement(i, element{i: g.key})
		indices[i], counts[i], remap[g.id] = g.first, g.count, int64(i)
	}
	for i, j := range inverse {
		inverse[i] = remap[j]
	}
	out.Values = values
	out.Indices, _ = WrapSlice(indices, n)
	out.Counts, _ = WrapSlice(counts, n)
	out.Inverse, e = WrapSlice(inverse, a.shape...)
	return out, e
}

func uniqueAxis(a *Array, axis int) (UniqueResult, error) {
	var out UniqueResult
	ax, e := NormalizeAxis(axis, a.NDim())
	if e != nil {
		return out, e
	}
	if a.HasObject() {
		return out, ErrDType
	}
	v, e := MoveAxis(a, ax, 0)
	if e != nil {
		return out, e
	}
	cmp, e := arrayOrder(v, nil)
	if e != nil {
		return out, e
	}
	n := v.shape[0]
	width, e := shapeSizeWidth(1, v.shape[1:])
	if e != nil {
		return out, e
	}
	at := func(row, j int) int { return v.offset + row*v.strides[0] + flatOffset(j, v.shape[1:], v.strides[1:]) }
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		for k := 0; k < width; k++ {
			x, y := at(order[i], k), at(order[j], k)
			if cmp.less(x, y) {
				return true
			}
			if cmp.less(y, x) {
				return false
			}
		}
		return false
	})
	var first, counts []int64
	inverse := make([]int64, n)
	for _, r := range order {
		same := len(first) > 0
		if same {
			for j := 0; j < width; j++ {
				if !cmp.equal(at(int(first[len(first)-1]), j), at(r, j)) {
					same = false
					break
				}
			}
		}
		if !same {
			first = append(first, int64(r))
			counts = append(counts, 0)
		}
		counts[len(counts)-1]++
		inverse[r] = int64(len(counts) - 1)
	}
	out.Indices, _ = WrapSlice(first, len(first))
	out.Counts, _ = WrapSlice(counts, len(counts))
	out.Inverse, _ = WrapSlice(inverse, n)
	out.Values, e = Take(a, out.Indices, &ax)
	return out, e
}
func UniqueValues(a *Array) (*Array, error) {
	no := false
	v, err := Unique(a, UniqueOptions{EqualNaN: &no})
	return v.Values, err
}
func Isin(a, b *Array, invert bool) (*Array, error) {
	if a.DType() == StringDType || b.DType() == StringDType {
		x, y, e := stringPair(a, b)
		if e != nil {
			return nil, e
		}
		set := make(map[string]struct{}, y.size)
		// NumPy's StringDType isin uses scalar membership comparisons.
		// Each test scalar is inferred as fixed Unicode, stripping trailing
		// NUL padding even though array-to-array equality preserves it.
		y.walk(func(_, off int) {
			v := y.elementAt(off)
			if !stringNaN(v) {
				set[strings.TrimRight(flexKey(v), "\x00")] = struct{}{}
			}
		})
		return mapSpecialUnary(x, Bool, func(e element) (element, error) {
			_, ok := set[flexKey(e)]
			ok = ok && !stringNaN(e)
			return boolElement(ok != invert), nil
		})
	}
	if a.DType().IsFlexible() || b.DType().IsFlexible() {
		d := a.DType()
		if d == VoidDType || b.DType() == VoidDType {
			if d != b.DType() || a.ItemSize() != b.ItemSize() {
				return nil, ErrDType
			}
		}
		set := make(map[string]struct{}, b.size)
		if d == b.DType() {
			b.walk(func(_, off int) { set[flexOrderKey(b.elementAt(off), d)] = struct{}{} })
		}
		return mapSpecialUnary(a, Bool, func(e element) (element, error) {
			_, ok := set[flexOrderKey(e, d)]
			return boolElement(ok != invert), nil
		})
	}
	if a.DType() == RecordDType || b.DType() == RecordDType {
		if a.DType() != b.DType() {
			return nil, ErrDType
		}
		x, e := a.Reshape(-1)
		if e != nil {
			return nil, e
		}
		y, e := b.Reshape(-1)
		if e != nil {
			return nil, e
		}
		all, e := Concatenate([]*Array{x, y}, 0)
		if e != nil {
			return nil, e
		}
		cmp, e := arrayOrder(all, nil)
		if e != nil {
			return nil, e
		}
		out, e := Zeros(Bool, a.shape...)
		if e != nil {
			return nil, e
		}
		if cmp.searchable && b.size > 8 {
			order := make([]int, b.size)
			for i := range order {
				order[i] = a.size + i
			}
			sort.Slice(order, func(i, j int) bool { return cmp.less(order[i], order[j]) })
			for i := 0; i < a.size; i++ {
				j := sort.Search(len(order), func(j int) bool { return !cmp.less(order[j], i) })
				found := j < len(order) && cmp.equal(i, order[j])
				out.setElement(i, boolElement(found != invert))
			}
			return out, nil
		}
		for i := 0; i < a.size; i++ {
			found := false
			for j := 0; j < b.size; j++ {
				if cmp.equal(i, a.size+j) {
					found = true
					break
				}
			}
			out.setElement(i, boolElement(found != invert))
		}
		return out, nil
	}
	if a.DType().IsTemporal() || b.DType().IsTemporal() {
		x, y, e := temporalSearchPair(a, b)
		if e != nil {
			if e != ErrDType {
				return nil, e
			}
			if a.DType() != Object && b.DType() != Object {
				return Full(Bool, invert, a.shape...)
			}
		} else {
			set := make(map[int64]struct{}, y.size)
			y.walk(func(_, off int) {
				v := temporalTicks(y.elementAt(off))
				if v != NaT {
					set[v] = struct{}{}
				}
			})
			return mapSpecialUnary(x, Bool, func(v element) (element, error) {
				_, ok := set[temporalTicks(v)]
				return boolElement(ok != invert), nil
			})
		}
	}
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if d.IsExtended() {
		key := func(v element, from DType) ExtendedComplex {
			z := extendedValue(v, from)
			if z.Real.IsZero() {
				z.Real = ExtendedFloat{}
			}
			if z.Imag.IsZero() {
				z.Imag = ExtendedFloat{}
			}
			return z
		}
		set := make(map[ExtendedComplex]struct{}, b.size)
		b.walk(func(_, off int) {
			z := key(b.elementAt(off), b.DType())
			if !z.Real.IsNaN() && !z.Imag.IsNaN() {
				set[z] = struct{}{}
			}
		})
		return mapSpecialUnary(a, Bool, func(v element) (element, error) {
			_, ok := set[key(v, a.DType())]
			return boolElement(ok != invert), nil
		})
	}
	if d == Object {
		// Go object values can contain slices and maps, which cannot be map keys.
		test := make([]any, b.size)
		b.walk(func(i, off int) { test[i] = castElement(b.elementAt(off), b.DType(), Object).extra })
		return mapSpecialUnary(a, Bool, func(v element) (element, error) {
			x := castElement(v, a.DType(), Object).extra
			found := slices.ContainsFunc(test, func(y any) bool { return objectEqual(x, y) })
			return boolElement(found != invert), nil
		})
	}
	if d == Float64 && a.DType().IsInteger() && b.DType().IsInteger() {
		type key struct {
			negative bool
			bits     uint64
		}
		makeKey := func(e element, d DType) key { return key{d.IsSigned() && int64(e.i) < 0, e.i} }
		set := make(map[key]struct{}, b.size)
		b.walk(func(_, off int) { set[makeKey(b.elementAt(off), b.DType())] = struct{}{} })
		return unary(a, Bool, func(e element, from DType) (element, error) {
			_, ok := set[makeKey(e, from)]
			return boolElement(ok != invert), nil
		})
	}
	set := make(map[element]struct{}, b.size)
	b.walk(func(_, off int) { set[castElement(b.elementAt(off), b.DType(), d)] = struct{}{} })
	return unary(a, Bool, func(e element, from DType) (element, error) {
		_, ok := set[castElement(e, from, d)]
		return boolElement(ok != invert), nil
	})
}
func sortedUnique(a *Array) (*Array, error) { r, err := Unique(a); return r.Values, err }
func Union1D(a, b *Array) (*Array, error) {
	x, err := a.Reshape(-1)
	if err != nil {
		return nil, err
	}
	y, err := b.Reshape(-1)
	if err != nil {
		return nil, err
	}
	r, err := Concatenate([]*Array{x, y}, 0)
	if err != nil {
		return nil, err
	}
	return sortedUnique(r)
}
func Intersect1D(a, b *Array) (*Array, error) {
	x, err := sortedUnique(a)
	if err != nil {
		return nil, err
	}
	y, err := sortedUnique(b)
	if err != nil {
		return nil, err
	}
	d, err := PromoteTypes(x.DType(), y.DType())
	if err != nil {
		return nil, err
	}
	if d == RecordDType {
		s, e := PromoteRecordSchemas(x.data.schema, y.data.schema)
		if e != nil {
			return nil, e
		}
		x, e = x.CastRecords(s)
		if e != nil {
			return nil, e
		}
		y, e = y.CastRecords(s)
		if e != nil {
			return nil, e
		}
	} else if d.IsFlexible() {
		f, e := flexibleDescriptorFor([]*Array{x, y}, d)
		if e != nil {
			return nil, e
		}
		x, e = x.CastFlexible(f)
		if e != nil {
			return nil, e
		}
		y, e = y.CastFlexible(f)
		if e != nil {
			return nil, e
		}
	} else {
		x, err = x.Astype(d)
	}
	if err != nil {
		return nil, err
	}
	mask, err := Isin(x, y, false)
	if err != nil {
		return nil, err
	}
	return x.Index(mask)
}
func Setdiff1D(a, b *Array) (*Array, error) {
	x, err := sortedUnique(a)
	if err != nil {
		return nil, err
	}
	mask, err := Isin(x, b, true)
	if err != nil {
		return nil, err
	}
	return x.Index(mask)
}
func Setxor1D(a, b *Array) (*Array, error) {
	x, err := Setdiff1D(a, b)
	if err != nil {
		return nil, err
	}
	y, err := Setdiff1D(b, a)
	if err != nil {
		return nil, err
	}
	return Union1D(x, y)
}

// SearchSorted finds insertion positions in an ascending, one-dimensional array.
func SearchSorted(a, values *Array, right bool) (*Array, error) {
	return SearchSortedWithSorter(a, values, right, nil)
}

// Search and membership compare temporal values in the promoted tick unit.
// Unlike concatenate, comparing datetime to timedelta is not supported.
func temporalSearchPair(a, b *Array) (*Array, *Array, error) {
	d, e := PromoteTypes(a.DType(), b.DType())
	if e != nil || !d.IsTemporal() || a.DType().IsTemporal() && b.DType().IsTemporal() && a.DType() != b.DType() {
		return nil, nil, ErrDType
	}
	u := a.data.unit
	if !a.DType().IsTemporal() {
		u = b.data.unit
	}
	if a.DType().IsTemporal() && b.DType().IsTemporal() {
		u, e = commonDateUnit(a.data.unit, b.data.unit, d == DateTime64)
		if e != nil {
			return nil, nil, e
		}
	}
	if a.DType() != d || a.data.unit != u {
		a, e = a.CastTemporal(d, u.String())
		if e != nil {
			return nil, nil, e
		}
	}
	if b.DType() != d || b.data.unit != u {
		b, e = b.CastTemporal(d, u.String())
		if e != nil {
			return nil, nil, e
		}
	}
	return a, b, nil
}
func SearchSortedWithSorter(a, values *Array, right bool, sorter *Array) (*Array, error) {
	if a.DType() == 0 || values.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() != 1 {
		return nil, ErrShape
	}
	if sorter != nil {
		s := sorter
		if s.NDim() != 1 || s.size != a.size {
			return nil, ErrShape
		}
		if !s.DType().IsInteger() {
			return nil, ErrDType
		}
		for i := 0; i < s.size; i++ {
			x := s.elementAt(s.offset + i*s.strides[0])
			if x.i >= uint64(a.size) {
				return nil, ErrIndex
			}
		}
		var e error
		a, e = Take(a, s, Int(0))
		if e != nil {
			return nil, e
		}
	}
	d, err := PromoteTypes(a.DType(), values.DType())
	if err != nil {
		return nil, err
	}
	if d == RecordDType {
		v, e := values.Reshape(-1)
		if e != nil {
			return nil, e
		}
		all, e := Concatenate([]*Array{a, v}, 0)
		if e != nil {
			return nil, e
		}
		cmp, e := arrayOrder(all, nil)
		if e != nil {
			return nil, e
		}
		out, e := Zeros(Int64, values.shape...)
		if e != nil {
			return nil, e
		}
		for i := 0; i < v.size; i++ {
			j := sort.Search(a.size, func(j int) bool {
				if right {
					return cmp.less(a.size+i, j)
				}
				return !cmp.less(j, a.size+i)
			})
			out.setElement(i, element{i: uint64(j)})
		}
		return out, nil
	}
	sorted := a
	if d.IsFlexible() {
		f, e := flexibleDescriptorFor([]*Array{a, values}, d)
		if e != nil {
			return nil, e
		}
		sorted, err = a.CastFlexible(f)
		if err == nil {
			values, err = values.CastFlexible(f)
		}
	} else if d.IsTemporal() {
		sorted, values, err = temporalSearchPair(a, values)
	} else if a.DType() != d {
		sorted, err = a.Astype(d)
	}
	if err != nil {
		return nil, err
	}
	out, err := Zeros(Int64, values.shape...)
	if err != nil {
		return nil, err
	}
	values.walk(func(i, off int) {
		x := castElement(values.elementAt(off), values.DType(), d)
		j := sort.Search(sorted.size, func(j int) bool {
			y := sorted.elementAt(sorted.offset + j*sorted.strides[0])
			if right {
				return sortLess(x, y, d)
			}
			return !sortLess(y, x, d)
		})
		out.setElement(i, element{i: uint64(j)})
	})
	return out, nil
}
func Digitize(a, bins *Array, right bool) (*Array, error) {
	if bins.DType() == 0 || a.DType() == 0 || bins.DType().IsComplex() || a.DType().IsComplex() {
		return nil, ErrDType
	}
	if bins.NDim() != 1 {
		return nil, ErrShape
	}
	ascending, descending := true, true
	for i := 1; i < bins.size; i++ {
		x, y := bins.elementAt(bins.offset+(i-1)*bins.strides[0]), bins.elementAt(bins.offset+i*bins.strides[0])
		if sortLess(y, x, bins.DType()) {
			ascending = false
		}
		if sortLess(x, y, bins.DType()) {
			descending = false
		}
	}
	if !ascending && !descending {
		return nil, ErrValue
	}
	if ascending {
		return SearchSorted(bins, a, !right)
	}
	b, err := bins.Slice(Reverse())
	if err != nil {
		return nil, err
	}
	r, err := SearchSorted(b, a, !right)
	if err != nil {
		return nil, err
	}
	return Subtract(ScalarOf(int64(bins.size)), r)
}

func alongAxisLayout(a, indices *Array, axis *int) (*Array, *Array, int, []int, error) {
	if a.DType() == 0 || !indices.DType().IsInteger() {
		return nil, nil, 0, nil, ErrDType
	}
	if axis == nil {
		var err error
		a, err = a.Reshape(-1)
		if err != nil {
			return nil, nil, 0, nil, err
		}
		axis = Int(0)
	}
	if a.NDim() != indices.NDim() {
		return nil, nil, 0, nil, ErrShape
	}
	ax, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, nil, 0, nil, err
	}
	as, is := slices.Clone(a.shape), slices.Clone(indices.shape)
	as[ax], is[ax] = 1, 1
	shape, err := BroadcastShapes(as, is)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	shape[ax] = indices.shape[ax]
	return a, indices, ax, shape, nil
}
func alongOffsets(a, indices *Array, axis int, shape []int) ([]int, error) {
	size, err := a.shapeSize(shape)
	if err != nil {
		return nil, err
	}
	offsets := make([]int, size)
	coord := make([]int, len(shape))
	for i := range offsets {
		off, ioff := a.offset, indices.offset
		for j := range shape {
			if indices.shape[j] != 1 {
				ioff += coord[j] * indices.strides[j]
			}
			if j != axis && a.shape[j] != 1 {
				off += coord[j] * a.strides[j]
			}
		}
		index, err := indexInteger(indices, ioff, a.shape[axis])
		if err != nil {
			return nil, err
		}
		offsets[i] = off + index*a.strides[axis]
		for j := len(coord) - 1; j >= 0; j-- {
			coord[j]++
			if coord[j] < shape[j] {
				break
			}
			coord[j] = 0
		}
	}
	return offsets, nil
}

// TakeAlongAxis matches index slices to the corresponding source axis slices.
func TakeAlongAxis(a, indices *Array, axis *int) (*Array, error) {
	a, indices, ax, shape, err := alongAxisLayout(a, indices, axis)
	if err != nil {
		return nil, err
	}
	offsets, err := alongOffsets(a, indices, ax, shape)
	if err != nil {
		return nil, err
	}
	out, err := a.zerosLike(shape)
	if err != nil {
		return nil, err
	}
	for i, off := range offsets {
		out.setElement(i, a.elementAt(off))
	}
	return out, nil
}

// PutAlongAxis broadcasts and assigns corresponding axis indices.
func PutAlongAxis(a, indices, values *Array, axis *int) error {
	if !a.Writable() {
		return ErrReadOnly
	}
	if values.DType() == 0 {
		return ErrDType
	}
	if axis == nil && !a.IsCContiguous() {
		// Flattened writes must address original logical elements.
		flat, err := a.Copy()
		if err != nil {
			return err
		}
		if err = PutAlongAxis(flat, indices, values, axis); err != nil {
			return err
		}
		return a.Assign(nil, flat)
	}
	a, indices, ax, shape, err := alongAxisLayout(a, indices, axis)
	if err != nil {
		return err
	}
	offsets, err := alongOffsets(a, indices, ax, shape)
	if err != nil {
		return err
	}
	rhs, err := values.BroadcastTo(shape...)
	if err != nil {
		return err
	}
	rhs, err = rhs.Astype(a.DType())
	if err != nil {
		return err
	}
	for i, off := range offsets {
		a.setElement(off, rhs.elementAt(i))
	}
	return nil
}
