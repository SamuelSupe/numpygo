package np

import (
	"fmt"
	"slices"
	"sort"
)

// Numerical arrays may be used directly as integer-array or boolean-mask indices.
func (*Array) basicIndex() {}

type indexTerm struct {
	axis        int
	dim         int
	start, step int
	array       *Array
	scalar      bool
	newAxis     bool
	barrier     bool
	resultAxis  int
}

type advancedSelection struct {
	shape, offsets   []int
	canonical, order []int
}

func indexInteger(a *Array, offset, dim int) (int, error) {
	e := a.elementAt(offset)
	if a.DType().IsSigned() {
		return signedIndex(int64(e.i), dim)
	}
	if e.i >= uint64(dim) {
		return 0, ErrIndex
	}
	return int(e.i), nil
}
func signedIndex(v int64, dim int) (int, error) {
	if v < 0 {
		v += int64(dim)
	}
	if v < 0 || v >= int64(dim) {
		return 0, ErrIndex
	}
	return int(v), nil
}

func advancedInteger(a *Array, offset, dim int) (int, error) {
	// Fancy map iteration converts index arrays to signed intp. Unlike
	// callers such as Take, uint64 indices are interpreted modulo intp.
	if a.DType() == Uint64 {
		return signedIndex(int64(a.elementAt(offset).i), dim)
	}
	return indexInteger(a, offset, dim)
}

func indexCount(index Index) (int, error) {
	switch x := index.(type) {
	case Slice, AtIndex:
		return 1, nil
	case specialIndex:
		return 0, nil
	case *Array:
		if x.DType() == Bool {
			return x.NDim(), nil
		}
		if x.DType().IsInteger() {
			return 1, nil
		}
	}
	return 0, ErrIndex
}

// advancedLayout builds a gather/scatter plan with NumPy's separated-vs-adjacent
// advanced-index axis placement rules. Offsets are validated before any write.
func (a *Array) advancedLayout(indices []Index) (*advancedSelection, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	consumed, ellipses := 0, 0
	for _, x := range indices {
		n, err := indexCount(x)
		if err != nil {
			return nil, err
		}
		consumed += n
		if x == Ellipsis {
			ellipses++
		}
	}
	if consumed > a.NDim() || ellipses > 1 {
		return nil, ErrIndex
	}
	axis := 0
	var terms []indexTerm
	full := func(n int) {
		for i := 0; i < n; i++ {
			terms = append(terms, indexTerm{axis: axis, dim: a.shape[axis], step: 1})
			axis++
		}
	}
	for _, index := range indices {
		switch x := index.(type) {
		case Slice:
			start, n, step, err := x.normalize(a.shape[axis])
			if err != nil {
				return nil, err
			}
			if _, ok := checkedStride(a.strides[axis], step, a.strideUnit()); !ok {
				return nil, ErrShape
			}
			terms = append(terms, indexTerm{axis: axis, dim: n, start: start, step: step})
			axis++
		case AtIndex:
			i, err := normalizeIndex(int(x), a.shape[axis])
			if err != nil {
				return nil, err
			}
			terms = append(terms, indexTerm{axis: axis, start: i, scalar: true})
			axis++
		case specialIndex:
			switch x {
			case NewAxis:
				terms = append(terms, indexTerm{axis: -1, dim: 1, newAxis: true})
			case Ellipsis:
				terms = append(terms, indexTerm{axis: -1, barrier: true})
				full(a.NDim() - consumed)
			default:
				return nil, ErrIndex
			}
		case *Array:
			if x.DType() == Bool {
				if x.NDim() == 0 {
					n := 0
					if x.elementAt(x.offset).truth(Bool) {
						n = 1
					}
					coord, _ := Zeros(Int64, n)
					terms = append(terms, indexTerm{axis: -1, array: coord})
					continue
				}
				for i, dim := range x.shape {
					if dim != 0 && dim != a.shape[axis+i] {
						return nil, fmt.Errorf("%w: boolean mask shape mismatch", ErrIndex)
					}
				}
				coords, err := Nonzero(x)
				if err != nil {
					return nil, err
				}
				for _, coord := range coords {
					terms = append(terms, indexTerm{axis: axis, array: coord})
					axis++
				}
			} else {
				terms = append(terms, indexTerm{axis: axis, array: x})
				axis++
			}
		}
	}
	full(a.NDim() - axis)
	first, last := -1, -1
	var shapes [][]int
	for i, t := range terms {
		if t.array != nil || t.scalar {
			if first < 0 {
				first = i
			}
			last = i
		}
		if t.array != nil {
			shapes = append(shapes, t.array.shape)
		}
	}
	advShape, err := BroadcastShapes(shapes...)
	if err != nil {
		return nil, fmt.Errorf("%w: advanced index shapes", ErrIndex)
	}
	advSize, err := shapeSize(Int64, advShape)
	if err != nil {
		return nil, err
	}
	separated := false
	if first >= 0 {
		for i := first; i <= last; i++ {
			if terms[i].array == nil && !terms[i].scalar {
				separated = true
				break
			}
		}
	}
	var shape []int
	advAxis := 0
	if separated {
		shape = append(shape, advShape...)
	}
	for i := range terms {
		t := &terms[i]
		if i == first && !separated {
			advAxis = len(shape)
			shape = append(shape, advShape...)
		}
		if t.array != nil {
			// An empty advanced broadcast has no integer coordinates to
			// validate. Empty basic slices alone do not waive index bounds.
			if t.axis >= 0 && advSize != 0 {
				var validation error
				t.array.walk(func(_, off int) {
					if validation == nil {
						_, validation = advancedInteger(t.array, off, a.shape[t.axis])
					}
				})
				if validation != nil {
					return nil, validation
				}
			}
			t.array, err = t.array.BroadcastTo(advShape...)
			if err != nil {
				return nil, ErrIndex
			}
		} else if !t.scalar && !t.barrier {
			t.resultAxis = len(shape)
			shape = append(shape, t.dim)
		}
	}
	size, err := a.shapeSize(shape)
	if err != nil {
		return nil, err
	}
	offsets := make([]int, size)
	coord := make([]int, len(shape))
	for flat := 0; flat < size; flat++ {
		off := a.offset
		for _, t := range terms {
			if t.axis < 0 {
				continue
			}
			idx := t.start
			if t.array != nil {
				ioff := t.array.offset
				for k, stride := range t.array.strides {
					ioff += coord[advAxis+k] * stride
				}
				idx, _ = advancedInteger(t.array, ioff, a.shape[t.axis])
			} else if !t.scalar {
				idx += coord[t.resultAxis] * t.step
			}
			off += idx * a.strides[t.axis]
		}
		offsets[flat] = off
		for k := len(coord) - 1; k >= 0; k-- {
			coord[k]++
			if coord[k] < shape[k] {
				break
			}
			coord[k] = 0
		}
	}
	p := &advancedSelection{shape: shape, offsets: offsets}
	// Map iteration allocates the advanced axes first. When the user
	// inserts them after basic axes, the result is a view of that copy.
	advAxes := make([]int, len(advShape))
	for i := range advAxes {
		advAxes[i] = advAxis + i
	}
	basic := []int{}
	basicStrides := make([]int, len(shape))
	indexArrays := []*Array{}
	for _, t := range terms {
		if t.array != nil {
			indexArrays = append(indexArrays, t.array)
		} else if !t.scalar && !t.barrier {
			basic = append(basic, t.resultAxis)
			if t.axis >= 0 {
				basicStrides[t.resultAxis] = a.strides[t.axis] * t.step
			}
		}
	}
	p.canonical = append(slices.Clone(advAxes), basic...)
	basicOrder := slices.Clone(basic)
	// Basic subspaces preserve the input stride order, including
	// reversed, zero-stride and permuted axes, in a dense copy.
	sort.SliceStable(basicOrder, func(i, j int) bool {
		return absStride(basicStrides[basicOrder[i]]) > absStride(basicStrides[basicOrder[j]])
	})
	if advSize != 0 && size == advSize && len(advAxes) > 1 {
		// With no multi-element basic subspace, the index operands
		// themselves determine keep-order allocation.
		if len(indices) == 1 && len(basic) == 0 {
			if ix, ok := indices[0].(*Array); ok && ix.DType().IsInteger() && ix.IsFContiguous() && !ix.IsCContiguous() {
				slices.Reverse(advAxes)
			} else {
				axes := ufuncResultAxes(advShape, indexArrays, nil, 0, "K")
				for i, ax := range axes {
					advAxes[i] = advAxis + ax
				}
			}
		} else {
			axes := ufuncResultAxes(advShape, indexArrays, nil, 0, "K")
			for i, ax := range axes {
				advAxes[i] = advAxis + ax
			}
		}
	}
	p.order = append(advAxes, basicOrder...)
	return p, nil
}

func (a *Array) copyAdvanced(p *advancedSelection) (*Array, error) {
	return a.copySelection(p, false)
}
func (a *Array) copySelection(p *advancedSelection, preserveScalarEndian bool) (*Array, error) {
	shape := make([]int, len(p.shape))
	inverse := make([]int, len(shape))
	remap := false
	for i, ax := range p.canonical {
		shape[i], inverse[ax] = p.shape[ax], i
		remap = remap || i != ax
	}
	var out *Array
	var err error
	d := a.DType()
	// Fancy indexing preserves the actual descriptor, including byte
	// order; unlike ndarray results a scalar result is native.
	if (len(shape) != 0 || preserveScalarEndian) && !a.NativeByteOrder() && !d.IsFlexible() {
		n := len(p.offsets)
		out = owned(&storage{dtype: d, buffer: make([]byte, n*a.ItemSize()), order: storageOrder(a.data), unit: a.data.unit}, shape, n)
	} else {
		out, err = a.zerosLike(shape)
		if err != nil {
			return nil, err
		}
	}
	if out.size != 0 {
		step := 1
		for i := len(p.order) - 1; i >= 0; i-- {
			ax := p.order[i]
			out.strides[inverse[ax]] = step
			step *= p.shape[ax]
		}
	}
	if remap {
		out, err = out.Transpose(inverse...)
		if err != nil {
			return nil, err
		}
	}
	if out.size == 0 || out.ItemSize() == 0 {
		return out, nil
	}
	// Fixed-width gather is a byte copy, avoiding numeric conversions
	// that can change NaN payloads, padding or nonnative values.
	if !a.rawReferenceStorage() && !d.IsExtended() && a.NativeByteOrder() == out.NativeByteOrder() {
		src, dst := rawStorageBytes(a.data), rawStorageBytes(out.data)
		width := a.ItemSize()
		out.walk(func(i, off int) {
			start := p.offsets[i] * a.strideUnit()
			copy(dst[off*width:(off+1)*width], src[start:start+width])
		})
	} else {
		out.walk(func(i, off int) { out.setElement(off, a.elementAt(p.offsets[i])) })
	}
	return out, nil
}

// Index selects NumPy basic or advanced indices. Basic indexing shares storage;
// integer-array or boolean-array indices produce an independent copy, possibly
// with an axis-remapped view of that copy as the returned array.
func (a *Array) Index(indices ...Index) (*Array, error) {
	for _, x := range indices {
		if _, ok := x.(*Array); ok {
			p, err := a.advancedLayout(indices)
			if err != nil {
				return nil, err
			}
			return a.copyAdvanced(p)
		}
	}
	return a.Slice(indices...)
}

// Assign broadcasts values over the selected elements and snapshots the RHS
// when storage may overlap. Repeated indices receive the last C-order value.
func (a *Array) Assign(indices []Index, values *Array) error {
	if !a.Writable() {
		return ErrReadOnly
	}
	if values.DType() == 0 {
		return ErrDType
	}
	var shape []int
	var offsets []int
	var view *Array
	advanced := false
	for _, x := range indices {
		if _, ok := x.(*Array); ok {
			advanced = true
			break
		}
	}
	if advanced {
		p, err := a.advancedLayout(indices)
		if err != nil {
			return err
		}
		shape, offsets = p.shape, p.offsets
	} else {
		var err error
		view, err = a.Slice(indices...)
		if err != nil {
			return err
		}
		shape = view.shape
	}
	// NumPy allows leading singleton dimensions in assignment.
	for values.NDim() > len(shape) && values.shape[0] == 1 {
		var err error
		values, err = values.Slice(AtIndex(0))
		if err != nil {
			return err
		}
	}
	rhs, err := values.BroadcastTo(shape...)
	if err != nil {
		return err
	}
	// Disjoint native storage needs no RHS allocation or casting. Check
	// byte spans, not Array identity: independently wrapped slices can alias.
	if !advanced && nativeNumericStorage(view) && nativeNumericStorage(rhs) &&
		view.DType() == rhs.DType() && !MayShareMemory(view, rhs) &&
		(view.IsCContiguous() || rhs.IsCContiguous() || view.NDim() == 2 && view.IsFContiguous() || view.IsFContiguous() && rhs.IsFContiguous()) {
		copyNative(view, rhs)
		return nil
	}
	snapshot, err := a.castLike(rhs)
	if err != nil {
		return err
	}
	if !advanced {
		// CastLike already creates an independent C-order snapshot, even
		// for distinct Array handles backed by overlapping Go slices.
		if nativeNumericStorage(view) && nativeNumericStorage(snapshot) {
			copyNative(view, snapshot)
		} else {
			view.walk(func(i, off int) { a.setElement(off, snapshot.elementAt(i)) })
		}
		return nil
	}
	for i, off := range offsets {
		a.setElement(off, snapshot.elementAt(i))
	}
	return nil
}

// Nonzero returns one int64 coordinate array per axis, in C order.
func Nonzero(a *Array) ([]*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 0 {
		return nil, ErrShape
	}
	coords := make([][]int64, a.NDim())
	a.walk(func(flat, off int) {
		if !a.elementAt(off).truth(a.DType()) {
			return
		}
		for j := a.NDim() - 1; j >= 0; j-- {
			coords[j] = append(coords[j], int64(flat%a.shape[j]))
			flat /= a.shape[j]
		}
	})
	out := make([]*Array, len(coords))
	for i, c := range coords {
		out[i], _ = WrapSlice(c, len(c))
	}
	return out, nil
}

// Take gathers indices along one axis; nil axis gathers from the C-order flattening.
func Take(a, indices *Array, axis *int) (*Array, error) {
	if a.DType() == 0 || !indices.DType().IsInteger() {
		return nil, ErrDType
	}
	if axis == nil {
		var err error
		a, err = a.Reshape(-1)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	i, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, err
	}
	spec := make([]Index, a.NDim())
	for j := range spec {
		spec[j] = Slice{}
	}
	spec[i] = indices
	return a.Index(spec...)
}

type TakeOptions struct {
	Mode string // raise, wrap, clip
	Out  *Array
}

func indexMode(indices *Array, n int, mode string) (*Array, error) {
	if !indices.DType().IsInteger() {
		return nil, ErrDType
	}
	switch mode {
	case "", "raise":
		return indices, nil
	case "wrap", "clip":
	default:
		return nil, ErrValue
	}
	if n == 0 && indices.size != 0 {
		return nil, ErrIndex
	}
	out, e := Zeros(Int64, indices.shape...)
	if e != nil {
		return nil, e
	}
	indices.walk(func(i, off int) {
		x := indices.elementAt(off).i
		var v int64
		if indices.DType().IsSigned() {
			v = int64(x)
			if mode == "wrap" {
				v %= int64(n)
				if v < 0 {
					v += int64(n)
				}
			} else {
				v = max(0, min(v, int64(n-1)))
			}
		} else if mode == "wrap" {
			v = int64(x % uint64(n))
		} else {
			v = int64(min(x, uint64(n-1)))
		}
		out.setElement(i, element{i: uint64(v)})
	})
	return out, nil
}
func TakeWithOptions(a, indices *Array, axis *int, opt TakeOptions) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	n := a.size
	if axis != nil {
		ax, e := NormalizeAxis(*axis, a.NDim())
		if e != nil {
			return nil, e
		}
		n = a.shape[ax]
	}
	ix, e := indexMode(indices, n, opt.Mode)
	if e != nil {
		return nil, e
	}
	result, e := Take(a, ix, axis)
	if e != nil {
		return nil, e
	}
	return reductionOutput(opt.Out, result)
}

func FlatNonzero(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	v, err := a.Reshape(-1)
	if err != nil {
		return nil, err
	}
	c, err := Nonzero(v)
	if err != nil {
		return nil, err
	}
	return c[0], nil
}

// Where chooses between x and y with independent broadcasting of the condition.
func Where(condition, x, y *Array) (*Array, error) {
	if condition.DType() == 0 {
		return nil, ErrDType
	}
	d, err := PromoteTypes(x.DType(), y.DType())
	if err != nil {
		return nil, err
	}
	shape, err := BroadcastShapes(condition.shape, x.shape, y.shape)
	if err != nil {
		return nil, err
	}
	c, err := condition.BroadcastTo(shape...)
	if err != nil {
		return nil, err
	}
	x, err = x.BroadcastTo(shape...)
	if err != nil {
		return nil, err
	}
	y, err = y.BroadcastTo(shape...)
	if err != nil {
		return nil, err
	}
	out, err := allocationFor([]*Array{x, y}, d, shape)
	if err != nil {
		return nil, err
	}
	if d.IsSpecial() || x.DType().IsSpecial() || y.DType().IsSpecial() {
		x, err = out.castLike(x)
		if err != nil {
			return nil, err
		}
		y, err = out.castLike(y)
		if err != nil {
			return nil, err
		}
	}
	for i := 0; i < out.size; i++ {
		source := y
		if c.elementAt(c.offset + flatOffset(i, shape, c.strides)).truth(c.DType()) {
			source = x
		}
		out.setElement(i, castElement(source.elementAt(source.offset+flatOffset(i, shape, source.strides)), source.DType(), d))
	}
	return out, nil
}

// Concatenate joins arrays along a named axis, promoting their dtypes.
func Concatenate(arrays []*Array, axis int) (*Array, error) {
	if len(arrays) == 0 {
		return nil, ErrShape
	}
	a := arrays[0]
	if a.DType() == 0 {
		return nil, ErrDType
	}
	axis, err := NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	shape := slices.Clone(a.shape)
	shape[axis] = 0
	d := a.DType()
	for _, v := range arrays {
		if v.DType() == 0 || v.NDim() != len(shape) {
			return nil, ErrShape
		}
		for j, dim := range v.shape {
			if j != axis && shape[j] != dim {
				return nil, ErrShape
			}
		}
		if shape[axis] > maxInt-v.shape[axis] {
			return nil, ErrShape
		}
		shape[axis] += v.shape[axis]
		d, err = PromoteTypes(d, v.DType())
		if err != nil {
			return nil, err
		}
	}
	out, err := allocationFor(arrays, d, shape)
	if err != nil {
		return nil, err
	}
	start := 0
	for _, v := range arrays {
		spec := make([]Index, axis+1)
		for j := range spec {
			spec[j] = Slice{}
		}
		spec[axis] = Range(start, start+v.shape[axis])
		if err = out.Assign(spec, v); err != nil {
			return nil, err
		}
		start += v.shape[axis]
	}
	return out, nil
}

func Stack(arrays []*Array, axis int) (*Array, error) {
	if len(arrays) == 0 || arrays[0].DType() == 0 {
		return nil, ErrShape
	}
	n := arrays[0].NDim() + 1
	axis, err := NormalizeAxis(axis, n)
	if err != nil {
		return nil, err
	}
	expanded := make([]*Array, len(arrays))
	for i, a := range arrays {
		if !slices.Equal(a.shape, arrays[0].shape) {
			return nil, ErrShape
		}
		spec := make([]Index, axis+1)
		for j := range spec {
			spec[j] = Slice{}
		}
		spec[axis] = NewAxis
		expanded[i], err = a.Slice(spec...)
		if err != nil {
			return nil, err
		}
	}
	return Concatenate(expanded, axis)
}
