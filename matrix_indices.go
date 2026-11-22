package np

import "slices"

func Tri(rows, cols, k int, dtype DType) (*Array, error) {
	out, err := Zeros(dtype, rows, cols)
	if err != nil {
		return nil, err
	}
	one := castElement(element{i: 1}, Int64, dtype)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if j-i <= k {
				out.setElement(i*cols+j, one)
			}
		}
	}
	return out, nil
}
func Tril(a *Array, k int) (*Array, error) { return triangular(a, k, false) }
func Triu(a *Array, k int) (*Array, error) { return triangular(a, k, true) }
func triangular(a *Array, k int, upper bool) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 0 {
		return nil, ErrShape
	}
	if a.NDim() == 1 {
		var err error
		a, err = a.BroadcastTo(a.size, a.size)
		if err != nil {
			return nil, err
		}
	}
	shape := a.shape
	m, n := shape[len(shape)-2], shape[len(shape)-1]
	out, err := a.zerosLike(shape)
	if err != nil {
		return nil, err
	}
	if out.size == 0 {
		return out, nil
	}
	a.walk(func(i, off int) {
		diff := i%n - i/n%m
		if upper && diff >= k || !upper && diff <= k {
			out.setElement(i, a.elementAt(off))
		}
	})
	return out, nil
}

func Diag(a *Array, k int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 2 {
		return Diagonal(a, k, 0, 1)
	}
	if a.NDim() != 1 {
		return nil, ErrShape
	}
	if k == -maxInt-1 {
		return nil, ErrShape
	}
	offset := max(k, -k)
	if a.size > maxInt-offset {
		return nil, ErrShape
	}
	n := a.size + offset
	out, err := a.zerosLike([]int{n, n})
	if err != nil {
		return nil, err
	}
	row, col := 0, 0
	if k < 0 {
		row = -k
	} else {
		col = k
	}
	a.walk(func(i, off int) { out.setElement((row+i)*n+col+i, a.elementAt(off)) })
	return out, nil
}
func Diagflat(a *Array, k int) (*Array, error) {
	x, err := Ravel(a)
	if err != nil {
		return nil, err
	}
	return Diag(x, k)
}
func DiagIndices(n, ndim int) ([]*Array, error) {
	if n < 0 || ndim < 1 || ndim > maxDims {
		return nil, ErrShape
	}
	idx, err := ArangeInt64(0, int64(n), 1)
	if err != nil {
		return nil, err
	}
	out := make([]*Array, ndim)
	for i := range out {
		out[i] = idx
	}
	return out, nil
}
func DiagIndicesFrom(a *Array) ([]*Array, error) {
	if a.DType() == 0 || a.NDim() < 2 {
		return nil, ErrShape
	}
	n := a.shape[0]
	for _, dim := range a.shape {
		if dim != n {
			return nil, ErrShape
		}
	}
	return DiagIndices(n, a.NDim())
}
func TrilIndices(rows, k, cols int) ([]*Array, error) { return triIndices(rows, cols, k, false) }
func TriuIndices(rows, k, cols int) ([]*Array, error) { return triIndices(rows, cols, k, true) }
func triIndices(rows, cols, k int, upper bool) ([]*Array, error) {
	if _, err := ElementCount(Int64, rows, cols); err != nil {
		return nil, err
	}
	var ri, ci []int64
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := j - i
			if upper && diff >= k || !upper && diff <= k {
				ri = append(ri, int64(i))
				ci = append(ci, int64(j))
			}
		}
	}
	r, _ := WrapSlice(ri, len(ri))
	c, _ := WrapSlice(ci, len(ci))
	return []*Array{r, c}, nil
}
func TrilIndicesFrom(a *Array, k int) ([]*Array, error) {
	if a.DType() == 0 || a.NDim() != 2 {
		return nil, ErrShape
	}
	return TrilIndices(a.shape[0], k, a.shape[1])
}
func TriuIndicesFrom(a *Array, k int) ([]*Array, error) {
	if a.DType() == 0 || a.NDim() != 2 {
		return nil, ErrShape
	}
	return TriuIndices(a.shape[0], k, a.shape[1])
}

// FillDiagonal writes repeated values on a 2D or equal-sided ND diagonal.
func FillDiagonal(a, values *Array, wrap bool) error {
	if !a.Writable() {
		return ErrReadOnly
	}
	if a.NDim() < 2 || values.DType() == 0 {
		return ErrShape
	}
	if values.size == 0 {
		return nil
	}
	v, err := a.castLike(values)
	if err != nil {
		return err
	}
	if a.NDim() == 2 {
		rows, cols := a.shape[0], a.shape[1]
		if rows == 0 || cols == 0 {
			return nil
		}
		count := min(rows, cols)
		if wrap && rows > cols {
			step := cols + 1
			flat := 0
			for i := 0; flat < a.size; i++ {
				a.setElement(a.offset+flatOffset(flat, a.shape, a.strides), v.elementAt(i%v.size))
				if flat > maxInt-step {
					break
				}
				flat += step
			}
		} else {
			for i := 0; i < count; i++ {
				a.setElement(a.offset+i*a.strides[0]+i*a.strides[1], v.elementAt(i%v.size))
			}
		}
		return nil
	}
	n := a.shape[0]
	stride := 0
	for i, d := range a.shape {
		if d != n {
			return ErrShape
		}
		stride += a.strides[i]
	}
	for i := 0; i < n; i++ {
		a.setElement(a.offset+i*stride, v.elementAt(i%v.size))
	}
	return nil
}
func Vander(x *Array, n int, increasing bool) (*Array, error) {
	if x.DType() == 0 || x.NDim() != 1 || n < 0 {
		return nil, ErrShape
	}
	d, err := PromoteTypes(x.DType(), Int64)
	if err != nil {
		return nil, err
	}
	out, err := Zeros(d, x.size, n)
	if err != nil {
		return nil, err
	}
	x, err = x.Astype(d)
	if err != nil {
		return nil, err
	}
	for i := 0; i < x.size; i++ {
		v := castElement(element{i: 1}, Int64, d)
		for j := 0; j < n; j++ {
			col := j
			if !increasing {
				col = n - 1 - j
			}
			out.setElement(i*n+col, v)
			v = multiplyScalar(v, x.elementAt(i), d)
		}
	}
	return out, nil
}

func Indices(dimensions []int, sparse bool) ([]*Array, error) {
	if _, err := ElementCount(Int64, dimensions...); err != nil {
		return nil, err
	}
	out := make([]*Array, len(dimensions))
	for i, n := range dimensions {
		shape := make([]int, len(dimensions))
		for j := range shape {
			shape[j] = 1
		}
		shape[i] = n
		v, err := ArangeInt64(0, int64(n), 1)
		if err != nil {
			return nil, err
		}
		v, err = v.Reshape(shape...)
		if err != nil {
			return nil, err
		}
		if !sparse {
			v, err = v.BroadcastTo(dimensions...)
			if err != nil {
				return nil, err
			}
			v, err = v.Copy()
			if err != nil {
				return nil, err
			}
		}
		out[i] = v
	}
	return out, nil
}
func DenseIndices(dimensions []int) (*Array, error) {
	if len(dimensions) == 0 {
		return Zeros(Int64, 0)
	}
	axes, err := Indices(dimensions, false)
	if err != nil {
		return nil, err
	}
	return Stack(axes, 0)
}
func Meshgrid(indexing string, sparse bool, points ...*Array) ([]*Array, error) {
	if indexing == "" {
		indexing = "xy"
	}
	if indexing != "xy" && indexing != "ij" {
		return nil, ErrValue
	}
	shape := make([]int, len(points))
	axes := make([]int, len(points))
	for i, p := range points {
		if p.DType() == 0 || p.NDim() != 1 {
			return nil, ErrShape
		}
		shape[i] = p.size
		axes[i] = i
	}
	if indexing == "xy" && len(points) >= 2 {
		shape[0], shape[1] = shape[1], shape[0]
		axes[0], axes[1] = 1, 0
	}
	out := make([]*Array, len(points))
	for i, p := range points {
		sh := make([]int, len(points))
		for j := range sh {
			sh[j] = 1
		}
		sh[axes[i]] = p.size
		v, err := p.Reshape(sh...)
		if err != nil {
			return nil, err
		}
		if !sparse {
			v, err = v.BroadcastTo(shape...)
			if err != nil {
				return nil, err
			}
		}
		out[i], err = v.Copy()
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
func Ix(arrays ...*Array) ([]*Array, error) {
	out := make([]*Array, len(arrays))
	for i, a := range arrays {
		if a.DType() == 0 || a.NDim() != 1 {
			return nil, ErrShape
		}
		if a.DType() == Bool {
			var err error
			a, err = FlatNonzero(a)
			if err != nil {
				return nil, err
			}
		} else if !a.DType().IsInteger() {
			return nil, ErrDType
		}
		shape := make([]int, len(arrays))
		for j := range shape {
			shape[j] = 1
		}
		shape[i] = a.size
		var err error
		out[i], err = a.Reshape(shape...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
func indexStrides(shape []int, order string) ([]int, error) {
	if order == "" || order == "C" {
		r := make([]int, len(shape))
		s := 1
		for i := len(shape) - 1; i >= 0; i-- {
			r[i] = s
			s *= shape[i]
		}
		return r, nil
	}
	if order == "F" {
		r := make([]int, len(shape))
		s := 1
		for i, d := range shape {
			r[i] = s
			s *= d
		}
		return r, nil
	}
	return nil, ErrValue
}
func RavelMultiIndex(coords []*Array, shape []int, mode, order string) (*Array, error) {
	if len(coords) != len(shape) || len(coords) == 0 {
		return nil, ErrShape
	}
	if _, err := ElementCount(Int64, shape...); err != nil {
		return nil, err
	}
	strides, err := indexStrides(shape, order)
	if err != nil {
		return nil, err
	}
	shapes := make([][]int, len(coords))
	for i, c := range coords {
		if !c.DType().IsInteger() {
			return nil, ErrIndex
		}
		shapes[i] = c.shape
	}
	sh, err := BroadcastShapes(shapes...)
	if err != nil {
		return nil, err
	}
	out, err := Zeros(Int64, sh...)
	if err != nil {
		return nil, err
	}
	for ax, c := range coords {
		v, err := c.BroadcastTo(sh...)
		if err != nil {
			return nil, err
		}
		for i := 0; i < out.size; i++ {
			e := v.elementAt(v.offset + flatOffset(i, sh, v.strides))
			if mode == "" || mode == "raise" {
				if c.DType().IsSigned() && int64(e.i) < 0 || e.i >= uint64(shape[ax]) {
					return nil, ErrIndex
				}
			}
			idx, err := indexForPut(e, c.DType(), shape[ax], mode)
			if err != nil {
				return nil, err
			}
			out.setElement(i, element{i: out.elementAt(i).i + uint64(idx*strides[ax])})
		}
	}
	return out, nil
}
func UnravelIndex(indices *Array, shape []int, order string) ([]*Array, error) {
	if !indices.DType().IsInteger() {
		return nil, ErrDType
	}
	total, err := ElementCount(Int64, shape...)
	if err != nil {
		return nil, err
	}
	strides, err := indexStrides(shape, order)
	if err != nil {
		return nil, err
	}
	out := make([]*Array, len(shape))
	for i := range out {
		out[i], err = Zeros(Int64, indices.shape...)
		if err != nil {
			return nil, err
		}
	}
	var failure error
	indices.walk(func(flat, off int) {
		if failure != nil {
			return
		}
		v := indices.elementAt(off)
		if indices.DType().IsSigned() && int64(v.i) < 0 || v.i >= uint64(total) {
			failure = ErrIndex
			return
		}
		for ax, dim := range shape {
			out[ax].setElement(flat, element{i: uint64(int(v.i) / strides[ax] % dim)})
		}
	})
	if failure != nil {
		return nil, failure
	}
	return out, nil
}

func UniqueAll(a *Array) (UniqueResult, error) {
	no := false
	return Unique(a, UniqueOptions{EqualNaN: &no})
}
func UniqueCounts(a *Array) (*Array, *Array, error) {
	r, err := UniqueAll(a)
	return r.Values, r.Counts, err
}
func UniqueInverse(a *Array) (*Array, *Array, error) {
	r, err := UniqueAll(a)
	return r.Values, r.Inverse, err
}

func TrimZeros(a *Array, trimFront, trimBack bool) (*Array, error) {
	if a.DType() == 0 || a.NDim() != 1 {
		return nil, ErrShape
	}
	start, end := 0, a.size
	if trimFront {
		for start < end && !a.elementAt(a.offset+start*a.strides[0]).truth(a.DType()) {
			start++
		}
	}
	if trimBack {
		for end > start && !a.elementAt(a.offset+(end-1)*a.strides[0]).truth(a.DType()) {
			end--
		}
	}
	return a.Slice(Range(start, end))
}
func Ediff1D(a, begin, end *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	x, err := Ravel(a)
	if err != nil {
		return nil, err
	}
	d, err := Diff(x, 1, 0)
	if err != nil {
		return nil, err
	}
	arrays := []*Array{d}
	if begin != nil {
		b, err := Ravel(begin)
		if err != nil {
			return nil, err
		}
		b, err = b.Astype(a.DType())
		if err != nil {
			return nil, err
		}
		arrays = slices.Insert(arrays, 0, b)
	}
	if end != nil {
		b, err := Ravel(end)
		if err != nil {
			return nil, err
		}
		b, err = b.Astype(a.DType())
		if err != nil {
			return nil, err
		}
		arrays = append(arrays, b)
	}
	return Concatenate(arrays, 0)
}
