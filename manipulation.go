package np

import (
	"math"
	"slices"
)

func Full(dtype DType, value any, shape ...int) (*Array, error) {
	if dtype.IsSpecial() {
		out, e := Zeros(dtype, shape...)
		if e != nil {
			return nil, e
		}
		if e = out.Fill(value); e != nil {
			return nil, e
		}
		return out, nil
	}
	from := dtypeOf(value)
	if from == 0 {
		return nil, ErrDType
	}
	out, err := Zeros(dtype, shape...)
	if err != nil {
		return nil, err
	}
	v := castElement(elementFrom(value), from, dtype)
	for i := 0; i < out.size; i++ {
		out.setElement(i, v)
	}
	return out, nil
}
func Ones(dtype DType, shape ...int) (*Array, error) { return Full(dtype, int64(1), shape...) }
func ZerosLike(a *Array) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	return ZerosLikeWithShape(a, 0, "K", a.shape)
}
func OnesLike(a *Array) (*Array, error) {
	if a.DType() == StringDType {
		return FullLike(a, "1")
	}
	return FullLike(a, int64(1))
}
func FullLike(a *Array, value any) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	return FullLikeWithShape(a, value, 0, "K", a.shape)
}

// ZerosLikeWithShape, OnesLikeWithShape and FullLikeWithShape use the
// same explicit dtype, order and target-shape contract as EmptyLikeWithShape.
func ZerosLikeWithShape(a *Array, dtype DType, order string, shape []int) (*Array, error) {
	v, e := EmptyLikeWithShape(a, dtype, order, shape)
	if e == nil && v.DType() == Object {
		e = v.Fill(int64(0))
	}
	return v, e
}
func OnesLikeWithShape(a *Array, dtype DType, order string, shape []int) (*Array, error) {
	if dtype == StringDType || dtype == 0 && a.DType() == StringDType {
		return FullLikeWithShape(a, "1", dtype, order, shape)
	}
	return FullLikeWithShape(a, int64(1), dtype, order, shape)
}
func FullLikeWithShape(a *Array, value any, dtype DType, order string, shape []int) (*Array, error) {
	out, err := EmptyLikeWithShape(a, dtype, order, shape)
	if err != nil {
		return nil, err
	}
	if out.DType().IsSpecial() {
		if e := out.Fill(value); e != nil {
			return nil, e
		}
		return out, nil
	}
	from := dtypeOf(value)
	if from == 0 || from.IsTemporal() {
		return nil, ErrDType
	}
	v := castElement(elementFrom(value), from, out.DType())
	out.walk(func(_, off int) { out.setElement(off, v) })
	return out, nil
}

func Eye(rows, cols, k int, dtype DType) (*Array, error) {
	out, err := Zeros(dtype, rows, cols)
	if err != nil {
		return nil, err
	}
	one := castElement(element{i: 1}, Int64, dtype)
	for i := 0; i < rows; i++ {
		if k >= -i && k < cols-i {
			out.setElement(i*cols+i+k, one)
		}
	}
	return out, nil
}
func Identity(n int, dtype DType) (*Array, error) { return Eye(n, n, 0, dtype) }

// ArangeInt64 constructs the half-open integer sequence without a float conversion.
func ArangeInt64(start, stop, step int64) (*Array, error) {
	if step == 0 {
		return nil, ErrValue
	}
	var count uint64
	if step > 0 && start < stop {
		count = 1 + (uint64(stop)-uint64(start)-1)/uint64(step)
	}
	if step < 0 && start > stop {
		count = 1 + (uint64(start)-uint64(stop)-1)/(0-uint64(step))
	}
	if count > uint64(maxInt/8) {
		return nil, ErrShape
	}
	v := make([]int64, int(count))
	for i := range v {
		v[i] = start + int64(i)*step
	}
	return WrapSlice(v, len(v))
}

// Arange builds a half-open floating-point sequence with an explicit output dtype.
func Arange(start, stop, step float64, dtype DType) (*Array, error) {
	if step == 0 || math.IsNaN(step) || math.IsNaN(start) || math.IsNaN(stop) {
		return nil, ErrValue
	}
	count := math.Max(0, math.Ceil((stop-start)/step))
	if math.IsInf(count, 0) || count > float64(maxInt/16) {
		return nil, ErrShape
	}
	out, err := Zeros(dtype, int(count))
	if err != nil {
		return nil, err
	}
	// NumPy arange uses the delta after casting start and start+step to the dtype.
	if dtype.IsFloat() || dtype.IsInteger() {
		tmp, err := FromFloat64([]float64{start, start + step}, 2)
		if err != nil {
			return nil, err
		}
		tmp, err = tmp.Astype(dtype)
		if err != nil {
			return nil, err
		}
		x, y := tmp.elementAt(0), tmp.elementAt(1)
		if dtype.IsInteger() {
			for i := 0; i < out.size; i++ {
				out.setElement(i, element{i: x.i + uint64(i)*(y.i-x.i)})
			}
			return out, nil
		}
		start, step = real(x.z), real(y.z)-real(x.z)
	}
	for i := 0; i < out.size; i++ {
		out.setElement(i, castElement(element{z: complex(start+float64(i)*step, 0)}, Float64, dtype))
	}
	return out, nil
}
func Linspace(start, stop float64, num int, endpoint bool, dtype DType) (*Array, error) {
	out, err := Zeros(dtype, num)
	if err != nil {
		return nil, err
	}
	div := num
	if endpoint {
		div--
	}
	for i := 0; i < num; i++ {
		v := start
		if div > 0 {
			v = start + (stop-start)*float64(i)/float64(div)
		}
		if endpoint && i == num-1 && num > 1 {
			v = stop
		}
		if dtype.IsInteger() {
			v = math.Floor(v)
		}
		out.setElement(i, castElement(element{z: complex(v, 0)}, Float64, dtype))
	}
	return out, nil
}
func Logspace(start, stop float64, num int, endpoint bool, base float64, dtype DType) (*Array, error) {
	x, err := Linspace(start, stop, num, endpoint, Float64)
	if err != nil {
		return nil, err
	}
	x, err = Power(Scalar(base), x)
	if err != nil {
		return nil, err
	}
	return x.Astype(dtype)
}

func Ravel(a *Array) (*Array, error) { return a.Reshape(-1) }
func Flatten(a *Array) (*Array, error) {
	out, err := Ravel(a)
	if err != nil {
		return nil, err
	}
	return out.Copy()
}
func ExpandDims(a *Array, axis int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	ax, err := NormalizeAxis(axis, a.NDim()+1)
	if err != nil {
		return nil, err
	}
	shape := slices.Clone(a.shape)
	shape = slices.Insert(shape, ax, 1)
	return a.Reshape(shape...)
}
func Squeeze(a *Array, axes ...int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	remove := make([]bool, a.NDim())
	if len(axes) == 0 {
		for i, d := range a.shape {
			remove[i] = d == 1
		}
	} else {
		for _, ax := range axes {
			i, err := NormalizeAxis(ax, a.NDim())
			if err != nil || remove[i] || a.shape[i] != 1 {
				return nil, ErrAxis
			}
			remove[i] = true
		}
	}
	var spec []Index
	for i := range a.shape {
		if remove[i] {
			spec = append(spec, AtIndex(0))
		} else {
			spec = append(spec, Slice{})
		}
	}
	return a.Slice(spec...)
}
func SwapAxes(a *Array, axis1, axis2 int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	x, err := NormalizeAxis(axis1, a.NDim())
	if err != nil {
		return nil, err
	}
	y, err := NormalizeAxis(axis2, a.NDim())
	if err != nil {
		return nil, err
	}
	axes := make([]int, a.NDim())
	for i := range axes {
		axes[i] = i
	}
	axes[x], axes[y] = axes[y], axes[x]
	return a.Transpose(axes...)
}
func MoveAxis(a *Array, source, destination int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	x, err := NormalizeAxis(source, a.NDim())
	if err != nil {
		return nil, err
	}
	y, err := NormalizeAxis(destination, a.NDim())
	if err != nil {
		return nil, err
	}
	axes := make([]int, 0, a.NDim())
	for i := 0; i < a.NDim(); i++ {
		if i != x {
			axes = append(axes, i)
		}
	}
	axes = slices.Insert(axes, y, x)
	return a.Transpose(axes...)
}
func Flip(a *Array, axes ...int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	index := make([]Index, a.NDim())
	for i := range index {
		index[i] = Slice{}
	}
	if len(axes) == 0 {
		for i := range index {
			index[i] = Reverse()
		}
	} else {
		seen := make([]bool, a.NDim())
		for _, ax := range axes {
			i, err := NormalizeAxis(ax, a.NDim())
			if err != nil || seen[i] {
				return nil, ErrAxis
			}
			seen[i] = true
			index[i] = Reverse()
		}
	}
	return a.Slice(index...)
}
func Roll(a *Array, shift int, axis *int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	orig := a.shape
	flatten := axis == nil
	if flatten {
		var err error
		a, err = a.Reshape(-1)
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
	idx := make([]int64, n)
	if n != 0 {
		shift %= n
		if shift < 0 {
			shift += n
		}
		for i := range idx {
			idx[i] = int64((i - shift + n) % n)
		}
	}
	index, _ := WrapSlice(idx, n)
	out, err := Take(a, index, &ax)
	if err != nil {
		return nil, err
	}
	if flatten {
		return out.Reshape(orig...)
	}
	return out, nil
}
func Repeat(a *Array, repeats int, axis *int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if repeats < 0 {
		return nil, ErrValue
	}
	if axis == nil {
		var err error
		a, err = a.Reshape(-1)
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
	if repeats != 0 && n > maxInt/8/repeats {
		return nil, ErrShape
	}
	indices := make([]int64, n*repeats)
	for i := range indices {
		indices[i] = int64(i / repeats)
	}
	idx, _ := WrapSlice(indices, len(indices))
	return Take(a, idx, &ax)
}
func Tile(a *Array, reps ...int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	rank := max(len(reps), a.NDim())
	shape := make([]int, rank)
	base := make([]int, rank)
	stride := make([]int, rank)
	for i := range base {
		base[i] = 1
	}
	copy(base[rank-a.NDim():], a.shape)
	copy(stride[rank-a.NDim():], a.strides)
	for i := range shape {
		r := 1
		if i >= rank-len(reps) {
			r = reps[i-rank+len(reps)]
		}
		if r < 0 || r != 0 && base[i] > maxInt/r {
			return nil, ErrShape
		}
		shape[i] = base[i] * r
	}
	out, err := a.zerosLike(shape)
	if err != nil {
		return nil, err
	}
	for i := 0; i < out.size; i++ {
		flat, off := i, a.offset
		for j := rank - 1; j >= 0; j-- {
			off += (flat % shape[j] % base[j]) * stride[j]
			flat /= shape[j]
		}
		out.setElement(i, a.elementAt(off))
	}
	return out, nil
}
func Diff(a *Array, n, axis int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if n < 0 {
		return nil, ErrValue
	}
	if n == 0 {
		return a, nil
	}
	ax, err := NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	for k := 0; k < n; k++ {
		length := a.shape[ax]
		if length == 0 {
			return a.Copy()
		}
		left, right := make([]Index, ax+1), make([]Index, ax+1)
		for j := 0; j < ax; j++ {
			left[j], right[j] = Slice{}, Slice{}
		}
		left[ax], right[ax] = Range(0, length-1), Range(1, length)
		x, err := a.Slice(left...)
		if err != nil {
			return nil, err
		}
		y, err := a.Slice(right...)
		if err != nil {
			return nil, err
		}
		if a.DType() == Bool {
			a, err = NotEqual(y, x)
		} else {
			a, err = Subtract(y, x)
		}
		if err != nil {
			return nil, err
		}
	}
	return a, nil
}

// Diagonal exposes the selected diagonal as a read-only view (NumPy diagonal semantics).
func Diagonal(a *Array, offset, axis1, axis2 int) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() < 2 {
		return nil, ErrShape
	}
	x, err := NormalizeAxis(axis1, a.NDim())
	if err != nil {
		return nil, err
	}
	y, err := NormalizeAxis(axis2, a.NDim())
	if err != nil || x == y {
		return nil, ErrAxis
	}
	sx, sy := 0, 0
	if offset < 0 {
		if offset == -maxInt-1 {
			sx = a.shape[x]
		} else {
			sx = min(-offset, a.shape[x])
		}
	} else {
		sy = min(offset, a.shape[y])
	}
	length := min(a.shape[x]-sx, a.shape[y]-sy)
	var shape, strides []int
	for i, d := range a.shape {
		if i != x && i != y {
			shape = append(shape, d)
			strides = append(strides, a.strides[i])
		}
	}
	shape = append(shape, length)
	stride := a.strides[x] + a.strides[y]
	if _, ok := checkedStride(stride, 1, a.strideUnit()); !ok {
		return nil, ErrShape
	}
	strides = append(strides, stride)
	size, err := a.shapeSize(shape)
	if err != nil {
		return nil, err
	}
	off := 0
	if size > 0 {
		off = a.offset + sx*a.strides[x] + sy*a.strides[y]
	}
	return &Array{data: a.data, shape: shape, strides: strides, offset: off, size: size, readOnly: true, base: a.viewBase()}, nil
}
func Trace(a *Array, offset, axis1, axis2 int, dtype DType) (*Array, error) {
	diag, err := Diagonal(a, offset, axis1, axis2)
	if err != nil {
		return nil, err
	}
	return Sum(diag, SumOptions{Axes: []int{-1}, DType: dtype})
}
