package np

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"

	numkernel "github.com/SamuelSupe/numpygo/internal/numeric"
)

type op uint8

const (
	addOp op = iota
	subOp
	mulOp
	divOp
	floorOp
	remOp
	powOp
	andOp
	orOp
	xorOp
)

func Add(a, b *Array) (*Array, error)         { return arithmetic(a, b, addOp) }
func Subtract(a, b *Array) (*Array, error)    { return arithmetic(a, b, subOp) }
func Multiply(a, b *Array) (*Array, error)    { return arithmetic(a, b, mulOp) }
func Divide(a, b *Array) (*Array, error)      { return arithmetic(a, b, divOp) }
func FloorDivide(a, b *Array) (*Array, error) { return arithmetic(a, b, floorOp) }
func Remainder(a, b *Array) (*Array, error)   { return arithmetic(a, b, remOp) }
func Power(a, b *Array) (*Array, error)       { return arithmetic(a, b, powOp) }
func BitwiseAnd(a, b *Array) (*Array, error)  { return arithmetic(a, b, andOp) }
func BitwiseOr(a, b *Array) (*Array, error)   { return arithmetic(a, b, orOp) }
func BitwiseXor(a, b *Array) (*Array, error)  { return arithmetic(a, b, xorOp) }

func arithmetic(a, b *Array, o op) (*Array, error) {
	if a.DType().IsSpecial() || b.DType().IsSpecial() {
		return specialArithmetic(a, b, o)
	}
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if o == divOp && (d.IsInteger() || d == Bool) {
		d = Float64
	}
	if d == Bool && (o == floorOp || o == remOp || o == powOp) {
		d = Int8
	}
	if (d == Bool && o == subOp) || (d.IsComplex() && (o == floorOp || o == remOp)) || (o >= andOp && !d.IsInteger() && d != Bool) {
		return nil, ErrDType
	}
	if o <= divOp && (d == Float32 || d == Float64) {
		if v, e := denseArithmetic(a, b, d, o); v != nil || e != nil {
			return v, e
		}
	}
	if o <= divOp && (d == Complex64 || d == Complex128) {
		if v, e := packedBinaryResult(a, b, d, 19+uint8(o)); v != nil || e != nil {
			return v, e
		}
	}
	return binaryMap(a, b, d, d, func(x, y element) (element, error) {
		return arithmeticScalar(x, y, d, o)
	})
}

func arithmeticScalar(x, y element, d DType, o op) (element, error) {
	if d.IsExtended() {
		return extendedArithmetic(x, y, d, o)
	}
	if d.IsInteger() || d == Bool {
		v, err := integerOp(x.i, y.i, d, o)
		return element{i: v}, err
	}
	if d.IsComplex() {
		switch o {
		case addOp:
			return element{z: x.z + y.z}, nil
		case subOp:
			return element{z: x.z - y.z}, nil
		case mulOp:
			return element{z: x.z * y.z}, nil
		case divOp:
			return element{z: complexDivide(x.z, y.z)}, nil
		case powOp:
			return element{z: complexPower(x.z, y.z)}, nil
		}
	}
	u, v := real(x.z), real(y.z)
	var z float64
	switch o {
	case addOp:
		z = u + v
	case subOp:
		z = u - v
	case mulOp:
		z = u * v
	case divOp:
		z = u / v
	case floorOp:
		z, _ = realDivmod(u, v)
	case remOp:
		_, z = realDivmod(u, v)
	case powOp:
		z = math.Pow(u, v)
	}
	return element{z: complex(z, 0)}, nil
}

// Native real arrays can share a typed loop over contiguous or strided lanes.
// Nonnative storage and mixed dtypes retain the general broadcast kernel.
func denseArithmetic(a, b *Array, d DType, o op) (*Array, error) {
	if a.DType() != d || b.DType() != d || a.data.buffer != nil || b.data.buffer != nil ||
		a.data.part != nil || b.data.part != nil {
		return nil, nil
	}
	shape := a.shape
	as, bs := 1, 1
	if a.NDim() == 0 {
		as = 0
		shape = b.shape
	}
	if b.NDim() == 0 {
		bs = 0
	}
	if !a.IsCContiguous() || !b.IsCContiguous() || (as != 0 && bs != 0 && !slices.Equal(a.shape, b.shape)) {
		return stridedArithmetic(a, b, d, o)
	}
	out, e := Zeros(d, shape...)
	if e != nil {
		return nil, e
	}
	if d == Float64 {
		denseLoop(out.data.real, a.data.real[a.offset:a.offset+a.size], b.data.real[b.offset:b.offset+b.size], as, bs, o)
	} else {
		x, y := a.data.raw.([]float32), b.data.raw.([]float32)
		denseLoop(out.data.raw.([]float32), x[a.offset:a.offset+a.size], y[b.offset:b.offset+b.size], as, bs, o)
	}
	return out, nil
}
func denseLoop[T float32 | float64](out, a, b []T, as, bs int, o op) {
	if len(out) >= 32 {
		switch z := any(out).(type) {
		case []float64:
			if numkernel.Lane64(z, any(a).([]float64), any(b).([]float64), as, bs, uint8(o)) {
				return
			}
		case []float32:
			if numkernel.Lane32(z, any(a).([]float32), any(b).([]float32), as, bs, uint8(o)) {
				return
			}
		}
	}
	switch o {
	case addOp:
		for i := range out {
			out[i] = a[i*as] + b[i*bs]
		}
	case subOp:
		for i := range out {
			out[i] = a[i*as] - b[i*bs]
		}
	case mulOp:
		for i := range out {
			out[i] = a[i*as] * b[i*bs]
		}
	case divOp:
		for i := range out {
			out[i] = a[i*as] / b[i*bs]
		}
	}
}

func stridedArithmetic(a, b *Array, d DType, o op) (*Array, error) {
	shape, e := BroadcastShapes(a.shape, b.shape)
	if e != nil {
		return nil, e
	}
	out, e := Zeros(d, shape...)
	if e != nil || out.size == 0 {
		return out, e
	}
	a, e = a.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	b, e = b.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	if d == Float64 {
		stridedRealLoop(out.data.real, a.data.real, b.data.real, a, b, shape, o)
	} else {
		stridedRealLoop(out.data.raw.([]float32), a.data.raw.([]float32), b.data.raw.([]float32), a, b, shape, o)
	}
	return out, nil
}

// Offsets are resolved once per inner lane, including negative and zero
// broadcast strides; each value thereafter is a direct typed operation.
func stridedRealLoop[T float32 | float64](out, x, y []T, a, b *Array, shape []int, o op) {
	ax := len(shape) - 1
	width := shape[ax]
	as, bs := a.strides[ax], b.strides[ax]
	if ax == 1 && shape[0] >= 32 && width >= 32 &&
		((a.strides[0] == 1 || a.strides[0] == -1) && as != 0 && as != 1 || (b.strides[0] == 1 || b.strides[0] == -1) && bs != 0 && bs != 1) {
		// Pack spatially close tiles for transposed operands while keeping
		// the result in the root arithmetic API's logical C order.
		const tile = 32
		for r := 0; r < shape[0]; r += tile {
			for c := 0; c < width; c += tile {
				end := min(c+tile, width)
				for row := r; row < min(r+tile, shape[0]); row++ {
					ai := a.offset + row*a.strides[0] + c*as
					bi := b.offset + row*b.strides[0] + c*bs
					realLane(out[row*width+c:row*width+end], x, y, ai, bi, as, bs, o)
				}
			}
		}
		return
	}
	for row := 0; row < len(out)/width; row++ {
		ai := a.offset + flatOffset(row, shape[:ax], a.strides[:ax])
		bi := b.offset + flatOffset(row, shape[:ax], b.strides[:ax])
		z := out[row*width : (row+1)*width]
		realLane(z, x, y, ai, bi, as, bs, o)
	}
}
func realLane[T float32 | float64](z, x, y []T, ai, bi, as, bs int, o op) {
	if len(z) >= 32 && (as == 0 || as == 1) && (bs == 0 || bs == 1) {
		nx, ny := 1+(len(z)-1)*as, 1+(len(z)-1)*bs
		denseLoop(z, x[ai:ai+nx], y[bi:bi+ny], as, bs, o)
		return
	}
	switch o {
	case addOp:
		for i := range z {
			z[i] = x[ai+i*as] + y[bi+i*bs]
		}
	case subOp:
		for i := range z {
			z[i] = x[ai+i*as] - y[bi+i*bs]
		}
	case mulOp:
		for i := range z {
			z[i] = x[ai+i*as] * y[bi+i*bs]
		}
	case divOp:
		for i := range z {
			z[i] = x[ai+i*as] / y[bi+i*bs]
		}
	}
}
func integerOp(x, y uint64, d DType, o op) (uint64, error) {
	switch o {
	case addOp:
		if d == Bool {
			return x | y, nil
		}
		return x + y, nil
	case subOp:
		return x - y, nil
	case mulOp:
		return x * y, nil
	case andOp:
		return x & y, nil
	case orOp:
		return x | y, nil
	case xorOp:
		return x ^ y, nil
	case powOp:
		if d.IsSigned() && int64(y) < 0 {
			return 0, fmt.Errorf("%w: negative integer exponent", ErrValue)
		}
		z := uint64(1)
		for y != 0 {
			if y&1 != 0 {
				z *= x
			}
			x *= x
			y >>= 1
		}
		return z, nil
	case floorOp, remOp:
		if y == 0 {
			return 0, nil
		}
		if d.IsSigned() {
			a, b := int64(x), int64(y)
			q, r := a/b, a%b
			if r != 0 && (r < 0) != (b < 0) {
				q--
				r += b
			}
			if o == remOp {
				return uint64(r), nil
			}
			return uint64(q), nil
		}
		if o == remOp {
			return x % y, nil
		}
		return x / y, nil
	}
	return 0, ErrDType
}
func realDivmod(x, y float64) (float64, float64) {
	r := math.Mod(x, y)
	if y == 0 {
		return math.Floor(x / y), r
	}
	div := (x - r) / y
	if r != 0 {
		if math.Signbit(r) != math.Signbit(y) {
			r += y
			div--
		}
	} else {
		r = math.Copysign(0, y)
	}
	if div == 0 {
		return math.Copysign(0, x/y), r
	}
	q := math.Floor(div)
	if div-q > 0.5 {
		q++
	}
	return q, r
}

func complexDivide(x, y complex128) complex128 {
	return numkernel.ComplexDivide(x, y)
}

func complexPower(x, y complex128) complex128 {
	// Positive real bases with finite real exponents have zero imaginary
	// part even if the magnitude overflows. cmplx.Pow's polar formula
	// instead multiplies infinity by sin(0), manufacturing an imaginary NaN.
	if real(x) > 0 && !math.IsInf(real(x), 0) && imag(x) == 0 && imag(y) == 0 && !math.IsInf(real(y), 0) && !math.IsNaN(real(y)) {
		return complex(math.Pow(real(x), real(y)), 0)
	}
	return cmplx.Pow(x, y)
}

// binaryMap applies a promoted pairwise kernel in broadcast order.
func binaryMap(a, b *Array, common, out DType, fn func(element, element) (element, error)) (*Array, error) {
	if a.DType().IsSpecial() || b.DType().IsSpecial() || common.IsSpecial() || out.IsSpecial() {
		return nil, ErrDType
	}
	return binaryMapCore(a, b, common, out, fn)
}
func binaryMapCore(a, b *Array, common, out DType, fn func(element, element) (element, error)) (*Array, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return nil, ErrDType
	}
	shape, err := BroadcastShapes(a.shape, b.shape)
	if err != nil {
		return nil, err
	}
	result, err := Zeros(out, shape...)
	if err != nil {
		return nil, err
	}
	a, err = a.BroadcastTo(shape...)
	if err != nil {
		return nil, err
	}
	b, err = b.BroadcastTo(shape...)
	if err != nil {
		return nil, err
	}
	oa, ob := a.offset, b.offset
	index := make([]int, len(shape))
	for i := 0; i < result.size; i++ {
		x, y := a.elementAt(oa), b.elementAt(ob)
		if common != 0 {
			x = castElement(x, a.DType(), common)
			y = castElement(y, b.DType(), common)
		}
		v, err := fn(x, y)
		if err != nil {
			return nil, err
		}
		result.setElement(i, v)
		for axis := len(shape) - 1; axis >= 0; axis-- {
			if index[axis]+1 < shape[axis] {
				index[axis]++
				oa += a.strides[axis]
				ob += b.strides[axis]
				break
			}
			oa -= index[axis] * a.strides[axis]
			ob -= index[axis] * b.strides[axis]
			index[axis] = 0
		}
	}
	return result, nil
}
func lessElement(x, y element, d DType) bool {
	if d == StringDType {
		return flexKey(x) < flexKey(y)
	}
	if d.IsFlexible() {
		return flexOrderKey(x, d) < flexOrderKey(y, d)
	}
	if d.IsExtended() {
		return extendedLess(x, y, d)
	}
	if d.IsTemporal() {
		a, b := temporalTicks(x), temporalTicks(y)
		return a != NaT && b != NaT && a < b
	}
	if d == Object {
		return objectLess(x.extra, y.extra)
	}
	if d.IsSigned() {
		return int64(x.i) < int64(y.i)
	}
	if d.IsUnsigned() || d == Bool {
		return x.i < y.i
	}
	if real(x.z) == real(y.z) && d.IsComplex() {
		return imag(x.z) < imag(y.z)
	}
	return real(x.z) < real(y.z)
}
func equalElement(x, y element, d DType) bool {
	if d == StringDType {
		return !stringNaN(x) && !stringNaN(y) && flexKey(x) == flexKey(y)
	}
	if d.IsFlexible() {
		return flexOrderKey(x, d) == flexOrderKey(y, d)
	}
	if d.IsExtended() {
		return extendedValue(x, d).Equal(extendedValue(y, d))
	}
	if d.IsTemporal() {
		a, b := temporalTicks(x), temporalTicks(y)
		return a != NaT && b != NaT && a == b
	}
	if d == Object {
		return objectEqual(x.extra, y.extra)
	}
	if d == RecordDType {
		return recordEqual(x, y)
	}
	if d.IsInteger() || d == Bool {
		return x.i == y.i
	}
	return x.z == y.z
}
func nanElement(x element, d DType) bool {
	if d == StringDType {
		return stringNaN(x)
	}
	if d.IsExtended() {
		z := extendedValue(x, d)
		return z.Real.IsNaN() || z.Imag.IsNaN()
	}
	if d.IsTemporal() {
		return temporalTicks(x) == NaT
	}
	if d == Object {
		f := dtypeOf(x.extra)
		if f != 0 {
			return nanElement(elementFrom(x.extra), f)
		}
	}
	return (d.IsFloat() || d.IsComplex()) && (math.IsNaN(real(x.z)) || math.IsNaN(imag(x.z)))
}
func boolElement(b bool) element {
	if b {
		return element{i: 1}
	}
	return element{}
}
func compare(a, b *Array, ordered bool, fn func(element, element, DType) bool) (*Array, error) {
	if a.DType().IsSpecial() || b.DType().IsSpecial() {
		return specialComparison(a, b, ordered, fn)
	}
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	// A mixed int64/uint64 comparison must not round through float64.
	if d == Float64 && a.DType().IsInteger() && b.DType().IsInteger() {
		return binaryMap(a, b, 0, Bool, func(x, y element) (element, error) {
			if a.DType().IsSigned() && int64(x.i) < 0 {
				x.i, y.i = 0, 1
			} else if b.DType().IsSigned() && int64(y.i) < 0 {
				x.i, y.i = 1, 0
			}
			return boolElement(fn(x, y, Uint64)), nil
		})
	}
	return binaryMap(a, b, d, Bool, func(x, y element) (element, error) { return boolElement(fn(x, y, d)), nil })
}
func Equal(a, b *Array) (*Array, error) { return compare(a, b, false, equalElement) }
func NotEqual(a, b *Array) (*Array, error) {
	return compare(a, b, false, func(x, y element, d DType) bool { return !equalElement(x, y, d) })
}
func Less(a, b *Array) (*Array, error) { return compare(a, b, true, lessElement) }
func Greater(a, b *Array) (*Array, error) {
	return compare(a, b, true, func(x, y element, d DType) bool { return lessElement(y, x, d) })
}
func LessEqual(a, b *Array) (*Array, error) {
	return compare(a, b, true, func(x, y element, d DType) bool { return equalElement(x, y, d) || lessElement(x, y, d) })
}
func GreaterEqual(a, b *Array) (*Array, error) {
	return compare(a, b, true, func(x, y element, d DType) bool { return equalElement(x, y, d) || lessElement(y, x, d) })
}
func Minimum(a, b *Array) (*Array, error) { return extrema(a, b, false, false) }
func Maximum(a, b *Array) (*Array, error) { return extrema(a, b, true, false) }
func FMin(a, b *Array) (*Array, error)    { return extrema(a, b, false, true) }
func FMax(a, b *Array) (*Array, error)    { return extrema(a, b, true, true) }
func extrema(a, b *Array, maximum, ignoreNaN bool) (*Array, error) {
	if a.DType().IsFlexible() || b.DType().IsFlexible() {
		return nil, ErrDType
	}
	if a.DType().IsTemporal() || b.DType().IsTemporal() {
		if a.DType() != b.DType() {
			return nil, ErrDType
		}
		d := a.DType()
		u, e := commonDateUnit(a.data.unit, b.data.unit, d == DateTime64)
		if e != nil {
			return nil, e
		}
		a, e = a.CastTemporal(d, u.String())
		if e != nil {
			return nil, e
		}
		b, e = b.CastTemporal(d, u.String())
		if e != nil {
			return nil, e
		}
		return temporalMap(a, b, u, d, func(x, y element) (element, error) { return selectExtrema(x, y, d, maximum, ignoreNaN), nil })
	}
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	if d.IsExtended() {
		return binaryMapCore(a, b, d, d, func(x, y element) (element, error) { return selectExtrema(x, y, d, maximum, ignoreNaN), nil })
	}
	if !ignoreNaN && (d == Float32 || d == Float64) {
		code := uint8(15)
		if maximum {
			code = 16
		}
		if v, e := packedBinaryResult(a, b, d, code); v != nil || e != nil {
			return v, e
		}
	}
	if d == StringDType {
		x, y, e := stringPair(a, b)
		if e != nil {
			return nil, e
		}
		r, e := binaryMapCore(x, y, d, d, func(x, y element) (element, error) {
			if stringBadOrder(x) || stringBadOrder(y) {
				return element{}, ErrValue
			}
			return selectExtrema(x, y, d, maximum, ignoreNaN || !maximum), nil
		})
		if e == nil {
			r.setStringDescriptor(x.stringDescriptor())
		}
		return r, e
	}
	return binaryMap(a, b, d, d, func(x, y element) (element, error) { return selectExtrema(x, y, d, maximum, ignoreNaN), nil })
}
func selectExtrema(x, y element, d DType, maximum, ignoreNaN bool) element {
	if nanElement(x, d) {
		if ignoreNaN && !nanElement(y, d) {
			return y
		}
		return x
	}
	if nanElement(y, d) {
		if ignoreNaN {
			return x
		}
		return y
	}
	if (ignoreNaN && equalElement(x, y, d)) || maximum && lessElement(y, x, d) || !maximum && lessElement(x, y, d) {
		return x
	}
	return y
}
func logical(a, b *Array, fn func(bool, bool) bool) (*Array, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return nil, ErrDType
	}
	return binaryMapCore(a, b, 0, Bool, func(x, y element) (element, error) {
		return boolElement(fn(x.truth(a.DType()), y.truth(b.DType()))), nil
	})
}
func LogicalAnd(a, b *Array) (*Array, error) {
	return logical(a, b, func(x, y bool) bool { return x && y })
}
func LogicalOr(a, b *Array) (*Array, error) {
	return logical(a, b, func(x, y bool) bool { return x || y })
}
func LogicalXor(a, b *Array) (*Array, error) {
	return logical(a, b, func(x, y bool) bool { return x != y })
}

// SumOptions controls reduction axes, output dtype and remaining dimensions.
// Nil Axes reduces all; non-nil empty reduces none.
type SumOptions struct {
	Axes     []int
	KeepDims bool
	DType    DType
}
type reductionLayout struct {
	shape, outputStrides, reduceShape, reduceStrides []int
	reduceAxes                                       []int
	count                                            int
	pairwise                                         bool
}

func reduction(a *Array, options []SumOptions) (reductionLayout, SumOptions, error) {
	if a.DType().IsSpecial() {
		return reductionLayout{}, SumOptions{}, ErrDType
	}
	return reductionAny(a, options)
}
func reductionAny(a *Array, options []SumOptions) (reductionLayout, SumOptions, error) {
	var r reductionLayout
	var opt SumOptions
	if a.DType() == 0 {
		return r, opt, ErrDType
	}
	if len(options) > 1 {
		return r, opt, ErrAxis
	}
	if len(options) == 1 {
		opt = options[0]
	}
	reduced := make([]bool, a.NDim())
	if opt.Axes == nil {
		for i := range reduced {
			reduced[i] = true
		}
	} else {
		for _, axis := range opt.Axes {
			axis, err := NormalizeAxis(axis, a.NDim())
			if err != nil || reduced[axis] {
				return r, opt, ErrAxis
			}
			reduced[axis] = true
		}
	}
	var reduceAxes []int
	for axis, dim := range a.shape {
		if reduced[axis] {
			reduceAxes = append(reduceAxes, axis)
			if opt.KeepDims {
				r.shape = append(r.shape, 1)
				r.outputStrides = append(r.outputStrides, 0)
			}
		} else {
			r.shape = append(r.shape, dim)
			r.outputStrides = append(r.outputStrides, a.strides[axis])
		}
	}
	absStride := func(axis int) int { return max(a.strides[axis], -a.strides[axis]) }
	sort.SliceStable(reduceAxes, func(i, j int) bool { return absStride(reduceAxes[i]) > absStride(reduceAxes[j]) })
	r.reduceAxes = reduceAxes
	r.count = 1
	for _, axis := range reduceAxes {
		r.reduceShape = append(r.reduceShape, a.shape[axis])
		r.reduceStrides = append(r.reduceStrides, a.strides[axis])
		r.count *= a.shape[axis]
	}
	r.pairwise = len(reduceAxes) == a.NDim()
	if len(reduceAxes) > 0 {
		fast := absStride(reduceAxes[len(reduceAxes)-1])
		r.pairwise = true
		for axis, dim := range a.shape {
			if !reduced[axis] && dim > 1 && absStride(axis) < fast {
				r.pairwise = false
				break
			}
		}
	}
	return r, opt, nil
}
func (r reductionLayout) base(a *Array, i int) int {
	return a.offset + flatOffset(i, r.shape, r.outputStrides)
}
func (r reductionLayout) offset(base, j int) int {
	return base + flatOffset(j, r.reduceShape, r.reduceStrides)
}

// companion aligns a broadcast mask or weights array with the exact reduction
// order selected for the data, even when their strides differ.
func (r reductionLayout) companion(a *Array, keepDims bool) reductionLayout {
	out := r
	out.outputStrides = nil
	out.reduceStrides = nil
	reduced := make([]bool, a.NDim())
	for _, axis := range r.reduceAxes {
		reduced[axis] = true
		out.reduceStrides = append(out.reduceStrides, a.strides[axis])
	}
	for axis, s := range a.strides {
		if reduced[axis] {
			if keepDims {
				out.outputStrides = append(out.outputStrides, 0)
			}
		} else {
			out.outputStrides = append(out.outputStrides, s)
		}
	}
	return out
}
func sumDType(d DType) DType {
	if d == Bool || d.IsSigned() {
		return Int64
	}
	if d.IsUnsigned() {
		return Uint64
	}
	return d
}
func Sum(a *Array, options ...SumOptions) (*Array, error) {
	if a.DType().IsSpecial() {
		return specialReduce(a, options, "sum", false)
	}
	r, opt, err := reduction(a, options)
	if err != nil {
		return nil, err
	}
	d := opt.DType
	if d == 0 {
		d = sumDType(a.DType())
	}
	out, err := Zeros(d, r.shape...)
	if err != nil {
		return nil, err
	}
	if nativeRealReduction(a, out, r, false) {
		return out, nil
	}
	if a.size == 0 {
		return out, nil
	}
	for i := 0; i < out.size; i++ {
		base := r.base(a, i)
		at := func(j int) element { return castElement(a.elementAt(r.offset(base, j)), a.DType(), d) }
		out.setElement(i, sumGroup(r.count, at, d, r.pairwise))
	}
	return out, nil
}

func sumGroup(n int, at func(int) element, d DType, pairwise bool) element {
	var z element
	switch {
	case d.IsExtended():
		for j := 0; j < n; j++ {
			z = addScalar(z, at(j), d)
		}
	case d == Bool:
		for j := 0; j < n; j++ {
			z.i |= at(j).i
		}
	case d.IsInteger():
		for j := 0; j < n; j++ {
			z.i += at(j).i
		}
	case d == Float16 || d == Float32:
		z.z = complex(float64(sumValues(n, func(j int) float32 { return float32(real(at(j).z)) }, pairwise)), 0)
	case d == Float64:
		z.z = complex(sumValues(n, func(j int) float64 { return real(at(j).z) }, pairwise), 0)
	case d == Complex64:
		z.z = complex128(sumValues(n, func(j int) complex64 { return complex64(at(j).z) }, pairwise))
	case d == Complex128:
		z.z = sumValues(n, func(j int) complex128 { return at(j).z }, pairwise)
	}
	return roundElement(z, d)
}

func flatOffset(flat int, shape, strides []int) int {
	offset := 0
	for i := len(shape) - 1; i >= 0; i-- {
		offset += (flat % shape[i]) * strides[i]
		flat /= shape[i]
	}
	return offset
}

type numeric interface {
	float32 | float64 | complex64 | complex128
}

func sumValues[T numeric](n int, at func(int) T, pairwise bool) T {
	var total T
	if pairwise {
		return total + pairwiseSum(0, n, at)
	}
	for i := 0; i < n; i++ {
		total += at(i)
	}
	return total
}

// The eight accumulators and bounded pairwise blocks reduce roundoff from long
// sequential sums and mirror NumPy's real fast-axis reduction strategy.
func pairwiseSum[T numeric](start, n int, at func(int) T) T {
	if n < 8 {
		var total T
		for i := 0; i < n; i++ {
			total += at(start + i)
		}
		return total
	}
	if n > 128 {
		middle := n / 2
		middle -= middle % 8
		return pairwiseSum(start, middle, at) + pairwiseSum(start+middle, n-middle, at)
	}
	var partial [8]T
	for i := range partial {
		partial[i] = at(start + i)
	}
	i := 8
	for ; i+7 < n; i += 8 {
		for j := range partial {
			partial[j] += at(start + i + j)
		}
	}
	total := ((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]))
	for ; i < n; i++ {
		total += at(start + i)
	}
	return total
}
