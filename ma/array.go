// Package ma implements numeric masked arrays over numpygo's shared strided
// arrays. True mask elements are missing; data and masks follow the same shape.
package ma

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

// ErrMA is the base category for masked-array-specific errors. ErrMask also
// matches np.ErrShape for callers already handling shape errors.
var (
	ErrMA   = errors.New("masked array error")
	ErrMask = fmt.Errorf("%w: %w", ErrMA, np.ErrShape)
)

type Array struct {
	data, mask   *np.Array
	fill         *any
	explicitMask bool
	hard         bool
	shared       bool
	customFill   bool
}
type Options struct {
	Copy      bool
	FillValue any
	HardMask  bool
}

func New(data, mask *np.Array, options ...Options) (*Array, error) {
	if data.DType() == 0 || len(options) > 1 {
		return nil, np.ErrDType
	}
	var opt Options
	if len(options) == 1 {
		opt = options[0]
	}
	var err error
	if opt.Copy {
		data, err = data.Copy()
		if err != nil {
			return nil, err
		}
	}
	explicit := mask != nil
	if mask == nil {
		mask, err = np.Zeros(np.Bool, data.Shape()...)
	} else {
		if mask.DType() == 0 {
			return nil, np.ErrDType
		}
		if !slices.Equal(mask.Shape(), data.Shape()) {
			if mask.Size() == 1 {
				mask, err = mask.Reshape()
				if err == nil {
					mask, err = mask.BroadcastTo(data.Shape()...)
				}
			} else if mask.Size() == data.Size() {
				mask, err = mask.Reshape(data.Shape()...)
			} else {
				return nil, ErrMask
			}
			if err != nil {
				return nil, err
			}
		}
		if mask.DType() != np.Bool || opt.Copy || !mask.Writable() {
			mask, err = mask.Astype(np.Bool)
		}
	}
	if err != nil {
		return nil, err
	}
	fill := DefaultFillValue(data.DType())
	a := &Array{data: data, mask: mask, fill: &fill, explicitMask: explicit, hard: opt.HardMask, shared: explicit && !opt.Copy}
	if opt.FillValue != nil {
		if err = a.SetFillValue(opt.FillValue); err != nil {
			return nil, err
		}
	}
	return a, nil
}
func valid(a *Array) bool        { return a != nil && a.data.DType() != 0 }
func (a *Array) Data() *np.Array { return a.data }
func (a *Array) Mask() *np.Array { return a.mask }
func (a *Array) Shape() []int    { return a.data.Shape() }
func (a *Array) DType() np.DType { return a.data.DType() }
func (a *Array) NDim() int       { return a.data.NDim() }
func (a *Array) Size() int       { return a.data.Size() }
func (a *Array) HardMask() bool  { return a.hard }
func (a *Array) HardenMask()     { a.hard = true }
func (a *Array) SoftenMask()     { a.hard = false }
func (a *Array) FillValue() any  { return *a.fill }
func (a *Array) SetFillValue(value any) error {
	v, err := np.Full(a.DType(), value)
	if err != nil {
		return err
	}
	*a.fill, err = v.AtValue()
	if err == nil {
		a.customFill = true
	}
	return err
}
func GetData(a *Array) *np.Array      { return a.data }
func GetMaskArray(a *Array) *np.Array { return a.mask }
func GetMask(a *Array) *np.Array {
	if !a.explicitMask && !IsMasked(a) {
		return nil
	}
	return a.mask
}
func IsMasked(a *Array) bool {
	if !valid(a) {
		return false
	}
	v, err := np.Any(a.mask)
	if err != nil {
		return false
	}
	x, _ := v.AtValue()
	return x.(bool)
}
func IsMask(a *np.Array) bool { return a.DType() == np.Bool }
func MakeMask(a *np.Array) (*np.Array, error) {
	if a.DType() == 0 {
		return nil, np.ErrDType
	}
	return a.Astype(np.Bool)
}
func MakeMaskNone(shape ...int) (*np.Array, error) { return np.Zeros(np.Bool, shape...) }
func MaskOr(a, b *np.Array) (*np.Array, error) {
	if a == nil {
		if b == nil {
			return nil, nil
		}
		return b.Astype(np.Bool)
	}
	if b == nil {
		return a.Astype(np.Bool)
	}
	return np.LogicalOr(a, b)
}
func DefaultFillValue(d np.DType) any {
	switch {
	case d == np.Bool:
		return true
	case d.IsInteger():
		return int64(999999)
	case d.IsComplex():
		return complex128(1e20)
	case d.IsFloat():
		return float64(1e20)
	default:
		return nil
	}
}

// MinimumFillValue is a large sentinel for computing a minimum.
func MinimumFillValue(d np.DType) any {
	switch d {
	case np.Bool:
		return true
	case np.Int8:
		return int8(math.MaxInt8)
	case np.Int16:
		return int16(math.MaxInt16)
	case np.Int32:
		return int32(math.MaxInt32)
	case np.Int64:
		return int64(math.MaxInt64)
	case np.Uint8:
		return uint8(math.MaxUint8)
	case np.Uint16:
		return uint16(math.MaxUint16)
	case np.Uint32:
		return uint32(math.MaxUint32)
	case np.Uint64:
		return uint64(math.MaxUint64)
	case np.Float16, np.Float32, np.Float64:
		return math.Inf(1)
	case np.Complex64, np.Complex128:
		return complex(math.Inf(1), math.Inf(1))
	}
	return nil
}
func MaximumFillValue(d np.DType) any {
	switch d {
	case np.Bool:
		return false
	case np.Int8:
		return int8(math.MinInt8)
	case np.Int16:
		return int16(math.MinInt16)
	case np.Int32:
		return int32(math.MinInt32)
	case np.Int64:
		return int64(math.MinInt64)
	case np.Uint8:
		return uint8(0)
	case np.Uint16:
		return uint16(0)
	case np.Uint32:
		return uint32(0)
	case np.Uint64:
		return uint64(0)
	case np.Float16, np.Float32, np.Float64:
		return math.Inf(-1)
	case np.Complex64, np.Complex128:
		return complex(math.Inf(-1), math.Inf(-1))
	}
	return nil
}
func (a *Array) Filled(values ...any) (*np.Array, error) {
	if !valid(a) || len(values) > 1 {
		return nil, np.ErrValue
	}
	if !IsMasked(a) {
		return a.data, nil
	}
	fill := a.FillValue()
	if len(values) == 1 {
		fill = values[0]
	}
	f, err := np.Full(a.DType(), fill)
	if err != nil {
		return nil, err
	}
	return np.Where(a.mask, f, a.data)
}
func Filled(a *Array, values ...any) (*np.Array, error) { return a.Filled(values...) }
func Compressed(a *Array) (*np.Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	data, err := np.Ravel(a.data)
	if err != nil {
		return nil, err
	}
	mask, err := np.Ravel(a.mask)
	if err != nil {
		return nil, err
	}
	inverse, err := np.LogicalNot(mask)
	if err != nil {
		return nil, err
	}
	return data.Index(inverse)
}
func (a *Array) Copy() (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	data, err := a.data.Copy()
	if err != nil {
		return nil, err
	}
	mask, err := a.mask.Copy()
	if err != nil {
		return nil, err
	}
	return a.with(data, mask), nil
}
func (a *Array) with(data, mask *np.Array) *Array {
	fill := a.fill
	if data.DType() != a.DType() {
		v := DefaultFillValue(data.DType())
		fill = &v
	}
	return &Array{data: data, mask: mask, fill: fill, explicitMask: a.explicitMask, hard: a.hard,
		shared: a.explicitMask || a.shared, customFill: a.customFill && data.DType() == a.DType()}
}
func (a *Array) Index(indices ...np.Index) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	data, err := a.data.Index(indices...)
	if err != nil {
		return nil, err
	}
	mask, err := a.mask.Index(indices...)
	if err != nil {
		return nil, err
	}
	return a.with(data, mask), nil
}
func (a *Array) Assign(indices []np.Index, values *Array) error {
	if !valid(a) || !valid(values) {
		return np.ErrDType
	}
	if !a.data.Writable() || !a.mask.Writable() {
		return np.ErrReadOnly
	}
	old, err := a.Index(indices...)
	if err != nil {
		return err
	}
	data, err := values.data.BroadcastTo(old.Shape()...)
	if err != nil {
		return err
	}
	mask, err := values.mask.BroadcastTo(old.Shape()...)
	if err != nil {
		return err
	}
	if a.hard {
		mask, err = np.LogicalOr(old.mask, mask)
		if err != nil {
			return err
		}
		// NumPy hard masks protect the payload of both previously masked
		// slots and slots newly masked by the RHS.
		data, err = np.Where(mask, old.data, data)
		if err != nil {
			return err
		}
	}
	data, err = data.Astype(a.DType())
	if err != nil {
		return err
	}
	mask, err = mask.Copy()
	if err != nil {
		return err
	}
	if err = a.data.Assign(indices, data); err != nil {
		return err
	}
	if err = a.mask.Assign(indices, mask); err != nil {
		return err
	}
	if values.explicitMask || a.hard {
		a.explicitMask = true
	}
	return nil
}

func MaskedAll(dtype np.DType, shape ...int) (*Array, error) {
	data, err := np.Zeros(dtype, shape...)
	if err != nil {
		return nil, err
	}
	mask, err := np.Ones(np.Bool, shape...)
	if err != nil {
		return nil, err
	}
	return New(data, mask)
}
func MaskedAllLike(a *Array) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	return MaskedAll(a.DType(), a.Shape()...)
}
func MaskedWhere(condition *np.Array, a *Array) (*Array, error) {
	if !valid(a) || condition.DType() == 0 {
		return nil, np.ErrDType
	}
	c, err := condition.BroadcastTo(a.Shape()...)
	if err != nil {
		return nil, err
	}
	mask, err := np.LogicalOr(a.mask, c)
	if err != nil {
		return nil, err
	}
	data, err := a.data.Copy()
	if err != nil {
		return nil, err
	}
	out := a.with(data, mask)
	out.explicitMask = true
	return out, nil
}
func maskedCompare(a *Array, value *np.Array, cmp func(*np.Array, *np.Array) (*np.Array, error)) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	mask, err := cmp(a.data, value)
	if err != nil {
		return nil, err
	}
	return MaskedWhere(mask, a)
}
func MaskedEqual(a *Array, value *np.Array) (*Array, error) { return maskedCompare(a, value, np.Equal) }
func MaskedNotEqual(a *Array, value *np.Array) (*Array, error) {
	return maskedCompare(a, value, np.NotEqual)
}
func MaskedGreater(a *Array, value *np.Array) (*Array, error) {
	return maskedCompare(a, value, np.Greater)
}
func MaskedGreaterEqual(a *Array, value *np.Array) (*Array, error) {
	return maskedCompare(a, value, np.GreaterEqual)
}
func MaskedLess(a *Array, value *np.Array) (*Array, error) { return maskedCompare(a, value, np.Less) }
func MaskedLessEqual(a *Array, value *np.Array) (*Array, error) {
	return maskedCompare(a, value, np.LessEqual)
}
func MaskedInside(a *Array, lo, hi *np.Array) (*Array, error)  { return maskedRange(a, lo, hi, false) }
func MaskedOutside(a *Array, lo, hi *np.Array) (*Array, error) { return maskedRange(a, lo, hi, true) }
func maskedRange(a *Array, lo, hi *np.Array, outside bool) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	minimum, err := np.Minimum(lo, hi)
	if err != nil {
		return nil, err
	}
	maximum, err := np.Maximum(lo, hi)
	if err != nil {
		return nil, err
	}
	l, err := np.GreaterEqual(a.data, minimum)
	if err != nil {
		return nil, err
	}
	h, err := np.LessEqual(a.data, maximum)
	if err != nil {
		return nil, err
	}
	m, err := np.LogicalAnd(l, h)
	if err != nil {
		return nil, err
	}
	if outside {
		m, err = np.LogicalNot(m)
		if err != nil {
			return nil, err
		}
	}
	return MaskedWhere(m, a)
}
func MaskedValues(a *Array, value *np.Array, rtol, atol float64) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	m, err := np.IsClose(a.data, value, rtol, atol, false)
	if err != nil {
		return nil, err
	}
	out, err := MaskedWhere(m, a)
	if err != nil {
		return nil, err
	}
	if value.Size() == 1 {
		v, err := value.Reshape()
		if err != nil {
			return nil, err
		}
		x, err := v.AtValue()
		if err != nil {
			return nil, err
		}
		if err = out.SetFillValue(x); err != nil {
			return nil, err
		}
	}
	return out, nil
}
func MaskedInvalid(a *Array) (*Array, error) {
	if !valid(a) {
		return nil, np.ErrDType
	}
	finite, err := np.IsFinite(a.data)
	if err != nil {
		return nil, err
	}
	invalid, err := np.LogicalNot(finite)
	if err != nil {
		return nil, err
	}
	return MaskedWhere(invalid, a)
}
func FixInvalid(a *Array, fill any) (*Array, error) {
	out, err := MaskedInvalid(a)
	if err != nil {
		return nil, err
	}
	if fill == nil {
		fill = a.FillValue()
	}
	finite, err := np.IsFinite(a.data)
	if err != nil {
		return nil, err
	}
	invalid, err := np.LogicalNot(finite)
	if err != nil {
		return nil, err
	}
	v, err := np.Full(a.DType(), fill)
	if err != nil {
		return nil, err
	}
	data, err := np.Where(invalid, v, a.data)
	if err != nil {
		return nil, err
	}
	out.data = data
	return out, nil
}
