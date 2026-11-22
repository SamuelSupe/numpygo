package np

import "encoding/binary"

type VoidFlags struct {
	CContiguous, FContiguous, OwnData, Writable, Aligned, Writeback bool
}

// The standalone void scalar owns its byte payload. Its setfield method
// validates against a temporary array, as NumPy does for standalone scalars.
func (v VoidScalar) DType() FlexibleDType { return FixedVoid(len(v)) }
func (v VoidScalar) Base() *Array         { return nil }
func (v VoidScalar) Flags() VoidFlags     { return VoidFlags{true, true, true, true, true, false} }
func (v VoidScalar) GetField(dtype DType, offset int, order binary.ByteOrder) (any, error) {
	if !dtype.IsNativeNumeric() && !dtype.IsExtended() {
		return nil, ErrDType
	}
	a, e := FromBuffer(v, dtype, 1, offset, order)
	if e != nil {
		return nil, e
	}
	return a.AtValue(0)
}
func (v VoidScalar) SetField(value any, dtype DType, offset int, order binary.ByteOrder) error {
	if !dtype.IsNativeNumeric() && !dtype.IsExtended() {
		return ErrDType
	}
	b := append([]byte(nil), v...)
	a, e := FromBuffer(b, dtype, 1, offset, order)
	if e != nil {
		return e
	}
	return a.SetValue(value, 0)
}
