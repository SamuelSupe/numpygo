package np

import (
	"encoding/binary"
	"reflect"
	"slices"
)

// boundedLayout checks the full byte/slot interval without multiplying an
// untrusted stride until it is known to fit in the backing allocation.
func boundedLayout(length, offset, width, size int, shape, strides []int) error {
	if width < 0 || offset < 0 || offset > length || len(shape) != len(strides) {
		return ErrShape
	}
	if size == 0 {
		return nil
	}
	before, after := offset, length-offset-width
	if after < 0 {
		return ErrShape
	}
	for i, dim := range shape {
		if dim < 2 {
			continue
		}
		s, extent := strides[i], dim-1
		if s >= 0 {
			if s > after/extent {
				return ErrShape
			}
			after -= s * extent
		} else {
			mag := uint64(-(s + 1)) + 1
			if mag > uint64(before/extent) {
				return ErrShape
			}
			before -= int(mag) * extent
		}
	}
	return nil
}

// AsStrided creates a shared view with byte strides, including overlap, zero
// and negative strides. Unlike NumPy's unchecked as_strided, every addressed
// element must fit inside the underlying allocation. Pointer-bearing slots
// may only be addressed at element boundaries.
func AsStrided(a *Array, shape, strides []int, writable bool) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if r := a.mappingRegion(); r != nil && r.closed.Load() {
		return nil, ErrClosedMapping
	}
	if shape == nil {
		shape = a.Shape()
	}
	if strides == nil {
		strides = a.Strides()
	}
	n, e := a.shapeSize(shape)
	if e != nil {
		return nil, e
	}
	width := a.ItemSize()
	if a.DType() == Object && a.data.slots != nil {
		if e = boundedLayout(len(a.data.buffer), a.offset, width, n, shape, strides); e != nil {
			return nil, e
		}
		v := &Array{data: a.data, shape: slices.Clone(shape), strides: slices.Clone(strides), offset: a.offset, size: n, readOnly: a.readOnly || !writable, base: a.viewBase()}
		if !validObjectSlots(v) {
			return nil, ErrShape
		}
		return v, nil
	}
	if a.DType() == Object || a.DType() == StringDType || width == 0 {
		if len(shape) != len(strides) {
			return nil, ErrShape
		}
		step := max(width, 1)
		ss := slices.Clone(strides)
		for i, s := range ss {
			if width == 0 && s != 0 || s%step != 0 {
				return nil, ErrShape
			}
			ss[i] = s / step
		}
		length := 0
		if width != 0 {
			length = reflect.ValueOf(storageSlice(a.data)).Len()
		} else {
			// Zero-width items do not address any bytes.
			return &Array{data: a.data, shape: slices.Clone(shape), strides: ss, size: n, readOnly: a.readOnly || !writable, base: a.viewBase()}, nil
		}
		if e = boundedLayout(length, a.offset, 1, n, shape, ss); e != nil {
			return nil, e
		}
		return &Array{data: a.data, shape: slices.Clone(shape), strides: ss, offset: a.offset, size: n, readOnly: a.readOnly || !writable, base: a.viewBase()}, nil
	}
	f := StructuredField{DType: a.DType(), Schema: a.data.schema, Flexible: a.data.flex, Unit: a.data.unit, Order: storageOrder(a.data)}
	v, e := a.ViewLayout(f, shape, strides)
	if e != nil {
		return nil, e
	}
	v.readOnly = v.readOnly || !writable
	return v, nil
}

// ViewLayout reinterprets fixed-width storage at a byte offset relative to
// the source's first logical element. All addressed items must fit the backing
// allocation; the view retains its source's lifetime and write protection.
// This is used by record conversion routines, not for pointer-managed dtypes.
func (a *Array) ViewLayout(f StructuredField, shape, strides []int) (*Array, error) {
	if a.DType() == 0 || a.DType() == Object || a.DType() == StringDType || f.DType == Object || f.DType == StringDType {
		return nil, ErrDType
	}
	if a.rawReferenceStorage() && f.DType != RecordDType || schemaHasObjects(f.Schema) && a.data.slots == nil {
		return nil, ErrDType
	}
	if r := a.mappingRegion(); r != nil && r.closed.Load() {
		return nil, ErrClosedMapping
	}
	width := f.DType.ItemSize()
	switch {
	case f.DType == RecordDType:
		if f.Schema == nil || f.Schema.ItemSize() < 0 {
			return nil, ErrDType
		}
		width = f.Schema.ItemSize()
	case f.DType.IsFlexible():
		var e error
		width, e = f.Flexible.width()
		if e != nil || f.Flexible.Kind != f.DType {
			return nil, ErrDType
		}
	case f.DType.IsTemporal():
		if u, e := ParseDateUnit(f.Unit.String()); e != nil || u != f.Unit {
			return nil, ErrDType
		}
	case width == 0:
		return nil, ErrDType
	}
	o := f.Order
	if o == nil {
		o = binary.NativeEndian
	}
	if f.DType.IsExtended() && o.Uint16([]byte{1, 0}) != 1 {
		return nil, ErrDType
	}
	n, e := shapeSizeWidth(max(width, 1), shape)
	if e != nil {
		return nil, e
	}
	raw := rawStorageBytes(a.data)
	offset := a.offset * a.strideUnit()
	if n != 0 {
		if f.Offset < -offset || f.Offset > len(raw)-offset {
			return nil, ErrShape
		}
		offset += f.Offset
	} else {
		offset = 0
	}
	if e = boundedLayout(len(raw), offset, width, n, shape, strides); e != nil {
		return nil, e
	}
	v := &Array{data: &storage{dtype: f.DType, schema: f.Schema, flex: f.Flexible, unit: f.Unit, buffer: raw, order: o, byteIndexed: true},
		shape: slices.Clone(shape), strides: slices.Clone(strides), offset: offset, size: n, readOnly: a.readOnly, base: a.viewBase()}
	if f.DType == RecordDType && a.data.slots != nil {
		v.data.slots = a.data.slots
		if schemaHasObjects(f.Schema) {
			offsets := f.Schema.(managedRecordSchema).ObjectOffsets()
			valid := true
			v.walk(func(_, off int) {
				for _, rel := range offsets {
					if _, ok := v.data.slots.Get(off + rel); !ok {
						valid = false
					}
				}
			})
			if !valid {
				return nil, ErrShape
			}
		}
	}
	return v, nil
}

// SlidingWindowView appends one dimension per window, in axes order. A nil
// axes selects every input dimension; duplicate axes reduce it repeatedly.
// The default NumPy view is read-only (pass writable=false).
func SlidingWindowView(a *Array, window, axes []int, writable bool) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if axes == nil {
		axes = make([]int, a.NDim())
		for i := range axes {
			axes[i] = i
		}
	}
	if len(window) != len(axes) {
		return nil, ErrShape
	}
	shape, strides := a.Shape(), a.Strides()
	extra := make([]int, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax += a.NDim()
		}
		if ax < 0 || ax >= a.NDim() {
			return nil, ErrAxis
		}
		w := window[i]
		if w < 0 || w > shape[ax] || w == 0 && shape[ax] == maxInt {
			return nil, ErrValue
		}
		shape[ax] -= w - 1
		extra[i] = strides[ax]
	}
	return AsStrided(a, append(shape, window...), append(strides, extra...), writable)
}
