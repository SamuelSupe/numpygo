package np

// RecordSlots contains GC-owned object references indexed by their logical
// byte positions in a structured allocation. The shadow bytes are never
// used to hold Go or foreign pointers.
type RecordSlots struct {
	data   []byte
	values map[int]any
}

type managedRecordSchema interface {
	StructuredSchema
	HasObject() bool
	ObjectOffsets() []int
	DecodeManagedScalar([]byte, *RecordSlots, int, bool) any
	EncodeManagedScalar(any) ([]byte, *RecordSlots, error)
}

func schemaHasObjects(schema StructuredSchema) bool {
	s, ok := schema.(managedRecordSchema)
	return ok && s.HasObject()
}

// NewRecordSlots initializes a single structured record's object slots.
// The offsets are byte positions, and must address complete object slots.
func NewRecordSlots(data []byte, offsets []int) (*RecordSlots, error) {
	return newRecordSlots(data, offsets, 1, len(data))
}

func newRecordSlots(data []byte, offsets []int, count, width int) (*RecordSlots, error) {
	s := &RecordSlots{data: data, values: make(map[int]any)}
	for _, off := range offsets {
		if off < 0 || off > width-Object.ItemSize() {
			return nil, ErrShape
		}
	}
	if len(offsets) == 0 {
		return s, nil
	}
	for i := 0; i < count; i++ {
		for _, off := range offsets {
			s.values[i*width+off] = int64(0)
		}
	}
	return s, nil
}

func (s *RecordSlots) Get(offset int) (any, bool) {
	if s == nil {
		return nil, false
	}
	v, ok := s.values[offset]
	return v, ok
}
func (s *RecordSlots) Set(offset int, v any) error {
	if _, ok := s.Get(offset); !ok {
		return ErrIndex
	}
	s.values[offset] = v
	return nil
}

// RecordObjectView exposes shaped object fields of a structured scalar.
// Only previously registered slots can be addressed.
func RecordObjectView(s *RecordSlots, offset int, shape []int, readonly bool) (*Array, error) {
	if s == nil {
		return nil, ErrDType
	}
	n, e := shapeSize(Object, shape)
	if e != nil {
		return nil, e
	}
	strides := contiguousStrides(shape, n)
	for i := range strides {
		strides[i] *= Object.ItemSize()
	}
	if e = boundedLayout(len(s.data), offset, Object.ItemSize(), n, shape, strides); e != nil {
		return nil, e
	}
	v := &Array{data: &storage{dtype: Object, buffer: s.data, slots: s, byteIndexed: true}, shape: append([]int(nil), shape...), strides: strides, offset: offset, size: n, readOnly: readonly}
	if !validObjectSlots(v) {
		return nil, ErrShape
	}
	return v, nil
}

func validObjectSlots(a *Array) bool {
	ok := true
	a.walk(func(_, off int) {
		if _, found := a.data.slots.Get(off * a.strideUnit()); !found {
			ok = false
		}
	})
	return ok
}

type managedRecordValue struct {
	bytes  []byte
	slots  *RecordSlots
	offset int
	schema managedRecordSchema
}

func (a *Array) managedRecord(off int) managedRecordValue {
	return managedRecordValue{a.recordBytes(off), a.data.slots, off * a.strideUnit(), a.data.schema.(managedRecordSchema)}
}

func (a *Array) storeManagedRecord(off int, e element) {
	v := e.extra.(managedRecordValue)
	dst := a.managedRecord(off)
	copy(dst.bytes, v.bytes)
	for _, rel := range dst.schema.ObjectOffsets() {
		x, _ := v.slots.Get(v.offset + rel)
		_ = dst.slots.Set(dst.offset+rel, x)
	}
}

func decodeRecordElement(schema StructuredSchema, e element) any {
	if v, ok := e.extra.(managedRecordValue); ok {
		return v.schema.DecodeManagedScalar(v.bytes, v.slots, v.offset, true)
	}
	return schema.DecodeScalar(e.extra.([]byte), true)
}

// HasObject reports whether a dtype owns Go-managed object or variable
// length reference slots, including those nested in structured fields.
func (a *Array) HasObject() bool {
	d := a.DType()
	return d == Object || d == StringDType || d == RecordDType && schemaHasObjects(a.data.schema)
}

func (a *Array) rawReferenceStorage() bool {
	return a.HasObject() || a.DType() == RecordDType && a.data.slots != nil
}
