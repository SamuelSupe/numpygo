package records

import "github.com/SamuelSupe/numpygo"

type fieldStorage struct {
	data   []byte
	slots  *np.RecordSlots
	offset int
}

func (b fieldStorage) sub(off, width int) fieldStorage {
	return fieldStorage{b.data[off : off+width], b.slots, b.offset + off}
}
func (r *Record) storage() fieldStorage {
	return fieldStorage{r.data, r.slots, r.start}
}
func copyRecordStorage(dst, src *Record) {
	copy(dst.data, src.data)
	for _, off := range dst.schema.objectOffsets {
		v, _ := src.slots.Get(src.start + off)
		_ = dst.slots.Set(dst.start+off, v)
	}
}

func readStoredValue(t Type, b fieldStorage, readonly bool) (any, error) {
	if !t.HasObject() {
		return readValue(t, b.data, readonly)
	}
	if t.Kind == ObjectKind {
		if len(t.Shape) == 0 {
			v, ok := b.slots.Get(b.offset)
			if !ok {
				return nil, np.ErrDType
			}
			return v, nil
		}
		return np.RecordObjectView(b.slots, b.offset, t.Shape, readonly)
	}
	if len(t.Shape) == 0 {
		return &Record{schema: t.Schema, data: b.data, readOnly: readonly, slots: b.slots, start: b.offset}, nil
	}
	n, e := np.ElementCount(np.Bool, t.Shape...)
	if e != nil {
		return nil, e
	}
	w := t.Schema.ItemSize()
	out := make([]any, n)
	for i := range out {
		out[i], e = readStoredValue(t.scalar(), b.sub(i*w, w), readonly)
		if e != nil {
			return nil, e
		}
	}
	return out, nil
}

func writeStoredValue(t Type, b fieldStorage, v any) error {
	if !t.HasObject() {
		return writeValue(t, b.data, v)
	}
	if t.Kind == ObjectKind {
		if len(t.Shape) == 0 {
			return b.slots.Set(b.offset, v)
		}
		a, e := np.RecordObjectView(b.slots, b.offset, t.Shape, false)
		if e != nil {
			return e
		}
		if x, ok := v.(*np.Array); ok {
			return a.Assign(nil, x)
		}
		values, e := unpackValue(v, a.Size())
		if e != nil {
			return e
		}
		x, e := np.FromObjects(values, t.Shape...)
		if e != nil {
			return e
		}
		return a.Assign(nil, x)
	}
	if len(t.Shape) == 0 {
		r, e := t.Schema.encodeRecord(v)
		if e != nil {
			return e
		}
		copyRecordStorage(&Record{schema: t.Schema, data: b.data, slots: b.slots, start: b.offset}, r)
		return nil
	}
	n, e := np.ElementCount(np.Bool, t.Shape...)
	if e != nil {
		return e
	}
	var values []any
	if a, ok := v.(*np.Array); ok {
		a, e = a.BroadcastTo(t.Shape...)
		if e != nil {
			return e
		}
		it, e := np.NDIndex(t.Shape...)
		if e != nil {
			return e
		}
		for ix := range it {
			x, e := a.AtValue(ix...)
			if e != nil {
				return e
			}
			values = append(values, x)
		}
	} else {
		values, e = unpackValue(v, n)
		if e != nil {
			return e
		}
	}
	// Prepare all elements before touching the destination.
	rows := make([]*Record, n)
	for i, x := range values {
		rows[i], e = t.Schema.encodeRecord(x)
		if e != nil {
			return e
		}
	}
	w := t.Schema.ItemSize()
	for i, r := range rows {
		q := b.sub(i*w, w)
		copyRecordStorage(&Record{schema: t.Schema, data: q.data, slots: q.slots, start: q.offset}, r)
	}
	return nil
}
