package records

import "github.com/SamuelSupe/numpygo"

// Structured casts match fields by position, not by name. A shape-changing
// unsafe subarray cast is right-aligned: singleton dimensions broadcast and
// destinations outside a nonsingleton source dimension are zero-filled.
func (s *Schema) castRecord(src *Record) (*Record, error) {
	if len(s.fields) != len(src.schema.fields) {
		return nil, np.ErrDType
	}
	out, e := NewRecord(s)
	if e != nil {
		return nil, e
	}
	for i, d := range s.fields {
		f := src.schema.fields[i]
		dw, e := d.Type.Size()
		if e != nil {
			return nil, e
		}
		sw, e := f.Type.Size()
		if e != nil {
			return nil, e
		}
		if e = castField(d.Type, out.storage().sub(d.Offset, dw), f.Type, src.storage().sub(f.Offset, sw)); e != nil {
			return nil, e
		}
	}
	return out, nil
}

func castField(dst Type, db fieldStorage, src Type, sb fieldStorage) error {
	d, s := dst.scalar(), src.scalar()
	dw, e := d.Size()
	if e != nil {
		return e
	}
	sw, e := s.Size()
	if e != nil {
		return e
	}
	it, e := np.NDIndex(dst.Shape...)
	if e != nil {
		return e
	}
	dflat := 0
	for ix := range it {
		sflat := 0
		valid := true
		for j, n := range src.Shape {
			k := j + len(dst.Shape) - len(src.Shape)
			coord := 0
			if k >= 0 && n != 1 {
				coord = ix[k]
			}
			if coord >= n {
				valid = false
				break
			}
			sflat = sflat*n + coord
		}
		if valid {
			if e = castLeaf(d, db.sub(dflat*dw, dw), s, sb.sub(sflat*sw, sw)); e != nil {
				return e
			}
		}
		dflat++
	}
	return nil
}

func leafArray(t Type, b []byte) (*np.Array, error) {
	if t.Kind == TemporalKind {
		return np.FromTemporalBuffer(b, t.DType, t.Unit.String(), nil, false, t.byteOrder())
	}
	if f, ok := t.flexibleType(); ok {
		return np.FromFlexibleBuffer(b, f, nil, false)
	}
	if t.Kind == NumericKind {
		a, e := np.FromBuffer(b, t.DType, 1, 0, t.byteOrder())
		if e != nil {
			return nil, e
		}
		return a.Reshape()
	}
	return nil, np.ErrDType
}

func castLeaf(dst Type, db fieldStorage, src Type, sb fieldStorage) error {
	if dst.Kind == RecordKind || dst.Kind == ObjectKind || src.Kind == ObjectKind {
		v, e := readStoredValue(src, sb, true)
		if e != nil {
			return e
		}
		return writeStoredValue(dst, db, v)
	}
	if src.Kind == RecordKind {
		if len(src.Schema.fields) != 1 {
			return np.ErrDType
		}
		f := src.Schema.fields[0]
		w, e := f.Type.Size()
		if e != nil {
			return e
		}
		return castField(dst, db, f.Type, sb.sub(f.Offset, w))
	}
	if src.Kind == dst.Kind && src.DType == dst.DType && src.Length == dst.Length && src.Unit == dst.Unit &&
		src.byteOrder().Uint16([]byte{1, 0}) == dst.byteOrder().Uint16([]byte{1, 0}) {
		copy(db.data, sb.data)
		return nil
	}
	a, e := leafArray(src, sb.data)
	if e != nil {
		return e
	}
	b, e := leafArray(dst, db.data)
	if e != nil {
		return e
	}
	return b.Assign(nil, a)
}
