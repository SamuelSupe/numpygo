package dtypes

import (
	"encoding/binary"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/records"
	"slices"
)

func Promote(a, b Descriptor) (Descriptor, error) {
	if a.sub != nil || b.sub != nil {
		if a.DType() == np.Object || b.DType() == np.Object {
			return New("ObjectDType", Options{})
		}
		if a.sub == nil || b.sub == nil || !slices.Equal(*a.shape, *b.shape) {
			return Descriptor{}, np.ErrDType
		}
		base, e := Promote(*a.sub, *b.sub)
		if e != nil {
			return Descriptor{}, e
		}
		return Subarray(base, *a.shape...)
	}
	t, e := np.PromoteTypes(a.DType(), b.DType())
	if e != nil {
		return Descriptor{}, e
	}
	opt := Options{}
	switch t {
	case np.RecordDType:
		s, e := np.PromoteRecordSchemas(a.schema, b.schema)
		if e != nil {
			return Descriptor{}, e
		}
		opt.Schema = s.(*records.Schema)
	case np.VoidDType:
		if a.flex.Length != b.flex.Length {
			return Descriptor{}, np.ErrDType
		}
		opt.Width = a.flex.Length
	case np.BytesDType, np.UnicodeDType:
		for _, x := range []Descriptor{a, b} {
			width := x.flex.Length
			if !x.DType().IsCharacter() {
				width, e = np.FlexibleStringWidth(x.DType())
				if e != nil {
					return Descriptor{}, e
				}
			}
			opt.Width = max(opt.Width, width)
		}
	case np.DateTime64, np.TimeDelta64:
		if a.DType().IsTemporal() && b.DType().IsTemporal() {
			u, e := np.PromoteDateUnits(a.unit, b.unit, t == np.DateTime64)
			if e != nil {
				return Descriptor{}, e
			}
			opt.Unit = u.String()
		} else if a.DType().IsTemporal() {
			opt.Unit = a.unit.String()
		} else {
			opt.Unit = b.unit.String()
		}
	case np.StringDType:
		x, e := a.Zeros(0)
		if e != nil {
			return Descriptor{}, e
		}
		y, e := b.Zeros(0)
		if e != nil {
			return Descriptor{}, e
		}
		opt.String, e = np.PromoteStringDescriptors(x, y)
		if e != nil {
			return Descriptor{}, e
		}
	}
	class := classFor(t)
	// The two C long long types share Go int64 storage but have
	// higher NumPy type numbers, which promotion retains.
	for _, x := range []Descriptor{a, b} {
		if x.DType() == t && (x.def.num == 9 || x.def.num == 10) {
			class = x.def.name
		}
	}
	return New(class, opt)
}

func equivalent(a, b Descriptor, ignoreOrder bool) bool {
	if a.sub != nil || b.sub != nil {
		return a.sub != nil && b.sub != nil && slices.Equal(a.Shape(), b.Shape()) && equivalent(*a.sub, *b.sub, ignoreOrder)
	}
	t := a.DType()
	if t == 0 || t != b.DType() {
		return false
	}
	if t == np.StringDType && a.str.HasNA && !b.str.HasNA {
		return false
	}
	if t == np.RecordDType {
		if a.schema.ItemSize() != b.schema.ItemSize() {
			return false
		}
		af, bf := a.schema.Fields(), b.schema.Fields()
		if len(af) != len(bf) {
			return false
		}
		for i, x := range af {
			y := bf[i]
			if x.Name != y.Name || x.Title != y.Title || x.Offset != y.Offset || !slices.Equal(x.Type.Shape, y.Type.Shape) {
				return false
			}
			p, e := fieldDescriptor(x.Type)
			if e != nil {
				return false
			}
			q, e := fieldDescriptor(y.Type)
			if e != nil || !equivalent(p, q, ignoreOrder) {
				return false
			}
		}
		return true
	}
	if t.IsTemporal() && a.unit != b.unit {
		return false
	}
	if t.IsFlexible() && a.flex.Length != b.flex.Length && b.flex.Length != 0 {
		return false
	}
	if !ignoreOrder && a.Metadata().ByteOrder != '|' && b.Metadata().ByteOrder != '|' && a.order != b.order {
		return false
	}
	// NumPy reports no/equiv for T regardless of the sentinel descriptor.
	return true
}

// CanCast compares complete descriptors, including capacity, units and
// byte order. As in NumPy, a permitted unsafe cast need not succeed for
// every concrete element value.
func CanCast(from, to Descriptor, rule string) (bool, error) {
	a, b := from.DType(), to.DType()
	if a == 0 || b == 0 {
		return false, np.ErrDType
	}
	switch rule {
	case "", "safe":
		rule = "safe"
	case "same_kind", "unsafe", "no", "equiv":
	default:
		return false, np.ErrValue
	}
	if from.sub != nil || to.sub != nil {
		if rule == "unsafe" {
			return true, nil
		}
		if from.sub != nil && to.sub != nil {
			if !slices.Equal(*from.shape, *to.shape) {
				return false, nil
			}
			return CanCast(*from.sub, *to.sub, rule)
		}
		if rule == "no" || rule == "equiv" {
			return false, nil
		}
		if from.sub != nil {
			// A subarray scalar is a void element, not its base scalar.
			return b == np.Object || b == np.VoidDType && to.flex.Length == 0, nil
		}
		// A non-subarray scalar is replicated into every subarray element.
		if a == np.VoidDType && to.sub.DType() == np.VoidDType {
			return false, nil
		}
		return CanCast(from, *to.sub, rule)
	}
	if a == np.RecordDType && b == np.VoidDType && to.flex.Length == 0 {
		// V0 is an unspecified void descriptor, so a structured
		// input retains its original schema.
		return true, nil
	}
	switch rule {
	case "no":
		return equivalent(from, to, false), nil
	case "equiv":
		return equivalent(from, to, true), nil
	case "unsafe":
		if a == np.RecordDType && b == np.RecordDType {
			return len(from.schema.Fields()) == len(to.schema.Fields()), nil
		}
		if a == np.RecordDType && b != np.Object {
			return len(from.schema.Fields()) == 1, nil
		}
		return true, nil
	}
	if b == np.Object {
		return true, nil
	}
	if a == np.Object {
		return false, nil
	}
	if a == np.RecordDType || b == np.RecordDType {
		if a == np.RecordDType && b == np.RecordDType {
			af, bf := from.schema.Fields(), to.schema.Fields()
			if len(af) != len(bf) {
				return false, nil
			}
			for i, x := range af {
				y := bf[i]
				p, e := fieldDescriptor(x.Type)
				if e != nil {
					return false, e
				}
				q, e := fieldDescriptor(y.Type)
				if e != nil {
					return false, e
				}
				ok, e := CanCast(p, q, rule)
				if e != nil || !ok {
					return ok, e
				}
			}
			return true, nil
		}
		return false, nil
	}
	if a == np.StringDType && b == np.StringDType && from.str.HasNA && !to.str.HasNA {
		return false, nil
	}
	if a == np.StringDType || b == np.StringDType {
		return np.CanCast(a, b, rule)
	}
	if b == np.VoidDType {
		if a == np.VoidDType && rule == "same_kind" {
			return true, nil
		}
		return to.flex.Length == 0 || to.ItemSize() >= from.ItemSize(), nil
	}
	if a == np.VoidDType {
		return false, nil
	}
	if b.IsCharacter() {
		if a == np.UnicodeDType && b == np.BytesDType {
			return false, nil
		}
		n := from.flex.Length
		if !a.IsCharacter() {
			var e error
			n, e = np.FlexibleStringWidth(a)
			if e != nil {
				return false, nil
			}
		}
		return rule == "same_kind" || to.flex.Length == 0 || to.flex.Length >= n, nil
	}
	if a.IsCharacter() {
		return false, nil
	}
	if a.IsTemporal() && a == b {
		return np.CanCastDateUnits(from.unit, to.unit, a, rule)
	}
	return np.CanCast(a, b, rule)
}

// Cast materializes a descriptor-selected ndarray. It uses the existing
// array kernels and applies an explicit casting rule before conversion.
// Pointer-managed slots are never exported as unstructured raw bytes.
func (d Descriptor) Cast(a *np.Array, rule string) (*np.Array, error) {
	src, e := FromArray(a)
	if e != nil {
		return nil, e
	}
	if rule == "" {
		rule = "unsafe"
	}
	ok, e := CanCast(src, d, rule)
	if e != nil {
		return nil, e
	}
	if !ok {
		return nil, np.ErrDType
	}
	if d.sub != nil {
		// Array allocation expands the subarray axes. Casting converts
		// the original scalar once, then broadcasts it across those axes.
		if d.ItemSize() == 0 {
			v, e := New("VoidDType", Options{Width: a.ItemSize()})
			if e != nil {
				return nil, e
			}
			return v.Cast(a, "unsafe")
		}
		leaf := d
		var trailing []int
		for leaf.sub != nil {
			trailing = append(trailing, leaf.Shape()...)
			leaf = *leaf.sub
		}
		base, e := leaf.Cast(a, rule)
		if e != nil {
			return nil, e
		}
		shape := base.Shape()
		for range trailing {
			shape = append(shape, 1)
		}
		base, e = base.Reshape(shape...)
		if e != nil {
			return nil, e
		}
		out, e := d.Zeros(a.Shape()...)
		if e != nil {
			return nil, e
		}
		if e = out.Assign(nil, base); e != nil {
			return nil, e
		}
		return out, nil
	}
	t := d.DType()
	if src.DType() == np.RecordDType && t == np.VoidDType && d.flex.Length == 0 {
		return a.Copy()
	}
	if src.DType() == np.RecordDType && t != np.RecordDType && t != np.Object {
		fields := src.schema.Fields()
		if len(fields) != 1 {
			return nil, np.ErrDType
		}
		v, e := a.Field(fields[0].Name)
		if e != nil {
			return nil, e
		}
		if len(fields[0].Type.Shape) > 0 {
			indices := make([]np.Index, a.NDim())
			for i := range indices {
				indices[i] = np.Slice{}
			}
			for range fields[0].Type.Shape {
				indices = append(indices, np.AtIndex(0))
			}
			v, e = v.Slice(indices...)
			if e != nil {
				return nil, e
			}
		}
		return d.Cast(v, "unsafe")
	}
	switch {
	case t == np.RecordDType:
		return a.CastRecords(d.schema)
	case t == np.StringDType:
		return a.CastStringDescriptor(d.str)
	case t == np.VoidDType:
		w := d.flex.Length
		if w == 0 {
			w = a.ItemSize()
		}
		if w != 0 && a.Size() > int(^uint(0)>>1)/w {
			return nil, np.ErrShape
		}
		// NumPy's astype to void uses the native scalar representation,
		// not the original buffer's non-native byte order.
		if !a.NativeByteOrder() {
			if a.DType().IsFlexible() {
				f, e := a.FlexibleType()
				if e != nil {
					return nil, e
				}
				f.Order = binary.NativeEndian
				a, e = a.CastFlexible(f)
				if e != nil {
					return nil, e
				}
			} else {
				a, e = a.Astype(a.DType())
				if e != nil {
					return nil, e
				}
			}
		}
		raw, e := a.ToBytes("C")
		if e != nil {
			return nil, e
		}
		out := make([]byte, a.Size()*w)
		for i := 0; i < a.Size(); i++ {
			copy(out[i*w:(i+1)*w], raw[i*a.ItemSize():(i+1)*a.ItemSize()])
		}
		return np.FromFlexibleBuffer(out, np.FixedVoid(w), a.Shape(), false)
	case t.IsCharacter():
		f := d.flex
		if f.Length == 0 {
			v, e := a.Astype(t)
			if e != nil {
				return nil, e
			}
			f, e = v.FlexibleType()
			if e != nil {
				return nil, e
			}
			f.Order = d.order
		}
		return a.CastFlexible(f)
	default:
		var v *np.Array
		if t.IsTemporal() {
			v, e = a.CastTemporal(t, d.unit.String())
		} else {
			v, e = a.Astype(t)
		}
		if e != nil {
			return nil, e
		}
		if d.Metadata().ByteOrder == '=' || d.Metadata().ByteOrder == '|' {
			return v, nil
		}
		out, e := d.Zeros(a.Shape()...)
		if e != nil {
			return nil, e
		}
		if e = out.Assign(nil, v); e != nil {
			return nil, e
		}
		return out, nil
	}
}

// WithByteOrder returns a descriptor with explicit byte order. Structured
// fields retain their own layout; use their schema to change field order.
func (d Descriptor) WithByteOrder(order binary.ByteOrder) Descriptor {
	d.order = normalizeOrder(order)
	d.byteMarker = 0
	if d.DType() == np.UnicodeDType {
		d.flex.Order = d.order
	}
	return d
}
