// Package recfunctions implements NumPy's structured-array utilities on
// native ndarray descriptors and typed Go field names.
package recfunctions

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/records"
	"slices"
)

type Name struct {
	Name     string
	Children []Name
}

func GetNames(s *records.Schema) []Name {
	out := make([]Name, 0, len(s.Fields()))
	for _, f := range s.Fields() {
		n := Name{Name: f.Name}
		if f.Type.Kind == records.RecordKind && len(f.Type.Shape) == 0 {
			n.Children = GetNames(f.Type.Schema)
		}
		out = append(out, n)
	}
	return out
}
func GetNamesFlat(s *records.Schema) []string {
	out := []string{}
	for _, f := range s.Fields() {
		out = append(out, f.Name)
		if f.Type.Kind == records.RecordKind && len(f.Type.Shape) == 0 {
			out = append(out, GetNamesFlat(f.Type.Schema)...)
		}
	}
	return out
}

// FlattenDescr retains scalar and subarray types while flattening nested
// (non-subarray) records. Offsets are deliberately discarded.
func FlattenDescr(s *records.Schema) []records.FieldSpec {
	out := []records.FieldSpec{}
	for _, f := range s.Fields() {
		if f.Type.Kind == records.RecordKind && len(f.Type.Shape) == 0 {
			out = append(out, FlattenDescr(f.Type.Schema)...)
		} else {
			out = append(out, records.FieldSpec{Name: f.Name, Type: f.Type})
		}
	}
	return out
}
func GetFieldStructure(s *records.Schema) map[string][]string {
	out := map[string][]string{}
	var visit func(*records.Schema, string)
	visit = func(s *records.Schema, parent string) {
		for _, f := range s.Fields() {
			parents := []string{}
			if parent != "" {
				if f.Type.Kind != records.RecordKind || len(f.Type.Shape) > 0 {
					parents = append(parents, out[parent]...)
				}
				parents = append(parents, parent)
			}
			out[f.Name] = parents
			if f.Type.Kind == records.RecordKind && len(f.Type.Shape) == 0 {
				visit(f.Type.Schema, f.Name)
			}
		}
	}
	visit(s, "")
	return out
}

func schemaOf(a *np.Array) (*records.Schema, error) {
	s, ok := a.StructuredSchema().(*records.Schema)
	if !ok {
		return nil, np.ErrDType
	}
	return s, nil
}

func zeroField(a *np.Array) error {
	if a.DType() != np.RecordDType {
		return a.Fill(0)
	}
	s, e := schemaOf(a)
	if e != nil {
		return e
	}
	for _, f := range s.Fields() {
		v, e := a.Field(f.Name)
		if e != nil {
			return e
		}
		if e = zeroField(v); e != nil {
			return e
		}
	}
	return nil
}

// AssignFieldsByName recursively copies matching fields (including subarray
// fields) and optionally sets every unassigned field to scalar zero.
func AssignFieldsByName(dst, src *np.Array, zeroUnassigned bool) error {
	if !dst.Writable() {
		return np.ErrReadOnly
	}
	if dst.DType() != np.RecordDType {
		return dst.Assign(nil, src)
	}
	ds, e := schemaOf(dst)
	if e != nil {
		return e
	}
	ss, e := schemaOf(src)
	if e != nil {
		return e
	}
	// Field permutations on overlapping views must read the original slots.
	if np.SharesMemory(dst, src) {
		src, e = src.Copy()
		if e != nil {
			return e
		}
	}
	for _, f := range ds.Fields() {
		d, e := dst.Field(f.Name)
		if e != nil {
			return e
		}
		if !slices.Contains(ss.Names(), f.Name) {
			if zeroUnassigned {
				if e = zeroField(d); e != nil {
					return e
				}
			}
			continue
		}
		s, e := src.Field(f.Name)
		if e != nil {
			return e
		}
		if e = AssignFieldsByName(d, s, zeroUnassigned); e != nil {
			return e
		}
	}
	return nil
}

func RequireFields(a *np.Array, required *records.Schema) (*np.Array, error) {
	out, e := np.ZerosRecords(required, a.Shape()...)
	if e != nil {
		return nil, e
	}
	return out, AssignFieldsByName(out, a, true)
}

// RecursiveFillFields copies the source's leading rows into matching
// destination fields. Unmatched fields and remaining rows stay unchanged.
func RecursiveFillFields(src, dst *np.Array) (*np.Array, error) {
	if src.NDim() == 0 || dst.NDim() == 0 || src.Shape()[0] > dst.Shape()[0] {
		return nil, np.ErrShape
	}
	v, e := dst.Slice(np.Slice{Stop: np.Int(src.Shape()[0])})
	if e != nil {
		return nil, e
	}
	return dst, AssignFieldsByName(v, src, false)
}

func dropSchema(s *records.Schema, names []string) (*records.Schema, error) {
	specs := []records.FieldSpec{}
	for _, f := range s.Fields() {
		if slices.Contains(names, f.Name) {
			continue
		}
		t := f.Type
		if t.Kind == records.RecordKind && len(t.Shape) == 0 {
			var e error
			t.Schema, e = dropSchema(t.Schema, names)
			if e != nil {
				return nil, e
			}
			if len(t.Schema.Fields()) == 0 {
				continue
			}
		}
		specs = append(specs, records.FieldSpec{Name: f.Name, Type: t})
	}
	return records.NewSchema(specs, false, 0)
}
func DropFields(a *np.Array, names ...string) (*np.Array, error) {
	s, e := schemaOf(a)
	if e != nil {
		return nil, e
	}
	s, e = dropSchema(s, names)
	if e != nil {
		return nil, e
	}
	return RequireFields(a, s)
}
func RecDropFields(a *np.Array, names ...string) (*records.Array, error) {
	v, e := DropFields(a, names...)
	if e != nil {
		return nil, e
	}
	return records.FromNDArray(v)
}

func renameSchema(s *records.Schema, mapping map[string]string) (*records.Schema, error) {
	specs := []records.FieldSpec{}
	for _, f := range s.Fields() {
		name := f.Name
		if n, ok := mapping[name]; ok {
			name = n
		}
		t := f.Type
		if t.Kind == records.RecordKind && len(t.Shape) == 0 {
			var e error
			t.Schema, e = renameSchema(t.Schema, mapping)
			if e != nil {
				return nil, e
			}
		}
		specs = append(specs, records.FieldSpec{Name: name, Type: t})
	}
	return records.NewSchema(specs, false, 0)
}
func RenameFields(a *np.Array, mapping map[string]string) (*np.Array, error) {
	s, e := schemaOf(a)
	if e != nil {
		return nil, e
	}
	s, e = renameSchema(s, mapping)
	if e != nil {
		return nil, e
	}
	shape, strides := a.Shape(), a.Strides()
	if s.ItemSize() != a.ItemSize() {
		if len(shape) == 0 || s.ItemSize() == 0 {
			return nil, np.ErrShape
		}
		ax := len(shape) - 1
		if (shape[ax] > 1 && strides[ax] != a.ItemSize()) || shape[ax]*a.ItemSize()%s.ItemSize() != 0 {
			return nil, np.ErrShape
		}
		shape[ax] = shape[ax] * a.ItemSize() / s.ItemSize()
		strides[ax] = s.ItemSize()
	}
	return a.ViewLayout(np.StructuredField{DType: np.RecordDType, Schema: s}, shape, strides)
}

func RepackSchema(s *records.Schema, align, recurse bool) (*records.Schema, error) {
	specs := []records.FieldSpec{}
	for _, f := range s.Fields() {
		t := f.Type
		if recurse && t.Kind == records.RecordKind && len(t.Shape) == 0 {
			var e error
			t.Schema, e = RepackSchema(t.Schema, align, true)
			if e != nil {
				return nil, e
			}
		}
		specs = append(specs, records.FieldSpec{Name: f.Name, Type: t, Title: f.Title})
	}
	out, e := records.NewSchema(specs, align, 0)
	if e != nil {
		return nil, e
	}
	return out.AsVoidDescriptor(), nil
}
func RepackFields(a *np.Array, align, recurse bool) (*np.Array, error) {
	s, e := schemaOf(a)
	if e != nil {
		if a.DType() != 0 {
			return a, nil
		}
		return nil, e
	}
	t, e := RepackSchema(s, align, recurse)
	if e != nil {
		return nil, e
	}
	if t.EqualSchema(s) {
		return a, nil
	}
	return RequireFields(a, t)
}
