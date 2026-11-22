package records

import (
	"encoding/binary"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"

	"github.com/SamuelSupe/numpygo"
)

// NPYDescriptor emits the Python data literal used by NumPy's dtype.descr.
// Ordered non-overlapping fields and unnamed void padding preserve offsets.
// NumPy itself cannot serialize overlapping/out-of-order dtype.descr fields.
func (s *Schema) NPYDescriptor() (string, error) {
	parts := make([]string, 0, len(s.fields)+1)
	pos := 0
	pad := func(n int) {
		if n > 0 {
			parts = append(parts, fmt.Sprintf("('', '|V%d')", n))
		}
	}
	for _, f := range s.fields {
		if f.Offset < pos {
			return "", np.ErrDType
		}
		pad(f.Offset - pos)
		desc := strconv.QuoteToASCII(f.Type.Descriptor())
		if f.Type.Kind == RecordKind {
			v, e := f.Type.Schema.NPYDescriptor()
			if e != nil {
				return "", e
			}
			desc = v
		}
		fieldShape := f.Type.Shape
		if len(f.Type.NestedShapes) > 0 {
			for i := len(f.Type.NestedShapes) - 1; i >= 1; i-- {
				desc = "(" + desc + ", " + shapeLiteral(f.Type.NestedShapes[i]) + ")"
			}
			fieldShape = f.Type.NestedShapes[0]
		}
		name := strconv.QuoteToASCII(f.Name)
		if f.Title != nil {
			var title string
			switch v := f.Title.(type) {
			case string:
				title = strconv.QuoteToASCII(v)
			case int:
				title = strconv.Itoa(v)
			case bool:
				title = "False"
				if v {
					title = "True"
				}
			default:
				return "", np.ErrDType
			}
			name = "(" + title + ", " + name + ")"
		}
		entry := "(" + name + ", " + desc
		if len(fieldShape) > 0 {
			entry += ", " + shapeLiteral(fieldShape)
		}
		parts = append(parts, entry+")")
		width, e := f.Type.Size()
		if e != nil {
			return "", e
		}
		pos = f.Offset + width
	}
	pad(s.itemsize - pos)
	return "[" + strings.Join(parts, ", ") + "]", nil
}

func shapeLiteral(dims []int) string {
	s := make([]string, len(dims))
	for i, n := range dims {
		s[i] = strconv.Itoa(n)
	}
	v := strings.Join(s, ", ")
	if len(s) == 1 {
		v += ","
	}
	return "(" + v + ")"
}
func descriptorShape(v any) ([]int, bool) {
	items, ok := v.([]any)
	if !ok {
		return nil, false
	}
	dims := make([]int, len(items))
	for i, x := range items {
		n, ok := x.(int)
		if !ok || n < 0 {
			return nil, false
		}
		dims[i] = n
	}
	return dims, true
}

func typeFromNPY(v any) (Type, error) {
	if seq, ok := v.([]any); ok {
		if len(seq) == 2 {
			if dims, ok := descriptorShape(seq[1]); ok {
				t, e := typeFromNPY(seq[0])
				if e != nil {
					return Type{}, e
				}
				return WrapSubarray(t, dims...), nil
			}
		}
		s, e := schemaFromNPY(seq)
		if e != nil {
			return Type{}, e
		}
		return Struct(s), nil
	}
	desc, ok := v.(string)
	if ok && (desc == "|O" || desc == "O" || desc == "|O8") {
		return Object(), nil
	}
	if !ok || len(desc) < 3 {
		return Type{}, np.ErrDType
	}
	switch desc[1] {
	case 'M', 'm':
		d, u, o, e := np.ParseTemporalDescriptor(desc)
		if e != nil {
			return Type{}, e
		}
		return Temporal(d, u.String(), o)
	case 'S', 'U', 'V':
		width, e := strconv.Atoi(desc[2:])
		if e != nil || width < 0 {
			return Type{}, np.ErrDType
		}
		switch desc[0] {
		case '<', '>', '|', '=':
		default:
			return Type{}, np.ErrDType
		}
		switch desc[1] {
		case 'S':
			return FixedBytes(width), nil
		case 'V':
			return Opaque(width), nil
		case 'U':
			var order binary.ByteOrder = binary.LittleEndian
			if desc[0] == '>' {
				order = binary.BigEndian
			} else if desc[0] == '=' {
				order = binary.NativeEndian
			}
			return FixedUnicode(width, order), nil
		}
	}
	d, o, e := np.ParseNPYScalarDescriptor(desc)
	if e != nil {
		return Type{}, e
	}
	return Numeric(d, o), nil
}

// SchemaFromNPYDescriptor removes unnamed padding fields while retaining
// their offsets, titles, nested descriptors and subarray shapes.
func SchemaFromNPYDescriptor(list []any) (*Schema, error) { return schemaFromNPY(list) }

func schemaFromNPY(list []any) (*Schema, error) {
	specs := make([]FieldSpec, 0, len(list))
	offset := 0
	for _, v := range list {
		entry, ok := v.([]any)
		if !ok || (len(entry) != 2 && len(entry) != 3) {
			return nil, np.ErrDType
		}
		var title any
		name, ok := entry[0].(string)
		if !ok {
			tuple, ok := entry[0].([]any)
			if !ok || len(tuple) != 2 {
				return nil, np.ErrDType
			}
			title = tuple[0]
			name, ok = tuple[1].(string)
			if !ok || name == "" {
				return nil, np.ErrDType
			}
		}
		t, e := typeFromNPY(entry[1])
		if e != nil {
			return nil, e
		}
		if len(entry) == 3 {
			dims, ok := descriptorShape(entry[2])
			if !ok {
				return nil, np.ErrShape
			}
			t = WrapSubarray(t, dims...)
		}
		width, e := t.Size()
		if e != nil {
			return nil, e
		}
		if width > int(^uint(0)>>1)-offset {
			return nil, np.ErrShape
		}
		if name == "" {
			if t.Kind != VoidKind || len(t.Shape) != 0 {
				return nil, np.ErrDType
			}
		} else {
			start := offset
			specs = append(specs, FieldSpec{Name: name, Type: t, Offset: &start, Title: title})
		}
		offset += width
	}
	// dtype.descr preserves padding but not the aligned flag itself.
	return NewSchema(specs, false, offset)
}

// LoadNPY reads a non-object structured NPY dtype (including nested fields,
// padding, subarrays and endian-specific data) into the unified ndarray.
// This entry point lives in records to avoid a dependency cycle in np.
func LoadNPY(r io.Reader) (*np.Array, error) {
	h, e := np.ReadNPYHeader(r)
	if e != nil {
		return nil, e
	}
	return loadStructuredPayload(r, h)
}
func LoadAnyNPY(r io.Reader) (*np.Array, error) {
	return np.LoadNPYWithRecords(r, loadStructuredPayload)
}

// LoadNPZ automatically resolves structured, flexible and numerical members
// of the same archive through the shared CRC-checking NPY reader.
func LoadNPZ(r io.ReaderAt, size int64) (map[string]*np.Array, error) {
	return np.LoadNPZWithRecords(r, size, loadStructuredPayload)
}
func loadStructuredPayload(r io.Reader, h np.NPYHeader) (*np.Array, error) {
	list, ok := h.Descriptor.([]any)
	if !ok {
		return nil, np.ErrDType
	}
	s, e := schemaFromNPY(list)
	if e != nil {
		return nil, e
	}
	if s.HasObject() {
		return nil, np.ErrDType
	}
	n, e := np.ElementCount(np.Bool, h.Shape...)
	if e != nil {
		return nil, e
	}
	if s.ItemSize() < 0 || n > int(^uint(0)>>1)/max(s.ItemSize(), 1) {
		return nil, np.ErrShape
	}
	b := make([]byte, n*s.ItemSize())
	if _, e = io.ReadFull(r, b); e != nil {
		return nil, e
	}
	shape := slices.Clone(h.Shape)
	if h.FortranOrder {
		slices.Reverse(shape)
	}
	a, e := np.FromRecordBuffer(s, b, shape, false)
	if e != nil {
		return nil, e
	}
	if h.FortranOrder {
		return a.Transpose()
	}
	return a, nil
}
