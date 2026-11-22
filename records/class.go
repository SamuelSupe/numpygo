package records

import (
	"encoding/binary"
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"strconv"
	"strings"
	"unicode/utf8"
)

// BindBase is called by ndarray scalar extraction to retain the shared
// record storage and its originating view.
func (r *Record) BindBase(a *np.Array) { r.base = a; r.owns = false }
func (r *Record) Base() *np.Array      { return r.base }
func (r *Record) DType() *Schema       { return r.schema }
func (r *Record) Flags() np.VoidFlags {
	return np.VoidFlags{CContiguous: true, FContiguous: true, OwnData: r.owns, Writable: !r.readOnly, Aligned: true}
}
func (r *Record) GetField(dtype np.DType, offset int, order binary.ByteOrder) (any, error) {
	if r.schema.HasObject() {
		return nil, np.ErrDType
	}
	return np.VoidScalar(r.data).GetField(dtype, offset, order)
}
func (r *Record) SetField(value any, dtype np.DType, offset int, order binary.ByteOrder) error {
	if r.readOnly {
		return np.ErrReadOnly
	}
	if r.schema.HasObject() || !dtype.IsNativeNumeric() && !dtype.IsExtended() {
		return np.ErrDType
	}
	a, e := np.FromBuffer(r.data, dtype, 1, offset, order)
	if e != nil {
		return e
	}
	return a.SetValue(value, 0)
}

func (s *Schema) fieldNameAt(index int) (string, error) {
	if index < 0 {
		index += len(s.fields)
	}
	if index < 0 || index >= len(s.fields) {
		return "", np.ErrIndex
	}
	return s.fields[index].Name, nil
}

// FieldAt and SetAt access structured scalar fields by their ordered
// position, independently of any Go or Python method name collision.
func (r *Record) FieldAt(index int) (any, error) {
	name, e := r.schema.fieldNameAt(index)
	if e != nil {
		return nil, e
	}
	return r.Field(name)
}
func (r *Record) SetAt(index int, value any) error {
	name, e := r.schema.fieldNameAt(index)
	if e != nil {
		return e
	}
	return r.Set(name, value)
}

// PPrint formats scalar numeric and text fields in descriptor order using
// NumPy's right-aligned field-name layout. Nested records and arbitrary
// objects belong to the caller's formatting policy.
func (r *Record) PPrint() (string, error) {
	names := r.schema.Names()
	if len(names) == 0 {
		return "", np.ErrValue
	}
	width := 0
	for _, n := range names {
		width = max(width, utf8.RuneCountInString(n))
	}
	lines := make([]string, len(names))
	for i, n := range names {
		v, e := r.Field(n)
		if e != nil {
			return "", e
		}
		var text string
		switch x := v.(type) {
		case bool:
			if x {
				text = "True"
			} else {
				text = "False"
			}
		case float32:
			text = recordFloat(float64(x), 32)
		case float64:
			text = recordFloat(x, 64)
		case int8, int16, int32, int64, uint8, uint16, uint32, uint64, Text:
			text = fmt.Sprint(x)
		case *np.Array:
			text, e = np.ArrayStr(x)
			if e != nil {
				return "", e
			}
		default:
			return "", np.ErrDType
		}
		lines[i] = fmt.Sprintf("%*s: %s", width, n, text)
	}
	return strings.Join(lines, "\n"), nil
}
func recordFloat(x float64, bits int) string {
	s := strconv.FormatFloat(x, 'g', -1, bits)
	if s == "+Inf" {
		return "inf"
	}
	if s == "-Inf" {
		return "-inf"
	}
	if s == "NaN" {
		return "nan"
	}
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

// FieldAt returns a storage-sharing column selected by field position.
func (a *Array) FieldAt(index int) (*Column, error) {
	name, e := a.schema.fieldNameAt(index)
	if e != nil {
		return nil, e
	}
	return a.Field(name)
}
func (a *Array) Index(indices ...np.Index) (*Array, error) {
	return wrap(a.nd.Index(indices...))
}
func (a *Array) Assign(indices []np.Index, values *Array) error {
	if values == nil {
		return np.ErrDType
	}
	return a.nd.Assign(indices, values.nd)
}

// SelectFields is a multi-field structured view, not a repacked copy. It
// preserves original offsets, byte order, record width and outer strides
// even when field order changes or a field is omitted.
func (a *Array) SelectFields(names ...string) (*Array, error) {
	specs := make([]FieldSpec, len(names))
	for i, name := range names {
		f, e := a.schema.Field(name)
		if e != nil || f.Name != name {
			return nil, np.ErrIndex
		}
		specs[i] = FieldSpec{Name: f.Name, Title: f.Title, Type: f.Type, Offset: np.Int(f.Offset)}
	}
	schema, e := NewSchema(specs, a.schema.Aligned(), a.schema.ItemSize())
	if e != nil {
		return nil, e
	}
	v, e := a.nd.ViewLayout(np.StructuredField{DType: np.RecordDType, Schema: schema}, a.Shape(), a.Strides())
	return wrap(v, e)
}
