package np

import (
	"encoding/binary"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode"
)

func pythonString(s string) string {
	quote := '\''
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}
	var b strings.Builder
	b.WriteRune(quote)
	for _, r := range s {
		switch r {
		case quote, '\\':
			b.WriteRune('\\')
			b.WriteRune(r)
		case '\n':
			b.WriteString("\\n")
		case '\r':
			b.WriteString("\\r")
		case '\t':
			b.WriteString("\\t")
		default:
			if unicode.IsPrint(r) {
				b.WriteRune(r)
			} else if r <= 255 {
				fmt.Fprintf(&b, "\\x%02x", r)
			} else if r <= 65535 {
				fmt.Fprintf(&b, "\\u%04x", r)
			} else {
				fmt.Fprintf(&b, "\\U%08x", r)
			}
		}
	}
	b.WriteRune(quote)
	return b.String()
}

func npyTuple(items []string) string {
	v := strings.Join(items, ", ")
	if len(items) == 1 {
		v += ","
	}
	return "(" + v + ")"
}
func npyTitle(v any) (string, error) {
	switch x := v.(type) {
	case string:
		return pythonString(x), nil
	case int:
		return strconv.Itoa(x), nil
	case bool:
		if x {
			return "True", nil
		}
		return "False", nil
	}
	return "", ErrDType
}
func npyDescr(v any, depth int) (string, error) {
	if depth > maxDims {
		return "", ErrDType
	}
	if x, ok := v.(string); ok {
		return pythonString(x), nil
	}
	fields, ok := v.([]any)
	if !ok {
		return "", ErrDType
	}
	// A nested subarray descriptor is a (dtype, shape) tuple rather
	// than a list of named fields. The parser stores both as []any.
	if len(fields) == 2 {
		if dims, ok := fields[1].([]any); ok {
			ss := make([]string, len(dims))
			valid := true
			for i, v := range dims {
				n, ok := v.(int)
				if !ok || n < 0 {
					valid = false
					break
				}
				ss[i] = strconv.Itoa(n)
			}
			if valid {
				leaf, e := npyDescr(fields[0], depth+1)
				if e != nil {
					return "", e
				}
				return npyTuple([]string{leaf, npyTuple(ss)}), nil
			}
		}
	}
	parts := make([]string, len(fields))
	for i, raw := range fields {
		f, ok := raw.([]any)
		if !ok || len(f) < 2 || len(f) > 3 {
			return "", ErrDType
		}
		name := ""
		switch x := f[0].(type) {
		case string:
			name = pythonString(x)
		case []any:
			if len(x) != 2 {
				return "", ErrDType
			}
			t, e := npyTitle(x[0])
			if e != nil {
				return "", e
			}
			n, ok := x[1].(string)
			if !ok {
				return "", ErrDType
			}
			name = npyTuple([]string{t, pythonString(n)})
		default:
			return "", ErrDType
		}
		d, e := npyDescr(f[1], depth+1)
		if e != nil {
			return "", e
		}
		items := []string{name, d}
		if len(f) == 3 {
			dims, ok := f[2].([]any)
			if !ok {
				return "", ErrShape
			}
			ss := make([]string, len(dims))
			for j, dim := range dims {
				q, ok := dim.(int)
				if !ok || q < 0 {
					return "", ErrShape
				}
				ss[j] = strconv.Itoa(q)
			}
			items = append(items, npyTuple(ss))
		}
		parts[i] = npyTuple(items)
	}
	return "[" + strings.Join(parts, ", ") + "]", nil
}

// NPYHeaderFromArray reports serializable dtype and shape metadata without
// converting the data. Object and variable string descriptors are marked
// as object exactly as NumPy does; the writer still rejects their payloads.
func NPYHeaderFromArray(a *Array) (NPYHeader, error) {
	if a.DType() == 0 {
		return NPYHeader{}, ErrDType
	}
	h := NPYHeader{Shape: a.Shape(), FortranOrder: a.IsFContiguous() && !a.IsCContiguous()}
	switch a.DType() {
	case Object, StringDType:
		h.Descriptor = "|O"
	case RecordDType:
		s, ok := a.StructuredSchema().(interface{ NPYDescriptor() (string, error) })
		if !ok {
			return h, ErrDType
		}
		desc, e := s.NPYDescriptor()
		if e != nil {
			return h, e
		}
		h.Descriptor, e = ParseNPYDescriptorLiteral(desc)
		return h, e
	case BytesDType, UnicodeDType, VoidDType:
		h.Descriptor = a.data.flex.String()
	case DateTime64, TimeDelta64:
		kind := "M"
		if a.DType() == TimeDelta64 {
			kind = "m"
		}
		d := "<" + kind + "8"
		if storageOrder(a.data).Uint16([]byte{1, 0}) != 1 {
			d = ">" + kind + "8"
		}
		if a.data.unit.Name != "generic" {
			d += "[" + a.data.unit.String() + "]"
		}
		h.Descriptor = d
	default:
		d := descriptor(a.DType())
		if d == "" {
			return h, ErrDType
		}
		if d[0] != '|' && storageOrder(a.data).Uint16([]byte{1, 0}) != 1 {
			d = ">" + d[1:]
		}
		h.Descriptor = d
	}
	return h, nil
}

// WriteNPYHeader writes magic, length and aligned data-literal metadata.
// Version 0.0 chooses the oldest representable version (1, 2, or 3).
func WriteNPYHeader(w io.Writer, h NPYHeader, version NPYVersion) error {
	if _, e := shapeSize(Bool, h.Shape); e != nil {
		return e
	}
	desc, e := npyDescr(h.Descriptor, 0)
	if e != nil {
		return e
	}
	order := "False"
	if h.FortranOrder {
		order = "True"
	}
	header := fmt.Sprintf("{'descr': %s, 'fortran_order': %s, 'shape': %s, }", desc, order, shapeText(h.Shape))
	if len(h.Shape) != 0 {
		axis := 0
		if h.FortranOrder {
			axis = len(h.Shape) - 1
		}
		header += strings.Repeat(" ", 21-len(strconv.Itoa(h.Shape[axis])))
	}
	latin := make([]byte, 0, len(header))
	isLatin := true
	for _, r := range header {
		if r > 255 {
			isLatin = false
			break
		}
		latin = append(latin, byte(r))
	}
	auto := version == (NPYVersion{})
	if auto {
		version = NPYVersion{1, 0}
		if !isLatin {
			version.Major = 3
		}
	}
	if version.Minor != 0 || version.Major < 1 || version.Major > 3 {
		return ErrValue
	}
	b := latin
	if version.Major == 3 {
		b = []byte(header)
	} else if !isLatin {
		return ErrValue
	}
	prefix := 12
	if version.Major == 1 {
		prefix = 10
	}
	npad := 64 - (prefix+len(b)+1)%64
	n := len(b) + npad + 1
	if version.Major == 1 && n > 65535 {
		if !auto {
			return ErrShape
		}
		version.Major, prefix = 2, 12
		npad = 64 - (prefix+len(b)+1)%64
		n = len(b) + npad + 1
	}
	if uint64(n) > uint64(^uint32(0)) {
		return ErrShape
	}
	magic, e := NPYMagic(version)
	if e != nil {
		return e
	}
	var frame []byte
	if version.Major == 1 {
		frame = binary.LittleEndian.AppendUint16(magic, uint16(n))
	} else {
		frame = binary.LittleEndian.AppendUint32(magic, uint32(n))
	}
	frame = append(frame, b...)
	frame = append(frame, strings.Repeat(" ", npad)...)
	frame = append(frame, '\n')
	nw, e := w.Write(frame)
	if e == nil && nw != len(frame) {
		e = io.ErrShortWrite
	}
	return e
}

// SaveNPYVersion preserves both byte order and the chosen file version.
func SaveNPYVersion(w io.Writer, a *Array, version NPYVersion) error {
	if a.DType() == Object || a.DType() == StringDType {
		return ErrDType
	}
	h, e := NPYHeaderFromArray(a)
	if e != nil {
		return e
	}
	order := "C"
	if h.FortranOrder {
		order = "F"
	}
	b, e := a.ToBytes(order)
	if e != nil {
		return e
	}
	if e = WriteNPYHeader(w, h, version); e != nil {
		return e
	}
	n, e := w.Write(b)
	if e == nil && n != len(b) {
		e = io.ErrShortWrite
	}
	return e
}
