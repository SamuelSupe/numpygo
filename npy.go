package np

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

const npyMagic = "\x93NUMPY"

func storageSlice(s *storage) any {
	switch s.dtype {
	case Float64:
		return s.real
	case Complex128:
		return s.cplx
	default:
		return s.raw
	}
}
func descriptor(d DType) string {
	switch d {
	case Bool:
		return "|b1"
	case Int8:
		return "|i1"
	case Uint8:
		return "|u1"
	case Int16:
		return "<i2"
	case Int32:
		return "<i4"
	case Int64:
		return "<i8"
	case Uint16:
		return "<u2"
	case Uint32:
		return "<u4"
	case Uint64:
		return "<u8"
	case Float16:
		return "<f2"
	case Float32:
		return "<f4"
	case Float64:
		return "<f8"
	case Complex64:
		return "<c8"
	case Complex128:
		return "<c16"
	case Float128:
		return "<f16"
	case Complex256:
		return "<c32"
	}
	return ""
}

func dtypeDescriptor(desc string) (DType, binary.ByteOrder, error) {
	if len(desc) < 3 {
		return 0, nil, ErrDType
	}
	var order binary.ByteOrder = binary.LittleEndian
	switch desc[0] {
	case '>':
		order = binary.BigEndian
	case '=':
		order = binary.NativeEndian
	case '<', '|':
	default:
		return 0, nil, ErrDType
	}
	for d := Float64; int(d) < len(dtypeNames); d++ {
		known := descriptor(d)
		if known != "" && known[1:] == desc[1:] {
			return d, order, nil
		}
	}
	return 0, nil, fmt.Errorf("%w: unsupported .npy descriptor %q", ErrDType, desc)
}

// ParseNPYScalarDescriptor parses a fixed-width numerical NPY dtype.
// Record loaders use it for leaf fields, preserving their byte order.
func ParseNPYScalarDescriptor(desc string) (DType, binary.ByteOrder, error) {
	return dtypeDescriptor(desc)
}

func shapeText(shape []int) string {
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = strconv.Itoa(d)
	}
	if len(parts) == 1 {
		return "(" + parts[0] + ",)"
	}
	return "(" + strings.Join(parts, ", ") + ")"
}

// SaveNPY writes a NumPy array using the oldest version that represents its
// descriptor. Fortran-contiguous views and declared byte order are preserved.
func SaveNPY(w io.Writer, a *Array) error {
	return SaveNPYVersion(w, a, NPYVersion{})
}

// LoadNPY reads a numeric NumPy v1, v2 or v3 file, including big-endian and
// Fortran-order payloads. Use records.LoadNPY for structured dtype payloads.
func LoadNPY(r io.Reader) (*Array, error) {
	return loadNPY(r, nil)
}

// RecordNPYDecoder lets the records package decode a structured payload
// without registering a global decoder or introducing a dependency cycle.
type RecordNPYDecoder func(io.Reader, NPYHeader) (*Array, error)

func LoadNPYWithRecords(r io.Reader, decode RecordNPYDecoder) (*Array, error) {
	if decode == nil {
		return nil, ErrValue
	}
	return loadNPY(r, decode)
}
func loadNPY(r io.Reader, decode RecordNPYDecoder) (*Array, error) {
	h, err := ReadNPYHeader(r)
	if err != nil {
		return nil, err
	}
	ds, ok := h.Descriptor.(string)
	if !ok {
		if decode != nil {
			return decode(r, h)
		}
		return nil, ErrDType
	}
	if f, e := ParseFlexibleDType(ds); e == nil {
		w := f.ItemSize()
		n, e := shapeSizeWidth(max(w, 1), h.Shape)
		if e != nil {
			return nil, e
		}
		b := make([]byte, n*w)
		if _, e = io.ReadFull(r, b); e != nil {
			return nil, e
		}
		a, e := FromFlexibleBuffer(b, f, h.Shape, false)
		if e != nil {
			return nil, e
		}
		if h.FortranOrder && a.size > 0 {
			stride := 1
			for i, d := range a.shape {
				a.strides[i] = stride
				stride *= d
			}
		}
		return a, nil
	}
	var unit DateUnit
	var temporal DType
	if len(ds) >= 3 && (ds[1:3] == "M8" || ds[1:3] == "m8") {
		var e error
		temporal, unit, _, e = ParseTemporalDescriptor(ds)
		if e != nil {
			return nil, e
		}
		ds = ds[:1] + "i8"
	}
	d, order, err := dtypeDescriptor(ds)
	if err != nil {
		return nil, err
	}
	shape := h.Shape
	a, err := Zeros(d, shape...)
	if err != nil {
		return nil, err
	}
	if d.IsExtended() {
		if order.Uint16([]byte{1, 0}) != 1 {
			return nil, ErrDType
		}
		b := make([]byte, a.NBytes())
		if _, err = io.ReadFull(r, b); err != nil {
			return nil, err
		}
		v, e := FromBuffer(b, d, a.size, 0, binary.LittleEndian)
		if e != nil {
			return nil, e
		}
		a, e = v.Reshape(shape...)
		if e != nil {
			return nil, e
		}
	} else {
		if err = binary.Read(r, order, storageSlice(a.data)); err != nil {
			return nil, err
		}
	}
	if h.FortranOrder && a.size > 0 {
		stride := 1
		for i, dim := range a.shape {
			a.strides[i] = stride
			stride *= dim
		}
	}
	if temporal != 0 {
		return a.retag(temporal, unit), nil
	}
	return a, nil
}

// SaveNPZ writes named arrays in the NumPy zip container format.
func SaveNPZ(w io.Writer, arrays map[string]*Array, compress bool) error {
	names := make([]string, 0, len(arrays))
	for name, a := range arrays {
		if name == "" || strings.ContainsRune(name, 0) || a.DType() == 0 {
			return ErrValue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	z := zip.NewWriter(w)
	method := uint16(zip.Store)
	if compress {
		method = zip.Deflate
	}
	for _, name := range names {
		entry, err := z.CreateHeader(&zip.FileHeader{Name: name + ".npy", Method: method})
		if err != nil {
			_ = z.Close()
			return err
		}
		if err = SaveNPY(entry, arrays[name]); err != nil {
			_ = z.Close()
			return err
		}
	}
	return z.Close()
}

// LoadNPZ decodes named numeric arrays without extracting files or loading pickle.
func LoadNPZ(r io.ReaderAt, size int64) (map[string]*Array, error) {
	return loadNPZ(r, size, nil)
}
func LoadNPZWithRecords(r io.ReaderAt, size int64, decode RecordNPYDecoder) (map[string]*Array, error) {
	if decode == nil {
		return nil, ErrValue
	}
	return loadNPZ(r, size, decode)
}
func loadNPZ(r io.ReaderAt, size int64, decode RecordNPYDecoder) (map[string]*Array, error) {
	if size < 0 {
		return nil, ErrShape
	}
	z, err := zip.NewReader(r, size)
	if err != nil {
		return nil, err
	}
	out := make(map[string]*Array, len(z.File))
	for _, entry := range z.File {
		if !strings.HasSuffix(entry.Name, ".npy") {
			return nil, ErrValue
		}
		name := strings.TrimSuffix(entry.Name, ".npy")
		if _, exists := out[name]; exists {
			return nil, ErrValue
		}
		reader, err := entry.Open()
		if err != nil {
			return nil, err
		}
		a, readErr := loadNPY(reader, decode)
		if readErr == nil {
			// Read to EOF to ensure ZIP CRC verification runs and catch extra bytes.
			var trailing [1]byte
			n, tailErr := reader.Read(trailing[:])
			if n != 0 || tailErr != io.EOF {
				readErr = ErrValue
			}
		}
		closeErr := reader.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		out[name] = a
	}
	return out, nil
}
