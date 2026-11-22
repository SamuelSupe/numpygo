// Package npyformat exposes NumPy's low-level NPY header, payload and mapping
// operations without Python object deserialization.
package npyformat

import (
	"bytes"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/records"
	"io"
	"os"
	"slices"
)

type Version = np.NPYVersion
type Header = np.NPYHeader

const DefaultMaxHeaderSize = 10000

func Magic(major, minor int) ([]byte, error) {
	return np.NPYMagic(Version{Major: major, Minor: minor})
}
func ReadMagic(r io.Reader) (Version, error)          { return np.ReadNPYMagic(r) }
func HeaderDataFromArray(a *np.Array) (Header, error) { return np.NPYHeaderFromArray(a) }

// DTypeToDescr returns the NPY-compatible scalar string or structured field
// list. Pointer-managed variable strings use the same object marker as NumPy;
// WriteArray does not serialize their payloads.
func DTypeToDescr(d dtypes.Descriptor) (any, error) {
	switch d.DType() {
	case 0:
		return nil, np.ErrDType
	case np.Object, np.StringDType:
		return "|O", nil
	case np.RecordDType:
		ds, e := d.Schema().NPYDescriptor()
		if e != nil {
			return nil, e
		}
		return np.ParseNPYDescriptorLiteral(ds)
	default:
		return d.String(), nil
	}
}
func DescrToDType(desc any) (dtypes.Descriptor, error) {
	switch d := desc.(type) {
	case string:
		return dtypes.Parse(d)
	case []any:
		s, e := records.SchemaFromNPYDescriptor(d)
		if e != nil {
			return dtypes.Descriptor{}, e
		}
		return dtypes.New("VoidDType", dtypes.Options{Schema: s})
	default:
		return dtypes.Descriptor{}, np.ErrDType
	}
}

// DropMetadata strips explicitly supplied user metadata, retaining layout.
func DropMetadata(d dtypes.Descriptor) dtypes.Descriptor { return d.WithoutMetadata() }

func ReadArrayHeader1(r io.Reader, maxHeaderSize int) (Header, error) {
	return np.ReadNPYHeaderVersion(r, Version{Major: 1}, headerLimit(maxHeaderSize))
}
func ReadArrayHeader2(r io.Reader, maxHeaderSize int) (Header, error) {
	return np.ReadNPYHeaderVersion(r, Version{Major: 2}, headerLimit(maxHeaderSize))
}
func WriteArrayHeader1(w io.Writer, h Header) error {
	return np.WriteNPYHeader(w, h, Version{Major: 1})
}
func WriteArrayHeader2(w io.Writer, h Header) error {
	return np.WriteNPYHeader(w, h, Version{Major: 2})
}
func headerLimit(n int) int {
	if n == 0 {
		return DefaultMaxHeaderSize
	}
	return n
}
func readHeader(r io.Reader, maxHeaderSize int) (Header, error) {
	v, e := ReadMagic(r)
	if e != nil {
		return Header{}, e
	}
	return np.ReadNPYHeaderVersion(r, v, headerLimit(maxHeaderSize))
}
func logicalSize(d dtypes.Descriptor, shape []int) (int, error) {
	if d.DType() == 0 || d.Metadata().HasObject {
		return 0, np.ErrDType
	}
	if d.DType().IsExtended() && d.Order().Uint16([]byte{1, 0}) != 1 {
		return 0, np.ErrDType
	}
	n, e := np.ElementCount(np.Bool, shape...)
	if e != nil {
		return 0, e
	}
	if n > int(^uint(0)>>1)/max(d.ItemSize(), 1) {
		return 0, np.ErrShape
	}
	return n * d.ItemSize(), nil
}
func flexible(d dtypes.Descriptor) (np.FlexibleDType, error) {
	return np.ParseFlexibleDType(d.String())
}

// ReadArray preserves the declared endian and C/Fortran layout. The
// descriptor and bounded data literal are decoded without executing pickle.
func ReadArray(r io.Reader, maxHeaderSize int) (*np.Array, error) {
	h, e := readHeader(r, maxHeaderSize)
	if e != nil {
		return nil, e
	}
	d, e := DescrToDType(h.Descriptor)
	if e != nil {
		return nil, e
	}
	n, e := logicalSize(d, h.Shape)
	if e != nil {
		return nil, e
	}
	// A short stream must not allocate an enormous claimed payload.
	b, e := io.ReadAll(io.LimitReader(r, int64(n)))
	if e != nil {
		return nil, e
	}
	if len(b) != n {
		return nil, io.ErrUnexpectedEOF
	}
	shape := slices.Clone(h.Shape)
	if h.FortranOrder {
		slices.Reverse(shape)
	}
	var a *np.Array
	switch t := d.DType(); {
	case t == np.RecordDType:
		a, e = np.FromRecordBuffer(d.Schema(), b, shape, false)
	case t.IsFlexible():
		f, er := flexible(d)
		if er != nil {
			return nil, er
		}
		a, e = np.FromFlexibleBuffer(b, f, shape, false)
	case t.IsTemporal():
		a, e = np.FromTemporalBuffer(b, t, d.Unit().String(), shape, false, d.Order())
	default:
		a, e = np.FromBuffer(b, t, -1, 0, d.Order())
		if e == nil {
			a, e = a.Reshape(shape...)
		}
	}
	if e != nil {
		return nil, e
	}
	if h.FortranOrder {
		return a.Transpose()
	}
	return a, nil
}
func WriteArray(w io.Writer, a *np.Array, v Version) error {
	return np.SaveNPYVersion(w, a, v)
}

// IsFileObj reports whether a Go value is an open OS file (an actual file
// descriptor rather than an in-memory stream). Opaque buffered wrappers
// cannot expose their underlying descriptor through Go's io interfaces.
func IsFileObj(v any) bool {
	f, ok := v.(*os.File)
	if !ok || f == nil {
		return false
	}
	_, e := f.Stat()
	return e == nil
}

type MemMapOptions struct {
	Mode          string // r, r+, c, w+
	DType         *dtypes.Descriptor
	Shape         []int
	FortranOrder  bool
	Version       Version
	MaxHeaderSize int
}

// OpenMemMap creates an NPY header in w+ mode, or reads the existing header
// otherwise. The mapping starts at its payload, not at file byte zero.
func OpenMemMap(path string, opt MemMapOptions) (*np.MappedArray, error) {
	mode := opt.Mode
	if mode == "" {
		mode = "r+"
	}
	if mode != "r" && mode != "r+" && mode != "c" && mode != "w+" {
		return nil, np.ErrValue
	}
	var h Header
	var d dtypes.Descriptor
	var offset int64
	var e error
	if mode == "w+" {
		if opt.Shape == nil {
			return nil, np.ErrShape
		}
		if opt.DType == nil {
			d, e = dtypes.Parse("f8")
		} else {
			d = *opt.DType
		}
		if e != nil {
			return nil, e
		}
		if _, e = logicalSize(d, opt.Shape); e != nil {
			return nil, e
		}
		if d.ItemSize() == 0 {
			return nil, np.ErrDType
		}
		desc, e := DTypeToDescr(d)
		if e != nil {
			return nil, e
		}
		h = Header{Descriptor: desc, Shape: slices.Clone(opt.Shape), FortranOrder: opt.FortranOrder}
		// Validate and stage the header before truncating an existing file.
		var b bytes.Buffer
		if e = np.WriteNPYHeader(&b, h, opt.Version); e != nil {
			return nil, e
		}
		if e = os.WriteFile(path, b.Bytes(), 0600); e != nil {
			return nil, e
		}
		offset = int64(b.Len())
		mode = "r+"
	} else {
		f, e := os.Open(path)
		if e != nil {
			return nil, e
		}
		defer f.Close()
		h, e = readHeader(f, opt.MaxHeaderSize)
		if e != nil {
			return nil, e
		}
		d, e = DescrToDType(h.Descriptor)
		if e != nil {
			return nil, e
		}
		if _, e = logicalSize(d, h.Shape); e != nil {
			return nil, e
		}
		offset, e = f.Seek(0, io.SeekCurrent)
		if e != nil {
			return nil, e
		}
	}
	shape := h.Shape
	if shape == nil {
		shape = []int{} // scalar, not inferred shape
	}
	opts := np.MemMapOptions{DType: d.DType(), Shape: shape, ByteOrder: d.Order(), Mode: mode, Offset: offset, Schema: d.Schema()}
	// A typed nil schema must not be stored in the interface for scalar dtypes.
	if d.DType() != np.RecordDType {
		opts.Schema = nil
	}
	if h.FortranOrder {
		opts.Order = "F"
	}
	if d.DType().IsTemporal() {
		opts.DateUnit = d.Unit().String()
	}
	if d.DType().IsFlexible() {
		f, e := flexible(d)
		if e != nil {
			return nil, e
		}
		opts.Flexible = &f
	}
	return np.MemMap(path, opts)
}
