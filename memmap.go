package np

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
)

var ErrClosedMapping = errors.New("memory mapping is closed")
var ErrMappingUnsupported = errors.New("memory mapping is unavailable on this platform")

type MemMapOptions struct {
	DType     DType
	Shape     []int  // nil infers a one-dimensional shape from the file length
	Mode      string // r, r+, w+, c (private copy-on-write); default r+
	Offset    int64
	Order     string // C or F
	ByteOrder binary.ByteOrder
	Flexible  *FlexibleDType   // required for S, U and V
	Schema    StructuredSchema // required for RecordDType
	DateUnit  string           // datetime64/timedelta64 unit; empty means generic
}

type mappedRegion struct {
	mu       sync.Mutex
	data     []byte
	mode     string
	filename string
	offset   int64
	closed   atomic.Bool
}

// MemMapInfo describes the original mapping, not the byte position of a view.
type MemMapInfo struct {
	Filename string
	Offset   int64
	Mode     string
}

// MappingInfo is retained by basic ndarray views. Independent copies do not
// have mapping metadata or a backing file to flush.
func (a *Array) MappingInfo() (MemMapInfo, bool) {
	r := a.mappingRegion()
	if r == nil {
		return MemMapInfo{}, false
	}
	return MemMapInfo{r.filename, r.offset, r.mode}, true
}

// FlushMapping flushes the owning shared mapping from any derived view.
// Non-mapped arrays have nothing to flush.
func (a *Array) FlushMapping() error {
	if r := a.mappingRegion(); r != nil {
		return r.flush()
	}
	return nil
}

func (a *Array) mappingRegion() *mappedRegion {
	if a == nil || a.data == nil {
		return nil
	}
	if a.data.mapping != nil {
		return a.data.mapping
	}
	if a.base != nil && a.base.data != nil {
		return a.base.data.mapping
	}
	return nil
}

// MappedArray owns the OS mapping. Close explicitly releases it and invalidates
// all derived views. Do not access any view concurrently with Close or remove/
// truncate the backing file while mapped. Always defer Close after opening.
type MappedArray struct {
	*Array
	region   *mappedRegion
	Filename string
	Offset   int64
	Mode     string
}

func MemMap(path string, opt MemMapOptions) (*MappedArray, error) {
	if !mmapSupported {
		return nil, ErrMappingUnsupported
	}
	path, pathErr := filepath.Abs(path)
	if pathErr != nil {
		return nil, pathErr
	}
	mode := opt.Mode
	if mode == "" {
		mode = "r+"
	}
	order := opt.Order
	if order == "" {
		order = "C"
	}
	if (mode != "r" && mode != "r+" && mode != "w+" && mode != "c") || (order != "C" && order != "F") || opt.Offset < 0 {
		return nil, ErrValue
	}
	width := opt.DType.ItemSize()
	var unit DateUnit
	switch {
	case opt.DType.IsFlexible():
		if opt.Flexible == nil || opt.Flexible.Kind != opt.DType {
			return nil, ErrDType
		}
		var e error
		width, e = opt.Flexible.width()
		if e != nil {
			return nil, e
		}
	case opt.DType == RecordDType:
		if opt.Schema == nil || schemaHasObjects(opt.Schema) {
			return nil, ErrDType
		}
		width = opt.Schema.ItemSize()
	case opt.DType.IsTemporal():
		var e error
		unit, e = ParseDateUnit(opt.DateUnit)
		if e != nil {
			return nil, e
		}
	}
	if width <= 0 || opt.DType == Object || opt.DType == StringDType {
		return nil, ErrDType
	}
	if opt.Flexible != nil && !opt.DType.IsFlexible() || opt.Schema != nil && opt.DType != RecordDType || opt.DateUnit != "" && !opt.DType.IsTemporal() {
		return nil, ErrDType
	}
	if opt.DType.IsExtended() && opt.ByteOrder != nil && opt.ByteOrder.Uint16([]byte{1, 0}) != 1 {
		return nil, ErrDType
	}
	shape := slices.Clone(opt.Shape)
	infer := opt.Shape == nil
	if infer && mode == "w+" {
		return nil, ErrShape
	}
	var size int
	if !infer {
		var e error
		size, e = shapeSizeWidth(width, shape)
		if e != nil {
			return nil, e
		}
		if int64(size*width) > math.MaxInt64-opt.Offset {
			return nil, ErrShape
		}
	}
	flag := os.O_RDONLY
	if mode == "r+" {
		flag = os.O_RDWR
	}
	if mode == "w+" {
		flag = os.O_RDWR | os.O_CREATE | os.O_TRUNC
	}
	f, e := os.OpenFile(path, flag, 0600)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	info, e := f.Stat()
	if e != nil {
		return nil, e
	}
	if infer {
		remaining := info.Size() - opt.Offset
		if remaining < 0 || remaining%int64(width) != 0 || remaining/int64(width) > int64(maxInt)/int64(width) {
			return nil, ErrShape
		}
		size = int(remaining / int64(width))
		shape = []int{size}
	}
	end := opt.Offset + int64(size*width)
	if end > info.Size() {
		if mode == "r" || mode == "c" {
			return nil, ErrValue
		}
		if e = f.Truncate(end); e != nil {
			return nil, e
		}
	}
	region := &mappedRegion{mode: mode, filename: path, offset: opt.Offset}
	buf := []byte{}
	if size > 0 {
		page := int64(os.Getpagesize())
		start := opt.Offset / page * page
		length := end - start
		if length > int64(maxInt) {
			return nil, ErrShape
		}
		region.data, e = mapFile(f, int(length), start, mode)
		if e != nil {
			return nil, e
		}
		buf = region.data[opt.Offset-start:]
	}
	var a *Array
	switch {
	case opt.DType.IsFlexible():
		a, e = FromFlexibleBuffer(buf, *opt.Flexible, []int{size}, mode == "r")
	case opt.DType == RecordDType:
		a, e = FromRecordBuffer(opt.Schema, buf, []int{size}, mode == "r")
	case opt.DType.IsTemporal():
		a, e = FromBuffer(buf, Int64, size, 0, opt.ByteOrder)
		if e == nil {
			a = a.retag(opt.DType, unit)
			a.base = nil
		}
	default:
		a, e = FromBuffer(buf, opt.DType, size, 0, opt.ByteOrder)
	}
	if e != nil {
		_ = unmapBytes(region.data)
		return nil, e
	}
	a.data.mapping = region
	a.shape = shape
	a.strides = contiguousStrides(shape, size)
	if order == "F" {
		stride := 1
		for i, d := range shape {
			a.strides[i] = stride
			stride *= d
		}
	}
	a.owner = false
	a.readOnly = mode == "r"
	return &MappedArray{a, region, path, opt.Offset, mode}, nil
}

func (m *MappedArray) Flush() error {
	return m.region.flush()
}
func (r *mappedRegion) flush() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed.Load() {
		return ErrClosedMapping
	}
	if r.mode == "r" || r.mode == "c" {
		return nil
	}
	return syncMappedBytes(r.data)
}
func (m *MappedArray) Close() error {
	r := m.region
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed.Load() {
		return nil
	}
	if r.mode != "r" && r.mode != "c" {
		if e := syncMappedBytes(r.data); e != nil {
			return e
		}
	}
	r.closed.Store(true)
	if e := unmapBytes(r.data); e != nil {
		r.closed.Store(false)
		return e
	}
	r.data = nil
	return nil
}
