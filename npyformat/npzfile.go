package npyformat

import (
	"archive/zip"
	"bufio"
	"bytes"
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"io"
	"iter"
	"os"
	"slices"
	"strings"
	"sync"
)

// NpzFile is an ordered, lazy NPZ mapping. Opening only reads the ZIP
// directory; individual members are decompressed and decoded on demand.
// Members are never extracted to the filesystem. Close releases an owned
// file; callers retain ownership of ReaderAt values passed to OpenNPZ.
type NpzFile struct {
	mu            sync.RWMutex
	archive       *zip.Reader
	fid           *os.File
	files         []string
	lookup        map[string]*zip.File
	maxHeaderSize int
}
type NPZMember struct {
	Name           string
	Size           uint64
	CompressedSize uint64
	Method         uint16
}
type NPZItem struct {
	Key   string
	Value any // *np.Array for NPY data, []byte for other members
}

func OpenNPZ(r io.ReaderAt, size int64, maxHeaderSize int) (*NpzFile, error) {
	if size < 0 || maxHeaderSize < 0 {
		return nil, np.ErrValue
	}
	z, e := zip.NewReader(r, size)
	if e != nil {
		return nil, e
	}
	f := &NpzFile{archive: z, maxHeaderSize: maxHeaderSize, lookup: map[string]*zip.File{}}
	for _, x := range z.File {
		n := strings.TrimSuffix(x.Name, ".npy")
		f.files = append(f.files, n)
		f.lookup[n] = x
	}
	// Full member names take precedence over stripped aliases.
	for _, x := range z.File {
		f.lookup[x.Name] = x
	}
	return f, nil
}
func OpenNPZPath(path string, maxHeaderSize int) (*NpzFile, error) {
	r, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	s, e := r.Stat()
	if e != nil {
		_ = r.Close()
		return nil, e
	}
	f, e := OpenNPZ(r, s.Size(), maxHeaderSize)
	if e != nil {
		_ = r.Close()
		return nil, e
	}
	f.fid = r
	return f, nil
}
func (f *NpzFile) Files() []string { return slices.Clone(f.files) }
func (f *NpzFile) Keys() []string  { return f.Files() }
func (f *NpzFile) Len() int        { return len(f.files) }

// String displays an archive identity and its first five member names.
func (f *NpzFile) String() string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	filename := "object"
	if f.fid != nil {
		filename = f.fid.Name()
	}
	names := strings.Join(f.files[:min(len(f.files), 5)], ", ")
	if len(f.files) > 5 {
		names += "..."
	}
	filename = strings.NewReplacer("\\", "\\\\", "'", "\\'").Replace(filename)
	return "NpzFile '" + filename + "' with keys: " + names
}
func (f *NpzFile) Contains(key string) bool { _, ok := f.lookup[key]; return ok }
func (f *NpzFile) FID() *os.File {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.fid
}

// Zip returns immutable archive entry metadata rather than an unsafe or
// mutable handle into the open ZIP reader.
func (f *NpzFile) Zip() []NPZMember {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.archive == nil {
		return nil
	}
	out := make([]NPZMember, len(f.archive.File))
	for i, x := range f.archive.File {
		out[i] = NPZMember{x.Name, x.UncompressedSize64, x.CompressedSize64, x.Method}
	}
	return out
}
func (f *NpzFile) At(key string) (any, error) {
	x, ok := f.lookup[key]
	if !ok {
		return nil, fmt.Errorf("%w: %q is not a file in the archive", np.ErrIndex, key)
	}
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.archive == nil {
		return nil, os.ErrClosed
	}
	r, e := x.Open()
	if e != nil {
		return nil, e
	}
	defer r.Close()
	b := bufio.NewReader(r)
	magic, _ := b.Peek(6)
	if bytes.Equal(magic, []byte("\x93NUMPY")) {
		a, e := ReadArray(b, f.maxHeaderSize)
		if e != nil {
			return nil, e
		}
		// The decompressor checks the member CRC when EOF is reached.
		if _, e = io.Copy(io.Discard, b); e != nil {
			return nil, e
		}
		return a, nil
	}
	return io.ReadAll(b)
}
func (f *NpzFile) Get(key string, fallback any) (any, error) {
	if !f.Contains(key) {
		return fallback, nil
	}
	return f.At(key)
}
func (f *NpzFile) Items() iter.Seq2[NPZItem, error] {
	return func(yield func(NPZItem, error) bool) {
		for _, k := range f.files {
			v, e := f.At(k)
			if !yield(NPZItem{k, v}, e) || e != nil {
				return
			}
		}
	}
}
func (f *NpzFile) Values() iter.Seq2[any, error] {
	return func(yield func(any, error) bool) {
		for x, e := range f.Items() {
			if !yield(x.Value, e) || e != nil {
				return
			}
		}
	}
}
func (f *NpzFile) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.archive = nil
	if f.fid == nil {
		return nil
	}
	e := f.fid.Close()
	f.fid = nil
	return e
}
