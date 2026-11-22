package npyformat

import (
	"compress/bzip2"
	"compress/gzip"
	"errors"
	"github.com/SamuelSupe/numpygo"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// DataSource locates local numerical data files (including compressed
// variants) and explicitly requested HTTP(S) data cached under its directory.
// An empty destination creates a private temporary cache removed by Close.
type DataSource struct {
	dest   string
	temp   bool
	client *http.Client
	mu     sync.Mutex
	closed bool
}

func NewDataSource(dest string) (*DataSource, error) {
	tmp := dest == ""
	var e error
	if tmp {
		dest, e = os.MkdirTemp("", "numpygo-data-")
	} else {
		dest, e = filepath.Abs(dest)
	}
	if e != nil {
		return nil, e
	}
	return &DataSource{dest: dest, temp: tmp, client: &http.Client{Timeout: 30 * time.Second}}, nil
}
func (d *DataSource) Destination() string { return d.dest }
func (d *DataSource) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	d.closed = true
	if d.temp {
		return os.RemoveAll(d.dest)
	}
	return nil
}
func remoteURL(path string) (*url.URL, bool) {
	u, e := url.Parse(path)
	return u, e == nil && u.Scheme != "" && u.Host != ""
}
func cleanRelative(s string) string {
	s = filepath.Clean(filepath.FromSlash(s))
	for {
		v := strings.TrimLeft(s, "/\\.")
		if v == s {
			break
		}
		s = v
	}
	return s
}

// AbsPath returns the local cache path, regardless of whether the requested
// file is already cached. Query/fragment components do not enter the path.
func (d *DataSource) AbsPath(path string) (string, error) {
	if filepath.IsAbs(path) {
		if rel, e := filepath.Rel(d.dest, path); e == nil && filepath.IsLocal(rel) {
			return filepath.Join(d.dest, rel), nil
		}
	}
	u, e := url.Parse(path)
	if e != nil {
		return "", np.ErrValue
	}
	host := cleanRelative(u.Host)
	name := u.Path
	if u.Host != "" {
		name = u.EscapedPath()
	} else if u.RawPath != "" {
		name = u.RawPath
	}
	name = cleanRelative(name)
	rel := filepath.Join(host, name)
	if rel == "." {
		return d.dest, nil
	}
	if !filepath.IsLocal(rel) {
		return "", np.ErrValue
	}
	return filepath.Join(d.dest, rel), nil
}
func fileExists(path string) bool { _, e := os.Stat(path); return e == nil }
func (d *DataSource) Exists(path string) (bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return false, os.ErrClosed
	}
	return d.exists(path)
}
func (d *DataSource) exists(path string) (bool, error) {
	if fileExists(path) {
		return true, nil
	}
	cache, e := d.AbsPath(path)
	if e != nil {
		return false, e
	}
	if fileExists(cache) {
		return true, nil
	}
	u, remote := remoteURL(path)
	if !remote {
		return false, nil
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return false, nil
	}
	r, e := d.client.Get(path)
	if e != nil {
		return false, nil
	}
	defer r.Body.Close()
	return r.StatusCode >= 200 && r.StatusCode < 300, nil
}

var compressedSuffixes = []string{".bz2", ".gz", ".xz", ".lzma"}

func compression(path string) string {
	for _, s := range compressedSuffixes {
		if strings.HasSuffix(path, s) {
			return s
		}
	}
	return ""
}
func possibleNames(path string) []string {
	out := []string{path}
	if compression(path) == "" {
		for _, s := range compressedSuffixes {
			out = append(out, path+s)
		}
	}
	return out
}
func (d *DataSource) cache(path string) (string, error) {
	local, e := d.AbsPath(path)
	if e != nil {
		return "", e
	}
	rel, e := filepath.Rel(d.dest, local)
	if e != nil || !filepath.IsLocal(rel) || rel == "." {
		return "", np.ErrValue
	}
	if e = os.MkdirAll(d.dest, 0700); e != nil {
		return "", e
	}
	root, e := os.OpenRoot(d.dest)
	if e != nil {
		return "", e
	}
	defer root.Close()
	// Stay inside the destination even if a cache directory contains a
	// symlink. Root.MkdirAll is newer than the module's Go 1.24 baseline.
	dir := ""
	for _, p := range strings.Split(filepath.Dir(rel), string(filepath.Separator)) {
		if p == "." || p == "" {
			continue
		}
		dir = filepath.Join(dir, p)
		if e = root.Mkdir(dir, 0700); e != nil && !errors.Is(e, os.ErrExist) {
			return "", e
		}
	}
	r, e := d.client.Get(path)
	if e != nil {
		return "", e
	}
	defer r.Body.Close()
	if r.StatusCode < 200 || r.StatusCode >= 300 {
		return "", os.ErrNotExist
	}
	f, e := root.OpenFile(rel, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if errors.Is(e, os.ErrExist) {
		return local, nil
	}
	if e != nil {
		return "", e
	}
	_, e = io.Copy(f, r.Body)
	e = errors.Join(e, f.Close())
	if e != nil {
		_ = root.Remove(rel)
		return "", e
	}
	return local, nil
}

type dataStream struct {
	reader io.Reader
	writer io.Writer
	close  func() error
}

func (s *dataStream) Read(p []byte) (int, error) {
	if s.reader == nil {
		return 0, os.ErrPermission
	}
	return s.reader.Read(p)
}
func (s *dataStream) Write(p []byte) (int, error) {
	if s.writer == nil {
		return 0, np.ErrReadOnly
	}
	return s.writer.Write(p)
}
func (s *dataStream) Close() error { return s.close() }

// Open opens existing data in r/w/a modes (+ for uncompressed data).
// Gzip supports reading/writing; the standard-library bzip2 reader is read
// only. XZ/LZMA and Python text encoding/newline options are not implemented.
// HTTP(S) URLs are downloaded on first read, never opened for writing.
func (d *DataSource) Open(path, mode string) (io.ReadWriteCloser, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, os.ErrClosed
	}
	if mode == "" {
		mode = "r"
	}
	mode = strings.NewReplacer("b", "", "t", "").Replace(mode)
	flag := 0
	switch mode {
	case "r":
		flag = os.O_RDONLY
	case "r+":
		flag = os.O_RDWR
	case "w":
		flag = os.O_WRONLY | os.O_TRUNC
	case "w+":
		flag = os.O_RDWR | os.O_TRUNC
	case "a":
		flag = os.O_WRONLY | os.O_APPEND
	case "a+":
		flag = os.O_RDWR | os.O_APPEND
	default:
		return nil, np.ErrValue
	}
	u, remote := remoteURL(path)
	if remote && (mode != "r" || u.Scheme != "http" && u.Scheme != "https") {
		return nil, np.ErrValue
	}
	cache, e := d.AbsPath(path)
	if e != nil {
		return nil, e
	}
	localCandidates := possibleNames(path)
	var candidates []string
	if remote {
		candidates = append(possibleNames(cache), localCandidates...)
	} else {
		candidates = append(localCandidates, possibleNames(cache)...)
	}
	found := ""
	inCache := false
	for i, p := range candidates {
		if _, r := remoteURL(p); r {
			if yes, _ := d.exists(p); !yes {
				continue
			}
			found, e = d.cache(p)
			if e != nil {
				return nil, e
			}
			inCache = true
			break
		}
		if fileExists(p) {
			found = p
			inCache = remote || i >= len(localCandidates)
			break
		}
	}
	if found == "" {
		return nil, os.ErrNotExist
	}
	kind := compression(found)
	if kind == ".xz" || kind == ".lzma" || kind == ".bz2" && mode != "r" || kind == ".gz" && strings.Contains(mode, "+") {
		return nil, np.ErrValue
	}
	var f *os.File
	if inCache {
		rel, e := filepath.Rel(d.dest, found)
		if e != nil || !filepath.IsLocal(rel) {
			return nil, np.ErrValue
		}
		root, e := os.OpenRoot(d.dest)
		if e != nil {
			return nil, e
		}
		f, e = root.OpenFile(rel, flag, 0)
		_ = root.Close()
		if e != nil {
			return nil, e
		}
	} else {
		f, e = os.OpenFile(found, flag, 0)
	}
	if e != nil {
		return nil, e
	}
	switch kind {
	case ".gz":
		if mode == "r" {
			z, e := gzip.NewReader(f)
			if e != nil {
				_ = f.Close()
				return nil, e
			}
			return &dataStream{reader: z, close: func() error { return errors.Join(z.Close(), f.Close()) }}, nil
		}
		z := gzip.NewWriter(f)
		return &dataStream{writer: z, close: func() error { return errors.Join(z.Close(), f.Close()) }}, nil
	case ".bz2":
		return &dataStream{reader: bzip2.NewReader(f), close: f.Close}, nil
	default:
		return f, nil
	}
}
