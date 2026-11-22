package dtypes

import (
	"github.com/SamuelSupe/numpygo"
	"sync"
)

type DLPackKey struct{ Code, Bits int }

// DLPackRegistry is an isolated pure-Go equivalent of the dtype registration
// table. It only describes Go-managed storage, not C capsules or uintptrs.
// Built-in import mappings take priority over registered mappings.
type DLPackRegistry struct {
	mu      sync.RWMutex
	imports map[DLPackKey]Descriptor
	exports map[string]DLPackKey
}

func NewDLPackRegistry() *DLPackRegistry {
	return &DLPackRegistry{imports: make(map[DLPackKey]Descriptor), exports: make(map[string]DLPackKey)}
}
func registryKey(d Descriptor) string { return d.Metadata().Class + ":" + d.String() }
func (r *DLPackRegistry) RegisterDLPackDType(key DLPackKey, d Descriptor) error {
	t := d.DType()
	if t == 0 || t == np.Object || t == np.StringDType || t == np.RecordDType {
		return np.ErrDType
	}
	if key.Code < 0 || key.Code > 255 || key.Bits <= 0 || key.Bits > 255 || d.ItemSize()*8 != key.Bits {
		return np.ErrValue
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.imports == nil {
		r.imports = make(map[DLPackKey]Descriptor)
		r.exports = make(map[string]DLPackKey)
	}
	name := registryKey(d)
	if old, ok := r.exports[name]; ok && old != key {
		return np.ErrValue
	}
	r.exports[name] = key
	if old, ok := r.imports[key]; ok && registryKey(old) != name {
		// NumPy registers the new export even when an import conflicts.
		return np.ErrValue
	}
	r.imports[key] = d
	return nil
}
func defaultDLPack(t np.DType) (DLPackKey, bool) {
	code := 0
	switch {
	case t.IsSigned():
		code = 0
	case t.IsUnsigned():
		code = 1
	case t == np.Float16 || t == np.Float32 || t == np.Float64:
		code = 2
	case t == np.Complex64 || t == np.Complex128:
		code = 5
	case t == np.Bool:
		code = 6
	default:
		return DLPackKey{}, false
	}
	return DLPackKey{code, t.ItemSize() * 8}, true
}
func (r *DLPackRegistry) Import(key DLPackKey) (Descriptor, error) {
	for _, v := range definitions {
		if k, ok := defaultDLPack(v.d); ok && k == key {
			return New(classFor(v.d), Options{})
		}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	d, ok := r.imports[key]
	if !ok {
		return Descriptor{}, np.ErrDType
	}
	return d, nil
}
func (r *DLPackRegistry) Export(d Descriptor) (DLPackKey, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if k, ok := r.exports[registryKey(d)]; ok {
		return k, nil
	}
	k, ok := defaultDLPack(d.DType())
	if !ok {
		return DLPackKey{}, np.ErrDType
	}
	return k, nil
}
