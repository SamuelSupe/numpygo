package random

import (
	"github.com/SamuelSupe/numpygo"
	"reflect"
)

// BitGenerator is the common Go contract implemented by all five streams.
type BitGenerator interface {
	Algorithm() string
	RandomRaw(...int) (*np.Array, error)
	State() State
	SetState(State) error
	WithLock(func(*LockedStream) error) error
}

var _ BitGenerator = (*Generator)(nil)

// ExternalBitSource is the pure Go counterpart of NumPy's bitgen callback
// table. Implementations own their raw algorithm state and any cached
// uint32 words. The wrapper serializes every callback under one mutex;
// callers must not use the source directly after handing it to the wrapper.
// Snapshot returns a self-contained immutable binary string. Restore must
// validate the snapshot and leave the source unchanged on error.
type ExternalBitSource interface {
	Algorithm() string
	NextUint32() uint32
	NextUint64() uint64
	NextDouble() float64 // [0,1), as required by the numerical distributions
	NextRaw() uint64
	Snapshot() string
	Restore(string) error
}

// Optional capabilities keep algorithm-specific jumps and child derivation
// with the source. Callbacks execute under the parent's lock. Advance256
// and Spawn must leave the source unchanged on error; Jumped never mutates
// the parent. Returned sources must be independent and use the same algorithm.
type ExternalAdvanceSource interface {
	Advance256([4]uint64) error
}
type ExternalJumpSource interface {
	Jumped(uint64) (ExternalBitSource, error)
}
type ExternalSpawnSource interface {
	Spawn(int) ([]ExternalBitSource, error)
}

func (g *Generator) externalChild(source ExternalBitSource) (*Generator, error) {
	child, err := NewExternalBitGenerator(source)
	if err != nil {
		return nil, err
	}
	if child.algorithm != g.algorithm || sameExternalSource(source, g.external) {
		return nil, np.ErrValue
	}
	return child, nil
}

func sameExternalSource(a, b ExternalBitSource) bool {
	return a != nil && reflect.TypeOf(a) == reflect.TypeOf(b) &&
		reflect.ValueOf(a).Comparable() && a == b
}

// NewExternalBitGenerator makes an independent external algorithm available
// to both Generator and RandomState using the same state and mutex.
func NewExternalBitGenerator(source ExternalBitSource) (*Generator, error) {
	if source == nil {
		return nil, np.ErrDType
	}
	v := reflect.ValueOf(source)
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan:
		if v.IsNil() {
			return nil, np.ErrDType
		}
	}
	name := source.Algorithm()
	switch name {
	case "", "PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64":
		return nil, np.ErrValue
	}
	return &Generator{bitStream: &bitStream{algorithm: name, external: source}}, nil
}

func NewBitGenerator(name string, seed uint64) (*Generator, error) {
	switch name {
	case "MT19937":
		return NewMT19937(seed), nil
	case "PCG64":
		return NewPCG64(seed), nil
	case "PCG64DXSM":
		return NewPCG64DXSM(seed), nil
	case "Philox":
		return NewPhilox(seed), nil
	case "SFC64":
		return NewSFC64(seed), nil
	default:
		return nil, np.ErrValue
	}
}

// RawHandle retains the generator and permits synchronized raw draws. It is
// the Go equivalent of the capsule, ctypes and cffi interfaces; no pointer
// escapes to an external runtime or bypasses the generator's lock.
type RawHandle struct{ generator *Generator }

func (g *Generator) Capsule() *RawHandle { return &RawHandle{g} }
func (g *Generator) CTypes() *RawHandle  { return &RawHandle{g} }
func (g *Generator) CFFI() *RawHandle    { return &RawHandle{g} }
func (h *RawHandle) Algorithm() string   { return h.generator.Algorithm() }
func (h *RawHandle) RandomRaw(shape ...int) (*np.Array, error) {
	return h.generator.RandomRaw(shape...)
}
func (h *RawHandle) WithLock(fn func(*LockedStream) error) error { return h.generator.WithLock(fn) }
func (h *RawHandle) State() State                                { return h.generator.State() }
