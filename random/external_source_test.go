package random

import (
	"encoding/binary"
	"errors"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

// This source is implemented outside the library's engines. Its immutable
// raw tape is generated independently by NumPy; the adapter owns its cursor
// and half-word cache, and supplies every bitgen callback.
type tapeSource struct {
	data   []uint64
	pos    int
	cached uint32
	has    bool
}

func (s *tapeSource) Algorithm() string  { return "numpy-reference-tape" }
func (s *tapeSource) NextUint64() uint64 { v := s.data[s.pos]; s.pos++; return v }
func (s *tapeSource) NextUint32() uint32 {
	if s.has {
		s.has = false
		return s.cached
	}
	v := s.NextUint64()
	s.cached = uint32(v >> 32)
	s.has = true
	return uint32(v)
}
func (s *tapeSource) NextDouble() float64 { return float64(s.NextUint64()>>11) * 0x1p-53 }
func (s *tapeSource) NextRaw() uint64     { return s.NextUint64() }
func (s *tapeSource) Snapshot() string {
	var b [13]byte
	binary.LittleEndian.PutUint64(b[:8], uint64(s.pos))
	binary.LittleEndian.PutUint32(b[8:12], s.cached)
	if s.has {
		b[12] = 1
	}
	return string(b[:])
}
func (s *tapeSource) Restore(state string) error {
	if len(state) != 13 {
		return np.ErrValue
	}
	b := []byte(state)
	p := binary.LittleEndian.Uint64(b[:8])
	if p > uint64(len(s.data)) || b[12] > 1 {
		return np.ErrValue
	}
	s.pos = int(p)
	s.cached = binary.LittleEndian.Uint32(b[8:12])
	s.has = b[12] == 1
	return nil
}
func TestExternalBitSourceNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mg := refcheck.RequireValue[*Generator](t)
	mr := refcheck.RequireValue[*RandomState](t)
	tape := refcheck.RequireValue[[]uint64](t)(np.Values[uint64](refcheck.Input(t, "external_rng_tape")))
	bit := mg(NewExternalBitGenerator(&tapeSource{data: tape}))
	g := mg(NewGeneratorFromBitGenerator(bit))
	r := mr(NewRandomStateFromBitGenerator(bit))
	r2 := mr(NewRandomStateFromBitGenerator(bit))
	check := func(name string, a *np.Array) { t.Helper(); refcheck.Check(t, "external_rng_"+name, a) }
	check("uint32", m(g.IntegersDType(0, 123456, np.Uint32, false, 3)))
	check("legacy", m(r.StandardNormal(1)))
	state := r.GetState()
	if !state.Stream.External || state.Stream.Algorithm != "numpy-reference-tape" {
		t.Fatal("external snapshot lost identity")
	}
	check("modern", m(g.Random(3)))
	check("raw", m(bit.RandomRaw(5)))
	check("other_legacy", m(r2.StandardNormal(1)))
	check("cached", m(r.StandardNormal(1)))
	check("integer", m(r.Randint(-15, 21, np.Int8, 7)))
	check("normal", m(g.StandardNormal(4)))
	check("other_cache", m(r2.StandardNormal(1)))
	if e := r.SetState(state); e != nil {
		t.Fatal(e)
	}
	check("replay_modern", m(g.Random(3)))
	check("replay_raw", m(bit.RandomRaw(5)))
	check("replay_cache", m(r.StandardNormal(1)))
	before := g.State()
	invalid := before
	invalid.ExternalState = "bad"
	if e := g.SetState(invalid); !errors.Is(e, np.ErrValue) {
		t.Fatalf("invalid external state: %v", e)
	}
	if e := r.Seed(1); !errors.Is(e, np.ErrDType) {
		t.Fatalf("external legacy reseed: %v", e)
	}
	if e := g.Advance(1); !errors.Is(e, np.ErrValue) {
		t.Fatalf("external advance: %v", e)
	}
	if _, e := g.Jumped(1); !errors.Is(e, np.ErrValue) {
		t.Fatalf("external jump: %v", e)
	}
	if _, e := g.Spawn(1); !errors.Is(e, np.ErrValue) {
		t.Fatalf("external spawn: %v", e)
	}
	if g.State() != before {
		t.Fatal("rejected operations changed external stream")
	}
}

type chaChaSource struct{ rng *rand.ChaCha8 }

type capabilityTape struct {
	tapeSource
	jump     []uint64
	children [][]uint64
}

func (s *capabilityTape) Advance256(step [4]uint64) error {
	if step[1] != 0 || step[2] != 0 || step[3] != 0 || step[0] > uint64(len(s.data)-s.pos) {
		return np.ErrValue
	}
	s.pos += int(step[0])
	s.has = false
	s.cached = 0
	return nil
}
func (s *capabilityTape) Jumped(n uint64) (ExternalBitSource, error) {
	if n != 2 || s.pos != 0 {
		return nil, np.ErrValue
	}
	return &tapeSource{data: s.jump}, nil
}
func (s *capabilityTape) Spawn(n int) ([]ExternalBitSource, error) {
	if n < 0 || n > len(s.children) {
		return nil, np.ErrValue
	}
	out := make([]ExternalBitSource, n)
	for i := range out {
		out[i] = &tapeSource{data: s.children[i]}
	}
	s.children = s.children[n:]
	return out, nil
}
func TestExternalCapabilitiesNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mg := refcheck.RequireValue[*Generator](t)
	tape := func(n string) []uint64 {
		return refcheck.RequireValue[[]uint64](t)(np.Values[uint64](refcheck.Input(t, "external_caps_"+n)))
	}
	source := func() *capabilityTape {
		return &capabilityTape{tapeSource: tapeSource{data: tape("tape")}, jump: tape("jump"), children: [][]uint64{tape("spawn0"), tape("spawn1")}}
	}
	g := mg(NewExternalBitGenerator(source()))
	if e := g.Advance(17); e != nil {
		t.Fatal(e)
	}
	refcheck.Check(t, "external_caps_advance", m(g.RandomRaw(5)))
	before := g.State()
	if e := g.Advance128(1, 0); !errors.Is(e, np.ErrValue) {
		t.Fatal(e)
	}
	if g.State() != before {
		t.Fatal("failed advance mutated state")
	}
	g = mg(NewExternalBitGenerator(source()))
	before = g.State()
	j := mg(g.Jumped(2))
	refcheck.Check(t, "external_caps_jump", m(j.RandomRaw(6)))
	if g.State() != before {
		t.Fatal("jump consumed parent")
	}
	refcheck.Check(t, "external_caps_parent", m(g.RandomRaw(4)))
	children := refcheck.RequireValue[[]*Generator](t)(g.Spawn(2))
	for i, c := range children {
		refcheck.Check(t, "external_caps_spawn"+string(rune('0'+i)), m(c.RandomRaw(8)))
	}
	if _, e := g.Spawn(1); !errors.Is(e, np.ErrValue) {
		t.Fatal(e)
	}
	if _, e := g.Spawn(-1); !errors.Is(e, np.ErrValue) {
		t.Fatal(e)
	}
}

// Returning an already-owned mutable source would create a second mutex
// protecting the same state. Reject this malformed capability result.
type aliasCapability struct{ tapeSource }

func (s *aliasCapability) Jumped(uint64) (ExternalBitSource, error) { return s, nil }
func (s *aliasCapability) Spawn(n int) ([]ExternalBitSource, error) {
	child := &tapeSource{data: s.data}
	return []ExternalBitSource{child, child}, nil
}
func TestExternalChildIndependence(t *testing.T) {
	s := &aliasCapability{tapeSource{data: []uint64{1, 2}}}
	g := refcheck.RequireValue[*Generator](t)(NewExternalBitGenerator(s))
	if _, e := g.Jumped(1); !errors.Is(e, np.ErrValue) {
		t.Fatal(e)
	}
	if _, e := g.Spawn(2); !errors.Is(e, np.ErrValue) {
		t.Fatal(e)
	}
}

func (s *chaChaSource) Algorithm() string   { return "Go-ChaCha8" }
func (s *chaChaSource) NextUint64() uint64  { return s.rng.Uint64() }
func (s *chaChaSource) NextUint32() uint32  { return uint32(s.NextUint64()) }
func (s *chaChaSource) NextDouble() float64 { return float64(s.NextUint64()>>11) * 0x1p-53 }
func (s *chaChaSource) NextRaw() uint64     { return s.NextUint64() ^ 0xa5 }
func (s *chaChaSource) Snapshot() string {
	b, e := s.rng.MarshalBinary()
	if e != nil {
		panic(e)
	}
	return string(b)
}
func (s *chaChaSource) Restore(state string) error {
	r := new(rand.ChaCha8)
	if e := r.UnmarshalBinary([]byte(state)); e != nil {
		return e
	}
	s.rng = r
	return nil
}
func TestIndependentExternalCallbacks(t *testing.T) {
	m := refcheck.Require(t)
	mg := refcheck.RequireValue[*Generator](t)
	var seed [32]byte
	seed[0] = 41
	bit := mg(NewExternalBitGenerator(&chaChaSource{rand.NewChaCha8(seed)}))
	other := mg(NewGeneratorFromBitGenerator(bit))
	oracle := rand.NewChaCha8(seed)
	saved := bit.State()
	if e := bit.WithLock(func(s *LockedStream) error {
		if s.Uint32() != uint32(oracle.Uint64()) || s.Uint64() != oracle.Uint64() || s.Double() != float64(oracle.Uint64()>>11)*0x1p-53 || s.Raw() != oracle.Uint64()^0xa5 {
			t.Fatal("raw callback routed incorrectly")
		}
		return nil
	}); e != nil {
		t.Fatal(e)
	}
	if e := other.SetState(saved); e != nil {
		t.Fatal(e)
	}
	oracle = rand.NewChaCha8(seed)
	raw := refcheck.RequireValue[[]uint64](t)(np.Values[uint64](m(bit.RandomRaw(3))))
	for _, v := range raw {
		if v != oracle.Uint64()^0xa5 {
			t.Fatal("raw draw didn't use NextRaw")
		}
	}
	// The two consumers serialize complete draw blocks, not just each word.
	expected := [2][2]uint64{{oracle.Uint64(), oracle.Uint64()}, {oracle.Uint64(), oracle.Uint64()}}
	var wg sync.WaitGroup
	out := make(chan [2]uint64, 2)
	for _, g := range []*Generator{bit, other} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var p [2]uint64
			e := g.WithLock(func(s *LockedStream) error { p = [2]uint64{s.Uint64(), s.Uint64()}; return nil })
			if e != nil {
				t.Error(e)
			}
			out <- p
		}()
	}
	wg.Wait()
	close(out)
	a, b := <-out, <-out
	if !(a == expected[0] && b == expected[1] || a == expected[1] && b == expected[0]) {
		t.Fatal("external block lock interleaved")
	}
}
