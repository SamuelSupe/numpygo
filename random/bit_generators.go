package random

import (
	"math/bits"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

// MTState stores the full MT19937 key and current position.
type MTState struct {
	Key [624]uint32
	Pos int
}

func (s *MTState) seed(seed uint32) {
	for i := range s.Key {
		s.Key[i] = seed
		seed = 1812433253*(seed^(seed>>30)) + uint32(i+1)
	}
	s.Pos = 624
}
func (s *MTState) seedArray(words []uint32) {
	s.seed(19650218)
	i, j := 1, 0
	for k := max(624, len(words)); k > 0; k-- {
		p := s.Key[i-1]
		s.Key[i] = (s.Key[i] ^ ((p ^ (p >> 30)) * 1664525)) + words[j] + uint32(j)
		i++
		j++
		if i == 624 {
			s.Key[0] = s.Key[623]
			i = 1
		}
		if j == len(words) {
			j = 0
		}
	}
	for k := 623; k > 0; k-- {
		p := s.Key[i-1]
		s.Key[i] = (s.Key[i] ^ ((p ^ (p >> 30)) * 1566083941)) - uint32(i)
		i++
		if i == 624 {
			s.Key[0] = s.Key[623]
			i = 1
		}
	}
	s.Key[0] = 0x80000000
}

// twist updates one state word with the circular recurrence.
func (s *MTState) twist(i int) {
	y := s.Key[i]&0x80000000 | s.Key[(i+1)%624]&0x7fffffff
	s.Key[i] = s.Key[(i+397)%624] ^ (y >> 1) ^ (-(y & 1) & 0x9908b0df)
}
func (s *MTState) next() uint32 {
	if s.Pos == 624 {
		// The circular recurrence has two contiguous spans and one wrap
		// element. Refill them without modulo in each generated word.
		for i := 0; i < 227; i++ {
			y := s.Key[i]&0x80000000 | s.Key[i+1]&0x7fffffff
			s.Key[i] = s.Key[i+397] ^ (y >> 1) ^ (-(y & 1) & 0x9908b0df)
		}
		for i := 227; i < 623; i++ {
			y := s.Key[i]&0x80000000 | s.Key[i+1]&0x7fffffff
			s.Key[i] = s.Key[i-227] ^ (y >> 1) ^ (-(y & 1) & 0x9908b0df)
		}
		s.twist(623)
		s.Pos = 0
	}
	y := s.Key[s.Pos]
	s.Pos++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	return y ^ (y >> 18)
}

func NewMT19937(seed uint64) *Generator { return MT19937FromSeedSequence(Seed(seed)) }
func MT19937FromSeedSequence(seed *SeedSequence) *Generator {
	w, _ := seed.GenerateState32(624)
	s := &MTState{Pos: 623}
	copy(s.Key[:], w)
	s.Key[0] = 0x80000000
	return &Generator{bitStream: &bitStream{seed: seed, algorithm: "MT19937", mt: s}}
}

// PhiloxState includes the complete block counter, key and buffered block.
type PhiloxState struct {
	Counter   [4]uint64
	Key       [2]uint64
	Buffer    [4]uint64
	BufferPos int
}

func (s *PhiloxState) advance(step [4]uint64) {
	carry := uint64(0)
	for i, v := range step {
		s.Counter[i], carry = bits.Add64(s.Counter[i], v, carry)
	}
	s.Buffer = [4]uint64{}
	s.BufferPos = 4
}
func (s *PhiloxState) next() uint64 {
	if s.BufferPos < 4 {
		v := s.Buffer[s.BufferPos]
		s.BufferPos++
		return v
	}
	// A block refill advances the counter once, and immediately replaces
	// the buffer. Stop at the first non-wrapping limb.
	for i := range s.Counter {
		s.Counter[i]++
		if s.Counter[i] != 0 {
			break
		}
	}
	c0, c1, c2, c3 := s.Counter[0], s.Counter[1], s.Counter[2], s.Counter[3]
	k0, k1 := s.Key[0], s.Key[1]
	for i := 0; i < 10; i++ {
		h0, l0 := bits.Mul64(0xd2e7470ee14c6c93, c0)
		h1, l1 := bits.Mul64(0xca5a826395121157, c2)
		c0, c1, c2, c3 = h1^c1^k0, l1, h0^c3^k1, l0
		k0 += 0x9e3779b97f4a7c15
		k1 += 0xbb67ae8584caa73b
	}
	s.Buffer = [4]uint64{c0, c1, c2, c3}
	s.BufferPos = 1
	return c0
}

func NewPhilox(seed uint64) *Generator { return PhiloxFromSeedSequence(Seed(seed)) }
func PhiloxFromSeedSequence(seed *SeedSequence) *Generator {
	words, _ := seed.GenerateState64(2)
	return &Generator{bitStream: &bitStream{seed: seed, algorithm: "Philox", philox: &PhiloxState{Key: [2]uint64{words[0], words[1]}, BufferPos: 4}}}
}

// PhiloxFromKey constructs a stream with an explicit counter and key, bypassing SeedSequence.
func PhiloxFromKey(key [2]uint64, counter [4]uint64) *Generator {
	return &Generator{bitStream: &bitStream{algorithm: "Philox", philox: &PhiloxState{Key: key, Counter: counter, BufferPos: 4}}}
}

func sfcNext(s *[4]uint64) uint64 {
	v := s[0] + s[1] + s[3]
	s[3]++
	s[0] = s[1] ^ (s[1] >> 11)
	s[1] = s[2] + (s[2] << 3)
	s[2] = bits.RotateLeft64(s[2], 24) + v
	return v
}
func NewSFC64(seed uint64) *Generator { return SFC64FromSeedSequence(Seed(seed)) }
func SFC64FromSeedSequence(seed *SeedSequence) *Generator {
	words, _ := seed.GenerateState64(3)
	s := &[4]uint64{words[0], words[1], words[2], 1}
	for i := 0; i < 12; i++ {
		sfcNext(s)
	}
	return &Generator{bitStream: &bitStream{seed: seed, algorithm: "SFC64", sfc: s}}
}

func (g *Generator) Algorithm() string {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.algorithm != "" {
		return g.algorithm
	}
	if g.dxsm {
		return "PCG64DXSM"
	}
	return "PCG64"
}
func (g *Generator) SeedSequence() *SeedSequence {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.seed
}

// LockedStream exposes raw draws during a Generator.WithLock callback.
// It must not be retained or shared beyond that callback.
type LockedStream struct{ g *Generator }

func (s *LockedStream) Uint32() uint32  { return s.g.next32() }
func (s *LockedStream) Uint64() uint64  { return s.g.next64() }
func (s *LockedStream) Double() float64 { return s.g.double() }
func (s *LockedStream) Raw() uint64     { return s.g.nextRaw() }

// WithLock serializes an entire sequence of low-level draws, corresponding
// to NumPy's BitGenerator.lock without exporting a Python/C pointer.
func (g *Generator) WithLock(fn func(*LockedStream) error) error {
	if fn == nil {
		return np.ErrValue
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	s := &LockedStream{g: g}
	defer func() { s.g = nil }()
	return fn(s)
}

// Jumped returns an independent advanced stream without consuming the original.
// MT19937 jumps by 2^128 raw uint32 draws each time; Philox and PCG use
// NumPy's generator-specific counter/LCG jump sizes.
func (g *Generator) Jumped(jumps uint64) (*Generator, error) {
	g.mu.Lock()
	if g.external != nil {
		defer g.mu.Unlock()
		if source, ok := g.external.(ExternalJumpSource); ok {
			child, err := source.Jumped(jumps)
			if err != nil {
				return nil, err
			}
			return g.externalChild(child)
		}
		return nil, np.ErrValue
	}
	state, seed := g.snapshot(), g.seed
	g.mu.Unlock()
	out := &Generator{bitStream: &bitStream{seed: seed, algorithm: state.Algorithm, dxsm: state.DXSM}}
	switch state.Algorithm {
	case "SFC64":
		return nil, np.ErrValue
	case "MT19937":
		out.mt = new(MTState)
	case "Philox":
		out.philox = new(PhiloxState)
	}
	if err := out.restore(state); err != nil {
		return nil, err
	}
	switch state.Algorithm {
	case "MT19937":
		for i := uint64(0); i < jumps; i++ {
			out.mt.jump()
		}
	case "Philox":
		return out, out.Advance256([4]uint64{0, 0, jumps, 0})
	default:
		d := u128{0x9e3779b97f4a7c15, 0xf39cc0605cedc835}.mul(u128{0, jumps})
		return out, out.Advance128(d.hi, d.lo)
	}
	return out, nil
}

// SeedSequence metadata accessors return independent copies.
func (s *SeedSequence) Entropy() []uint32  { return slices.Clone(s.entropy) }
func (s *SeedSequence) SpawnKey() []uint32 { return slices.Clone(s.spawn) }

// Pool retains the original four-word accessor; PoolWords returns all words
// for a configurable entropy pool.
func (s *SeedSequence) Pool() [4]uint32          { var p [4]uint32; copy(p[:], s.pool); return p }
func (s *SeedSequence) PoolWords() []uint32      { return slices.Clone(s.pool) }
func (s *SeedSequence) PoolSize() int            { return len(s.pool) }
func (s *SeedSequence) NChildrenSpawned() uint64 { s.mu.Lock(); defer s.mu.Unlock(); return s.children }

type SeedSequenceState struct {
	Entropy, SpawnKey []uint32
	PoolSize          int
	NChildrenSpawned  uint64
}

func (s *SeedSequence) State() SeedSequenceState {
	return SeedSequenceState{s.Entropy(), s.SpawnKey(), s.PoolSize(), s.NChildrenSpawned()}
}

func (s *MTState) jump() {
	if s.Pos >= 624 {
		s.Pos = 0
	}
	original, tmp := *s, *s
	coef := func(i int) bool { return mtJumpPolynomial[i/32]&(uint32(1)<<uint(i%32)) != 0 }
	i := 19936
	for !coef(i) {
		i--
	}
	next := func() { tmp.twist(tmp.Pos); tmp.Pos = (tmp.Pos + 1) % 624 }
	xor := func() {
		for j := 0; j < 624; j++ {
			tmp.Key[(j+tmp.Pos)%624] ^= original.Key[(j+original.Pos)%624]
		}
	}
	if i > 0 {
		next()
		for i--; i > 0; i-- {
			if coef(i) {
				xor()
			}
			next()
		}
		if coef(0) {
			xor()
		}
	}
	*s = tmp
}
