// Package random implements NumPy-style seeded random generators.
package random

import (
	"fmt"
	"math/bits"
	"slices"
	"sync"

	"github.com/SamuelSupe/numpygo"
)

// SeedSequence reproduces NumPy's entropy pool. Seeds are
// assembled as little-endian uint32 words. Spawn derives independent children.
type SeedSequence struct {
	entropy  []uint32
	spawn    []uint32
	pool     []uint32
	mu       sync.Mutex
	children uint64
}

func Seed(seed uint64) *SeedSequence {
	words := []uint32{uint32(seed)}
	if seed>>32 != 0 {
		words = append(words, uint32(seed>>32))
	}
	return NewSeedSequence(words...)
}

func NewSeedSequence(words ...uint32) *SeedSequence { return seedWithKey(words, nil, 4) }

// SeedSequenceFromState restores entropy, spawn key, pool size and the
// number of already-spawned children without consuming random bits.
func SeedSequenceFromState(state SeedSequenceState) (*SeedSequence, error) {
	if state.PoolSize < 4 || state.NChildrenSpawned > 1<<32 {
		return nil, np.ErrValue
	}
	if _, err := np.ElementCount(np.Uint32, state.PoolSize); err != nil {
		return nil, err
	}
	s := seedWithKey(state.Entropy, state.SpawnKey, state.PoolSize)
	s.children = state.NChildrenSpawned
	return s, nil
}

func seedWithKey(words, key []uint32, poolSize int) *SeedSequence {
	s := &SeedSequence{entropy: slices.Clone(words), spawn: slices.Clone(key), pool: make([]uint32, poolSize)}
	entropy := slices.Clone(words)
	if len(key) > 0 && len(entropy) < poolSize {
		entropy = append(entropy, make([]uint32, poolSize-len(entropy))...)
	}
	entropy = append(entropy, key...)
	hash := uint32(0x43b0d7e5)
	hashmix := func(x uint32) uint32 { x ^= hash; hash *= 0x931e8875; x *= hash; return x ^ (x >> 16) }
	mix := func(x, y uint32) uint32 { z := uint32(0xca01f9dd)*x - uint32(0x4973f715)*y; return z ^ (z >> 16) }
	for i := range s.pool {
		var x uint32
		if i < len(entropy) {
			x = entropy[i]
		}
		s.pool[i] = hashmix(x)
	}
	for i := range s.pool {
		for j := range s.pool {
			if i != j {
				s.pool[j] = mix(s.pool[j], hashmix(s.pool[i]))
			}
		}
	}
	for _, x := range entropy[min(len(entropy), poolSize):] {
		for j := range s.pool {
			s.pool[j] = mix(s.pool[j], hashmix(x))
		}
	}
	return s
}
func (s *SeedSequence) GenerateState32(n int) ([]uint32, error) {
	if _, err := np.ElementCount(np.Uint32, n); err != nil {
		return nil, err
	}
	out := make([]uint32, n)
	hash := uint32(0x8b51f9dd)
	for i := range out {
		x := s.pool[i%len(s.pool)] ^ hash
		hash *= 0x58f38ded
		x *= hash
		out[i] = x ^ (x >> 16)
	}
	return out, nil
}
func (s *SeedSequence) GenerateState64(n int) ([]uint64, error) {
	if _, err := np.ElementCount(np.Uint64, n); err != nil {
		return nil, err
	}
	words, err := s.GenerateState32(2 * n)
	if err != nil {
		return nil, err
	}
	out := make([]uint64, n)
	for i := range out {
		out[i] = uint64(words[2*i]) | uint64(words[2*i+1])<<32
	}
	return out, nil
}
func (s *SeedSequence) Spawn(n int) ([]*SeedSequence, error) {
	if n < 0 {
		return nil, np.ErrShape
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if uint64(n) > 1<<32-s.children {
		return nil, np.ErrValue
	}
	out := make([]*SeedSequence, n)
	for i := range out {
		key := append(slices.Clone(s.spawn), uint32(s.children))
		out[i] = seedWithKey(s.entropy, key, len(s.pool))
		s.children++
	}
	return out, nil
}

type u128 struct{ hi, lo uint64 }

func (a u128) add(b u128) u128 {
	lo, carry := bits.Add64(a.lo, b.lo, 0)
	return u128{a.hi + b.hi + carry, lo}
}
func (a u128) mul(b u128) u128 {
	hi, lo := bits.Mul64(a.lo, b.lo)
	return u128{hi + a.hi*b.lo + a.lo*b.hi, lo}
}
func (a u128) shiftRight() u128 { return u128{a.hi >> 1, a.lo>>1 | a.hi<<63} }

var pcgMultiplier = u128{2549297995355413924, 4865540595714422341}

const dxsmMultiplier uint64 = 0xda942042e4dd58b5

// State is a value snapshot of the selected bit generator and its buffered draws.
type State struct {
	StateHigh, StateLow         uint64
	IncrementHigh, IncrementLow uint64
	HasUint32                   bool
	Uint32                      uint32
	DXSM                        bool
	Algorithm                   string
	MT                          MTState
	Philox                      PhiloxState
	SFC                         [4]uint64
	External                    bool
	ExternalState               string
}

// bitStream owns the algorithm state and the lock shared by every consumer.
type bitStream struct {
	mu         sync.Mutex
	seed       *SeedSequence
	state, inc u128
	cached     uint32
	has32      bool
	dxsm       bool
	algorithm  string
	mt         *MTState
	philox     *PhiloxState
	sfc        *[4]uint64
	external   ExternalBitSource
}

// Generator has a sampling policy and per-consumer Gaussian cache; its
// bitStream can be shared with other Generator or RandomState consumers.
type Generator struct {
	*bitStream
	legacy   bool
	hasGauss bool
	gauss    float64
}

func DefaultRNG(seed uint64) *Generator   { return FromSeedSequence(Seed(seed), false) }
func NewPCG64(seed uint64) *Generator     { return DefaultRNG(seed) }
func NewPCG64DXSM(seed uint64) *Generator { return FromSeedSequence(Seed(seed), true) }

func FromSeedSequence(s *SeedSequence, dxsm bool) *Generator {
	words, _ := s.GenerateState64(4)
	g := &Generator{bitStream: &bitStream{seed: s, inc: u128{words[2]<<1 | words[3]>>63, words[3]<<1 | 1}, dxsm: dxsm}}
	// NumPy initializes both PCG variants with the original PCG64 multiplier.
	g.state = g.inc.add(u128{words[0], words[1]}).mul(pcgMultiplier).add(g.inc)
	return g
}

func (g *Generator) State() State {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.snapshot()
}
func (g *Generator) SetState(s State) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.restore(s)
}
func (g *Generator) snapshot() State {
	s := State{StateHigh: g.state.hi, StateLow: g.state.lo, IncrementHigh: g.inc.hi, IncrementLow: g.inc.lo,
		HasUint32: g.has32, Uint32: g.cached, DXSM: g.dxsm, Algorithm: g.algorithm}
	if g.external != nil {
		s.External = true
		s.ExternalState = g.external.Snapshot()
		return s
	}
	if g.mt != nil {
		s.MT = *g.mt
	}
	if g.philox != nil {
		s.Philox = *g.philox
	}
	if g.sfc != nil {
		s.SFC = *g.sfc
	}
	return s
}
func (g *Generator) restore(s State) error {
	if s.Algorithm != g.algorithm || s.External != (g.external != nil) {
		return np.ErrValue
	}
	if g.external != nil {
		return g.external.Restore(s.ExternalState)
	}
	if g.algorithm == "" && s.DXSM != g.dxsm {
		return np.ErrValue
	}
	switch g.algorithm {
	case "MT19937":
		if s.MT.Pos < 0 || s.MT.Pos > 624 {
			return np.ErrValue
		}
		*g.mt = s.MT
	case "Philox":
		if s.Philox.BufferPos < 0 || s.Philox.BufferPos > 4 {
			return np.ErrValue
		}
		*g.philox = s.Philox
	case "SFC64":
		*g.sfc = s.SFC
	default:
		if s.IncrementLow&1 == 0 {
			return fmt.Errorf("%w: PCG64 increment must be odd", np.ErrValue)
		}
	}
	g.state = u128{s.StateHigh, s.StateLow}
	g.inc = u128{s.IncrementHigh, s.IncrementLow}
	g.has32 = s.HasUint32
	g.cached = s.Uint32
	g.dxsm = s.DXSM
	return nil
}
func (g *Generator) multiplier() u128 {
	if g.dxsm {
		return u128{0, dxsmMultiplier}
	}
	return pcgMultiplier
}
func (g *Generator) next64() uint64 {
	if g.external != nil {
		return g.external.NextUint64()
	}
	switch g.algorithm {
	case "MT19937":
		return uint64(g.mt.next())<<32 | uint64(g.mt.next())
	case "Philox":
		return g.philox.next()
	case "SFC64":
		return sfcNext(g.sfc)
	}
	return g.pcgNext()
}
func (g *Generator) pcgNext() uint64 {
	// Retain the scalar generator path for raw draws and distribution
	// callbacks. Benchmarks show the local-state batch path is useful for
	// bounded and converted outputs, but not for a raw PCG64 fill.
	if g.dxsm {
		hi, lo := g.state.hi, g.state.lo|1
		hi ^= hi >> 32
		hi *= dxsmMultiplier
		hi ^= hi >> 48
		hi *= lo
		g.state = g.state.mul(g.multiplier()).add(g.inc)
		return hi
	}
	g.state = g.state.mul(pcgMultiplier).add(g.inc)
	return bits.RotateLeft64(g.state.hi^g.state.lo, -int(g.state.hi>>58))
}
func pcgStep(state, inc u128, dxsm bool) (u128, uint64) {
	if dxsm {
		return pcgDXSMStep(state, inc)
	}
	return pcg64Step(state, inc)
}
func pcg64Step(state, inc u128) (u128, uint64) {
	state = state.mul(pcgMultiplier).add(inc)
	return state, bits.RotateLeft64(state.hi^state.lo, -int(state.hi>>58))
}
func pcgDXSMStep(state, inc u128) (u128, uint64) {
	hi, lo := state.hi, state.lo|1
	hi ^= hi >> 32
	hi *= dxsmMultiplier
	hi ^= hi >> 48
	hi *= lo
	return state.mul(u128{0, dxsmMultiplier}).add(inc), hi
}
func (g *Generator) next32() uint32 {
	if g.external != nil {
		return g.external.NextUint32()
	}
	if g.algorithm == "MT19937" {
		return g.mt.next()
	}
	if g.has32 {
		g.has32 = false
		return g.cached
	}
	x := g.next64()
	g.cached = uint32(x >> 32)
	g.has32 = true
	return uint32(x)
}
func (g *Generator) double() float64 {
	if g.external != nil {
		return g.external.NextDouble()
	}
	if g.algorithm == "MT19937" {
		return float64(uint64(g.mt.next()>>5)<<26|uint64(g.mt.next()>>6)) * 0x1p-53
	}
	return float64(g.next64()>>11) * 0x1p-53
}
func (g *Generator) single() float32 { return float32(g.next32()>>8) * 0x1p-24 }

// Advance jumps the underlying bit generator by delta steps (not distribution draws).
func (g *Generator) Advance(delta uint64) error { return g.Advance128(0, delta) }
func (g *Generator) Advance128(high, low uint64) error {
	return g.Advance256([4]uint64{low, high})
}

// Advance256 accepts a little-endian unsigned step. PCG steps are modulo 2^128;
// Philox advances its 256-bit block counter. Other algorithms do not support it.
func (g *Generator) Advance256(step [4]uint64) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.external != nil {
		if source, ok := g.external.(ExternalAdvanceSource); ok {
			return source.Advance256(step)
		}
		return np.ErrValue
	}
	switch g.algorithm {
	case "Philox":
		g.philox.advance(step)
		g.has32 = false
		g.cached = 0
		return nil
	case "MT19937", "SFC64":
		return np.ErrValue
	}
	delta := u128{step[1], step[0]}
	mult, plus := g.multiplier(), g.inc
	accMult, accPlus := u128{0, 1}, u128{}
	for delta.hi != 0 || delta.lo != 0 {
		if delta.lo&1 != 0 {
			accMult = accMult.mul(mult)
			accPlus = accPlus.mul(mult).add(plus)
		}
		plus = mult.add(u128{0, 1}).mul(plus)
		mult = mult.mul(mult)
		delta = delta.shiftRight()
	}
	g.state = accMult.mul(g.state).add(accPlus)
	g.has32 = false
	g.cached = 0
	return nil
}

func (g *Generator) RandomRaw(shape ...int) (*np.Array, error) {
	n, err := np.ElementCount(np.Uint64, shape...)
	if err != nil {
		return nil, err
	}
	out := make([]uint64, n)
	g.mu.Lock()
	defer g.mu.Unlock()
	// Select the raw engine once per fill, not once per generated word.
	if g.external != nil {
		for i := range out {
			out[i] = g.external.NextRaw()
		}
	} else {
		switch g.algorithm {
		case "MT19937":
			for i := range out {
				out[i] = uint64(g.mt.next())
			}
		case "Philox":
			for i := range out {
				out[i] = g.philox.next()
			}
		case "SFC64":
			for i := range out {
				out[i] = sfcNext(g.sfc)
			}
		default:
			for i := range out {
				out[i] = g.pcgNext()
			}
		}
	}
	return np.WrapSlice(out, shape...)
}
func (g *Generator) nextRaw() uint64 {
	if g.external != nil {
		return g.external.NextRaw()
	}
	if g.algorithm == "MT19937" {
		return uint64(g.next32())
	}
	return g.next64()
}

func (g *Generator) Random(shape ...int) (*np.Array, error) {
	return g.RandomDType(np.Float64, shape...)
}
func (g *Generator) RandomDType(d np.DType, shape ...int) (*np.Array, error) {
	if d == np.Float64 {
		if g.external == nil && g.algorithm == "" {
			n, e := np.ElementCount(d, shape...)
			if e != nil {
				return nil, e
			}
			v := make([]float64, n)
			g.mu.Lock()
			state, inc, dxsm := g.state, g.inc, g.dxsm
			for i := range v {
				var word uint64
				state, word = pcgStep(state, inc, dxsm)
				v[i] = float64(word>>11) * 0x1p-53
			}
			g.state = state
			g.mu.Unlock()
			return np.WrapFloat64(v, shape...)
		}
		return g.fill(shape, g.double)
	}
	if d != np.Float32 {
		return nil, np.ErrDType
	}
	n, err := np.ElementCount(d, shape...)
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	g.mu.Lock()
	defer g.mu.Unlock()
	for i := range out {
		out[i] = g.single()
	}
	return np.WrapSlice(out, shape...)
}
func (g *Generator) fill(shape []int, sample func() float64) (*np.Array, error) {
	n, err := np.ElementCount(np.Float64, shape...)
	if err != nil {
		return nil, err
	}
	out := make([]float64, n)
	g.mu.Lock()
	defer g.mu.Unlock()
	for i := range out {
		out[i] = sample()
	}
	return np.WrapFloat64(out, shape...)
}

type subword struct {
	bits      uint32
	remaining int
}

func (g *Generator) nextWord(width int, buf *subword) uint64 {
	if width == 64 {
		return g.next64()
	}
	if width == 32 {
		return uint64(g.next32())
	}
	if buf.remaining == 0 {
		buf.bits = g.next32()
		buf.remaining = 32 / width
	}
	value := buf.bits & ((uint32(1) << width) - 1)
	buf.bits >>= width
	buf.remaining--
	return uint64(value)
}

type boundedRange struct {
	r, mask, rangeExcl, threshold uint64
	width                         int
	legacy                        bool
}

func planBoundedRange(r uint64, width int, legacy bool) boundedRange {
	if width == 64 && r <= 0xffffffff {
		width = 32
	}
	p := boundedRange{r: r, width: width, legacy: legacy}
	if r == 0 {
		return p
	}
	if legacy {
		p.mask = ^uint64(0) >> bits.LeadingZeros64(r)
		return p
	}
	p.mask = ^uint64(0) >> (64 - width)
	if r != p.mask {
		p.rangeExcl = r + 1
		p.threshold = (p.mask - r) % p.rangeExcl
	}
	return p
}
func (g *Generator) bounded(r uint64, width int, buf *subword) uint64 {
	return g.drawBounded(planBoundedRange(r, width, g.legacy), buf)
}
func (g *Generator) drawBounded(p boundedRange, buf *subword) uint64 {
	if p.r == 0 {
		return 0
	}
	if p.legacy {
		for {
			v := g.nextWord(p.width, buf) & p.mask
			if v <= p.r {
				return v
			}
		}
	}
	if p.r == p.mask {
		return g.nextWord(p.width, buf)
	}
	if p.width == 64 {
		for {
			hi, lo := bits.Mul64(g.next64(), p.rangeExcl)
			if lo >= p.threshold {
				return hi
			}
		}
	}
	for {
		product := g.nextWord(p.width, buf) * p.rangeExcl
		if product&p.mask >= p.threshold {
			return product >> p.width
		}
	}
}

// Integers samples int64 values in [low,high).
func (g *Generator) Integers(low, high int64, shape ...int) (*np.Array, error) {
	return g.IntegersDType(low, high, np.Int64, false, shape...)
}
func (g *Generator) IntegersDType(low, high int64, d np.DType, endpoint bool, shape ...int) (*np.Array, error) {
	if d.IsUnsigned() || d == np.Bool {
		if low < 0 || high < 0 {
			return nil, np.ErrValue
		}
		return g.UnsignedIntegers(uint64(low), uint64(high), d, endpoint, shape...)
	}
	if !d.IsSigned() {
		return nil, np.ErrDType
	}
	width := d.ItemSize() * 8
	if high < low || (!endpoint && high == low) {
		return nil, np.ErrValue
	}
	if width < 64 {
		minimum, maximum := -int64(1)<<(width-1), int64(1)<<(width-1)-1
		if low < minimum || low > maximum || high > maximum && !(high == maximum+1 && !endpoint) {
			return nil, np.ErrValue
		}
	}
	r := uint64(high) - uint64(low)
	if !endpoint {
		r--
	}
	return g.integerFill(uint64(low), r, d, width, shape)
}
func (g *Generator) UnsignedIntegers(low, high uint64, d np.DType, endpoint bool, shape ...int) (*np.Array, error) {
	if !d.IsUnsigned() && d != np.Bool {
		return nil, np.ErrDType
	}
	width := d.ItemSize() * 8
	if d == np.Bool {
		width = 1
	}
	if high < low || (!endpoint && high == low) {
		return nil, np.ErrValue
	}
	if width < 64 {
		limit := uint64(1) << width
		if low >= limit || high > limit || endpoint && high == limit {
			return nil, np.ErrValue
		}
	}
	r := high - low
	if !endpoint {
		r--
	}
	return g.integerFill(low, r, d, width, shape)
}
func (g *Generator) integerFill(low, r uint64, d np.DType, width int, shape []int) (*np.Array, error) {
	n, err := np.ElementCount(d, shape...)
	if err != nil {
		return nil, err
	}
	p := planBoundedRange(r, width, g.legacy)
	if !g.legacy && g.external == nil && g.algorithm == "" {
		switch d {
		case np.Int64:
			return fillPCG64Integers[int64](g, n, low, p, shape)
		case np.Uint64:
			return fillPCG64Integers[uint64](g, n, low, p, shape)
		}
	}
	return g.integerOutput(n, d, shape, func(_ int, buf *subword) uint64 { return low + g.drawBounded(p, buf) })
}

// Keep a scalar PCG fill's state and cached half-word local to the locked
// call. The exact Lemire rejection order is identical to drawBounded,
// including the effective 32-bit path for small 64-bit integer ranges.
func fillPCG64Integers[T int64 | uint64](g *Generator, n int, low uint64, p boundedRange, shape []int) (*np.Array, error) {
	out := make([]T, n)
	g.mu.Lock()
	if p.r == 0 {
		for i := range out {
			out[i] = T(low)
		}
		g.mu.Unlock()
		return np.WrapSlice(out, shape...)
	}
	state, inc, dxsm := g.state, g.inc, g.dxsm
	has, cached := g.has32, g.cached
	full := p.r == p.mask
	if p.width == 32 {
		for i := range out {
			for {
				var word uint32
				if has {
					word = cached
					has = false
				} else {
					var v uint64
					state, v = pcgStep(state, inc, dxsm)
					word, cached = uint32(v), uint32(v>>32)
					has = true
				}
				if full {
					out[i] = T(low + uint64(word))
					break
				}
				product := uint64(word) * p.rangeExcl
				if product&0xffffffff >= p.threshold {
					out[i] = T(low + (product >> 32))
					break
				}
			}
		}
	} else {
		for i := range out {
			for {
				var word uint64
				state, word = pcgStep(state, inc, dxsm)
				if full {
					out[i] = T(low + word)
					break
				}
				hi, lo := bits.Mul64(word, p.rangeExcl)
				if lo >= p.threshold {
					out[i] = T(low + hi)
					break
				}
			}
		}
	}
	g.state, g.has32, g.cached = state, has, cached
	g.mu.Unlock()
	return np.WrapSlice(out, shape...)
}

type integerWord interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

func fillIntegerWords[T integerWord](g *Generator, n int, shape []int, draw func(int, *subword) uint64) (*np.Array, error) {
	out := make([]T, n)
	buf := subword{}
	g.mu.Lock()
	for i := range out {
		out[i] = T(draw(i, &buf))
	}
	g.mu.Unlock()
	return np.WrapSlice(out, shape...)
}
func (g *Generator) integerOutput(n int, d np.DType, shape []int, draw func(int, *subword) uint64) (*np.Array, error) {
	switch d {
	case np.Int8:
		return fillIntegerWords[int8](g, n, shape, draw)
	case np.Int16:
		return fillIntegerWords[int16](g, n, shape, draw)
	case np.Int32:
		return fillIntegerWords[int32](g, n, shape, draw)
	case np.Int64:
		return fillIntegerWords[int64](g, n, shape, draw)
	case np.Uint8:
		return fillIntegerWords[uint8](g, n, shape, draw)
	case np.Uint16:
		return fillIntegerWords[uint16](g, n, shape, draw)
	case np.Uint32:
		return fillIntegerWords[uint32](g, n, shape, draw)
	case np.Uint64:
		return fillIntegerWords[uint64](g, n, shape, draw)
	case np.Bool:
		out := make([]bool, n)
		buf := subword{}
		g.mu.Lock()
		for i := range out {
			out[i] = draw(i, &buf) != 0
		}
		g.mu.Unlock()
		return np.WrapSlice(out, shape...)
	}
	return nil, np.ErrDType
}
