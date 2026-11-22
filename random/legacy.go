package random

import (
	"crypto/rand"
	"encoding/binary"
	"math"

	"github.com/SamuelSupe/numpygo"
)

// RandomState provides NumPy's legacy MT19937 seeding and sampling algorithms.
// The Gaussian cache is part of the state, unlike the modern Generator stream.
type RandomState struct{ *Generator }
type LegacyState struct {
	MT       MTState
	HasGauss bool
	Gauss    float64
	// Stream is populated for non-MT19937 consumers. MT retains the
	// historical tuple-compatible fields above.
	Stream State
}

// ChoiceArray retains RandomState's one-dimensional population contract
// (Generator supports an arbitrary population axis).
func (r *RandomState) ChoiceArray(a *np.Array, axis int, opt ChoiceOptions, shape ...int) (*np.Array, error) {
	if a.DType() == 0 {
		return nil, np.ErrDType
	}
	if a.NDim() != 1 {
		return nil, np.ErrShape
	}
	if axis != 0 && axis != -1 {
		return nil, np.ErrAxis
	}
	if opt.NoShuffle {
		return nil, np.ErrValue
	}
	return r.Generator.ChoiceArray(a, axis, opt, shape...)
}

func NewRandomState(seed uint32) *RandomState {
	r := &RandomState{&Generator{bitStream: &bitStream{algorithm: "MT19937", mt: new(MTState)}, legacy: true}}
	r.Seed(seed)
	return r
}

// NewRandomStateFromBitGenerator uses legacy distribution algorithms but
// shares exactly the supplied engine, uint32 cache and mutex, without reseeding.
func NewRandomStateFromBitGenerator(bit BitGenerator) (*RandomState, error) {
	g, e := NewGeneratorFromBitGenerator(bit)
	if e != nil {
		return nil, e
	}
	g.legacy = true
	return &RandomState{g}, nil
}

// NewGeneratorFromBitGenerator attaches a modern distribution consumer to
// the same engine independently of the legacy Gaussian cache.
func NewGeneratorFromBitGenerator(bit BitGenerator) (*Generator, error) {
	if bit == nil {
		return nil, np.ErrDType
	}
	if g, ok := bit.(*Generator); ok && (g == nil || g.bitStream == nil) {
		return nil, np.ErrDType
	}
	var stream *bitStream
	err := bit.WithLock(func(s *LockedStream) error {
		if s == nil || s.g == nil || s.g.bitStream == nil {
			return np.ErrDType
		}
		stream = s.g.bitStream
		return nil
	})
	if err != nil {
		return nil, err
	}
	if stream == nil {
		return nil, np.ErrDType
	}
	return &Generator{bitStream: stream}, nil
}
func NewRandomStateArray(words ...uint32) (*RandomState, error) {
	r := NewRandomState(0)
	if err := r.SeedArray(words...); err != nil {
		return nil, err
	}
	return r, nil
}
func (r *RandomState) Seed(seed uint32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.mt == nil {
		return np.ErrDType
	}
	r.mt.seed(seed)
	r.hasGauss = false
	r.gauss = 0
	return nil
}
func (r *RandomState) SeedInt(seed int64) error {
	if seed < 0 || seed > math.MaxUint32 {
		return np.ErrValue
	}
	return r.Seed(uint32(seed))
}
func (r *RandomState) SeedArray(words ...uint32) error {
	if len(words) == 0 {
		return np.ErrValue
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.mt == nil {
		return np.ErrDType
	}
	r.mt.seedArray(words)
	r.hasGauss = false
	r.gauss = 0
	return nil
}
func (r *RandomState) GetState() LegacyState {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := LegacyState{HasGauss: r.hasGauss, Gauss: r.gauss}
	if r.mt != nil {
		out.MT = *r.mt
	} else {
		out.Stream = r.snapshot()
	}
	return out
}
func (r *RandomState) SetState(state LegacyState) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.mt != nil {
		if state.Stream.Algorithm != "" || state.Stream.IncrementLow != 0 || state.Stream.External || state.MT.Pos < 0 || state.MT.Pos > 624 {
			return np.ErrValue
		}
		*r.mt = state.MT
	} else {
		if e := r.restore(state.Stream); e != nil {
			return e
		}
	}
	r.hasGauss = state.HasGauss
	r.gauss = state.Gauss
	return nil
}
func (g *Generator) legacyNormal() float64 {
	if g.hasGauss {
		v := g.gauss
		g.hasGauss = false
		g.gauss = 0
		return v
	}
	for {
		x, y := 2*g.double()-1, 2*g.double()-1
		r2 := x*x + y*y
		if r2 >= 1 || r2 == 0 {
			continue
		}
		f := math.Sqrt(-2 * math.Log(r2) / r2)
		g.gauss = f * x
		g.hasGauss = true
		return f * y
	}
}
func (r *RandomState) RandomSample(shape ...int) (*np.Array, error) { return r.Random(shape...) }
func (r *RandomState) Rand(shape ...int) (*np.Array, error)         { return r.Random(shape...) }
func (r *RandomState) Randn(shape ...int) (*np.Array, error)        { return r.StandardNormal(shape...) }
func (r *RandomState) Randint(low, high int64, dtype np.DType, shape ...int) (*np.Array, error) {
	if dtype == 0 {
		dtype = np.Int64
	}
	return r.IntegersDType(low, high, dtype, false, shape...)
}
func (r *RandomState) RandomIntegers(low, high int64, shape ...int) (*np.Array, error) {
	return r.IntegersDType(low, high, np.Int64, true, shape...)
}
func (r *RandomState) ToMaxInt(shape ...int) (*np.Array, error) {
	return r.fillInt(shape, func() int64 { return int64(r.next64() >> 1) })
}

// MultivariateNormal uses the legacy SVD-based sample transform. As in NumPy,
// the exact signs of SVD vectors can depend on the numerical backend.
func (r *RandomState) MultivariateNormal(mean []float64, cov *np.Array, shape ...int) (*np.Array, error) {
	return r.Generator.MultivariateNormalWithOptions(mean, cov, MultivariateNormalOptions{}, shape...)
}

var globalRandomState = func() *RandomState {
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		panic(err)
	}
	words := make([]uint32, 4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(bytes[i*4:])
	}
	r, _ := NewRandomStateArray(words...)
	return r
}()

// GlobalRandomState is the process-wide legacy singleton corresponding to
// numpy.random's module-level functions. Prefer independent generators.
func GlobalRandomState() *RandomState { return globalRandomState }
