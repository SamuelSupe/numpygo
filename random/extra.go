package random

import (
	"encoding/binary"
	"math"
	"reflect"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

func (g *Generator) Bytes(length int) ([]byte, error) {
	if length < 0 {
		return nil, np.ErrShape
	}
	out := make([]byte, length)
	g.mu.Lock()
	defer g.mu.Unlock()
	var word [4]byte
	for i := 0; i < length; i += 4 {
		binary.LittleEndian.PutUint32(word[:], g.next32())
		copy(out[i:], word[:])
	}
	return out, nil
}
func (g *Generator) Spawn(n int) ([]*Generator, error) {
	if n < 0 {
		return nil, np.ErrValue
	}
	g.mu.Lock()
	if g.external != nil {
		defer g.mu.Unlock()
		source, ok := g.external.(ExternalSpawnSource)
		if !ok {
			return nil, np.ErrValue
		}
		children, err := source.Spawn(n)
		if err != nil {
			return nil, err
		}
		if len(children) != n {
			return nil, np.ErrValue
		}
		out := make([]*Generator, n)
		seen := make(map[ExternalBitSource]struct{}, n)
		for i, child := range children {
			if child != nil && reflect.ValueOf(child).Comparable() {
				if _, ok := seen[child]; ok {
					return nil, np.ErrValue
				}
				seen[child] = struct{}{}
			}
			out[i], err = g.externalChild(child)
			if err != nil {
				return nil, err
			}
		}
		return out, nil
	}
	s, dxsm, algorithm := g.seed, g.dxsm, g.algorithm
	g.mu.Unlock()
	if s == nil {
		return nil, np.ErrValue
	}
	children, err := s.Spawn(n)
	if err != nil {
		return nil, err
	}
	out := make([]*Generator, len(children))
	for i, seed := range children {
		switch algorithm {
		case "MT19937":
			out[i] = MT19937FromSeedSequence(seed)
		case "Philox":
			out[i] = PhiloxFromSeedSequence(seed)
		case "SFC64":
			out[i] = SFC64FromSeedSequence(seed)
		default:
			out[i] = FromSeedSequence(seed, dxsm)
		}
	}
	return out, nil
}

func (g *Generator) Geometric(p float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("geometric", shape, realArgs(p))
}
func (g *Generator) geometric(p float64) int64 {
	if p >= 1.0/3 {
		u := g.double()
		sum, prod := p, p
		var x int64 = 1
		for u > sum {
			prod *= 1 - p
			sum += prod
			x++
		}
		return x
	}
	var z float64
	if g.legacy {
		z = math.Ceil(math.Log1p(-g.double()) / math.Log(1-p))
	} else {
		z = math.Ceil(-g.exponential() / math.Log1p(-p))
	}
	if z >= float64(math.MaxInt64) {
		return math.MaxInt64
	}
	return int64(z)
}
func (g *Generator) Triangular(left, mode, right float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("triangular", shape, realArgs(left, mode, right))
}
func (g *Generator) triangular(left, mode, right float64) float64 {
	base := right - left
	ratio := (mode - left) / base
	u := g.double()
	if u <= ratio {
		return left + math.Sqrt(u*(mode-left)*base)
	}
	return right - math.Sqrt((1-u)*(right-mode)*base)
}
func (g *Generator) Wald(mean, scale float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("wald", shape, realArgs(mean, scale))
}
func (g *Generator) wald(mean, scale float64) float64 {
	y := g.normal()
	y = mean * y * y
	var x float64
	if g.legacy {
		x = mean + mean/(2*scale)*(y-math.Sqrt(4*scale*y+y*y))
	} else {
		d := 1 + math.Sqrt(1+4*scale/y)
		x = mean * (1 - 2/d)
	}
	if g.double() <= mean/(mean+x) {
		return x
	}
	return mean * mean / x
}
func (g *Generator) Logseries(p float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("logseries", shape, realArgs(p))
}
func (g *Generator) logseries(p float64) int64 {
	r := math.Log1p(-p)
	if g.legacy {
		r = math.Log(1 - p)
	}
	for {
		v := g.double()
		if v >= p {
			return 1
		}
		u := g.double()
		q := -math.Expm1(r * u)
		if g.legacy {
			q = 1 - math.Exp(r*u)
		}
		if v <= q*q {
			if v == 0 {
				continue
			}
			result := int64(math.Floor(1 + math.Log(v)/math.Log(q)))
			if result < 1 {
				continue
			}
			return result
		}
		if v >= q {
			return 1
		}
		return 2
	}
}
func (g *Generator) Zipf(a float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("zipf", shape, realArgs(a))
}
func (g *Generator) zipf(a float64) int64 {
	if a >= 1025 && !g.legacy {
		return 1
	}
	am1 := a - 1
	b := math.Pow(2, am1)
	umin := math.Pow(float64(math.MaxInt64), -am1)
	for {
		u0 := g.double()
		u := u0*umin + (1 - u0)
		if g.legacy {
			u = 1 - u0
		}
		v := g.double()
		x := math.Floor(math.Pow(u, -1/am1))
		if x >= float64(math.MaxInt64) || x < 1 {
			continue
		}
		t := math.Pow(1+1/x, am1)
		if v*x*(t-1)/(b-1) <= t/b {
			return int64(x)
		}
	}
}
func (g *Generator) noncentralChi(df, nonc float64) float64 {
	if nonc == 0 {
		return 2 * g.gamma(df/2)
	}
	if df > 1 {
		x := 2 * g.gamma((df-1)/2)
		n := g.normal() + math.Sqrt(nonc)
		return x + n*n
	}
	i := g.poisson(nonc / 2)
	return 2 * g.gamma((df+2*float64(i))/2)
}
func (g *Generator) NoncentralChisquare(df, nonc float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("noncentral_chisquare", shape, realArgs(df, nonc))
}
func (g *Generator) NoncentralF(dfnum, dfden, nonc float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("noncentral_f", shape, realArgs(dfnum, dfden, nonc))
}
func (g *Generator) Vonmises(mu, kappa float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("vonmises", shape, realArgs(mu, kappa))
}
func (g *Generator) vonmises(mu, kappa float64) float64 {
	if kappa < 1e-8 {
		return math.Pi * (2*g.double() - 1)
	}
	if kappa > 1e6 && !g.legacy {
		x := mu + math.Sqrt(1/kappa)*g.normal()
		if x < -math.Pi {
			x += 2 * math.Pi
		}
		if x > math.Pi {
			x -= 2 * math.Pi
		}
		return x
	}
	var s float64
	if kappa < 1e-5 {
		s = 1/kappa + kappa
	} else {
		r := 1 + math.Sqrt(1+4*kappa*kappa)
		rho := (r - math.Sqrt(2*r)) / (2 * kappa)
		s = (1 + rho*rho) / (2 * rho)
	}
	var w float64
	for {
		z := math.Cos(math.Pi * g.double())
		w = (1 + s*z) / (s + z)
		y := kappa * (s - w)
		v := g.double()
		if y*(2-y)-v >= 0 || math.Log(y/v)+1-y >= 0 {
			break
		}
	}
	x := math.Acos(w)
	if g.double() < .5 {
		x = -x
	}
	x += mu
	mod := math.Mod(math.Abs(x)+math.Pi, 2*math.Pi) - math.Pi
	if x < 0 {
		mod = -mod
	}
	return mod
}

func (g *Generator) Multinomial(n int64, p []float64, shape ...int) (*np.Array, error) {
	if n < 0 || !validMultinomial(p) {
		return nil, np.ErrValue
	}
	outShape := append(slices.Clone(shape), len(p))
	count, err := np.ElementCount(np.Int64, outShape...)
	if err != nil {
		return nil, err
	}
	out := make([]int64, count)
	g.mu.Lock()
	defer g.mu.Unlock()
	for i := 0; i < count; i += len(p) {
		g.multinomial(n, p, out[i:i+len(p)])
	}
	return np.WrapSlice(out, outShape...)
}
func validMultinomial(p []float64) bool {
	if len(p) == 0 {
		return false
	}
	var sum float64
	for i, x := range p {
		if math.IsNaN(x) || x < 0 || x > 1 {
			return false
		}
		if i < len(p)-1 {
			sum += x
		}
	}
	return sum <= 1+1e-12
}
func (g *Generator) multinomial(n int64, p []float64, out []int64) {
	remaining, prob := n, 1.0
	for j := 0; j < len(p)-1; j++ {
		if remaining <= 0 {
			break
		}
		x := g.binomial(remaining, math.Min(p[j]/prob, 1))
		out[j] = x
		remaining -= x
		prob -= p[j]
	}
	out[len(p)-1] = remaining
}
func (g *Generator) Dirichlet(alpha []float64, shape ...int) (*np.Array, error) {
	for _, a := range alpha {
		if !nonnegative(a) || g.legacy && a == 0 {
			return nil, np.ErrValue
		}
	}
	k := len(alpha)
	outShape := append(slices.Clone(shape), k)
	count, err := np.ElementCount(np.Float64, outShape...)
	if err != nil {
		return nil, err
	}
	out := make([]float64, count)
	if k == 0 {
		return np.WrapSlice(out, outShape...)
	}
	maxAlpha := 0.0
	for _, a := range alpha {
		maxAlpha = max(maxAlpha, a)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if maxAlpha < .1 && !g.legacy {
		csum := make([]float64, k)
		cs := 0.0
		for j := k - 1; j >= 0; j-- {
			cs += alpha[j]
			csum[j] = cs
		}
		if cs > 0 {
			for i := 0; i < count; i += k {
				remaining := 1.0
				for j := 0; j < k-1; j++ {
					v := g.beta(alpha[j], csum[j+1])
					out[i+j] = remaining * v
					remaining *= 1 - v
					if csum[j+1] == 0 {
						break
					}
				}
				out[i+k-1] = remaining
			}
		}
	} else {
		for i := 0; i < count; i += k {
			sum := 0.0
			for j, a := range alpha {
				out[i+j] = g.gamma(a)
				sum += out[i+j]
			}
			inv := 1 / sum
			for j := range alpha {
				out[i+j] *= inv
			}
		}
	}
	return np.WrapSlice(out, outShape...)
}
