package random

import (
	"fmt"
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

type sampleArgs struct {
	f [3]float64
	i [3]int64
}

func realArgs(values ...float64) sampleArgs {
	var a sampleArgs
	copy(a.f[:], values)
	return a
}

// A parameterized distribution shares its validator and unlocked numerical
// kernel between scalar and array calls. The outer fill owns the stream lock.
type distributionKernel struct {
	n       int
	intMask uint8
	valid   func(sampleArgs) bool
	real    func(sampleArgs) float64
	integer func(sampleArgs) int64
}

func (g *Generator) distribution(name string) (distributionKernel, error) {
	f := func(n int, valid func(sampleArgs) bool, draw func(sampleArgs) float64) (distributionKernel, error) {
		return distributionKernel{n: n, valid: valid, real: draw}, nil
	}
	i := func(n int, mask uint8, valid func(sampleArgs) bool, draw func(sampleArgs) int64) (distributionKernel, error) {
		return distributionKernel{n: n, intMask: mask, valid: valid, integer: draw}, nil
	}
	p0 := func(a sampleArgs) bool { return positive(a.f[0]) }
	n0 := func(a sampleArgs) bool { return nonnegativeScale(a.f[0]) }
	n1 := func(a sampleArgs) bool { return nonnegativeScale(a.f[1]) }
	p01 := func(a sampleArgs) bool { return positive(a.f[0]) && positive(a.f[1]) }
	pLimit := float64(math.MaxInt64) - 10*math.Sqrt(float64(math.MaxInt64))
	switch name {
	case "normal":
		return f(2, n1, func(a sampleArgs) float64 { return a.f[0] + a.f[1]*g.normal() })
	case "uniform":
		return f(2, func(a sampleArgs) bool {
			r := a.f[1] - a.f[0]
			return nonnegativeScale(r) || g.legacy && !math.IsNaN(r) && !math.IsInf(r, 0)
		}, func(a sampleArgs) float64 { return a.f[0] + (a.f[1]-a.f[0])*g.double() })
	case "exponential":
		return f(1, n0, func(a sampleArgs) float64 { return a.f[0] * g.exponential() })
	case "gamma":
		return f(2, func(a sampleArgs) bool { return n0(a) && n1(a) }, func(a sampleArgs) float64 { return a.f[1] * g.gamma(a.f[0]) })
	case "standard_gamma":
		return f(1, n0, func(a sampleArgs) float64 { return g.gamma(a.f[0]) })
	case "beta":
		return f(2, p01, func(a sampleArgs) float64 { return g.beta(a.f[0], a.f[1]) })
	case "chisquare":
		return f(1, p0, func(a sampleArgs) float64 { return 2 * g.gamma(a.f[0]/2) })
	case "f":
		return f(2, p01, func(a sampleArgs) float64 {
			x := 2 * g.gamma(a.f[0]/2) * a.f[1]
			y := 2 * g.gamma(a.f[1]/2) * a.f[0]
			return x / y
		})
	case "standard_t":
		return f(1, p0, func(a sampleArgs) float64 {
			x := g.normal()
			return math.Sqrt(a.f[0]/2) * x / math.Sqrt(g.gamma(a.f[0]/2))
		})
	case "pareto":
		return f(1, p0, func(a sampleArgs) float64 {
			x := g.exponential() / a.f[0]
			if g.legacy {
				return math.Exp(x) - 1
			}
			return math.Expm1(x)
		})
	case "weibull":
		return f(1, n0, func(a sampleArgs) float64 {
			if a.f[0] == 0 {
				return 0
			}
			return math.Pow(g.exponential(), 1/a.f[0])
		})
	case "power":
		return f(1, p0, func(a sampleArgs) float64 {
			e := g.exponential()
			if g.legacy {
				return math.Pow(1-math.Exp(-e), 1/a.f[0])
			}
			return math.Pow(-math.Expm1(-e), 1/a.f[0])
		})
	case "laplace":
		return f(2, n1, func(a sampleArgs) float64 { return g.laplace(a.f[0], a.f[1]) })
	case "gumbel":
		return f(2, n1, func(a sampleArgs) float64 { return g.gumbel(a.f[0], a.f[1]) })
	case "logistic":
		return f(2, n1, func(a sampleArgs) float64 { return g.logistic(a.f[0], a.f[1]) })
	case "lognormal":
		return f(2, n1, func(a sampleArgs) float64 { return math.Exp(a.f[0] + a.f[1]*g.normal()) })
	case "rayleigh":
		return f(1, n0, func(a sampleArgs) float64 {
			if g.legacy {
				return a.f[0] * math.Sqrt(-2*math.Log1p(-g.double()))
			}
			return a.f[0] * math.Sqrt(2*g.exponential())
		})
	case "triangular":
		return f(3, func(a sampleArgs) bool {
			return a.f[0] <= a.f[1] && a.f[1] <= a.f[2] && a.f[0] < a.f[2] && !math.IsInf(a.f[2]-a.f[0], 0)
		}, func(a sampleArgs) float64 { return g.triangular(a.f[0], a.f[1], a.f[2]) })
	case "wald":
		return f(2, p01, func(a sampleArgs) float64 { return g.wald(a.f[0], a.f[1]) })
	case "noncentral_chisquare":
		return f(2, func(a sampleArgs) bool { return p0(a) && n1(a) && a.f[1]/2 < pLimit }, func(a sampleArgs) float64 { return g.noncentralChi(a.f[0], a.f[1]) })
	case "noncentral_f":
		return f(3, func(a sampleArgs) bool { return p01(a) && nonnegativeScale(a.f[2]) && a.f[2]/2 < pLimit }, func(a sampleArgs) float64 {
			t := g.noncentralChi(a.f[0], a.f[2]) * a.f[1]
			return t / (2 * g.gamma(a.f[1]/2) * a.f[0])
		})
	case "vonmises":
		return f(2, n1, func(a sampleArgs) float64 { return g.vonmises(a.f[0], a.f[1]) })
	case "poisson":
		return i(1, 0, func(a sampleArgs) bool { return nonnegative(a.f[0]) && a.f[0] < pLimit }, func(a sampleArgs) int64 { return g.poisson(a.f[0]) })
	case "negative_binomial":
		return i(2, 0, func(a sampleArgs) bool { return p01(a) && a.f[1] <= 1 && a.f[0]*(1-a.f[1])/a.f[1] < pLimit }, func(a sampleArgs) int64 { return g.poisson((1 - a.f[1]) / a.f[1] * g.gamma(a.f[0])) })
	case "binomial":
		return i(2, 1, func(a sampleArgs) bool { return a.i[0] >= 0 && a.f[1] >= 0 && a.f[1] <= 1 }, func(a sampleArgs) int64 { return g.binomial(a.i[0], a.f[1]) })
	case "geometric":
		return i(1, 0, func(a sampleArgs) bool { return p0(a) && a.f[0] <= 1 }, func(a sampleArgs) int64 { return g.geometric(a.f[0]) })
	case "logseries":
		return i(1, 0, func(a sampleArgs) bool { return a.f[0] >= 0 && a.f[0] < 1 }, func(a sampleArgs) int64 { return g.logseries(a.f[0]) })
	case "zipf":
		return i(1, 0, func(a sampleArgs) bool { return a.f[0] > 1 && (!g.legacy || a.f[0] < 1025) }, func(a sampleArgs) int64 { return g.zipf(a.f[0]) })
	case "hypergeometric":
		return i(3, 7, func(a sampleArgs) bool {
			x, y, z := a.i[0], a.i[1], a.i[2]
			return x >= 0 && y >= 0 && z >= 0 && x < 1_000_000_000 && y < 1_000_000_000 && z <= x+y
		}, func(a sampleArgs) int64 { return g.hypergeometric(a.i[0], a.i[1], a.i[2]) })
	default:
		return distributionKernel{}, fmt.Errorf("%w: unsupported parameterized distribution %q", np.ErrValue, name)
	}
}

// NumPy's continuous scale constraint checks the sign bit (including -0).
func nonnegativeScale(x float64) bool { return nonnegative(x) && !math.Signbit(x) }

// When an empty operand eliminates all broadcast lanes, independently
// constrained parameters still need validation before returning an empty result.
func validEmptyParameters(name string, p []parameter) bool {
	check := func(i int, pred func(float64) bool) bool {
		for _, x := range p[i].f {
			if !pred(x) {
				return false
			}
		}
		return true
	}
	switch name {
	case "normal", "laplace", "gumbel", "logistic", "lognormal", "vonmises":
		return check(1, nonnegativeScale)
	case "gamma":
		return check(0, nonnegativeScale) && check(1, nonnegativeScale)
	case "beta", "f", "wald":
		return check(0, positive) && check(1, positive)
	case "noncentral_chisquare":
		return check(0, positive) && check(1, nonnegativeScale)
	case "noncentral_f":
		return check(0, positive) && check(1, positive) && check(2, nonnegativeScale)
	case "negative_binomial":
		return check(0, positive) && check(1, func(x float64) bool { return positive(x) && x <= 1 })
	case "binomial":
		for _, n := range p[0].i {
			if n < 0 {
				return false
			}
		}
		return check(1, func(x float64) bool { return x >= 0 && x <= 1 })
	case "hypergeometric":
		for j, operand := range p {
			for _, x := range operand.i {
				if x < 0 || j < 2 && x >= 1_000_000_000 {
					return false
				}
			}
		}
	}
	return true
}

func (g *Generator) sampleScalar(name string, shape []int, a sampleArgs) (*np.Array, error) {
	k, e := g.distribution(name)
	if e != nil {
		return nil, e
	}
	if !k.valid(a) {
		return nil, np.ErrValue
	}
	if k.integer != nil {
		return g.fillInt(shape, func() int64 { return k.integer(a) })
	}
	return g.fill(shape, func() float64 { return k.real(a) })
}

// parameter holds a logical C-order snapshot of the original operand, not
// an expanded copy of the output broadcast. Validation precedes every draw.
type parameter struct {
	shape []int
	f     []float64
	i     []int64
}

func snapshotParameter(a *np.Array, integer bool) (parameter, error) {
	p := parameter{}
	if a.DType() == 0 || a.DType().IsComplex() {
		return p, np.ErrDType
	}
	p.shape = a.Shape()
	if integer {
		ok, e := np.CanCast(a.DType(), np.Int64, "safe")
		if e != nil || !ok {
			return p, np.ErrDType
		}
		b := a
		if a.DType() != np.Int64 {
			b, e = a.Astype(np.Int64)
			if e != nil {
				return p, e
			}
		}
		p.i, e = np.Values[int64](b)
		if e != nil {
			return p, e
		}
	} else {
		b := a
		var e error
		if a.DType() != np.Float64 {
			b, e = a.Astype(np.Float64)
			if e != nil {
				return p, e
			}
		}
		p.f, e = b.Float64s()
		if e != nil {
			return p, e
		}
	}
	return p, nil
}

func parameterIndex(flat int, target, shape []int) int {
	j, stride := 0, 1
	for k := len(shape) - 1; k >= 0; k-- {
		dim := target[len(target)-len(shape)+k]
		if shape[k] != 1 {
			j += flat % dim * stride
		}
		flat /= dim
		stride *= shape[k]
	}
	return j
}

// The cursor walks the broadcast in C order, resolving singleton strides
// once rather than dividing the flat index for every parameter and draw.
type parameterCursor struct {
	params       []parameter
	shape, index []int
	strides      [][3]int
	offsets      [3]int
}

func newParameterCursor(params []parameter, shape []int) parameterCursor {
	c := parameterCursor{params: params, shape: shape, index: make([]int, len(shape)), strides: make([][3]int, len(shape))}
	for k, p := range params {
		stride := 1
		for j := len(p.shape) - 1; j >= 0; j-- {
			if p.shape[j] != 1 {
				c.strides[len(shape)-len(p.shape)+j][k] = stride
			}
			stride *= p.shape[j]
		}
	}
	return c
}
func (c *parameterCursor) next() sampleArgs {
	var a sampleArgs
	for k, p := range c.params {
		j := c.offsets[k]
		if p.i != nil {
			a.i[k] = p.i[j]
		} else {
			a.f[k] = p.f[j]
		}
	}
	for axis := len(c.shape) - 1; axis >= 0; axis-- {
		c.index[axis]++
		if c.index[axis] < c.shape[axis] {
			for k := range c.params {
				c.offsets[k] += c.strides[axis][k]
			}
			break
		}
		c.index[axis] = 0
		for k := range c.params {
			c.offsets[k] -= (c.shape[axis] - 1) * c.strides[axis][k]
		}
	}
	return a
}

func sampleShape(size []int, d np.DType, shapes ...[]int) (natural, output []int, count int, err error) {
	natural, err = np.BroadcastShapes(shapes...)
	if err != nil {
		return
	}
	output = natural
	if size != nil {
		output, err = np.BroadcastShapes(natural, size)
		if err != nil {
			return
		}
		if !slices.Equal(output, size) {
			err = np.ErrShape
			return
		}
	}
	count, err = np.ElementCount(d, output...)
	return
}

// Sample draws from a parameterized distribution with NumPy-style array
// broadcasting. A nil size infers the broadcast shape; []int{} requests a
// scalar. The result is float64 or int64 as in NumPy. C-order output and
// whole-call locking preserve the stream and RandomState Gaussian cache.
func (g *Generator) Sample(name string, size []int, operands ...*np.Array) (*np.Array, error) {
	k, e := g.distribution(name)
	if e != nil {
		return nil, e
	}
	if len(operands) != k.n {
		return nil, np.ErrShape
	}
	params := make([]parameter, k.n)
	shapes := make([][]int, k.n)
	d := np.Float64
	if k.integer != nil {
		d = np.Int64
	}
	for j, a := range operands {
		p, e := snapshotParameter(a, k.intMask&(1<<j) != 0)
		if e != nil {
			return nil, e
		}
		params[j], shapes[j] = p, p.shape
	}
	natural, shape, count, e := sampleShape(size, d, shapes...)
	if e != nil {
		return nil, e
	}
	lanes, e := np.ElementCount(d, natural...)
	if e != nil {
		return nil, e
	}
	if lanes == 0 && !validEmptyParameters(name, params) {
		return nil, np.ErrValue
	}
	cursor := newParameterCursor(params, natural)
	for j := 0; j < lanes; j++ {
		if !k.valid(cursor.next()) {
			return nil, np.ErrValue
		}
	}
	cursor = newParameterCursor(params, shape)
	if k.integer != nil {
		out := make([]int64, count)
		g.mu.Lock()
		for j := range out {
			out[j] = k.integer(cursor.next())
		}
		g.mu.Unlock()
		return np.WrapSlice(out, shape...)
	}
	out := make([]float64, count)
	g.mu.Lock()
	for j := range out {
		out[j] = k.real(cursor.next())
	}
	g.mu.Unlock()
	return np.WrapFloat64(out, shape...)
}
