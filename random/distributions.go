package random

import (
	"math"

	"github.com/SamuelSupe/numpygo"
)

// Distribution kernels follow the NumPy 2.5.0 random algorithms, adapted for
// Go and the generator's uint32/uint64 stream. See THIRD_PARTY_LICENSES.txt.
func (g *Generator) normal() float64 {
	if g.legacy {
		return g.legacyNormal()
	}
	for {
		r := g.next64()
		idx := int(r & 255)
		r >>= 8
		mag := (r >> 1) & 0x000fffffffffffff
		x := float64(mag) * wi_double[idx]
		if r&1 != 0 {
			x = -x
		}
		if mag < ki_double[idx] {
			return x
		}
		if idx == 0 {
			for {
				xx := -ziggurat_nor_inv_r * math.Log1p(-g.double())
				yy := -math.Log1p(-g.double())
				if yy+yy > xx*xx {
					v := ziggurat_nor_r + xx
					if (mag>>8)&1 != 0 {
						v = -v
					}
					return v
				}
			}
		}
		if (fi_double[idx-1]-fi_double[idx])*g.double()+fi_double[idx] < math.Exp(-0.5*x*x) {
			return x
		}
	}
}
func (g *Generator) normal32() float32 {
	for {
		r := g.next32()
		idx := int(r & 255)
		mag := (r >> 9) & 0x0007fffff
		x := float32(mag) * wi_float[idx]
		if (r>>8)&1 != 0 {
			x = -x
		}
		if mag < ki_float[idx] {
			return x
		}
		if idx == 0 {
			for {
				xx := -ziggurat_nor_inv_r_f * float32(math.Log1p(-float64(g.single())))
				yy := -float32(math.Log1p(-float64(g.single())))
				if yy+yy > xx*xx {
					v := ziggurat_nor_r_f + xx
					if (mag>>8)&1 != 0 {
						v = -v
					}
					return v
				}
			}
		}
		if float64((fi_float[idx-1]-fi_float[idx])*g.single()+fi_float[idx]) < math.Exp(-0.5*float64(x)*float64(x)) {
			return x
		}
	}
}
func (g *Generator) exponential() float64 {
	if g.legacy {
		return -math.Log(1 - g.double())
	}
	for {
		r := g.next64() >> 3
		idx := int(r & 255)
		r >>= 8
		x := float64(r) * we_double[idx]
		if r < ke_double[idx] {
			return x
		}
		if idx == 0 {
			return ziggurat_exp_r - math.Log1p(-g.double())
		}
		if (fe_double[idx-1]-fe_double[idx])*g.double()+fe_double[idx] < math.Exp(-x) {
			return x
		}
	}
}
func (g *Generator) exponential32() float32 {
	for {
		r := g.next32() >> 1
		idx := int(r & 255)
		r >>= 8
		x := float32(r) * we_float[idx]
		if r < ke_float[idx] {
			return x
		}
		if idx == 0 {
			return ziggurat_exp_r_f - float32(math.Log1p(-float64(g.single())))
		}
		if (fe_float[idx-1]-fe_float[idx])*g.single()+fe_float[idx] < float32(math.Exp(-float64(x))) {
			return x
		}
	}
}

func (g *Generator) fill32(shape []int, sample func() float32) (*np.Array, error) {
	n, err := np.ElementCount(np.Float32, shape...)
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	g.mu.Lock()
	defer g.mu.Unlock()
	for i := range out {
		out[i] = sample()
	}
	return np.WrapSlice(out, shape...)
}

func (g *Generator) StandardNormal(shape ...int) (*np.Array, error) { return g.fill(shape, g.normal) }
func (g *Generator) StandardNormalDType(d np.DType, shape ...int) (*np.Array, error) {
	switch d {
	case np.Float64:
		return g.StandardNormal(shape...)
	case np.Float32:
		return g.fill32(shape, g.normal32)
	}
	return nil, np.ErrDType
}
func (g *Generator) StandardExponential(shape ...int) (*np.Array, error) {
	return g.fill(shape, g.exponential)
}
func (g *Generator) StandardExponentialDType(d np.DType, shape ...int) (*np.Array, error) {
	switch d {
	case np.Float64:
		return g.StandardExponential(shape...)
	case np.Float32:
		return g.fill32(shape, g.exponential32)
	}
	return nil, np.ErrDType
}
func (g *Generator) StandardExponentialInverse(shape ...int) (*np.Array, error) {
	return g.fill(shape, func() float64 { return -math.Log1p(-g.double()) })
}

func nonnegative(x float64) bool { return x >= 0 && !math.IsInf(x, 0) }
func positive(x float64) bool    { return x > 0 && !math.IsInf(x, 0) }

func (g *Generator) Normal(loc, scale float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("normal", shape, realArgs(loc, scale))
}
func (g *Generator) Uniform(low, high float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("uniform", shape, realArgs(low, high))
}
func (g *Generator) Exponential(scale float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("exponential", shape, realArgs(scale))
}

func (g *Generator) gamma(shape float64) float64 {
	switch {
	case shape == 0:
		return 0
	case shape == 1:
		return g.exponential()
	case shape < 1:
		for {
			u, v := g.double(), g.exponential()
			if u <= 1-shape {
				x := math.Pow(u, 1/shape)
				if x <= v {
					return x
				}
			} else {
				y := -math.Log((1 - u) / shape)
				x := math.Pow(1-shape+shape*y, 1/shape)
				if x <= v+y {
					return x
				}
			}
		}
	default:
		b := shape - 1.0/3
		c := 1 / math.Sqrt(9*b)
		for {
			x := g.normal()
			v := 1 + c*x
			if v <= 0 {
				continue
			}
			v = v * v * v
			u := g.double()
			xx := x * x
			if u < 1-0.0331*xx*xx || math.Log(u) < 0.5*xx+b*(1-v+math.Log(v)) {
				return b * v
			}
		}
	}
}
func (g *Generator) Gamma(k, scale float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("gamma", shape, realArgs(k, scale))
}
func (g *Generator) StandardGamma(k float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("standard_gamma", shape, realArgs(k))
}
func (g *Generator) beta(a, b float64) float64 {
	if a <= 1 && b <= 1 {
		if g.legacy {
			for {
				u, v := g.double(), g.double()
				x, y := math.Pow(u, 1/a), math.Pow(v, 1/b)
				if x+y <= 1 {
					if x+y > 0 {
						return x / (x + y)
					}
					lx, ly := math.Log(u)/a, math.Log(v)/b
					m := math.Max(lx, ly)
					lx -= m
					ly -= m
					return math.Exp(lx - math.Log(math.Exp(lx)+math.Exp(ly)))
				}
			}
		}
		if a < 3e-103 && b < 3e-103 {
			if (a+b)*g.double() < a {
				return 1
			}
			return 0
		}
		for {
			u, v := g.double(), g.double()
			x, y := math.Pow(u, 1/a), math.Pow(v, 1/b)
			if x+y <= 1 && u+v > 0 {
				if x > 0 && y > 0 {
					return x / (x + y)
				}
				delta := math.Log(u)/a - math.Log(v)/b
				if delta > 0 {
					return math.Exp(-math.Log1p(math.Exp(-delta)))
				}
				return math.Exp(delta - math.Log1p(math.Exp(delta)))
			}
		}
	}
	x, y := g.gamma(a), g.gamma(b)
	return x / (x + y)
}
func (g *Generator) Beta(a, b float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("beta", shape, realArgs(a, b))
}
func (g *Generator) Chisquare(df float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("chisquare", shape, realArgs(df))
}
func (g *Generator) F(dfnum, dfden float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("f", shape, realArgs(dfnum, dfden))
}
func (g *Generator) StandardCauchy(shape ...int) (*np.Array, error) {
	return g.fill(shape, func() float64 { x, y := g.normal(), g.normal(); return x / y })
}
func (g *Generator) StandardT(df float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("standard_t", shape, realArgs(df))
}
func (g *Generator) Pareto(a float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("pareto", shape, realArgs(a))
}
func (g *Generator) Weibull(a float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("weibull", shape, realArgs(a))
}
func (g *Generator) Power(a float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("power", shape, realArgs(a))
}
func (g *Generator) Laplace(loc, scale float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("laplace", shape, realArgs(loc, scale))
}
func (g *Generator) laplace(loc, scale float64) float64 {
	for {
		u := g.double()
		if u >= .5 {
			return loc - scale*math.Log(2-u-u)
		}
		if u > 0 {
			return loc + scale*math.Log(u+u)
		}
	}
}
func (g *Generator) Gumbel(loc, scale float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("gumbel", shape, realArgs(loc, scale))
}
func (g *Generator) gumbel(loc, scale float64) float64 {
	for {
		u := 1 - g.double()
		if u < 1 {
			return loc - scale*math.Log(-math.Log(u))
		}
	}
}
func (g *Generator) Logistic(loc, scale float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("logistic", shape, realArgs(loc, scale))
}
func (g *Generator) logistic(loc, scale float64) float64 {
	for {
		u := g.double()
		if u > 0 {
			return loc + scale*math.Log(u/(1-u))
		}
	}
}
func (g *Generator) Lognormal(mean, sigma float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("lognormal", shape, realArgs(mean, sigma))
}
func (g *Generator) Rayleigh(scale float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("rayleigh", shape, realArgs(scale))
}

func (g *Generator) fillInt(shape []int, sample func() int64) (*np.Array, error) {
	n, err := np.ElementCount(np.Int64, shape...)
	if err != nil {
		return nil, err
	}
	out := make([]int64, n)
	g.mu.Lock()
	defer g.mu.Unlock()
	for i := range out {
		out[i] = sample()
	}
	return np.WrapSlice(out, shape...)
}
func (g *Generator) poisson(lam float64) int64 {
	if lam == 0 {
		return 0
	}
	if lam < 10 {
		target := math.Exp(-lam)
		prod := 1.0
		var x int64
		for {
			prod *= g.double()
			if prod <= target {
				return x
			}
			x++
		}
	}
	slam, loglam := math.Sqrt(lam), math.Log(lam)
	b := 0.931 + 2.53*slam
	a := -0.059 + 0.02483*b
	invalpha := 1.1239 + 1.1328/(b-3.4)
	vr := 0.9277 - 3.6224/(b-2)
	for {
		u, v := g.double()-.5, g.double()
		us := .5 - math.Abs(u)
		if us == 0 {
			continue
		}
		k := int64(math.Floor((2*a/us+b)*u + lam + .43))
		if us >= .07 && v <= vr {
			return k
		}
		if k < 0 || us < .013 && v > us {
			continue
		}
		lg, _ := math.Lgamma(float64(k) + 1)
		if math.Log(v)+math.Log(invalpha)-math.Log(a/(us*us)+b) <= -lam+float64(k)*loglam-lg {
			return k
		}
	}
}
func (g *Generator) Poisson(lam float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("poisson", shape, realArgs(lam))
}
func (g *Generator) NegativeBinomial(n, p float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("negative_binomial", shape, realArgs(n, p))
}

// Binomial uses inversion for small means and BTPE for larger means.
func (g *Generator) Binomial(n int64, p float64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("binomial", shape, sampleArgs{f: [3]float64{0, p}, i: [3]int64{n}})
}
func (g *Generator) binomial(n int64, p float64) int64 {
	if (n == 0 || p == 0) && !g.legacy {
		return 0
	}
	r := math.Min(p, 1-p)
	nn := float64(n)
	var x int64
	if r*nn <= 30 {
		q := 1 - r
		qn := math.Exp(nn * math.Log1p(-r))
		if g.legacy {
			qn = math.Exp(nn * math.Log(q))
		}
		nr := nn * r
		bound := int64(math.Min(nn, nr+10*math.Sqrt(nr*q+1)))
		px, u := qn, g.double()
		for u > px {
			x++
			if x > bound {
				x = 0
				px = qn
				u = g.double()
			} else {
				u -= px
				px = float64(n-x+1) * r * px / (float64(x) * q)
			}
		}
	} else {
		x = g.binomialBTPE(n, r)
	}
	if p > 0.5 {
		return n - x
	}
	return x
}
func (g *Generator) binomialBTPE(n int64, r float64) int64 {
	nn := float64(n)
	q := 1 - r
	nrq := nn * r * q
	fm := nn*r + r
	m := math.Floor(fm)
	p1 := math.Floor(2.195*math.Sqrt(nrq)-4.6*q) + .5
	xm := m + .5
	xl, xr := xm-p1, xm+p1
	c := .134 + 20.5/(15.3+m)
	a := (fm - xl) / (fm - xl*r)
	laml := a * (1 + a/2)
	a = (xr - fm) / (xr * q)
	lamr := a * (1 + a/2)
	p2 := p1 * (1 + 2*c)
	p3 := p2 + c/laml
	p4 := p3 + c/lamr
	for {
		u, v := g.double()*p4, g.double()
		var y float64
		switch {
		case u <= p1:
			return int64(math.Floor(xm - p1*v + u))
		case u <= p2:
			x := xl + (u-p1)/c
			v = v*c + 1 - math.Abs(m-x+.5)/p1
			if v > 1 {
				continue
			}
			y = math.Floor(x)
		case u <= p3:
			if v == 0 {
				continue
			}
			y = math.Floor(xl + math.Log(v)/laml)
			if y < 0 {
				continue
			}
			v *= (u - p2) * laml
		default:
			if v == 0 {
				continue
			}
			y = math.Floor(xr - math.Log(v)/lamr)
			if y > nn {
				continue
			}
			v *= (u - p3) * lamr
		}
		k := math.Abs(y - m)
		if k <= 20 || k >= nrq/2-1 {
			s := r / q
			a := s * (nn + 1)
			f := 1.0
			if m < y {
				for i := m + 1; i <= y; i++ {
					f *= a/i - s
				}
			} else if m > y {
				for i := y + 1; i <= m; i++ {
					f /= a/i - s
				}
			}
			if v <= f {
				return int64(y)
			}
			continue
		}
		rho := k / nrq * ((k*(k/3+.625)+1.0/6)/nrq + .5)
		t := -k * k / (2 * nrq)
		logv := math.Log(v)
		if logv < t-rho {
			return int64(y)
		}
		if logv > t+rho {
			continue
		}
		x1, f1, z, w := y+1, m+1, nn+1-m, nn-y+1
		correction := func(v float64) float64 {
			v2 := v * v
			c := 13860.
			if g.legacy {
				c = 13680.
			}
			return (c - (462-(132-(99-140/v2)/v2)/v2)/v2) / v / 166320
		}
		last := correction(x1) + correction(w)
		if !g.legacy {
			last = -last
		}
		if logv <= xm*math.Log(f1/x1)+(nn-m+.5)*math.Log(z/w)+(y-m)*math.Log(w*r/(x1*q))+correction(f1)+correction(z)+last {
			return int64(y)
		}
	}
}
