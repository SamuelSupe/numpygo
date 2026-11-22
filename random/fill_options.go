package random

import (
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

type FillOptions struct {
	Size   []int
	DType  np.DType
	Out    *np.Array
	Method string // zig or inv for standard_exponential
}

// FillRandom supports the Generator methods exposing dtype and out. For
// standard_gamma provide one shape-parameter array, otherwise no parameters.
// A nil size uses out.shape, the parameter's broadcast shape, or scalar.
func (g *Generator) FillRandom(name string, opt FillOptions, params ...*np.Array) (*np.Array, error) {
	d := opt.DType
	if d == 0 {
		d = np.Float64
	}
	if d != np.Float32 && d != np.Float64 {
		return nil, np.ErrDType
	}
	if g.legacy && (opt.Out != nil || d == np.Float32) {
		return nil, np.ErrDType
	}
	gamma := name == "standard_gamma"
	if gamma && len(params) != 1 || !gamma && len(params) != 0 {
		return nil, np.ErrShape
	}
	if name != "random" && name != "standard_normal" && name != "standard_exponential" && !gamma {
		return nil, np.ErrValue
	}
	if opt.Method != "" && (name != "standard_exponential" || opt.Method != "zig" && opt.Method != "inv") {
		return nil, np.ErrValue
	}
	size := opt.Size
	fOrder := false
	if opt.Out != nil {
		out := opt.Out
		if out.DType() != d {
			return nil, np.ErrDType
		}
		if !out.Writable() {
			return nil, np.ErrReadOnly
		}
		if !out.NativeByteOrder() || out.Size() > 0 && out.DataIdentity()%uintptr(d.ItemSize()) != 0 {
			return nil, np.ErrValue
		}
		if !out.IsCContiguous() {
			if gamma || !out.IsFContiguous() {
				return nil, np.ErrValue
			}
			fOrder = true
		}
		if size != nil && !slices.Equal(size, out.Shape()) {
			return nil, np.ErrShape
		}
		size = out.Shape()
	}
	var a parameter
	var natural []int
	if gamma {
		var e error
		a, e = snapshotParameter(params[0], false)
		if e != nil {
			return nil, e
		}
		natural = a.shape
		for i, x := range a.f {
			if d == np.Float32 {
				x = float64(float32(x))
				a.f[i] = x
			}
			if !nonnegativeScale(x) {
				return nil, np.ErrValue
			}
		}
	}
	_, shape, count, e := sampleShape(size, d, natural)
	if e != nil {
		return nil, e
	}
	var result *np.Array
	if d == np.Float32 {
		out := make([]float32, count)
		g.mu.Lock()
		for i := range out {
			switch name {
			case "random":
				out[i] = g.single()
			case "standard_normal":
				out[i] = g.normal32()
			case "standard_exponential":
				if opt.Method == "inv" {
					out[i] = -float32(math.Log1p(-float64(g.single())))
				} else {
					out[i] = g.exponential32()
				}
			case "standard_gamma":
				out[i] = g.gamma32(float32(a.f[parameterIndex(i, shape, a.shape)]))
			}
		}
		g.mu.Unlock()
		result, e = np.WrapSlice(out, shape...)
	} else {
		out := make([]float64, count)
		g.mu.Lock()
		for i := range out {
			switch name {
			case "random":
				out[i] = g.double()
			case "standard_normal":
				out[i] = g.normal()
			case "standard_exponential":
				if opt.Method == "inv" {
					out[i] = -math.Log1p(-g.double())
				} else {
					out[i] = g.exponential()
				}
			case "standard_gamma":
				out[i] = g.gamma(a.f[parameterIndex(i, shape, a.shape)])
			}
		}
		g.mu.Unlock()
		result, e = np.WrapFloat64(out, shape...)
	}
	if e != nil || opt.Out == nil {
		return result, e
	}
	if fOrder {
		reverse := slices.Clone(shape)
		slices.Reverse(reverse)
		result, e = result.Reshape(reverse...)
		if e != nil {
			return nil, e
		}
		result, e = result.Transpose()
		if e != nil {
			return nil, e
		}
	}
	if e = np.CopyTo(opt.Out, result, nil); e != nil {
		return nil, e
	}
	return opt.Out, nil
}

// NumPy has a separate float32 gamma sampler, consuming uint32 words
// rather than casting the double-precision gamma stream.
func (g *Generator) gamma32(shape float32) float32 {
	logf := func(x float32) float32 { return float32(math.Log(float64(x))) }
	powf := func(x, y float32) float32 { return float32(math.Pow(float64(x), float64(y))) }
	switch {
	case shape == 0:
		return 0
	case shape == 1:
		return g.exponential32()
	case shape < 1:
		for {
			u, v := g.single(), g.exponential32()
			if u <= 1-shape {
				if x := powf(u, 1/shape); x <= v {
					return x
				}
			} else {
				y := -logf((1 - u) / shape)
				if x := powf(1-shape+shape*y, 1/shape); x <= v+y {
					return x
				}
			}
		}
	default:
		b := shape - float32(1./3)
		c := 1 / float32(math.Sqrt(float64(9*b)))
		for {
			x := g.normal32()
			v := 1 + c*x
			if v <= 0 {
				continue
			}
			v = v * v * v
			u := g.single()
			xx := x * x
			if u < 1-float32(.0331)*xx*xx || logf(u) < .5*x*x+b*(1-v+logf(v)) {
				return b * v
			}
		}
	}
}
