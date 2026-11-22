package random

import (
	"math"
	"math/bits"
	"slices"
	"sort"

	"github.com/SamuelSupe/numpygo"
)

func (g *Generator) interval(maximum uint64) uint64 {
	if maximum == 0 {
		return 0
	}
	mask := ^uint64(0) >> bits.LeadingZeros64(maximum)
	for {
		var v uint64
		if maximum <= 0xffffffff {
			v = uint64(g.next32()) & mask
		} else {
			v = g.next64() & mask
		}
		if v <= maximum {
			return v
		}
	}
}
func (g *Generator) Permutation(n int) (*np.Array, error) {
	if _, err := np.ElementCount(np.Int64, n); err != nil {
		return nil, err
	}
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(i)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	for i := n - 1; i > 0; i-- {
		j := int(g.interval(uint64(i)))
		out[i], out[j] = out[j], out[i]
	}
	return np.WrapSlice(out, n)
}
func (g *Generator) PermutationArray(a *np.Array, axis int) (*np.Array, error) {
	if a.DType() == 0 {
		return nil, np.ErrDType
	}
	ax, err := np.NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	p, err := g.Permutation(a.Shape()[ax])
	if err != nil {
		return nil, err
	}
	return np.Take(a, p, &ax)
}
func (g *Generator) Shuffle(a *np.Array, axis int) error {
	if !a.Writable() {
		return np.ErrReadOnly
	}
	out, err := g.PermutationArray(a, axis)
	if err != nil {
		return err
	}
	return a.Assign(nil, out)
}

// Permuted independently shuffles each one-dimensional slice along axis.
// A nil axis shuffles the flattened array and restores its shape.
func (g *Generator) Permuted(a *np.Array, axis *int) (*np.Array, error) {
	return g.PermutedWithOut(a, axis, nil)
}

// PermutedWithOut supports NumPy's independent slice permutations into
// an existing equally-shaped writable array with safe casting.
func (g *Generator) PermutedWithOut(a *np.Array, axis *int, out *np.Array) (*np.Array, error) {
	if a.DType() == 0 {
		return nil, np.ErrDType
	}
	var err error
	if axis != nil {
		if _, err = np.NormalizeAxis(*axis, a.NDim()); err != nil {
			return nil, err
		}
	}
	if out == nil {
		out, err = np.EmptyLike(a, a.DType(), "K")
		if err != nil {
			return nil, err
		}
	} else {
		if !out.Writable() {
			return nil, np.ErrReadOnly
		}
		if !slices.Equal(a.Shape(), out.Shape()) {
			return nil, np.ErrShape
		}
		ok, e := np.CanCast(a.DType(), out.DType(), "safe")
		if e != nil || !ok {
			return nil, np.ErrDType
		}
	}
	finish := func(v *np.Array, e error) (*np.Array, error) {
		if e != nil {
			return nil, e
		}
		if e = out.Assign(nil, v); e != nil {
			return nil, e
		}
		return out, nil
	}
	if axis == nil {
		fortran := out.IsFContiguous() && !out.IsCContiguous()
		if fortran {
			a, err = a.Transpose()
			if err != nil {
				return nil, err
			}
		}
		x, err := np.Ravel(a)
		if err != nil {
			return nil, err
		}
		p, err := g.PermutationArray(x, 0)
		if err != nil {
			return nil, err
		}
		p, err = p.Reshape(a.Shape()...)
		if err != nil {
			return nil, err
		}
		if fortran {
			p, err = p.Transpose()
		}
		return finish(p, err)
	}
	shape := a.Shape()
	ax, err := np.NormalizeAxis(*axis, len(shape))
	if err != nil {
		return nil, err
	}
	n := shape[ax]
	if a.Size() == 0 {
		return finish(a, nil)
	}
	stride := 1
	for _, v := range shape[ax+1:] {
		stride *= v
	}
	outer := a.Size() / (n * stride)
	indices := make([]int64, a.Size())
	order := make([]int64, n)
	g.mu.Lock()
	for batch := 0; batch < outer; batch++ {
		for lane := 0; lane < stride; lane++ {
			for i := range order {
				order[i] = int64(i)
			}
			for i := n - 1; i > 0; i-- {
				j := int(g.interval(uint64(i)))
				order[i], order[j] = order[j], order[i]
			}
			for i, v := range order {
				indices[(batch*n+i)*stride+lane] = v
			}
		}
	}
	g.mu.Unlock()
	idx, err := np.WrapSlice(indices, shape...)
	if err != nil {
		return nil, err
	}
	return finish(np.TakeAlongAxis(a, idx, &ax))
}

type ChoiceOptions struct {
	WithoutReplacement bool
	Probabilities      []float64
	NoShuffle          bool
}

// Choice samples indices from [0,population), optionally weighted or without replacement.
func (g *Generator) Choice(population int, opt ChoiceOptions, shape ...int) (*np.Array, error) {
	n, err := np.ElementCount(np.Int64, shape...)
	if err != nil {
		return nil, err
	}
	if population < 0 || population == 0 && n != 0 || opt.WithoutReplacement && n > population {
		return nil, np.ErrValue
	}
	p := slices.Clone(opt.Probabilities)
	if opt.Probabilities != nil {
		if len(p) != population {
			return nil, np.ErrShape
		}
		var sum float64
		nonzero := 0
		for _, x := range p {
			if x < 0 || math.IsNaN(x) || math.IsInf(x, 0) {
				return nil, np.ErrValue
			}
			sum += x
			if x > 0 {
				nonzero++
			}
		}
		if math.Abs(sum-1) > math.Sqrt(2.220446049250313e-16) || opt.WithoutReplacement && nonzero < n {
			return nil, np.ErrValue
		}
	}
	out := make([]int64, n)
	g.mu.Lock()
	defer g.mu.Unlock()
	buf := subword{}
	if p == nil {
		if !opt.WithoutReplacement {
			for i := range out {
				out[i] = int64(g.bounded(uint64(population-1), 64, &buf))
			}
		} else {
			if g.legacy {
				idx := make([]int64, population)
				for i := range idx {
					idx[i] = int64(i)
				}
				for i := population - 1; i > 0; i-- {
					j := int(g.interval(uint64(i)))
					idx[i], idx[j] = idx[j], idx[i]
				}
				copy(out, idx[:n])
				return np.WrapSlice(out, shape...)
			}
			cutoff := 50
			if opt.NoShuffle {
				cutoff = 20
			}
			if population > 10000 && n > population/cutoff {
				idx := make([]int64, population)
				for i := range idx {
					idx[i] = int64(i)
				}
				for i := population - 1; i >= max(population-n, 1); i-- {
					j := int(g.bounded(uint64(i), 64, &buf))
					idx[i], idx[j] = idx[j], idx[i]
				}
				copy(out, idx[population-n:])
			} else {
				seen := make(map[uint64]struct{}, n)
				for j := population - n; j < population; j++ {
					v := g.bounded(uint64(j), 64, &buf)
					if _, ok := seen[v]; ok {
						v = uint64(j)
					}
					seen[v] = struct{}{}
					out[j-population+n] = int64(v)
				}
				if !opt.NoShuffle {
					for i := n - 1; i > 0; i-- {
						j := int(g.bounded(uint64(i), 64, &buf))
						out[i], out[j] = out[j], out[i]
					}
				}
			}
		}
	} else {
		found := 0
		for found < n {
			cdf := make([]float64, len(p))
			var total float64
			for j, x := range p {
				total += x
				cdf[j] = total
			}
			for j := range cdf {
				cdf[j] /= total
			}
			draws := n - found
			seen := make(map[int]struct{}, draws)
			for i := 0; i < draws; i++ {
				u := g.double()
				idx := sort.Search(len(cdf), func(j int) bool { return cdf[j] > u })
				if !opt.WithoutReplacement {
					out[found] = int64(idx)
					found++
					continue
				}
				if _, ok := seen[idx]; ok {
					continue
				}
				seen[idx] = struct{}{}
				out[found] = int64(idx)
				found++
			}
			if opt.WithoutReplacement {
				for i := range seen {
					p[i] = 0
				}
			}
		}
	}
	return np.WrapSlice(out, shape...)
}

// ChoiceArray selects along an array axis, inserting the requested sample shape.
func (g *Generator) ChoiceArray(a *np.Array, axis int, opt ChoiceOptions, shape ...int) (*np.Array, error) {
	if a.DType() == 0 {
		return nil, np.ErrDType
	}
	ax, err := np.NormalizeAxis(axis, a.NDim())
	if err != nil {
		return nil, err
	}
	idx, err := g.Choice(a.Shape()[ax], opt, shape...)
	if err != nil {
		return nil, err
	}
	return np.Take(a, idx, &ax)
}
