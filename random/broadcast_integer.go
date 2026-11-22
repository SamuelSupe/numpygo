package random

import (
	"math/big"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

// IntegerOptions specifies result shape, dtype and whether high is inclusive.
// Nil Size infers the broadcast of the bound arrays.
type IntegerOptions struct {
	Size     []int
	DType    np.DType
	Endpoint bool
}

func integerValues(a *np.Array) ([]big.Int, error) {
	if !a.DType().IsInteger() && a.DType() != np.Bool {
		return nil, np.ErrDType
	}
	out := make([]big.Int, a.Size())
	if a.DType().IsUnsigned() || a.DType() == np.Bool {
		v, e := a.Astype(np.Uint64)
		if e != nil {
			return nil, e
		}
		u, e := np.Values[uint64](v)
		if e != nil {
			return nil, e
		}
		for i, x := range u {
			out[i].SetUint64(x)
		}
	} else {
		v, e := a.Astype(np.Int64)
		if e != nil {
			return nil, e
		}
		u, e := np.Values[int64](v)
		if e != nil {
			return nil, e
		}
		for i, x := range u {
			out[i].SetInt64(x)
		}
	}
	return out, nil
}

// IntegersArray supports exact fixed-width signed, unsigned and bool
// parameter arrays, including uint64 bounds beyond the float64 lattice.
// A nil high means [0,low), or [0,low] with Endpoint.
func (g *Generator) IntegersArray(low, high *np.Array, opt IntegerOptions) (*np.Array, error) {
	if high == nil {
		high = low
		low = np.ScalarOf(int64(0))
	}
	d := opt.DType
	if d == 0 {
		d = np.Int64
	}
	if !d.IsInteger() && d != np.Bool {
		return nil, np.ErrDType
	}
	lv, e := integerValues(low)
	if e != nil {
		return nil, e
	}
	hv, e := integerValues(high)
	if e != nil {
		return nil, e
	}
	ls, hs := low.Shape(), high.Shape()
	natural, shape, count, e := sampleShape(opt.Size, d, ls, hs)
	if e != nil {
		return nil, e
	}
	lanes, e := np.ElementCount(d, natural...)
	if e != nil {
		return nil, e
	}
	width := d.ItemSize() * 8
	if d == np.Bool {
		width = 1
	}
	limit := new(big.Int).Lsh(big.NewInt(1), uint(width))
	minimum := big.NewInt(0)
	maximum := new(big.Int).Sub(new(big.Int).Set(limit), big.NewInt(1))
	if d.IsSigned() {
		half := new(big.Int).Rsh(new(big.Int).Set(limit), 1)
		minimum.Neg(half)
		maximum.Sub(half, big.NewInt(1))
	}
	type bound struct {
		low uint64
		rng boundedRange
	}
	bounds := make([]bound, lanes)
	one := big.NewInt(1)
	// Validate the original bounds independently, including empty output.
	for j := range lv {
		if lv[j].Cmp(minimum) < 0 || lv[j].Cmp(maximum) > 0 {
			return nil, np.ErrValue
		}
	}
	for j := range hv {
		x := &hv[j]
		closed := new(big.Int).Set(x)
		if !opt.Endpoint {
			closed.Sub(closed, one)
		}
		if closed.Cmp(minimum) < 0 || closed.Cmp(maximum) > 0 {
			return nil, np.ErrValue
		}
	}
	for j := range bounds {
		l := &lv[parameterIndex(j, natural, ls)]
		h := new(big.Int).Set(&hv[parameterIndex(j, natural, hs)])
		if !opt.Endpoint {
			h.Sub(h, one)
		}
		if h.Cmp(l) < 0 {
			return nil, np.ErrValue
		}
		var bits uint64
		if l.Sign() < 0 {
			bits = uint64(l.Int64())
		} else {
			bits = l.Uint64()
		}
		bounds[j] = bound{bits, planBoundedRange(h.Sub(h, l).Uint64(), width, g.legacy)}
	}
	return g.integerOutput(count, d, shape, func(j int, buf *subword) uint64 {
		b := bounds[parameterIndex(j, shape, natural)]
		return b.low + g.drawBounded(b.rng, buf)
	})
}

// MultinomialArray broadcasts trial counts with the leading dimensions of
// pvals; the last dimension is the category vector, not a broadcast axis.
// RandomState retains NumPy's legacy scalar-n/one-vector restriction.
func (g *Generator) MultinomialArray(n, pvals *np.Array, size []int) (*np.Array, error) {
	if n.DType() == 0 {
		return nil, np.ErrDType
	}
	if pvals.DType() == 0 || pvals.DType().IsComplex() || pvals.NDim() == 0 {
		return nil, np.ErrShape
	}
	if g.legacy && (n.NDim() != 0 || pvals.NDim() != 1) {
		return nil, np.ErrShape
	}
	trials, e := snapshotParameter(n, true)
	if e != nil {
		return nil, e
	}
	probs, e := snapshotParameter(pvals, false)
	if e != nil {
		return nil, e
	}
	k := probs.shape[len(probs.shape)-1]
	if k == 0 {
		return nil, np.ErrValue
	}
	pShape := probs.shape[:len(probs.shape)-1]
	_, shape, count, e := sampleShape(size, np.Int64, trials.shape, pShape)
	if e != nil {
		return nil, e
	}
	for _, x := range trials.i {
		if x < 0 {
			return nil, np.ErrValue
		}
	}
	for i := 0; i < len(probs.f); i += k {
		if !validMultinomial(probs.f[i : i+k]) {
			return nil, np.ErrValue
		}
	}
	outShape := append(slices.Clone(shape), k)
	total, e := np.ElementCount(np.Int64, outShape...)
	if e != nil {
		return nil, e
	}
	out := make([]int64, total)
	g.mu.Lock()
	for i := 0; i < count; i++ {
		t := trials.i[parameterIndex(i, shape, trials.shape)]
		j := parameterIndex(i, shape, pShape) * k
		g.multinomial(t, probs.f[j:j+k], out[i*k:(i+1)*k])
	}
	g.mu.Unlock()
	return np.WrapSlice(out, outShape...)
}
