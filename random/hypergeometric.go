package random

import (
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo"
)

func logfactorial(k int64) float64 {
	if k < int64(len(logFactorialTable)) {
		return logFactorialTable[k]
	}
	x := float64(k)
	return (x+.5)*math.Log(x) - x + (.9189385332046728 + (1/x)*(1.0/12-1/(360*x*x)))
}
func (g *Generator) hypergeometric(good, bad, sample int64) int64 {
	total := good + bad
	if g.legacy && sample <= 10 {
		if sample == 0 {
			return 0
		}
		d1 := total - sample
		d2 := float64(min(good, bad))
		y := d2
		for k := sample; k > 0 && y > 0; k-- {
			y -= math.Floor(g.double() + y/float64(d1+k))
		}
		z := int64(d2 - y)
		if good > bad {
			z = sample - z
		}
		return z
	}
	if !g.legacy && (sample < 10 || sample > total-10) {
		computed := min(sample, total-sample)
		remainingTotal, remainingGood := total, good
		for computed > 0 && remainingGood > 0 && remainingTotal > remainingGood {
			remainingTotal--
			if int64(g.interval(uint64(remainingTotal))) < remainingGood {
				remainingGood--
			}
			computed--
		}
		if remainingTotal == remainingGood {
			remainingGood -= computed
		}
		if sample > total/2 {
			return remainingGood
		}
		return good - remainingGood
	}
	computed := min(sample, total-sample)
	mingb, maxgb := min(good, bad), max(good, bad)
	p, q := float64(mingb)/float64(total), float64(maxgb)/float64(total)
	mu := float64(computed) * p
	a := mu + .5
	variance := float64(total-computed) * float64(computed) * p * q / float64(total-1)
	if g.legacy {
		variance = float64(total-computed) * float64(sample) * p * q / float64(total-1)
	}
	c := math.Sqrt(variance + .5)
	h := 1.7155277699214135*c + .8989161620588988
	m := int64(math.Floor(float64(computed+1) * float64(mingb+1) / float64(total+2)))
	lf := func(x int64) float64 {
		return logfactorial(x) + logfactorial(mingb-x) + logfactorial(computed-x) + logfactorial(maxgb-computed+x)
	}
	gp := lf(m)
	b := math.Min(float64(min(computed, mingb)+1), math.Floor(a+16*c))
	var k int64
	for {
		u, v := g.double(), g.double()
		x := a + h*(v-.5)/u
		if x < 0 || x >= b || math.IsNaN(x) {
			continue
		}
		k = int64(math.Floor(x))
		t := gp - lf(k)
		if u*(4-u)-3 <= t {
			break
		}
		if u*(u-t) >= 1 {
			continue
		}
		if 2*math.Log(u) <= t {
			break
		}
	}
	if good > bad {
		k = computed - k
	}
	if computed < sample {
		k = good - k
	}
	return k
}

// Hypergeometric samples the number of good items selected without replacement.
func (g *Generator) Hypergeometric(good, bad, sample int64, shape ...int) (*np.Array, error) {
	return g.sampleScalar("hypergeometric", shape, sampleArgs{i: [3]int64{good, bad, sample}})
}

// MultivariateHypergeometric reproduces NumPy's marginals (default) and
// count algorithms. The latter keeps a shuffled catalog between draws.
func (g *Generator) MultivariateHypergeometric(colors []int64, sample int64, method string, shape ...int) (*np.Array, error) {
	if method == "" {
		method = "marginals"
	}
	if method != "marginals" && method != "count" || sample < 0 {
		return nil, np.ErrValue
	}
	var total int64
	for _, x := range colors {
		if x < 0 || x > math.MaxInt64-total {
			return nil, np.ErrValue
		}
		total += x
	}
	if sample > total || method == "marginals" && total >= 1_000_000_000 {
		return nil, np.ErrValue
	}
	k := len(colors)
	outShape := append(slices.Clone(shape), k)
	size, err := np.ElementCount(np.Int64, outShape...)
	if err != nil {
		return nil, err
	}
	out := make([]int64, size)
	if k == 0 || total == 0 || sample == 0 || size == 0 {
		return np.WrapSlice(out, outShape...)
	}
	more := sample > total/2
	computed := min(sample, total-sample)
	var choices []int
	if method == "count" {
		if total > int64(int(^uint(0)>>1))/8 {
			return nil, np.ErrShape
		}
		choices = make([]int, int(total))
		index := 0
		for i, n := range colors {
			for j := int64(0); j < n; j++ {
				choices[index] = i
				index++
			}
		}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	for off := 0; off < size; off += k {
		if method == "count" {
			for j := 0; j < int(computed); j++ {
				i := j + int(g.interval(uint64(total-int64(j)-1)))
				choices[i], choices[j] = choices[j], choices[i]
			}
			for _, x := range choices[:int(computed)] {
				out[off+x]++
			}
		} else {
			n, remaining := computed, total
			for j, c := range colors[:k-1] {
				if n == 0 {
					break
				}
				remaining -= c
				draw := g.hypergeometric(c, remaining, n)
				out[off+j] = draw
				n -= draw
			}
			out[off+k-1] = n
		}
		if more {
			for i, c := range colors {
				out[off+i] = c - out[off+i]
			}
		}
	}
	return np.WrapSlice(out, outShape...)
}
