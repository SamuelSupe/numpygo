package np

import (
	"math"
	"slices"
	"sort"
)

type EinsumPathOptions struct {
	Optimize    string // "greedy" (default), "optimal", or "none"
	MemoryLimit int64  // number of elements in a temporary; zero uses the largest input/output
}

// EinsumWithPath executes a precomputed contraction plan with the existing
// numerical kernels. Invalid or incomplete paths are rejected.
func EinsumWithPath(subscripts string, plan EinsumPlan, arrays ...*Array) (*Array, error) {
	spec, e := parseEinsum(subscripts, arrays)
	if e != nil {
		return nil, e
	}
	current := slices.Clone(arrays)
	labels := slices.Clone(spec.labels)
	neededOutput := toLabelSet(spec.output)
	for _, step := range plan.Steps {
		if len(step) == 0 {
			return nil, ErrValue
		}
		selected := map[int]bool{}
		contracted := labelSet{}
		var operands []*Array
		var inputLabels [][]int
		for _, i := range step {
			if i < 0 || i >= len(current) || selected[i] {
				return nil, ErrIndex
			}
			selected[i] = true
			operands = append(operands, current[i])
			inputLabels = append(inputLabels, labels[i])
			unionLabels(contracted, toLabelSet(labels[i]))
		}
		var rest []*Array
		var restLabels [][]int
		needed := labelSet{}
		unionLabels(needed, neededOutput)
		for i, a := range current {
			if !selected[i] {
				rest = append(rest, a)
				restLabels = append(restLabels, labels[i])
				unionLabels(needed, toLabelSet(labels[i]))
			}
		}
		var resultLabels []int
		if len(rest) == 0 {
			resultLabels = slices.Clone(spec.output)
		} else {
			for l := range contracted {
				if needed[l] {
					resultLabels = append(resultLabels, l)
				}
			}
			sort.Ints(resultLabels)
		}
		result, e := contractPlanned(operands, inputLabels, resultLabels)
		if e != nil {
			return nil, e
		}
		current = append(rest, result)
		labels = append(restLabels, resultLabels)
	}
	if len(current) != 1 || !slices.Equal(labels[0], spec.output) {
		return nil, ErrValue
	}
	return current[0], nil
}

type EinsumPlan struct {
	Steps                                         [][]int
	NaiveCost, OptimizedCost, LargestIntermediate int64
}
type labelSet map[int]bool

func toLabelSet(v []int) labelSet {
	s := labelSet{}
	for _, l := range v {
		s[l] = true
	}
	return s
}
func unionLabels(dst, src labelSet) {
	for l := range src {
		dst[l] = true
	}
}
func intersects(a, b labelSet) bool {
	for i := range a {
		if b[i] {
			return true
		}
	}
	return false
}
func saturateAdd(a, b int64) int64 {
	if a > math.MaxInt64-b {
		return math.MaxInt64
	}
	return a + b
}
func saturateMul(a, b int64) int64 {
	if a != 0 && b > math.MaxInt64/a {
		return math.MaxInt64
	}
	return a * b
}

// EinsumPath computes an executable contraction order using NumPy's
// memory-bounded greedy or exhaustive pair search. Operand positions refer
// to the remaining operands after each step (new results are appended).
func EinsumPath(subscripts string, options EinsumPathOptions, arrays ...*Array) (EinsumPlan, error) {
	var out EinsumPlan
	spec, e := parseEinsum(subscripts, arrays)
	if e != nil {
		return out, e
	}
	dims := map[int]int64{}
	sets := make([]labelSet, len(arrays))
	for i, a := range arrays {
		if a.DType() == 0 {
			return out, ErrDType
		}
		local := map[int]int{}
		for j, l := range spec.labels[i] {
			n := a.shape[j]
			if p, ok := local[l]; ok && p != n {
				return out, ErrShape
			}
			local[l] = n
			if p, ok := dims[l]; !ok || p == 1 {
				dims[l] = int64(n)
			} else if n != 1 && p != int64(n) {
				return out, ErrShape
			}
		}
		sets[i] = toLabelSet(spec.labels[i])
	}
	output := toLabelSet(spec.output)
	if len(output) != len(spec.output) {
		return out, ErrShape
	}
	for l := range output {
		if _, ok := dims[l]; !ok {
			return out, ErrShape
		}
	}
	size := func(s labelSet) int64 {
		n := int64(1)
		for l := range s {
			n = saturateMul(n, dims[l])
		}
		return n
	}
	limit := options.MemoryLimit
	if limit < 0 {
		return out, ErrValue
	}
	if limit == 0 {
		limit = size(output)
		for _, s := range sets {
			limit = max(limit, size(s))
		}
	}
	all := labelSet{}
	for _, s := range sets {
		unionLabels(all, s)
	}
	cost := func(contracted, result labelSet, n int) int64 {
		factor := int64(max(1, n-1))
		if len(contracted) > len(result) {
			factor++
		}
		return saturateMul(size(contracted), factor)
	}
	out.NaiveCost = cost(all, output, len(sets))
	type candidate struct {
		pair                        []int
		remaining                   []labelSet
		removed, cost, intermediate int64
	}
	find := func(current []labelSet, i, j int) candidate {
		used, needed := labelSet{}, labelSet{}
		unionLabels(used, current[i])
		unionLabels(used, current[j])
		unionLabels(needed, output)
		var remaining []labelSet
		for k, s := range current {
			if k != i && k != j {
				remaining = append(remaining, s)
				unionLabels(needed, s)
			}
		}
		result := labelSet{}
		for l := range used {
			if needed[l] {
				result[l] = true
			}
		}
		n := size(result)
		remaining = append(remaining, result)
		return candidate{[]int{i, j}, remaining, saturateAdd(size(current[i]), size(current[j])) - n, cost(used, result, 2), n}
	}
	contractAll := func(current []labelSet) []int {
		p := make([]int, len(current))
		for i := range p {
			p[i] = i
		}
		return p
	}
	mode := options.Optimize
	if mode == "" {
		mode = "greedy"
	}
	if mode != "greedy" && mode != "optimal" && mode != "none" {
		return out, ErrValue
	}
	if len(sets) <= 2 || mode == "none" {
		out.Steps = [][]int{contractAll(sets)}
		out.OptimizedCost = saturateAdd(out.NaiveCost, 1)
		out.LargestIntermediate = size(output)
		return out, nil
	}
	if mode == "optimal" {
		// Exhaustive path search grows factorially, as NumPy documents. Use
		// greedy for more operands.
		if len(sets) > 8 {
			return out, ErrValue
		}
		bestCost := int64(math.MaxInt64)
		var best [][]int
		var bestLargest int64
		var visit func([]labelSet, [][]int, int64, int64)
		visit = func(cur []labelSet, steps [][]int, sum, largest int64) {
			if len(cur) == 1 {
				if sum < bestCost {
					bestCost = sum
					best = slices.Clone(steps)
					bestLargest = largest
				}
				return
			}
			for i := 0; i < len(cur); i++ {
				for j := i + 1; j < len(cur); j++ {
					c := find(cur, i, j)
					s := saturateAdd(sum, c.cost)
					if c.intermediate > limit || s >= bestCost {
						continue
					}
					visit(c.remaining, append(steps, c.pair), s, max(largest, c.intermediate))
				}
			}
		}
		visit(sets, nil, 0, 0)
		if best != nil {
			out.Steps = best
			out.OptimizedCost = saturateAdd(bestCost, 1)
			out.LargestIntermediate = bestLargest
			return out, nil
		}
	}
	total := int64(0)
	for len(sets) > 1 {
		var best *candidate
		for _, outer := range []bool{false, true} {
			for i := 0; i < len(sets); i++ {
				for j := i + 1; j < len(sets); j++ {
					if !outer && !intersects(sets[i], sets[j]) {
						continue
					}
					c := find(sets, i, j)
					if c.intermediate > limit || saturateAdd(total, c.cost) > out.NaiveCost {
						continue
					}
					if best == nil || c.removed > best.removed || c.removed == best.removed && c.cost < best.cost {
						v := c
						best = &v
					}
				}
			}
			if best != nil {
				break
			}
		}
		if best == nil {
			p := contractAll(sets)
			out.Steps = append(out.Steps, p)
			u := labelSet{}
			for _, s := range sets {
				unionLabels(u, s)
			}
			total = saturateAdd(total, cost(u, output, len(sets)))
			out.LargestIntermediate = max(out.LargestIntermediate, size(output))
			break
		}
		out.Steps = append(out.Steps, best.pair)
		total = saturateAdd(total, best.cost)
		out.LargestIntermediate = max(out.LargestIntermediate, best.intermediate)
		sets = best.remaining
	}
	out.OptimizedCost = saturateAdd(total, 1)
	return out, nil
}
