package np

import (
	"slices"
	"sort"
)

// Insert inserts values before the indicated original axis positions. A scalar
// index follows NumPy's special axis-movement rule for a single insertion.
func Insert(a, indices, values *Array, axis *int) (*Array, error) {
	if a.DType() == 0 || indices.DType() == 0 || values.DType() == 0 {
		return nil, ErrDType
	}
	if axis == nil {
		var err error
		a, err = Ravel(a)
		if err != nil {
			return nil, err
		}
		values, err = Ravel(values)
		if err != nil {
			return nil, err
		}
		axis = Int(0)
	}
	ax, err := NormalizeAxis(*axis, a.NDim())
	if err != nil {
		return nil, err
	}
	n := a.shape[ax]
	scalar := indices.NDim() == 0
	if indices.DType() == Bool {
		if indices.NDim() != 1 || indices.size != n {
			return nil, ErrShape
		}
		indices, err = FlatNonzero(indices)
		if err != nil {
			return nil, err
		}
		scalar = false
	}
	if !indices.DType().IsInteger() || indices.NDim() > 1 {
		return nil, ErrIndex
	}
	if indices.size == 0 {
		return a.Copy()
	}
	positions := make([]int, indices.size)
	var failure error
	indices.walk(func(i, off int) {
		if failure != nil {
			return
		}
		e := indices.elementAt(off)
		if indices.DType().IsSigned() {
			v := int64(e.i)
			if v < 0 {
				v += int64(n)
			}
			if v < 0 || v > int64(n) {
				failure = ErrIndex
				return
			}
			positions[i] = int(v)
		} else {
			if e.i > uint64(n) {
				failure = ErrIndex
				return
			}
			positions[i] = int(e.i)
		}
	})
	if failure != nil {
		return nil, failure
	}
	if values.NDim() > a.NDim() {
		return nil, ErrShape
	}
	if values.NDim() < a.NDim() {
		sh := make([]int, a.NDim()-values.NDim())
		for i := range sh {
			sh[i] = 1
		}
		sh = append(sh, values.shape...)
		values, err = values.Reshape(sh...)
		if err != nil {
			return nil, err
		}
	}
	if scalar && len(positions) == 1 {
		values, err = MoveAxis(values, 0, ax)
		if err != nil {
			return nil, err
		}
	}
	if len(positions) == 1 {
		count := values.shape[ax]
		if count == 0 {
			return a.Copy()
		}
		p := positions[0]
		positions = make([]int, count)
		for i := range positions {
			positions[i] = p
		}
	}
	count := len(positions)
	if n > maxInt-count {
		return nil, ErrShape
	}
	vshape := slices.Clone(a.shape)
	vshape[ax] = count
	values, err = values.BroadcastTo(vshape...)
	if err != nil {
		return nil, err
	}
	// Stable ordering preserves the order of multiple insertions at one position.
	order := make([]int, count)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return positions[order[i]] < positions[order[j]] })
	shape := slices.Clone(a.shape)
	shape[ax] += count
	out, err := a.zerosLike(shape)
	if err != nil {
		return nil, err
	}
	spec := func(i int) []Index {
		p := make([]Index, ax+1)
		for j := range p {
			p[j] = Slice{}
		}
		p[ax] = AtIndex(i)
		return p
	}
	write := func(dst int, src *Array, idx int) error {
		v, err := src.Slice(spec(idx)...)
		if err != nil {
			return err
		}
		return out.Assign(spec(dst), v)
	}
	dst, ins := 0, 0
	for p := 0; p <= n; p++ {
		for ins < count && positions[order[ins]] == p {
			if err := write(dst, values, order[ins]); err != nil {
				return nil, err
			}
			dst++
			ins++
		}
		if p < n {
			if err := write(dst, a, p); err != nil {
				return nil, err
			}
			dst++
		}
	}
	return out, nil
}

// Pad supports constant, edge, wrap, reflect, and symmetric extension.
func Pad(a *Array, widths [][2]int, mode string, constant any) (*Array, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if len(widths) == 1 && a.NDim() > 1 {
		w := widths[0]
		widths = make([][2]int, a.NDim())
		for i := range widths {
			widths[i] = w
		}
	}
	if len(widths) != a.NDim() {
		return nil, ErrShape
	}
	if mode == "" {
		mode = "constant"
	}
	if mode != "constant" && mode != "edge" && mode != "wrap" && mode != "reflect" && mode != "symmetric" {
		opt := PadOptions{Mode: mode}
		if constant != nil && mode == "linear_ramp" {
			opt.EndValues = [][2]any{{constant, constant}}
		}
		return PadWithOptions(a, widths, opt)
	}
	shape := a.Shape()
	for i, w := range widths {
		if w[0] < 0 || w[1] < 0 || w[0] > maxInt-shape[i] || w[1] > maxInt-shape[i]-w[0] {
			return nil, ErrShape
		}
		if mode != "constant" && a.shape[i] == 0 && w[0]+w[1] > 0 {
			return nil, ErrValue
		}
		shape[i] += w[0] + w[1]
	}
	out, err := a.zerosLike(shape)
	if err != nil {
		return nil, err
	}
	var fill element
	if mode == "constant" && constant != nil {
		d := dtypeOf(constant)
		if d == 0 {
			return nil, ErrDType
		}
		fill = castElement(elementFrom(constant), d, a.DType())
	}
	for flat := 0; flat < out.size; flat++ {
		k, off := flat, a.offset
		outside := false
		for axis := len(shape) - 1; axis >= 0; axis-- {
			x := k%shape[axis] - widths[axis][0]
			k /= shape[axis]
			n := a.shape[axis]
			if x < 0 || x >= n {
				switch mode {
				case "constant":
					outside = true
				case "edge":
					x = max(0, min(n-1, x))
				case "wrap":
					x %= n
					if x < 0 {
						x += n
					}
				case "reflect", "symmetric":
					if n == 1 {
						x = 0
						break
					}
					// Reflect with a period of twice the effective width.
					m := n - 1
					if mode == "symmetric" {
						m = n
					}
					if m > maxInt/2 {
						return nil, ErrShape
					}
					p := 2 * m
					x %= p
					if x < 0 {
						x += p
					}
					if x >= n {
						if mode == "reflect" {
							x = p - x
						} else {
							x = p - 1 - x
						}
					}
				}
			}
			if outside {
				break
			}
			off += x * a.strides[axis]
		}
		v := fill
		if !outside {
			v = a.elementAt(off)
		}
		out.setElement(flat, v)
	}
	return out, nil
}
