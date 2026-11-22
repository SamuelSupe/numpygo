package np

import (
	"iter"
	"slices"
)

// Arrayterator iterates over bounded, storage-sharing multidimensional
// blocks. This permits processing large memory-mapped arrays without copying
// their full contents. A buffer size of zero selects the complete view.
type Arrayterator struct {
	array       *Array
	bufSize     int
	start, stop []int
	step        []int
}

func NewArrayterator(a *Array, bufSize int) (*Arrayterator, error) {
	if a.DType() == 0 {
		return nil, ErrDType
	}
	if bufSize < 0 {
		return nil, ErrValue
	}
	a.viewBase()
	v := &Arrayterator{array: a, bufSize: bufSize, start: make([]int, a.NDim()), stop: a.Shape(), step: make([]int, a.NDim())}
	for i := range v.step {
		v.step[i] = 1
	}
	return v, nil
}
func (a *Arrayterator) Var() *Array     { return a.array }
func (a *Arrayterator) BufferSize() int { return a.bufSize }
func (a *Arrayterator) Start() []int    { return slices.Clone(a.start) }
func (a *Arrayterator) Stop() []int     { return slices.Clone(a.stop) }
func (a *Arrayterator) Step() []int     { return slices.Clone(a.step) }
func (a *Arrayterator) Shape() []int {
	out := make([]int, len(a.start))
	for i, x := range a.start {
		d := a.stop[i] - x
		if d > 0 {
			out[i] = 1 + (d-1)/a.step[i]
		} else {
			out[i] = d / a.step[i]
		}
	}
	return out
}

// Slice follows Arrayterator's additive source-coordinate slicing, keeping
// integer-indexed axes of size one. Negative bounds/steps are rejected
// because the original iterator's forward-progress algorithm is not defined
// for them. A reversed ndarray may still be supplied as Var.
func (a *Arrayterator) Slice(indices ...Index) (*Arrayterator, error) {
	n := len(a.start)
	used, ellipses := 0, 0
	for _, v := range indices {
		switch v {
		case Ellipsis:
			ellipses++
		default:
			switch v.(type) {
			case Slice, AtIndex:
				used++
			default:
				return nil, ErrIndex
			}
		}
	}
	if used > n || ellipses > 1 {
		return nil, ErrIndex
	}
	fixed := make([]Slice, 0, n)
	for _, v := range indices {
		switch x := v.(type) {
		case Slice:
			fixed = append(fixed, x)
		case AtIndex:
			if int(x) < 0 || int(x) == maxInt {
				return nil, ErrIndex
			}
			fixed = append(fixed, Range(int(x), int(x)+1))
		case specialIndex:
			fixed = append(fixed, make([]Slice, n-used)...)
		}
	}
	fixed = append(fixed, make([]Slice, n-len(fixed))...)
	out := &Arrayterator{array: a.array, bufSize: a.bufSize, start: a.Start(), stop: a.Stop(), step: a.Step()}
	for i, s := range fixed {
		start, stop, step := 0, a.stop[i]-a.start[i], 1
		if s.Start != nil {
			start = *s.Start
		}
		if s.Stop != nil && *s.Stop != 0 {
			stop = *s.Stop
		}
		if s.Step != nil && *s.Step != 0 {
			step = *s.Step
		}
		if start < 0 || stop < 0 || step < 1 || start > maxInt-a.start[i] || step > maxInt/a.step[i] {
			return nil, ErrValue
		}
		out.start[i] = a.start[i] + start
		out.stop[i] = a.start[i] + min(stop, a.stop[i]-a.start[i])
		out.step[i] *= step
	}
	return out, nil
}

func (a *Arrayterator) view(start, stop []int) (*Array, error) {
	ix := make([]Index, len(start))
	for i := range ix {
		ix[i] = Strided(start[i], stop[i], a.step[i])
	}
	return a.array.Slice(ix...)
}

// Array returns the corresponding shared ndarray view without block copying.
func (a *Arrayterator) Array() (*Array, error) { return a.view(a.start, a.stop) }

// Blocks yields each block in logical C order. Every block contains at most
// BufferSize elements when a positive size was requested.
func (a *Arrayterator) Blocks() iter.Seq2[*Array, error] {
	return func(yield func(*Array, error) bool) {
		shape := a.Shape()
		n := len(shape)
		if n == 0 {
			v, e := a.Array()
			yield(v, e)
			return
		}
		for _, d := range shape {
			if d <= 0 {
				return
			}
		}
		start, stop := a.Start(), a.Stop()
		limit := a.bufSize
		if limit == 0 {
			var e error
			limit, e = a.array.shapeSize(shape)
			if e != nil {
				yield(nil, e)
				return
			}
		}
		for {
			count, running := limit, 0
			for i := n - 1; i >= 0; i-- {
				remaining := a.stop[i] - start[i]
				switch {
				case count == 0:
					stop[i] = start[i] + min(remaining, 1)
				case count <= shape[i]:
					// Clamp before multiplication to avoid overflow.
					stop[i] = a.stop[i]
					if count <= remaining/a.step[i] {
						stop[i] = start[i] + count*a.step[i]
					}
					running = i
				default:
					stop[i] = a.stop[i]
				}
				count /= shape[i]
			}
			v, e := a.view(start, stop)
			if !yield(v, e) || e != nil {
				return
			}
			start[running] = stop[running]
			for i := n - 1; i > 0; i-- {
				if start[i] >= a.stop[i] {
					start[i] = a.start[i]
					start[i-1] += min(a.step[i-1], a.stop[i-1]-start[i-1])
				}
			}
			if start[0] >= a.stop[0] {
				return
			}
		}
	}
}

func (a *Arrayterator) Flat() iter.Seq2[any, error] {
	return func(yield func(any, error) bool) {
		for v, e := range a.Blocks() {
			if e != nil {
				yield(nil, e)
				return
			}
			it, e := NDIter(v, "C")
			if e != nil {
				yield(nil, e)
				return
			}
			for x := range it {
				if !yield(x, nil) {
					return
				}
			}
		}
	}
}
