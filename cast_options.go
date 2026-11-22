package np

import (
	"sort"
	"strings"
)

// Copy is the corresponding NumPy top-level function; its default is K.
func Copy(a *Array, order string) (*Array, error) {
	if order == "" {
		order = "K"
	}
	return a.CopyWithOrder(order)
}

// CopyWithOrder makes an independent ndarray copy, retaining the complete
// descriptor and value bits. The default is C order, as ndarray.copy.
func (a *Array) CopyWithOrder(order string) (*Array, error) {
	if order == "" {
		order = "C"
	}
	return a.CopyLikeOrder(a, order)
}

// CopyLikeOrder packs a new independent array in the requested order.
// K follows like's outer axis order; extra trailing dimensions use C order
// unless like is Fortran-contiguous. This is also the layout rule for casts
// that expand fixed subarray descriptors.
func (a *Array) CopyLikeOrder(like *Array, order string) (*Array, error) {
	if a.DType() == 0 || like.DType() == 0 {
		return nil, ErrDType
	}
	if like.NDim() > a.NDim() {
		return nil, ErrShape
	}
	order = strings.ToUpper(order)
	switch order {
	case "", "K":
		order = "K"
		if like.IsFContiguous() && !like.IsCContiguous() {
			order = "F"
		}
	case "A":
		order = "C"
		if like.IsFContiguous() && !like.IsCContiguous() {
			order = "F"
		}
	case "C", "F":
	default:
		return nil, ErrValue
	}
	p := &advancedSelection{shape: a.Shape(), offsets: make([]int, a.Size()), canonical: make([]int, a.NDim())}
	for i := range p.canonical {
		p.canonical[i] = i
	}
	if order == "F" {
		p.order = make([]int, a.NDim())
		for i := range p.order {
			p.order[i] = a.NDim() - 1 - i
		}
	} else {
		p.order, _ = layoutAxes(like, order)
		for i := like.NDim(); i < a.NDim(); i++ {
			p.order = append(p.order, i)
		}
	}
	a.walk(func(i, off int) { p.offsets[i] = off })
	return a.copySelection(p, true)
}

// CastOptions controls dtype casting and the memory order of new storage.
// The defaults match ndarray.astype: unsafe casting, order K, and a copy.
// NoCopy allows reuse only when dtype, native byte order and layout match.
type CastOptions struct {
	Casting string
	Order   string
	NoCopy  bool
}

func (a *Array) AstypeWithOptions(dtype DType, opt CastOptions) (*Array, error) {
	rule := opt.Casting
	if rule == "" {
		rule = "unsafe"
	}
	ok, e := CanCast(a.DType(), dtype, rule)
	if e != nil {
		return nil, e
	}
	if !ok || rule == "no" && !a.NativeByteOrder() {
		return nil, ErrDType
	}
	order := opt.Order
	switch order {
	case "", "K":
		order = "K"
	case "A":
		order = "C"
		if a.IsFContiguous() && !a.IsCContiguous() {
			order = "F"
		}
	case "C", "F":
	default:
		return nil, ErrValue
	}
	if opt.NoCopy && a.DType() == dtype && a.NativeByteOrder() &&
		(order == "K" || order == "C" && a.IsCContiguous() || order == "F" && a.IsFContiguous()) {
		return a, nil
	}
	if order == "C" || a.NDim() < 2 || order == "K" && a.IsCContiguous() {
		return a.Astype(dtype)
	}
	axes, inverse := layoutAxes(a, order)
	v, e := a.Transpose(axes...)
	if e != nil {
		return nil, e
	}
	v, e = v.Astype(dtype)
	if e != nil {
		return nil, e
	}
	// v is a new allocation. Reorder its metadata directly instead of
	// exposing a view of an internal packed temporary.
	shape, strides := make([]int, len(inverse)), make([]int, len(inverse))
	for i, ax := range inverse {
		shape[i], strides[i] = v.shape[ax], v.strides[ax]
	}
	v.shape, v.strides = shape, strides
	return v, nil
}

// layoutAxes lists axes from slowest to fastest in newly packed storage.
func layoutAxes(a *Array, order string) (axes, inverse []int) {
	axes, inverse = make([]int, a.NDim()), make([]int, a.NDim())
	for i := range axes {
		axes[i] = i
	}
	if order == "F" || order == "K" && a.IsFContiguous() && !a.IsCContiguous() {
		for i := range axes {
			axes[i] = len(axes) - 1 - i
		}
	} else if order == "K" && !a.IsCContiguous() {
		// K packs the original axis order, removing gaps and negative strides.
		magnitude := func(s int) uint {
			if s < 0 {
				return uint(-(s + 1)) + 1
			}
			return uint(s)
		}
		sort.SliceStable(axes, func(i, j int) bool { return magnitude(a.strides[axes[i]]) > magnitude(a.strides[axes[j]]) })
	}
	for i, j := range axes {
		inverse[j] = i
	}
	return axes, inverse
}
