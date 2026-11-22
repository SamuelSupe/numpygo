package np

import (
	"slices"
	"strings"
)

func normalizeUFuncOrder(order string) (string, error) {
	switch strings.ToUpper(order) {
	case "", "K":
		return "K", nil
	case "C", "F", "A":
		return strings.ToUpper(order), nil
	default:
		return "", ErrValue
	}
}

func ufuncAnyOrder(order string, inputs []*Array) string {
	if order == "A" {
		for _, a := range inputs {
			if !a.IsFContiguous() {
				return "C"
			}
		}
		return "F"
	}
	return order
}

// NumPy's single-output contiguous fast path keeps F order, including
// singleton strides. Iteration (masking, broadcasting or casting) instead
// treats singleton axes as ambiguous when computing keep-order strides.
func (u *UFunc) elementwiseOrder(opt UFuncOptions, inputs []*Array, shape []int) string {
	if opt.Order != "K" || opt.Where != nil {
		return opt.Order
	}
	d := make([]DType, u.NArgs())
	for i, a := range inputs {
		if a.NDim() != 0 && !slices.Equal(a.shape, shape) || !a.IsFContiguous() || !a.NativeByteOrder() {
			return "K"
		}
		d[i] = a.DType()
	}
	loop, e := u.ResolveDTypes(d, u.callResolveOptions(opt.Casting, opt.DType, opt.Signature))
	if e == nil && slices.Equal(loop[:u.NIn], d[:u.NIn]) {
		for _, a := range inputs {
			if !a.IsCContiguous() {
				return "F"
			}
		}
	}
	return "K"
}

// ufuncResultAxes returns the slowest-to-fastest allocation axes in
// canonical (outer axes followed by core axes) coordinates. K keeps the
// compatible outer stride order of the operands, not their stride gaps.
func ufuncResultAxes(shape []int, inputs []*Array, cores []int, outCore int, order string) []int {
	axes := make([]int, len(shape))
	for i := range axes {
		axes[i] = i
	}
	order = ufuncAnyOrder(order, inputs)
	if order == "F" {
		slices.Reverse(axes)
		return axes
	}
	if order == "C" {
		return axes
	}
	batchRank := len(shape) - outCore
	strides := make([][]int, len(inputs))
	for i, a := range inputs {
		rank := a.NDim()
		if cores != nil {
			rank -= cores[i]
		}
		strides[i] = make([]int, batchRank)
		for j := 0; j < rank; j++ {
			if a.shape[j] != 1 {
				strides[i][batchRank-rank+j] = a.strides[j]
			}
		}
	}
	// NumPy's iterator insertion starts with the fastest C axis. A
	// broadcast-only comparison is ambiguous, so continue comparing
	// earlier axes rather than treating it as an ordering barrier.
	slices.Reverse(axes[:batchRank])
	for i := 1; i < batchRank; i++ {
		ax, pos := axes[i], i
		for j := i - 1; j >= 0; j-- {
			less, greater := false, false
			for _, s := range strides {
				x, y := absStride(s[ax]), absStride(s[axes[j]])
				if x != 0 && y != 0 {
					less = less || x < y
					greater = greater || x > y
				}
			}
			if greater {
				break
			}
			if less {
				pos = j
			}
		}
		copy(axes[pos+1:i+1], axes[pos:i])
		axes[pos] = ax
	}
	slices.Reverse(axes[:batchRank])
	return axes
}

func packUFuncResult(value *Array, axes []int) (*Array, error) {
	strides := make([]int, len(axes))
	if value.size != 0 {
		s := 1
		for i := len(axes) - 1; i >= 0; i-- {
			ax := axes[i]
			strides[ax] = s
			s *= value.shape[ax]
		}
	}
	if value.owner && value.base == nil && slices.Equal(value.strides, strides) {
		return value, nil
	}
	out, e := value.zerosLike(value.shape)
	if e != nil {
		return nil, e
	}
	out.strides = strides
	// Fresh independent storage: direct logical copy also preserves
	// descriptors and avoids an unnecessary cast/snapshot.
	value.walk(func(i, off int) {
		out.setElement(flatOffset(i, out.shape, out.strides), value.elementAt(off))
	})
	return out, nil
}
