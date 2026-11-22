package ma

import "github.com/SamuelSupe/numpygo"

type UniqueResult struct {
	Values           *Array
	Indices, Inverse *np.Array
}

// Unique sorts valid values and treats all masked elements as one final value.
func Unique(a *Array) (UniqueResult, error) {
	var out UniqueResult
	flat, e := Ravel(a)
	if e != nil {
		return out, e
	}
	good, e := np.LogicalNot(flat.mask)
	if e != nil {
		return out, e
	}
	positions, e := np.FlatNonzero(good)
	if e != nil {
		return out, e
	}
	data, e := flat.data.Index(good)
	if e != nil {
		return out, e
	}
	u, e := np.Unique(data)
	if e != nil {
		return out, e
	}
	first, e := np.Take(positions, u.Indices, nil)
	if e != nil {
		return out, e
	}
	inverse, e := np.Full(np.Int64, int64(u.Values.Size()), flat.Size())
	if e != nil {
		return out, e
	}
	if e = inverse.Assign([]np.Index{good}, u.Inverse); e != nil {
		return out, e
	}
	values, e := New(u.Values, nil)
	if e != nil {
		return out, e
	}
	if IsMasked(flat) {
		invalid, e := np.FlatNonzero(flat.mask)
		if e != nil {
			return out, e
		}
		pos, e := invalid.AtValue(0)
		if e != nil {
			return out, e
		}
		item, e := flat.Index(np.AtIndex(int(pos.(int64))))
		if e != nil {
			return out, e
		}
		item, e = Reshape(item, 1)
		if e != nil {
			return out, e
		}
		values, e = Concatenate([]*Array{values, item}, 0)
		if e != nil {
			return out, e
		}
		p, e := invalid.Slice(np.Range(0, 1))
		if e != nil {
			return out, e
		}
		first, e = np.Concatenate([]*np.Array{first, p}, 0)
		if e != nil {
			return out, e
		}
	}
	inverse, e = inverse.Reshape(a.Shape()...)
	if e != nil {
		return out, e
	}
	return UniqueResult{values, first, inverse}, nil
}

func setOperation(a, b *Array, op func(*np.Array, *np.Array) (*np.Array, error), includeMasked func(bool, bool) bool) (*Array, error) {
	x, e := Compressed(a)
	if e != nil {
		return nil, e
	}
	y, e := Compressed(b)
	if e != nil {
		return nil, e
	}
	data, e := op(x, y)
	if e != nil {
		return nil, e
	}
	out, e := New(data, nil)
	if e != nil {
		return nil, e
	}
	if includeMasked(IsMasked(a), IsMasked(b)) {
		m, e := MaskedAll(data.DType(), 1)
		if e != nil {
			return nil, e
		}
		return Concatenate([]*Array{out, m}, 0)
	}
	return out, nil
}
func Union1D(a, b *Array) (*Array, error) {
	return setOperation(a, b, np.Union1D, func(a, b bool) bool { return a || b })
}
func Intersect1D(a, b *Array) (*Array, error) {
	return setOperation(a, b, np.Intersect1D, func(a, b bool) bool { return a && b })
}
func Setdiff1D(a, b *Array) (*Array, error) {
	return setOperation(a, b, np.Setdiff1D, func(a, b bool) bool { return a && !b })
}
func Setxor1D(a, b *Array) (*Array, error) {
	return setOperation(a, b, np.Setxor1D, func(a, b bool) bool { return a != b })
}

// In1D follows the legacy masked stable-sort membership algorithm, including
// its propagated masks at boundaries adjacent to masked entries.
func In1D(a, b *Array, invert bool) (*Array, error) {
	ua, e := Unique(a)
	if e != nil {
		return nil, e
	}
	ub, e := Unique(b)
	if e != nil {
		return nil, e
	}
	both, e := Concatenate([]*Array{ua.Values, ub.Values}, 0)
	if e != nil {
		return nil, e
	}
	order, e := Argsort(both, np.Int(0), true)
	if e != nil {
		return nil, e
	}
	s, e := both.Index(order)
	if e != nil {
		return nil, e
	}
	permutation, e := np.Values[int64](order)
	if e != nil {
		return nil, e
	}
	values, masks := make([]bool, ua.Values.Size()), make([]bool, ua.Values.Size())
	for i, id := range permutation {
		if id >= int64(len(values)) {
			continue
		}
		if i+1 == len(permutation) {
			values[id] = invert
			continue
		}
		x, e := s.data.Slice(np.AtIndex(i))
		if e != nil {
			return nil, e
		}
		y, e := s.data.Slice(np.AtIndex(i + 1))
		if e != nil {
			return nil, e
		}
		eq, e := np.ArrayEqual(x, y, false)
		if e != nil {
			return nil, e
		}
		m1, _ := s.mask.AtValue(i)
		m2, _ := s.mask.AtValue(i + 1)
		values[id] = eq != invert
		masks[id] = m1.(bool) || m2.(bool)
	}
	d, e := np.FromSlice(values, len(values))
	if e != nil {
		return nil, e
	}
	m, e := np.FromSlice(masks, len(masks))
	if e != nil {
		return nil, e
	}
	index, e := np.Ravel(ua.Inverse)
	if e != nil {
		return nil, e
	}
	d, e = np.Take(d, index, nil)
	if e != nil {
		return nil, e
	}
	m, e = np.Take(m, index, nil)
	if e != nil {
		return nil, e
	}
	return New(d, m)
}
func Isin(a, b *Array, invert bool) (*Array, error) {
	out, e := In1D(a, b, invert)
	if e != nil {
		return nil, e
	}
	return Reshape(out, a.Shape()...)
}
