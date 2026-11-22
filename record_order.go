package np

import (
	"bytes"
	"slices"
)

// Ordering uses declared fields. NumPy compares subarray fields as opaque
// byte sequences for sorting, but compares their scalar values for equality.
type offsetOrder struct {
	less       func(int, int) bool
	equal      func(int, int) bool
	searchable bool
}

func arrayOrder(a *Array, fields []string) (offsetOrder, error) {
	if a.DType() != RecordDType {
		if len(fields) != 0 {
			return offsetOrder{}, ErrDType
		}
		return offsetOrder{
			less:  func(x, y int) bool { return sortLess(a.elementAt(x), a.elementAt(y), a.DType()) },
			equal: func(x, y int) bool { return equalElement(a.elementAt(x), a.elementAt(y), a.DType()) },
		}, nil
	}
	type key struct {
		a     *Array
		rel   int
		width int
	}
	var keys []key
	var collect func(*Array, int, int, []string, bool) error
	collect = func(v *Array, parentRank, rel int, first []string, rawOrder bool) error {
		n, e := shapeSizeWidth(1, v.shape[parentRank:])
		if e != nil {
			return e
		}
		if rawOrder && v.NDim() > parentRank && !v.HasObject() {
			if n*v.ItemSize() > 0 {
				keys = append(keys, key{a: v, rel: rel, width: n * v.ItemSize()})
			}
			return nil
		}
		if v.DType() != RecordDType {
			for j := 0; j < n; j++ {
				keys = append(keys, key{a: v, rel: rel + flatOffset(j, v.shape[parentRank:], v.strides[parentRank:])})
			}
			return nil
		}
		s, ok := v.StructuredSchema().(interface{ Names() []string })
		if !ok {
			return ErrDType
		}
		names := s.Names()
		order := slices.Clone(first)
		for i, n := range first {
			if !slices.Contains(names, n) || slices.Contains(first[:i], n) {
				return ErrValue
			}
		}
		for _, n := range names {
			if !slices.Contains(order, n) {
				order = append(order, n)
			}
		}
		children := make([]*Array, len(order))
		for i, name := range order {
			f, e := v.Field(name)
			if e != nil {
				return e
			}
			children[i] = f
		}
		for j := 0; j < n; j++ {
			offset := rel + flatOffset(j, v.shape[parentRank:], v.strides[parentRank:])
			for _, f := range children {
				if e = collect(f, v.NDim(), offset, nil, rawOrder); e != nil {
					return e
				}
			}
		}
		return nil
	}
	if e := collect(a, a.NDim(), 0, fields, true); e != nil {
		return offsetOrder{}, e
	}
	orderKeys := keys
	keys = nil
	if e := collect(a, a.NDim(), 0, fields, false); e != nil {
		return offsetOrder{}, e
	}
	compare := func(x, y int, equality bool) bool {
		selected := orderKeys
		if equality {
			selected = keys
		}
		for _, k := range selected {
			v := k.a
			px := v.offset + (x-a.offset)*a.strideUnit() + k.rel
			qx := v.offset + (y-a.offset)*a.strideUnit() + k.rel
			if k.width > 0 {
				if c := bytes.Compare(v.data.buffer[px:px+k.width], v.data.buffer[qx:qx+k.width]); c != 0 {
					return c < 0
				}
				continue
			}
			p := v.elementAt(px)
			q := v.elementAt(qx)
			if equality {
				if !equalElement(p, q, v.DType()) {
					return false
				}
			} else {
				if sortLess(p, q, v.DType()) {
					return true
				}
				if sortLess(q, p, v.DType()) {
					return false
				}
			}
		}
		return equality
	}
	searchable := !a.HasObject()
	for _, k := range orderKeys {
		if k.width > 0 {
			searchable = false
		}
	}
	return offsetOrder{
		less:       func(x, y int) bool { return compare(x, y, false) },
		equal:      func(x, y int) bool { return compare(x, y, true) },
		searchable: searchable,
	}, nil
}
