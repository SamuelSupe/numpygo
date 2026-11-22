package recfunctions

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/records"
	"slices"
)

func scalarColumns(a *np.Array) ([]*np.Array, error) {
	if a.DType() != np.RecordDType {
		if a.NDim() != 1 {
			return nil, np.ErrShape
		}
		return []*np.Array{a}, nil
	}
	s, e := schemaOf(a)
	if e != nil {
		return nil, e
	}
	ls, e := leaves(s, 0)
	if e != nil {
		return nil, e
	}
	out := make([]*np.Array, len(ls))
	for i, l := range ls {
		f, e := layout(l.typ, l.offset)
		if e != nil {
			return nil, e
		}
		out[i], e = a.ViewLayout(f, a.Shape(), a.Strides())
		if e != nil {
			return nil, e
		}
	}
	return out, nil
}
func keyCols(a *np.Array, keys []string) ([]*np.Array, error) {
	if len(keys) == 0 {
		return scalarColumns(a)
	}
	s, e := schemaOf(a)
	if e != nil {
		return nil, e
	}
	parents := GetFieldStructure(s)
	out := []*np.Array{}
	for _, k := range keys {
		p, ok := parents[k]
		if !ok {
			return nil, np.ErrValue
		}
		f, e := a.Field(append(slices.Clone(p), k)...)
		if e != nil {
			return nil, e
		}
		c, e := scalarColumns(f)
		if e != nil {
			return nil, e
		}
		out = append(out, c...)
	}
	return out, nil
}

func indexArray(idx []int64) (*np.Array, error) { return np.FromSlice(idx, len(idx)) }
func take(a Array, idx []int64) (Array, error) {
	i, e := indexArray(idx)
	if e != nil {
		return Array{}, e
	}
	d, e := np.Take(a.Data, i, np.Int(0))
	if e != nil {
		return Array{}, e
	}
	var m *np.Array
	if a.Mask != nil {
		m, e = np.Take(a.Mask, i, np.Int(0))
	}
	return Array{d, m}, e
}

// lexOrder composes stable scalar argsorts from the least-significant
// field to the most-significant. This preserves integer precision and
// NumPy's floating/character/temporal ordering.
func lexOrder(cols []*np.Array, n int) ([]int64, error) {
	idx := make([]int64, n)
	for i := range idx {
		idx[i] = int64(i)
	}
	for k := len(cols) - 1; k >= 0; k-- {
		i, e := indexArray(idx)
		if e != nil {
			return nil, e
		}
		c, e := np.Take(cols[k], i, np.Int(0))
		if e != nil {
			return nil, e
		}
		p, e := np.Argsort(c, np.Int(0))
		if e != nil {
			return nil, e
		}
		order, e := np.Values[int64](p)
		if e != nil {
			return nil, e
		}
		next := make([]int64, n)
		for j, x := range order {
			next[j] = idx[x]
		}
		idx = next
	}
	return idx, nil
}
func adjacentEqual(cols []*np.Array, order []int64) ([]bool, error) {
	if len(order) == 0 {
		return []bool{}, nil
	}
	flags := make([]bool, len(order)-1)
	for i := range flags {
		flags[i] = true
	}
	if len(order) == 1 {
		return flags, nil
	}
	a, e := indexArray(order[:len(order)-1])
	if e != nil {
		return nil, e
	}
	b, e := indexArray(order[1:])
	if e != nil {
		return nil, e
	}
	for _, col := range cols {
		x, e := np.Take(col, a, np.Int(0))
		if e != nil {
			return nil, e
		}
		y, e := np.Take(col, b, np.Int(0))
		if e != nil {
			return nil, e
		}
		q, e := np.Equal(x, y)
		if e != nil {
			return nil, e
		}
		v, e := np.Values[bool](q)
		if e != nil {
			return nil, e
		}
		for i := range flags {
			flags[i] = flags[i] && v[i]
		}
	}
	return flags, nil
}

func FindDuplicates(a Array, keys []string, ignoreMask bool) (Array, *np.Array, error) {
	a, e := ravel(a)
	if e != nil {
		return Array{}, nil, e
	}
	work := a
	var mask []bool
	if a.Mask != nil {
		work.Data, e = a.Data.Copy()
		if e != nil {
			return Array{}, nil, e
		}
		if e = fillMissing(work, nil, nil); e != nil {
			return Array{}, nil, e
		}
		c, e := keyCols(a.Mask, keys)
		if e != nil {
			return Array{}, nil, e
		}
		mask = make([]bool, a.Data.Size())
		for i := range mask {
			mask[i] = true
		}
		for _, col := range c {
			v, e := np.Values[bool](col)
			if e != nil {
				return Array{}, nil, e
			}
			for i := range mask {
				mask[i] = mask[i] && v[i]
			}
		}
	}
	cols, e := keyCols(work.Data, keys)
	if e != nil {
		return Array{}, nil, e
	}
	order, e := lexOrder(cols, a.Data.Size())
	if e != nil {
		return Array{}, nil, e
	}
	if mask != nil {
		// NumPy masked argsort places fully masked records last.
		slices.SortStableFunc(order, func(x, y int64) int {
			if mask[x] == mask[y] {
				return 0
			}
			if mask[x] {
				return 1
			}
			return -1
		})
	}
	match, e := adjacentEqual(cols, order)
	if e != nil {
		return Array{}, nil, e
	}
	dup := make([]bool, len(order))
	for i, ok := range match {
		if ok && !(ignoreMask && mask != nil && mask[order[i+1]]) {
			dup[i], dup[i+1] = true, true
		}
	}
	idx := []int64{}
	for i, yes := range dup {
		if yes {
			idx = append(idx, order[i])
		}
	}
	out, e := take(a, idx)
	if e != nil {
		return Array{}, nil, e
	}
	ix, e := indexArray(idx)
	return out, ix, e
}

// JoinBy supports inner, outer and leftouter joins of unique structured keys.
// NumPy's result for non-unique keys is undefined; reject them rather than
// silently producing shifted or paired unrelated rows.
func JoinBy(keys []string, left, right Array, jointype, leftPostfix, rightPostfix string, opt Options) (Array, error) {
	if jointype == "" {
		jointype = "inner"
	}
	if len(keys) == 0 || jointype != "inner" && jointype != "outer" && jointype != "leftouter" {
		return Array{}, np.ErrValue
	}
	l, e := ravel(left)
	if e != nil {
		return Array{}, e
	}
	r, e := ravel(right)
	if e != nil {
		return Array{}, e
	}
	ls, e := schemaOf(l.Data)
	if e != nil {
		return Array{}, e
	}
	rs, e := schemaOf(r.Data)
	if e != nil {
		return Array{}, e
	}
	for i, k := range keys {
		if slices.Contains(keys[:i], k) || !slices.Contains(ls.Names(), k) || !slices.Contains(rs.Names(), k) {
			return Array{}, np.ErrValue
		}
	}
	lk, e := keyCols(l.Data, keys)
	if e != nil {
		return Array{}, e
	}
	rk, e := keyCols(r.Data, keys)
	if e != nil || len(lk) != len(rk) {
		return Array{}, np.ErrDType
	}
	for _, a := range []Array{l, r} {
		if a.Mask != nil {
			c, e := keyCols(a.Mask, keys)
			if e != nil {
				return Array{}, e
			}
			for _, x := range c {
				v, e := np.Any(x)
				if e != nil {
					return Array{}, e
				}
				z, e := v.AtValue()
				if e != nil || z.(bool) {
					return Array{}, np.ErrValue // join keys must not be missing
				}
			}
		}
	}
	c := make([]*np.Array, len(lk))
	for i := range c {
		x, e := dtypes.FromArray(lk[i])
		if e != nil {
			return Array{}, e
		}
		y, e := dtypes.FromArray(rk[i])
		if e != nil {
			return Array{}, e
		}
		d, e := dtypes.Promote(x, y)
		if e != nil {
			return Array{}, e
		}
		p, e := d.Cast(lk[i], "unsafe")
		if e != nil {
			return Array{}, e
		}
		q, e := d.Cast(rk[i], "unsafe")
		if e != nil {
			return Array{}, e
		}
		c[i], e = np.Concatenate([]*np.Array{p, q}, 0)
		if e != nil {
			return Array{}, e
		}
	}
	n := int64(l.Data.Size())
	order, e := lexOrder(c, int(n)+r.Data.Size())
	if e != nil {
		return Array{}, e
	}
	match, e := adjacentEqual(c, order)
	if e != nil {
		return Array{}, e
	}
	type pair struct{ left, right int64 }
	pairs := []pair{}
	for i := 0; i < len(order); i++ {
		x := order[i]
		if i < len(match) && match[i] {
			y := order[i+1]
			if (x < n) == (y < n) || i+1 < len(match) && match[i+1] {
				return Array{}, np.ErrValue
			}
			if x >= n {
				x, y = y, x
			}
			pairs = append(pairs, pair{x, y - n})
			i++
		} else if x < n && jointype != "inner" {
			pairs = append(pairs, pair{x, -1})
		} else if x >= n && jointype == "outer" {
			pairs = append(pairs, pair{-1, x - n})
		}
	}
	spec := []records.FieldSpec{}
	lmap, rmap := map[string]string{}, map[string]string{}
	for _, f := range ls.Fields() {
		if slices.Contains(keys, f.Name) {
			g, e := rs.Field(f.Name)
			if e != nil {
				return Array{}, e
			}
			t, e := largestType(f.Type, g.Type, true)
			if e != nil {
				return Array{}, e
			}
			spec = append(spec, records.FieldSpec{Name: f.Name, Type: t})
			lmap[f.Name], rmap[f.Name] = f.Name, f.Name
		}
	}
	for _, f := range ls.Fields() {
		if slices.Contains(keys, f.Name) {
			continue
		}
		if slices.Contains(rs.Names(), f.Name) {
			if leftPostfix == "" && rightPostfix == "" {
				return Array{}, np.ErrValue
			}
			g, e := rs.Field(f.Name)
			if e != nil {
				return Array{}, e
			}
			x, y := f.Name+leftPostfix, f.Name+rightPostfix
			spec = append(spec, records.FieldSpec{Name: x, Type: f.Type}, records.FieldSpec{Name: y, Type: g.Type})
			lmap[f.Name], rmap[g.Name] = x, y
		} else {
			spec = append(spec, records.FieldSpec{Name: f.Name, Type: f.Type, Title: f.Title})
			lmap[f.Name] = f.Name
		}
	}
	for _, f := range rs.Fields() {
		if _, ok := rmap[f.Name]; !ok {
			spec = append(spec, records.FieldSpec{Name: f.Name, Type: f.Type, Title: f.Title})
			rmap[f.Name] = f.Name
		}
	}
	s, e := records.NewSchema(spec, false, 0)
	if e != nil {
		return Array{}, e
	}
	out, e := newMasked(s, len(pairs))
	if e != nil {
		return Array{}, e
	}
	for i, p := range pairs {
		for k, src := range []Array{l, r} {
			j, names := p.left, lmap
			if k == 1 {
				j, names = p.right, rmap
			}
			if j < 0 {
				continue
			}
			for from, to := range names {
				u, e := selectField(src, from)
				if e != nil {
					return Array{}, e
				}
				u, e = rangeArray(u, int(j), int(j)+1)
				if e != nil {
					return Array{}, e
				}
				v, e := selectField(out, to)
				if e != nil {
					return Array{}, e
				}
				if e = copyField(v, u, i); e != nil {
					return Array{}, e
				}
			}
		}
	}
	return finish(out, opt, nil)
}

func RecJoin(keys []string, left, right *np.Array, jointype, leftPostfix, rightPostfix string, defaults map[string]any) (*records.Array, error) {
	out, e := JoinBy(keys, Array{Data: left}, Array{Data: right}, jointype, leftPostfix, rightPostfix, Options{Defaults: defaults})
	if e != nil {
		return nil, e
	}
	return records.FromNDArray(out.Data)
}
