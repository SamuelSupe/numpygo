package npstrings

import "github.com/SamuelSupe/numpygo"

// textWithInts broadcasts both character operands and integer-valued
// options, preserving the character descriptor separately.
func textWithInts(text []*np.Array, ints ...*np.Array) (characters, [][]int, error) {
	all := append(append([]*np.Array(nil), text...), ints...)
	views, e := np.BroadcastArrays(all...)
	if e != nil {
		return characters{}, nil, e
	}
	c, e := textInputs(views[:len(text)]...)
	if e != nil {
		return c, nil, e
	}
	values := make([][]int, len(ints))
	for i, a := range views[len(text):] {
		if !a.DType().IsInteger() {
			return c, nil, np.ErrDType
		}
		if a.DType().IsUnsigned() {
			v, e := a.Astype(np.Uint64)
			if e != nil {
				return c, nil, e
			}
			u, e := np.Values[uint64](v)
			if e != nil {
				return c, nil, e
			}
			for _, x := range u {
				if x > uint64(^uint(0)>>1) {
					return c, nil, np.ErrValue
				}
			}
		}
		v, e := a.Astype(np.Int64)
		if e != nil {
			return c, nil, e
		}
		x, e := np.Values[int64](v)
		if e != nil {
			return c, nil, e
		}
		values[i] = make([]int, len(x))
		for j, y := range x {
			if int64(int(y)) != y {
				return c, nil, np.ErrValue
			}
			values[i][j] = int(y)
		}
	}
	return c, values, nil
}
func scalarText(s string, d np.DType) (*np.Array, error) {
	switch d {
	case np.BytesDType:
		return np.FromByteStrings([]string{s}, max(1, len(s)))
	case np.UnicodeDType:
		return np.FromUnicodeStrings([]string{s}, max(1, length(s, d)))
	case np.StringDType:
		return np.FromStrings([]string{s})
	default:
		return nil, np.ErrDType
	}
}
