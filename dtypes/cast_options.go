package dtypes

import (
	"strings"

	"github.com/SamuelSupe/numpygo"
)

// CastWithOptions extends descriptor casts to NumPy's casting, allocation
// order and copy options. NoCopy can reuse only an already matching array.
// Subarray casts expand new axes and always allocate.
func (d Descriptor) CastWithOptions(a *np.Array, opt np.CastOptions) (*np.Array, error) {
	src, e := FromArray(a)
	if e != nil {
		return nil, e
	}
	rule := opt.Casting
	if rule == "" {
		rule = "unsafe"
	}
	ok, e := CanCast(src, d, rule)
	if e != nil {
		return nil, e
	}
	if !ok {
		return nil, np.ErrDType
	}
	order := strings.ToUpper(opt.Order)
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
		return nil, np.ErrValue
	}
	same := d.Equal(src)
	if d.sub == nil && d.DType() == np.VoidDType && d.flex.Length == 0 && src.DType() == np.RecordDType {
		same = true
	}
	if d.sub == nil && d.DType().IsFlexible() && d.DType() == src.DType() && d.flex.Length == 0 && d.IsNative() == src.IsNative() {
		same = true
	}
	if opt.NoCopy && same && (order == "K" || order == "C" && a.IsCContiguous() || order == "F" && a.IsFContiguous()) {
		return a, nil
	}
	v, e := d.Cast(a, rule)
	if e != nil {
		return nil, e
	}
	if v.Base() == nil && v.Writable() && (order == "C" || order == "K" && a.IsCContiguous()) && v.IsCContiguous() {
		return v, nil
	}
	return v.CopyLikeOrder(a, order)
}
