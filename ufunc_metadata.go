package np

import (
	"slices"
	"strings"
)

type ufuncReferenceMeta struct {
	nin, nout int
	signature string
	identity  any
	types     []string
}

// Types reports the pinned upstream loop declarations, including platform
// aliases and object loops. It is compatibility metadata, not a promise
// that every Python/C loop can execute in Go.
func (u *UFunc) Types() []string   { return slices.Clone(ufuncReference[u.Name].types) }
func (u *UFunc) NTypes() int       { return len(ufuncReference[u.Name].types) }
func (u *UFunc) NArgs() int        { return u.NIn + u.NOut }
func (u *UFunc) Signature() string { return ufuncReference[u.Name].signature }

func loopType(c byte) DType {
	switch c {
	case '?':
		return Bool
	case 'b':
		return Int8
	case 'B':
		return Uint8
	case 'h':
		return Int16
	case 'H':
		return Uint16
	case 'i':
		return Int32
	case 'I':
		return Uint32
	case 'l', 'q', 'p':
		return Int64
	case 'L', 'Q', 'P':
		return Uint64
	case 'e':
		return Float16
	case 'f':
		return Float32
	case 'd':
		return Float64
	case 'g':
		return Float128
	case 'F':
		return Complex64
	case 'D':
		return Complex128
	case 'G':
		return Complex256
	case 'M':
		return DateTime64
	case 'm':
		return TimeDelta64
	case 'O':
		return Object
	case 'S':
		return BytesDType
	case 'U':
		return UnicodeDType
	case 'V':
		return VoidDType
	case 'T':
		return StringDType
	}
	return 0
}
func parseLoop(s string) []DType {
	s = strings.ReplaceAll(s, "->", "")
	d := make([]DType, len(s))
	for i, c := range []byte(s) {
		d[i] = loopType(c)
	}
	return d
}

// ResolveDTypesOptions controls strong dtype resolution. WeakInputs
// marks the type-only Python int/float/complex equivalents; Signature
// constrains loop arguments (0 means unconstrained).
type ResolveDTypesOptions struct {
	Casting    string
	Signature  []DType
	WeakInputs []bool
	Reduction  bool
}

// ResolveDTypes resolves native dtype families and returns input and output
// loop dtypes. It does not include temporal units or flexible widths,
// which remain descriptor-level ndarray operations.
func (u *UFunc) ResolveDTypes(args []DType, opt ResolveDTypesOptions) ([]DType, error) {
	if u == nil || len(args) != u.NArgs() || len(opt.Signature) != 0 && len(opt.Signature) != u.NArgs() ||
		len(opt.WeakInputs) != 0 && len(opt.WeakInputs) != u.NIn {
		return nil, ErrShape
	}
	args = slices.Clone(args)
	rule := opt.Casting
	if rule == "" {
		rule = "same_kind"
		if opt.Reduction {
			rule = "unsafe"
		}
	}
	if _, e := CanCast(Bool, Bool, rule); e != nil {
		return nil, e
	}
	if opt.Reduction {
		if u.NIn != 2 || u.NOut != 1 || u.Signature() != "" || args[1] == 0 {
			return nil, ErrValue
		}
		d := args[0]
		if d == 0 {
			d = args[2]
		}
		if d == 0 {
			d = args[1]
			if u.Name == "add" || u.Name == "multiply" {
				d = sumDType(d)
			}
		}
		if len(opt.Signature) > 0 {
			for _, x := range opt.Signature {
				if x != 0 {
					d = x
					break
				}
			}
		}
		args[0], args[1], args[2] = d, d, d
		opt.Reduction = false
	}
	if len(opt.WeakInputs) > 0 {
		strong := DType(0)
		kind := 0
		for i := 0; i < u.NIn; i++ {
			if opt.WeakInputs[i] {
				if args[i] != Int64 && args[i] != Float64 && args[i] != Complex128 {
					return nil, ErrDType
				}
				kind = max(kind, dtypeKind(args[i]))
			} else {
				if args[i] == 0 {
					return nil, ErrDType
				}
				if strong == 0 {
					strong = args[i]
				} else {
					d, e := PromoteTypes(strong, args[i])
					if e != nil {
						return nil, e
					}
					strong = d
				}
			}
		}
		d := weakPromotion(strong, kind)
		for i, w := range opt.WeakInputs {
			if w {
				args[i] = d
			}
		}
	}
	for i := 0; i < u.NIn; i++ {
		if args[i] == 0 {
			return nil, ErrDType
		}
	}
	if len(opt.Signature) != 0 {
		return u.resolveSignature(args, opt.Signature, rule)
	}
	// NumPy explicitly rejects boolean negative/positive/subtract rather
	// than choosing a cast integer loop.
	if (u.Name == "negative" || u.Name == "positive") && args[0] == Bool {
		return nil, ErrDType
	}
	if u.Name == "subtract" && args[0] == Bool && args[1] == Bool {
		return nil, ErrDType
	}
	inputRule := "safe"
	if rule == "no" || rule == "equiv" {
		inputRule = rule
	}
	work := slices.Clone(args)
	if u.NIn == 2 {
		a, b := args[0], args[1]
		switch u.Name {
		case "logical_and", "logical_or", "logical_xor":
			if a != b && a != Object && b != Object {
				return resolveOutput([]DType{Bool, Bool, Bool}, args, u.NIn, rule)
			}
		case "divide":
			if (a.IsInteger() || a == Bool) && (b.IsInteger() || b == Bool) {
				work[0], work[1] = Float64, Float64
			}
		case "ldexp":
		case "less", "less_equal", "greater", "greater_equal", "equal", "not_equal":
			if a == Int64 && b == Uint64 || a == Uint64 && b == Int64 {
				// Mixed int64 comparisons have dedicated exact loops.
				for _, s := range u.Types() {
					d := parseLoop(s)
					if len(d) == 3 && d[0] == a && d[1] == b {
						return resolveOutput(d, args, u.NIn, rule)
					}
				}
			}
			if (a.IsNativeNumeric() || a.IsExtended()) && (b.IsNativeNumeric() || b.IsExtended()) {
				d, e := PromoteTypes(a, b)
				if e != nil {
					return nil, e
				}
				work[0], work[1] = d, d
			}
		default:
			if (a.IsNativeNumeric() || a.IsExtended()) && (b.IsNativeNumeric() || b.IsExtended()) {
				d, e := PromoteTypes(a, b)
				if e != nil {
					return nil, e
				}
				work[0], work[1] = d, d
			}
		}
	}
	for _, s := range u.Types() {
		d := parseLoop(s)
		if len(d) != u.NArgs() {
			continue
		}
		ok := true
		for i, x := range d {
			if x == 0 {
				ok = false
				break
			}
			if x == Object {
				// Do not fall back to an unavailable Python object loop
				// for incompatible numerical inputs.
				hasObject := false
				for _, a := range args[:u.NIn] {
					hasObject = hasObject || a == Object
				}
				if !hasObject {
					ok = false
					break
				}
			}
			if i < u.NIn {
				yes, _ := CanCast(work[i], x, inputRule)
				if !yes {
					ok = false
					break
				}
				yes, _ = CanCast(args[i], x, rule)
				if !yes {
					ok = false
					break
				}
			}
		}
		if !ok {
			continue
		}
		if v, e := resolveOutput(d, args, u.NIn, rule); e == nil {
			return v, nil
		}
	}
	return nil, ErrDType
}

func (u *UFunc) callResolveOptions(rule string, dtype DType, signature []DType) ResolveDTypesOptions {
	if dtype != 0 {
		signature = make([]DType, u.NArgs())
		for i := u.NIn; i < len(signature); i++ {
			signature[i] = dtype
		}
	}
	return ResolveDTypesOptions{Casting: rule, Signature: signature}
}

// Resolve explicitly constrained loops before validating output casts.
// Prefer the ordinary resolution, then safe conversions for unspecified
// inputs. A dtype restriction may force narrowing (e.g. add(dtype=float32)),
// but must not replace an exact complex absolute loop with a real one.
func (u *UFunc) resolveSignature(args, signature []DType, rule string) ([]DType, error) {
	matches := func(d []DType) bool {
		for i, v := range signature {
			if v != 0 && v != d[i] {
				return false
			}
		}
		return true
	}
	validate := func(d []DType) ([]DType, error) {
		for i := 0; i < u.NIn; i++ {
			ok, _ := CanCast(args[i], d[i], rule)
			if !ok {
				return nil, ErrDType
			}
		}
		return resolveOutput(d, args, u.NIn, rule)
	}
	naturalArgs := slices.Clone(args)
	clear(naturalArgs[u.NIn:])
	if d, e := u.ResolveDTypes(naturalArgs, ResolveDTypesOptions{}); e == nil && matches(d) {
		return validate(d)
	}
	var first []DType
	hasObject := slices.Contains(args[:u.NIn], Object)
	for _, text := range u.Types() {
		d := parseLoop(text)
		if len(d) != len(args) || slices.Contains(d, 0) || !matches(d) || !hasObject && slices.Contains(d, Object) {
			continue
		}
		if first == nil {
			first = d
		}
		safe := true
		for i := 0; i < u.NIn; i++ {
			if signature[i] == 0 {
				ok, _ := CanCast(args[i], d[i], "safe")
				if !ok {
					safe = false
					break
				}
			}
		}
		if safe {
			return validate(d)
		}
	}
	if first != nil {
		return validate(first)
	}
	return nil, ErrDType
}

func resolveOutput(loop, args []DType, nin int, rule string) ([]DType, error) {
	for i := nin; i < len(loop); i++ {
		if args[i] != 0 {
			ok, e := CanCast(loop[i], args[i], rule)
			if e != nil || !ok {
				return nil, ErrDType
			}
		}
	}
	return loop, nil
}
