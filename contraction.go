package np

import (
	"slices"
	"sort"
	"strings"
)

func multiplyScalar(x, y element, d DType) element {
	if d.IsExtended() {
		v, _ := extendedArithmetic(x, y, d, mulOp)
		return v
	}
	if d == Bool || d.IsInteger() {
		return element{i: x.i * y.i}
	}
	if d.IsComplex() {
		return element{z: x.z * y.z}
	}
	return element{z: complex(real(x.z)*real(y.z), 0)}
}
func addScalar(x, y element, d DType) element {
	if d.IsExtended() {
		v, _ := extendedArithmetic(x, y, d, addOp)
		return v
	}
	if d == Bool {
		return element{i: x.i | y.i}
	}
	if d.IsInteger() {
		return element{i: x.i + y.i}
	}
	if d.IsComplex() {
		return element{z: x.z + y.z}
	}
	return element{z: complex(real(x.z)+real(y.z), 0)}
}

// contract evaluates an explicit index-label contraction. Shared labels
// broadcast size-one dimensions. Repeated labels within an operand select a diagonal.
func contract(arrays []*Array, labels [][]int, output []int) (*Array, error) {
	if len(arrays) == 0 || len(arrays) != len(labels) {
		return nil, ErrShape
	}
	dims := map[int]int{}
	var order []int
	dtype := arrays[0].DType()
	if dtype == 0 {
		return nil, ErrDType
	}
	for i, a := range arrays {
		if a.DType() == 0 || a.NDim() != len(labels[i]) {
			return nil, ErrShape
		}
		var err error
		dtype, err = PromoteTypes(dtype, a.DType())
		if err != nil {
			return nil, err
		}
		local := map[int]int{}
		for j, label := range labels[i] {
			dim := a.shape[j]
			if prior, ok := local[label]; ok && prior != dim {
				return nil, ErrShape
			}
			local[label] = dim
			prior, ok := dims[label]
			if !ok {
				dims[label] = dim
				order = append(order, label)
			} else if prior == 1 {
				dims[label] = dim
			} else if dim != 1 && dim != prior {
				return nil, ErrShape
			}
		}
	}
	if dtype.IsTemporal() || dtype == Object || dtype == StringDType || dtype == RecordDType || dtype.IsFlexible() {
		return nil, ErrDType
	}
	seen := map[int]bool{}
	shape := make([]int, len(output))
	for i, label := range output {
		dim, ok := dims[label]
		if !ok || seen[label] {
			return nil, ErrShape
		}
		seen[label] = true
		shape[i] = dim
	}
	var reduceLabels, reduceShape []int
	for _, label := range order {
		if !seen[label] {
			reduceLabels = append(reduceLabels, label)
			reduceShape = append(reduceShape, dims[label])
		}
	}
	count, err := shapeSize(dtype, reduceShape)
	if err != nil {
		return nil, err
	}
	size, err := shapeSize(dtype, shape)
	if err != nil {
		return nil, err
	}
	if len(arrays) == 1 && len(reduceLabels) == 0 {
		a := arrays[0]
		strides := make([]int, len(output))
		for i, label := range output {
			for j, l := range labels[0] {
				if label == l {
					strides[i] += a.strides[j]
				}
			}
		}
		for _, s := range strides {
			if _, ok := checkedStride(s, 1, dtype.ItemSize()); !ok {
				return nil, ErrShape
			}
		}
		off := a.offset
		if size == 0 {
			off = 0
		}
		return &Array{data: a.data, shape: shape, strides: strides, offset: off, size: size, readOnly: a.readOnly, base: a.viewBase()}, nil
	}
	out, err := Zeros(dtype, shape...)
	if err != nil {
		return nil, err
	}
	type operand struct {
		a                            *Array
		outputStrides, reduceStrides []int
		base                         int
	}
	operands := make([]operand, len(arrays))
	for i, a := range arrays {
		o := operand{a: a, outputStrides: make([]int, len(output)), reduceStrides: make([]int, len(reduceLabels))}
		for ax, label := range labels[i] {
			if a.shape[ax] == 1 && dims[label] != 1 {
				continue
			}
			if j := slices.Index(output, label); j >= 0 {
				o.outputStrides[j] += a.strides[ax]
			} else {
				j := slices.Index(reduceLabels, label)
				o.reduceStrides[j] += a.strides[ax]
			}
		}
		operands[i] = o
	}
	for i := 0; i < size; i++ {
		for j := range operands {
			o := &operands[j]
			o.base = o.a.offset + flatOffset(i, shape, o.outputStrides)
		}
		z := element{}
		for k := 0; k < count; k++ {
			var term element
			for j, o := range operands {
				e := o.a.elementAt(o.base + flatOffset(k, reduceShape, o.reduceStrides))
				v := castElement(e, o.a.DType(), dtype)
				if j == 0 {
					term = v
				} else {
					term = multiplyScalar(term, v, dtype)
				}
			}
			if len(reduceLabels) == 0 {
				z = term
			} else {
				z = addScalar(z, term, dtype)
			}
		}
		out.setElement(i, z)
	}
	return out, nil
}

func Einsum(subscripts string, arrays ...*Array) (*Array, error) {
	spec, err := parseEinsum(subscripts, arrays)
	if err != nil {
		return nil, err
	}
	return contract(arrays, spec.labels, spec.output)
}

// EinsumOptions selects the calculation dtype, checked casting, allocation
// order, output storage and optional contraction optimization.
type EinsumOptions struct {
	DType                    DType
	Casting, Order, Optimize string
	Out                      *Array
	Path                     *EinsumPlan
}

func EinsumWithOptions(subscripts string, opt EinsumOptions, arrays ...*Array) (*Array, error) {
	rule := opt.Casting
	if rule == "" {
		rule = "safe"
	}
	if _, e := CanCast(Bool, Bool, rule); e != nil {
		return nil, e
	}
	order, e := normalizeUFuncOrder(opt.Order)
	if e != nil {
		return nil, e
	}
	if len(arrays) == 0 {
		return nil, ErrShape
	}
	inputs := slices.Clone(arrays)
	if opt.DType != 0 {
		for i, a := range inputs {
			if a.DType() != opt.DType {
				inputs[i], e = a.AstypeWithOptions(opt.DType, CastOptions{Casting: rule, Order: "K"})
				if e != nil {
					return nil, e
				}
			}
		}
	}
	var out *Array
	if opt.Path != nil {
		out, e = EinsumWithPath(subscripts, *opt.Path, inputs...)
	} else if opt.Optimize != "" && opt.Optimize != "none" {
		plan, err := EinsumPath(subscripts, EinsumPathOptions{Optimize: opt.Optimize}, inputs...)
		if err != nil {
			return nil, err
		}
		out, e = EinsumWithPath(subscripts, plan, inputs...)
	} else {
		out, e = Einsum(subscripts, inputs...)
	}
	if e != nil {
		return nil, e
	}
	if opt.Out != nil {
		ok, err := CanCast(out.DType(), opt.Out.DType(), rule)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, ErrDType
		}
		return reductionOutput(opt.Out, out)
	}
	if out.Base() != nil {
		return out, nil
	}
	if order == "A" {
		order = "F"
		for _, a := range arrays {
			if !a.IsFContiguous() {
				order = "C"
				break
			}
		}
	}
	if order == "F" {
		return out.CopyWithOrder("F")
	}
	if order == "K" {
		spec, err := parseEinsum(subscripts, inputs)
		if err != nil {
			return nil, err
		}
		// Map each operand's output labels onto result axes before choosing
		// its stride-compatible allocation order.
		views := make([]*Array, len(inputs))
		for i, a := range inputs {
			strides := make([]int, len(spec.output))
			for j, label := range spec.labels[i] {
				if k := slices.Index(spec.output, label); k >= 0 && a.shape[j] > 1 {
					strides[k] += a.strides[j]
				}
			}
			views[i] = &Array{data: a.data, shape: out.shape, strides: strides, size: out.size}
		}
		return packUFuncResult(out, ufuncResultAxes(out.shape, views, nil, 0, "K"))
	}
	return out, nil
}

type einsumSpec struct {
	labels [][]int
	output []int
}

func parseEinsum(subscripts string, arrays []*Array) (*einsumSpec, error) {
	text := strings.ReplaceAll(subscripts, " ", "")
	parts := strings.Split(text, "->")
	if len(parts) > 2 {
		return nil, ErrValue
	}
	specs := strings.Split(parts[0], ",")
	if len(specs) != len(arrays) || len(arrays) == 0 {
		return nil, ErrShape
	}
	type parsed struct {
		prefix, suffix []int
		ell            int
		hasEll         bool
	}
	parse := func(s string) (parsed, error) {
		var p parsed
		parts := strings.Split(s, "...")
		if len(parts) > 2 {
			return p, ErrValue
		}
		p.hasEll = len(parts) == 2
		decode := func(s string) ([]int, error) {
			out := make([]int, 0, len(s))
			for _, c := range s {
				if (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') {
					return nil, ErrValue
				}
				out = append(out, int(c))
			}
			return out, nil
		}
		var err error
		p.prefix, err = decode(parts[0])
		if err != nil {
			return p, err
		}
		if p.hasEll {
			p.suffix, err = decode(parts[1])
		}
		return p, err
	}
	parsedInputs := make([]parsed, len(arrays))
	maxEll := 0
	counts := map[int]int{}
	for i, s := range specs {
		p, err := parse(s)
		if err != nil {
			return nil, err
		}
		p.ell = arrays[i].NDim() - len(p.prefix) - len(p.suffix)
		if p.ell < 0 || !p.hasEll && p.ell != 0 {
			return nil, ErrShape
		}
		maxEll = max(maxEll, p.ell)
		parsedInputs[i] = p
		for _, l := range p.prefix {
			counts[l]++
		}
		for _, l := range p.suffix {
			counts[l]++
		}
	}
	expand := func(p parsed, ell int) []int {
		result := slices.Clone(p.prefix)
		if p.hasEll {
			for i := maxEll - ell; i < maxEll; i++ {
				result = append(result, 1000+i)
			}
		}
		return append(result, p.suffix...)
	}
	labels := make([][]int, len(arrays))
	for i, p := range parsedInputs {
		labels[i] = expand(p, p.ell)
	}
	var output []int
	if len(parts) == 2 {
		p, err := parse(parts[1])
		if err != nil {
			return nil, err
		}
		output = expand(p, maxEll)
		if !p.hasEll && maxEll > 0 {
			// NumPy's explicit output must retain broadcast ellipsis dimensions.
			return nil, ErrShape
		}
	} else {
		for i := 0; i < maxEll; i++ {
			output = append(output, 1000+i)
		}
		var named []int
		for l, count := range counts {
			if count == 1 {
				named = append(named, l)
			}
		}
		sort.Ints(named)
		output = append(output, named...)
	}
	return &einsumSpec{labels, output}, nil
}

// Tensordot contracts matching axes, preserving the order of uncontracted axes.
func Tensordot(a, b *Array, axesA, axesB []int) (*Array, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return nil, ErrDType
	}
	if len(axesA) != len(axesB) {
		return nil, ErrAxis
	}
	al, bl := make([]int, a.NDim()), make([]int, b.NDim())
	for i := range al {
		al[i] = i
	}
	for i := range bl {
		bl[i] = len(al) + i
	}
	usedA, usedB := make([]bool, len(al)), make([]bool, len(bl))
	contractA, contractB := make([]int, len(axesA)), make([]int, len(axesB))
	for i, axis := range axesA {
		x, err := NormalizeAxis(axis, len(al))
		if err != nil || usedA[x] {
			return nil, ErrAxis
		}
		y, err := NormalizeAxis(axesB[i], len(bl))
		if err != nil || usedB[y] {
			return nil, ErrAxis
		}
		if a.shape[x] != b.shape[y] {
			return nil, ErrShape
		}
		usedA[x], usedB[y] = true, true
		contractA[i], contractB[i] = x, y
		bl[y] = al[x]
	}
	var output []int
	for i, l := range al {
		if !usedA[i] {
			output = append(output, l)
		}
	}
	for i, l := range bl {
		if !usedB[i] {
			output = append(output, l)
		}
	}
	if d, e := PromoteTypes(a.DType(), b.DType()); e == nil && (d == Float32 || d == Float64 || d == Complex64 || d == Complex128) {
		// Tensor contraction is ordinary matrix multiplication after moving
		// the contracted axes next to one another.
		return tensorGemm(a, b, d, contractA, contractB, usedA, usedB)
	}
	return contract([]*Array{a, b}, [][]int{al, bl}, output)
}
func Dot(a, b *Array) (*Array, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 0 || b.NDim() == 0 {
		return Multiply(a, b)
	}
	baxis := 0
	if b.NDim() > 1 {
		baxis = b.NDim() - 2
	}
	return Tensordot(a, b, []int{a.NDim() - 1}, []int{baxis})
}
func Inner(a, b *Array) (*Array, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return nil, ErrDType
	}
	if a.NDim() == 0 || b.NDim() == 0 {
		return Multiply(a, b)
	}
	return Tensordot(a, b, []int{-1}, []int{-1})
}
func Outer(a, b *Array) (*Array, error) {
	x, err := a.Reshape(-1)
	if err != nil {
		return nil, err
	}
	y, err := b.Reshape(-1)
	if err != nil {
		return nil, err
	}
	return Tensordot(x, y, nil, nil)
}
func VDot(a, b *Array) (*Array, error) {
	x, err := a.Reshape(-1)
	if err != nil {
		return nil, err
	}
	y, err := b.Reshape(-1)
	if err != nil {
		return nil, err
	}
	if x.DType().IsComplex() {
		x, err = Conjugate(x)
		if err != nil {
			return nil, err
		}
	}
	return Dot(x, y)
}

// Cross computes the cross product of three-component vectors along the designated axes.
func Cross(a, b *Array, axisA, axisB, axisC int) (*Array, error) {
	if a.DType() == 0 || b.DType() == 0 {
		return nil, ErrDType
	}
	x, err := MoveAxis(a, axisA, -1)
	if err != nil {
		return nil, err
	}
	y, err := MoveAxis(b, axisB, -1)
	if err != nil {
		return nil, err
	}
	if x.NDim() == 0 || y.NDim() == 0 {
		return nil, ErrShape
	}
	nx, ny := x.shape[x.NDim()-1], y.shape[y.NDim()-1]
	if nx != 3 || ny != 3 {
		return nil, ErrShape
	}
	component := func(v *Array, i int) (*Array, error) {
		return v.Slice(Ellipsis, AtIndex(i))
	}
	var u, v [3]*Array
	for i := range u {
		u[i], err = component(x, i)
		if err != nil {
			return nil, err
		}
		v[i], err = component(y, i)
		if err != nil {
			return nil, err
		}
	}
	difference := func(i, j int) (*Array, error) {
		p, err := Multiply(u[i], v[j])
		if err != nil {
			return nil, err
		}
		q, err := Multiply(u[j], v[i])
		if err != nil {
			return nil, err
		}
		return Subtract(p, q)
	}
	z, err := difference(0, 1)
	if err != nil {
		return nil, err
	}
	first, err := difference(1, 2)
	if err != nil {
		return nil, err
	}
	second, err := difference(2, 0)
	if err != nil {
		return nil, err
	}
	out, err := Stack([]*Array{first, second, z}, -1)
	if err != nil {
		return nil, err
	}
	return MoveAxis(out, -1, axisC)
}
func Kron(a, b *Array) (*Array, error) {
	d, err := PromoteTypes(a.DType(), b.DType())
	if err != nil {
		return nil, err
	}
	rank := max(a.NDim(), b.NDim())
	as, bs := make([]int, rank), make([]int, rank)
	ast, bst := make([]int, rank), make([]int, rank)
	for i := range as {
		as[i], bs[i] = 1, 1
	}
	copy(as[rank-a.NDim():], a.shape)
	copy(bs[rank-b.NDim():], b.shape)
	copy(ast[rank-a.NDim():], a.strides)
	copy(bst[rank-b.NDim():], b.strides)
	shape := make([]int, rank)
	for i := range shape {
		if bs[i] != 0 && as[i] > maxInt/bs[i] {
			return nil, ErrShape
		}
		shape[i] = as[i] * bs[i]
	}
	out, err := Zeros(d, shape...)
	if err != nil {
		return nil, err
	}
	for i := 0; i < out.size; i++ {
		k, ao, bo := i, a.offset, b.offset
		for j := rank - 1; j >= 0; j-- {
			v := k % shape[j]
			ao += (v / bs[j]) * ast[j]
			bo += (v % bs[j]) * bst[j]
			k /= shape[j]
		}
		out.setElement(i, multiplyScalar(castElement(a.elementAt(ao), a.DType(), d), castElement(b.elementAt(bo), b.DType(), d), d))
	}
	return out, nil
}
