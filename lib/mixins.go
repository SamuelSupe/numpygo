package lib

import "github.com/SamuelSupe/numpygo"

// UFuncOverride is the Go counterpart of __array_ufunc__ dispatch. An
// in-place call receives the retained value in outputs; other calls have
// no explicit outputs. The callback decides how to handle custom values,
// dtype promotion, multiple outputs and unsupported operand types.
type UFuncOverride func(name string, inputs, outputs []any) (any, error)

// NDArrayOperatorsMixin supplies explicit methods for the operators that
// NumPy's Python mixin forwards to universal functions. Go does not
// overload native +, -, or comparison syntax.
type NDArrayOperatorsMixin struct {
	value    any
	dispatch UFuncOverride
}

func NewNDArrayOperatorsMixin(value any, dispatch UFuncOverride) (*NDArrayOperatorsMixin, error) {
	if dispatch == nil {
		return nil, np.ErrValue
	}
	return &NDArrayOperatorsMixin{value, dispatch}, nil
}
func (m *NDArrayOperatorsMixin) Value() any { return m.value }

var mixinComparisons = map[string]string{"lt": "less", "le": "less_equal", "eq": "equal", "ne": "not_equal", "gt": "greater", "ge": "greater_equal"}
var mixinBinary = map[string]string{
	"add": "add", "sub": "subtract", "mul": "multiply", "matmul": "matmul", "truediv": "divide",
	"floordiv": "floor_divide", "mod": "remainder", "divmod": "divmod", "pow": "power",
	"lshift": "left_shift", "rshift": "right_shift", "and": "bitwise_and", "xor": "bitwise_xor", "or": "bitwise_or",
}
var mixinUnary = map[string]string{"neg": "negative", "pos": "positive", "abs": "absolute", "invert": "invert"}

func (m *NDArrayOperatorsMixin) Unary(op string) (any, error) {
	name, ok := mixinUnary[op]
	if !ok {
		return nil, np.ErrValue
	}
	return m.dispatch(name, []any{m.value}, nil)
}
func (m *NDArrayOperatorsMixin) Binary(op string, other any) (any, error) {
	name, ok := mixinBinary[op]
	if !ok {
		name, ok = mixinComparisons[op]
	}
	if !ok {
		return nil, np.ErrValue
	}
	return m.dispatch(name, []any{m.value, other}, nil)
}
func (m *NDArrayOperatorsMixin) Reflected(op string, other any) (any, error) {
	name, ok := mixinBinary[op]
	if !ok {
		return nil, np.ErrValue
	}
	return m.dispatch(name, []any{other, m.value}, nil)
}
func (m *NDArrayOperatorsMixin) InPlace(op string, other any) (any, error) {
	name, ok := mixinBinary[op]
	if !ok || op == "divmod" {
		return nil, np.ErrValue
	}
	return m.dispatch(name, []any{m.value, other}, []any{m.value})
}
