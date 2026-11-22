package np

import (
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"os"
	"sync"

	"github.com/SamuelSupe/numpygo/internal/x87"
)

type ErrorMode string

const (
	Ignore   ErrorMode = "ignore"
	Warn     ErrorMode = "warn"
	Raise    ErrorMode = "raise"
	Call     ErrorMode = "call"
	Print    ErrorMode = "print"
	LogError ErrorMode = "log"
)

type FPFlag uint8

const (
	FPDivide FPFlag = 1 << iota
	FPOverflow
	FPUnderflow
	FPInvalid
)

func (flag FPFlag) String() string {
	switch flag {
	case FPDivide:
		return "divide by zero"
	case FPOverflow:
		return "overflow"
	case FPUnderflow:
		return "underflow"
	case FPInvalid:
		return "invalid value"
	}
	return "floating-point error"
}

type ErrorPolicy struct{ Divide, Over, Under, Invalid ErrorMode }
type ErrorOptions struct{ All, Divide, Over, Under, Invalid ErrorMode }
type ErrorHandler struct {
	Callback func(string, int)
	Writer   io.Writer
}
type FloatingPointError struct {
	Kind      FPFlag
	Flags     FPFlag
	Operation string
}

var ErrFloatingPoint = errors.New("floating-point error")

func (e *FloatingPointError) Unwrap() error { return ErrFloatingPoint }

func (e *FloatingPointError) Error() string {
	return fmt.Sprintf("%s encountered in %s", e.Kind, e.Operation)
}

// ErrorContext is the explicit Go equivalent of NumPy's per-thread floating
// point settings. Independent contexts avoid process-global changes leaking
// across goroutines. Raw numerical functions remain policy-free kernels.
// Construct contexts with NewErrorContext; the zero value is not initialized.
type ErrorContext struct {
	mu       sync.RWMutex
	policy   ErrorPolicy
	handler  *ErrorHandler
	bufsize  int
	warnings io.Writer
}

func NewErrorContext() *ErrorContext {
	return &ErrorContext{policy: ErrorPolicy{Warn, Warn, Ignore, Warn}, bufsize: 8192, warnings: os.Stderr}
}

var defaultErrorContext = NewErrorContext()

func DefaultErrorContext() *ErrorContext    { return defaultErrorContext }
func (c *ErrorContext) GetErr() ErrorPolicy { c.mu.RLock(); defer c.mu.RUnlock(); return c.policy }
func validMode(m ErrorMode) bool {
	switch m {
	case Ignore, Warn, Raise, Call, Print, LogError:
		return true
	}
	return false
}
func (c *ErrorContext) SetErr(opt ErrorOptions) (ErrorPolicy, error) {
	for _, mode := range []ErrorMode{opt.All, opt.Divide, opt.Over, opt.Under, opt.Invalid} {
		if mode != "" && !validMode(mode) {
			return ErrorPolicy{}, ErrValue
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	old := c.policy
	if opt.All != "" {
		c.policy = ErrorPolicy{opt.All, opt.All, opt.All, opt.All}
	}
	if opt.Divide != "" {
		c.policy.Divide = opt.Divide
	}
	if opt.Over != "" {
		c.policy.Over = opt.Over
	}
	if opt.Under != "" {
		c.policy.Under = opt.Under
	}
	if opt.Invalid != "" {
		c.policy.Invalid = opt.Invalid
	}
	return old, nil
}
func (c *ErrorContext) GetErrCall() *ErrorHandler {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.handler
}
func (c *ErrorContext) SetErrCall(h *ErrorHandler) *ErrorHandler {
	c.mu.Lock()
	defer c.mu.Unlock()
	old := c.handler
	c.handler = h
	return old
}
func (c *ErrorContext) GetBufSize() int { c.mu.RLock(); defer c.mu.RUnlock(); return c.bufsize }
func (c *ErrorContext) SetBufSize(n int) (int, error) {
	if n < 16 || n%16 != 0 || n > 10000000 {
		return 0, ErrValue
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	old := c.bufsize
	c.bufsize = n
	return old, nil
}
func (c *ErrorContext) SetWarningWriter(w io.Writer) {
	if w == nil {
		w = io.Discard
	}
	c.mu.Lock()
	c.warnings = w
	c.mu.Unlock()
}

// ErrState executes with a derived context. Restoration is automatic even
// when fn returns an error or panics; no global/thread-local state is mutated.
func (c *ErrorContext) ErrState(opt ErrorOptions, fn func(*ErrorContext) error) error {
	if fn == nil {
		return ErrValue
	}
	c.mu.RLock()
	child := &ErrorContext{policy: c.policy, handler: c.handler, bufsize: c.bufsize, warnings: c.warnings}
	c.mu.RUnlock()
	if _, e := child.SetErr(opt); e != nil {
		return e
	}
	return fn(child)
}
func GetErr() ErrorPolicy                          { return defaultErrorContext.GetErr() }
func SetErr(opt ErrorOptions) (ErrorPolicy, error) { return defaultErrorContext.SetErr(opt) }
func GetErrCall() *ErrorHandler                    { return defaultErrorContext.GetErrCall() }
func SetErrCall(h *ErrorHandler) *ErrorHandler     { return defaultErrorContext.SetErrCall(h) }
func GetBufSize() int                              { return defaultErrorContext.GetBufSize() }
func SetBufSize(n int) (int, error)                { return defaultErrorContext.SetBufSize(n) }
func ErrState(opt ErrorOptions, fn func(*ErrorContext) error) error {
	return defaultErrorContext.ErrState(opt, fn)
}

func hasNaN(z complex128) bool { return math.IsNaN(real(z)) || math.IsNaN(imag(z)) }
func hasInf(z complex128) bool { return math.IsInf(real(z), 0) || math.IsInf(imag(z), 0) }
func tinyFor(d DType) float64 {
	switch d {
	case Float16:
		return math.Ldexp(1, -14)
	case Float32, Complex64:
		return math.Ldexp(1, -126)
	case Float64, Complex128:
		return math.Ldexp(1, -1022)
	}
	return 0
}
func inexactTiny(name string, x, y, z float64) bool {
	if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
		return false
	}
	rx, ry := new(big.Rat).SetFloat64(x), new(big.Rat).SetFloat64(y)
	var r *big.Rat
	switch name {
	case "add":
		r = rx.Add(rx, ry)
	case "subtract":
		r = rx.Sub(rx, ry)
	case "multiply":
		r = rx.Mul(rx, ry)
	case "square":
		r = rx.Mul(rx, rx)
	case "divide":
		if y == 0 {
			return false
		}
		r = rx.Quo(rx, ry)
	case "reciprocal":
		if x == 0 {
			return false
		}
		r = rx.Inv(rx)
	case "exp":
		return x != 0
	case "expm1":
		return x != 0 && z != x
	case "power":
		return x != 0 && y != 0
	default:
		return false
	}
	return r.Sign() != 0 && r.Cmp(new(big.Rat).SetFloat64(z)) != 0
}
func flagsFor(name string, d DType, x, y, z complex128, arity int, complexInput bool) FPFlag {
	finite := !hasNaN(x) && !hasInf(x) && (arity == 1 || !hasNaN(y) && !hasInf(y))
	inputNaN := hasNaN(x) || arity == 2 && hasNaN(y)
	flags := FPFlag(0)
	switch name {
	case "divide", "floor_divide":
		if y == 0 && !hasNaN(x) && !hasInf(x) && x != 0 {
			flags |= FPDivide
		}
		if y == 0 && (d.IsInteger() || d == Bool) {
			flags |= FPDivide
		}
	case "remainder":
		if y == 0 && (d.IsInteger() || d == Bool) {
			flags |= FPDivide
		}
	case "reciprocal", "log", "log2", "log10":
		if x == 0 {
			flags |= FPDivide
		}
	case "log1p":
		if x == -1 {
			flags |= FPDivide
		}
	case "power":
		if x == 0 && imag(y) == 0 && real(y) < 0 {
			flags |= FPDivide
		}
	}
	if !inputNaN && hasNaN(z) {
		flags |= FPInvalid
	}
	if finite && hasInf(z) && flags&FPDivide == 0 {
		flags |= FPOverflow
	}
	threshold := tinyFor(d)
	if threshold != 0 && finite && !hasNaN(z) && !hasInf(z) && flags&FPDivide == 0 && !complexInput {
		r := real(z)
		if math.Abs(r) < threshold && inexactTiny(name, real(x), real(y), r) {
			flags |= FPUnderflow
		}
	}
	return flags
}

func extendedFlags(name string, d DType, x, y, z x87.Complex, arity int, complexInput bool) FPFlag {
	zero := func(v x87.Complex) bool { return v.Real.IsZero() && v.Imag.IsZero() }
	nan := func(v x87.Complex) bool { return v.Real.IsNaN() || v.Imag.IsNaN() }
	inf := func(v x87.Complex) bool { return v.Real.IsInf() || v.Imag.IsInf() }
	finite := !nan(x) && !inf(x) && (arity == 1 || !nan(y) && !inf(y))
	flags := FPFlag(0)
	switch name {
	case "divide", "floor_divide":
		if zero(y) && !zero(x) && !nan(x) && !inf(x) {
			flags |= FPDivide
		}
	case "reciprocal", "log", "log2", "log10":
		if zero(x) {
			flags |= FPDivide
		}
	case "log1p":
		if x.Real.Equal(x87.FromInt64(-1)) && x.Imag.IsZero() {
			flags |= FPDivide
		}
	case "power", "float_power":
		if zero(x) && y.Imag.IsZero() && y.Real.Signbit() && !y.Real.IsZero() {
			flags |= FPDivide
		}
	}
	if !nan(x) && (arity == 1 || !nan(y)) && nan(z) {
		flags |= FPInvalid
	}
	if finite && inf(z) && flags&FPDivide == 0 {
		flags |= FPOverflow
	}
	if finite && d == Float128 && !complexInput && !nan(z) && !inf(z) && flags&FPDivide == 0 {
		if cmp, _ := z.Real.Abs().Compare(x87.Info().Tiny); cmp < 0 {
			a, b := x.Real.Rat(), y.Real.Rat()
			var r *big.Rat
			switch name {
			case "add":
				r = a.Add(a, b)
			case "subtract":
				r = a.Sub(a, b)
			case "multiply":
				r = a.Mul(a, b)
			case "square":
				r = a.Mul(a, a)
			case "divide":
				if b.Sign() != 0 {
					r = a.Quo(a, b)
				}
			case "reciprocal":
				if a.Sign() != 0 {
					r = a.Inv(a)
				}
			case "exp", "exp2":
				if !x.Real.IsZero() {
					flags |= FPUnderflow
				}
			case "expm1":
				if !x.Real.IsZero() && !z.Real.Equal(x.Real) {
					flags |= FPUnderflow
				}
			case "power", "float_power":
				if !zero(x) && !zero(y) {
					flags |= FPUnderflow
				}
			}
			if r != nil && r.Sign() != 0 && r.Cmp(z.Real.Rat()) != 0 {
				flags |= FPUnderflow
			}
		}
	}
	return flags
}

func (c *ErrorContext) handle(name string, flags FPFlag, policy ErrorPolicy, h *ErrorHandler, w io.Writer) error {
	for i, flag := range []FPFlag{FPDivide, FPOverflow, FPUnderflow, FPInvalid} {
		if flags&flag == 0 {
			continue
		}
		mode := []ErrorMode{policy.Divide, policy.Over, policy.Under, policy.Invalid}[i]
		message := flag.String() + " encountered in " + name
		switch mode {
		case Ignore:
		case Raise:
			return &FloatingPointError{flag, flags, name}
		case Warn:
			if _, e := fmt.Fprintln(w, "RuntimeWarning:", message); e != nil {
				return e
			}
		case Print:
			if _, e := fmt.Fprintln(w, "Warning:", message); e != nil {
				return e
			}
		case Call:
			if h == nil || h.Callback == nil {
				return ErrValue
			}
			h.Callback(flag.String(), int(flags))
		case LogError:
			if h == nil || h.Writer == nil {
				return ErrValue
			}
			if _, e := fmt.Fprintln(h.Writer, "Warning:", message); e != nil {
				return e
			}
		}
	}
	return nil
}

// Call applies a numerical ufunc and reports the four standard FP conditions
// once per operation. It detects result-level conditions and real inexact
// underflow; it is not a processor fenv or a promise about every intermediate.
func (c *ErrorContext) Call(name string, args ...*Array) (*Array, error) {
	u, e := LookupUFunc(name)
	if e != nil {
		return nil, e
	}
	out, e := u.Call(args...)
	if e != nil {
		return nil, e
	}
	return c.reportResult(u, out, nil, args)
}

// CallWithOptions applies the same explicit floating-point policy to masked
// evaluation, selected dtypes and output views. Snapshot input values before
// writes so FP diagnostics remain correct when out aliases an operand.
func (c *ErrorContext) CallWithOptions(name string, opt UFuncOptions, args ...*Array) (*Array, error) {
	u, e := LookupUFunc(name)
	if e != nil {
		return nil, e
	}
	if u.Signature() != "" || u.NOut != 1 {
		return nil, ErrValue
	}
	inputs := args
	mask := opt.Where
	if opt.Out != nil {
		inputs = make([]*Array, len(args))
		for i, a := range args {
			if nativeNumericStorage(opt.Out) && nativeNumericStorage(a) && !MayShareMemory(opt.Out, a) {
				inputs[i] = a
				continue
			}
			inputs[i], e = a.Copy()
			if e != nil {
				return nil, e
			}
		}
		if mask != nil && !(nativeNumericStorage(opt.Out) && nativeNumericStorage(mask) && !MayShareMemory(opt.Out, mask)) {
			mask, e = mask.Copy()
			if e != nil {
				return nil, e
			}
		}
	}
	out, e := u.CallWithOptions(opt, args...)
	if e != nil {
		return nil, e
	}
	return c.reportResult(u, out, mask, inputs)
}

func (c *ErrorContext) reportResult(u *UFunc, out, where *Array, args []*Array) (*Array, error) {
	var e error
	c.mu.RLock()
	policy, h, buf, w := c.policy, c.handler, c.bufsize, c.warnings
	c.mu.RUnlock()
	if buf < 16 || w == nil {
		return nil, ErrValue
	}
	if policy == (ErrorPolicy{Ignore, Ignore, Ignore, Ignore}) {
		return out, nil
	}
	// Comparisons/logical/bitwise operations do not raise FP flags for NaNs.
	switch u.Name {
	case "equal", "logical_and", "logical_or", "bitwise_and", "bitwise_or", "bitwise_xor", "minimum", "maximum", "positive", "negative", "absolute":
		return out, nil
	}
	a, e := args[0].BroadcastTo(out.Shape()...)
	if e != nil {
		return nil, e
	}
	var b *Array
	if u.NIn == 2 {
		b, e = args[1].BroadcastTo(out.Shape()...)
		if e != nil {
			return nil, e
		}
	}
	var mask *Array
	if where != nil {
		mask, e = where.BroadcastTo(out.shape...)
		if e != nil {
			return nil, e
		}
	}
	flags := FPFlag(0)
	complexInput := args[0].DType().IsComplex() || b != nil && args[1].DType().IsComplex()
	// Honor the per-context scratch block size without altering numerical order.
	for start := 0; start < out.size; start += buf {
		end := min(out.size, start+buf)
		for i := start; i < end; i++ {
			if mask != nil && mask.elementAt(mask.offset+flatOffset(i, mask.shape, mask.strides)).i == 0 {
				continue
			}
			xe := a.elementAt(a.offset + flatOffset(i, a.shape, a.strides))
			ze := out.elementAt(out.offset + flatOffset(i, out.shape, out.strides))
			if out.DType().IsExtended() {
				x, y, z := extendedValue(xe, a.DType()), x87.Complex{}, extendedValue(ze, out.DType())
				if b != nil {
					y = extendedValue(b.elementAt(b.offset+flatOffset(i, b.shape, b.strides)), b.DType())
				}
				flags |= extendedFlags(u.Name, out.DType(), x, y, z, u.NIn, complexInput)
				continue
			}
			x := xe.number(a.DType())
			y := complex128(0)
			if b != nil {
				y = b.elementAt(b.offset + flatOffset(i, b.shape, b.strides)).number(b.DType())
			}
			z := ze.number(out.DType())
			flags |= flagsFor(u.Name, out.DType(), x, y, z, u.NIn, complexInput)
		}
	}
	if e = c.handle(u.Name, flags, policy, h, w); e != nil {
		return nil, e
	}
	return out, nil
}
func CallUFunc(name string, args ...*Array) (*Array, error) {
	return defaultErrorContext.Call(name, args...)
}
func CallUFuncWithOptions(name string, opt UFuncOptions, args ...*Array) (*Array, error) {
	return defaultErrorContext.CallWithOptions(name, opt, args...)
}
