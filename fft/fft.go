// Package fft implements NumPy's one-axis discrete Fourier transforms.
package fft

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/numeric"

	"gonum.org/v1/gonum/dsp/fourier"
)

type Norm string

const (
	Backward Norm = "backward" // inverse scaled by 1/n (default)
	Forward  Norm = "forward"  // forward scaled by 1/n
	Ortho    Norm = "ortho"    // both scaled by 1/sqrt(n)
)

var ErrNorm = errors.New("invalid FFT normalization")

// Options controls transform length, axis and normalization. Nil Axis selects
// the final axis. Nil N selects its length, except IRFFT where it selects twice
// the input length minus two. Explicit N truncates or zero-pads the input.
type Options struct {
	N    *int
	Axis *int
	Norm Norm
	Out  *np.Array
}

type transformKind uint8

const (
	forwardComplex transformKind = iota
	inverseComplex
	forwardReal
	inverseReal
)

// FFT computes the complex spectrum of a real or complex input array.
func FFT(a *np.Array, options ...Options) (*np.Array, error) {
	return transform(a, forwardComplex, options)
}

// IFFT computes the inverse complex transform.
func IFFT(a *np.Array, options ...Options) (*np.Array, error) {
	return transform(a, inverseComplex, options)
}

// RFFT computes the nonnegative-frequency spectrum of a real input.
func RFFT(a *np.Array, options ...Options) (*np.Array, error) {
	return transform(a, forwardReal, options)
}

// IRFFT reconstructs a real signal from its nonnegative-frequency spectrum.
// Pass N explicitly to recover an odd-length original signal.
func IRFFT(a *np.Array, options ...Options) (*np.Array, error) {
	return transform(a, inverseReal, options)
}

func normalization(norm Norm, n int, inverse bool, dtype np.DType) (float64, error) {
	round := func(x float64) float64 {
		switch dtype {
		case np.Float16:
			return np.HalfFromFloat64(x).Float64()
		case np.Float32, np.Complex64:
			return float64(float32(x))
		default:
			return x
		}
	}
	switch norm {
	case "", Backward:
		if inverse {
			return round(1 / round(float64(n))), nil
		}
	case Forward:
		if !inverse {
			return round(1 / round(float64(n))), nil
		}
	case Ortho:
		return round(1 / round(math.Sqrt(round(float64(n))))), nil
	default:
		return 0, fmt.Errorf("%w: %q", ErrNorm, norm)
	}
	return 1, nil
}

func fftRealInput(a *np.Array) ([]float64, error) {
	if v, ok := np.NativeSliceForKernel[float64](a, numeric.KernelAccess{}, false); ok {
		return v, nil
	}
	if a.DType() == np.Float64 {
		return a.Float64s()
	}
	if a.DType() == np.Float32 {
		v, ok := np.NativeSliceForKernel[float32](a, numeric.KernelAccess{}, false)
		if !ok {
			var e error
			v, e = np.Values[float32](a)
			if e != nil {
				return nil, e
			}
		}
		out := make([]float64, len(v))
		numeric.Widen(out, v)
		return out, nil
	}
	v, e := a.Astype(np.Float64)
	if e != nil {
		return nil, e
	}
	if x, ok := np.NativeSliceForKernel[float64](v, numeric.KernelAccess{}, false); ok {
		return x, nil
	}
	return v.Float64s()
}

// The owned flag distinguishes a detached work slice that may be reused
// as output from a read-only borrow of the caller's native storage.
func fftComplexInput(a *np.Array) ([]complex128, bool, error) {
	if v, ok := np.NativeSliceForKernel[complex128](a, numeric.KernelAccess{}, false); ok {
		return v, false, nil
	}
	v, e := a.Complex128s()
	return v, true, e
}

func selectDirectOut(a, out *np.Array, dtype np.DType, shape []int) *np.Array {
	if out == nil || dtype != np.Float64 && dtype != np.Complex128 && dtype != np.Float32 && dtype != np.Complex64 || out.DType() != dtype || !slices.Equal(out.Shape(), shape) {
		return nil
	}
	if np.MayShareMemory(a, out) {
		// Exact complex aliases may be processed in place: each lane is
		// gathered before writes, and FFTPACK supports identical slices.
		// Any shifted/cropped/reinterpreted alias keeps the snapshot path.
		if (dtype != np.Complex128 && dtype != np.Complex64) || a.DType() != dtype || !slices.Equal(a.Shape(), shape) || a.DataIdentity() != out.DataIdentity() {
			return nil
		}
		if dtype == np.Complex128 {
			if _, ok := np.NativeSliceForKernel[complex128](a, numeric.KernelAccess{}, false); !ok {
				return nil
			}
		} else {
			if _, ok := np.NativeSliceForKernel[complex64](a, numeric.KernelAccess{}, false); !ok {
				return nil
			}
		}
	}
	if dtype == np.Complex128 {
		if _, ok := np.NativeSliceForKernel[complex128](out, numeric.KernelAccess{}, true); ok {
			return out
		}
	} else if dtype == np.Float64 {
		if _, ok := np.NativeSliceForKernel[float64](out, numeric.KernelAccess{}, true); ok {
			return out
		}
	} else if dtype == np.Complex64 {
		if _, ok := np.NativeSliceForKernel[complex64](out, numeric.KernelAccess{}, true); ok {
			return out
		}
	} else if dtype == np.Float32 {
		if _, ok := np.NativeSliceForKernel[float32](out, numeric.KernelAccess{}, true); ok {
			return out
		}
	}
	return nil
}

func fftBuffer[T sample](direct *np.Array, size int) []T {
	if direct != nil {
		v, ok := np.NativeSliceForKernel[T](direct, numeric.KernelAccess{}, true)
		if ok && len(v) == size {
			return v
		}
		// The single-precision destination retains a binary64 working
		// buffer; the final cast writes straight into the supplied array.
		if direct.DType() != np.Float32 && direct.DType() != np.Complex64 {
			panic("invalid direct FFT destination")
		}
	}
	return make([]T, size)
}

func transform(a *np.Array, kind transformKind, options []Options) (*np.Array, error) {
	if (!a.DType().IsNativeNumeric() && !a.DType().IsExtended()) || (kind == forwardReal && a.DType().IsComplex()) {
		return nil, np.ErrDType
	}
	if len(options) > 1 {
		return nil, np.ErrShape
	}
	var opt Options
	if len(options) == 1 {
		opt = options[0]
	}
	shape := a.Shape()
	axis := -1
	if opt.Axis != nil {
		axis = *opt.Axis
	}
	axis, err := np.NormalizeAxis(axis, len(shape))
	if err != nil {
		return nil, err
	}
	inLen := shape[axis]
	n := inLen
	if opt.N != nil {
		n = *opt.N
	} else if kind == inverseReal {
		// The source array's shape validation ensures doubling inLen fits int.
		n = 2 * (inLen - 1)
	}
	if n <= 0 {
		return nil, fmt.Errorf("%w: FFT length must be positive", np.ErrShape)
	}
	scale, err := normalization(opt.Norm, n, kind == inverseComplex || kind == inverseReal, a.DType())
	if err != nil {
		return nil, err
	}
	if _, err := np.ElementCount(np.Complex128, n); err != nil {
		return nil, err
	}
	outLen := n
	dtype := np.Complex128
	single := a.DType() == np.Float16 || a.DType() == np.Float32 || a.DType() == np.Complex64
	if single {
		dtype = np.Complex64
	}
	if a.DType().IsExtended() {
		dtype = np.Complex256
	}
	if kind == forwardReal {
		outLen = n/2 + 1
	} else if kind == inverseReal {
		dtype = np.Float64
		if single {
			dtype = np.Float32
		}
		if a.DType() == np.Float16 {
			dtype = np.Float16
		}
		if a.DType().IsExtended() {
			dtype = np.Float128
		}
	}
	shape[axis] = outLen
	if opt.Out != nil {
		if err := validateOut(opt.Out, dtype, shape, axis); err != nil {
			return nil, err
		}
		// A real input may select a wider FFT loop when requested by out.
		// Complex input retains its original transform precision.
		if !a.DType().IsComplex() && !a.DType().IsExtended() {
			od := opt.Out.DType()
			if kind == inverseReal {
				if od == np.Float128 || od == np.Complex256 || od == np.Float64 || od == np.Complex128 {
					dtype = np.Float64
				} else if od != np.Float16 && single {
					dtype = np.Float32
				}
			} else if od == np.Complex256 {
				dtype = np.Complex128
			} else if od == np.Complex128 && !(kind == forwardReal && a.DType() == np.Float32) {
				dtype = np.Complex128
			}
		}
	}
	direct := selectDirectOut(a, opt.Out, dtype, shape)
	finish := func(result *np.Array, e error) (*np.Array, error) {
		if e != nil {
			return nil, e
		}
		out := opt.Out
		if out != nil && result == out {
			return out, nil
		}
		if out == nil {
			if a.IsCContiguous() {
				return result, nil
			}
			out, e = np.EmptyLikeWithShape(a, dtype, "K", shape)
			if e != nil {
				return nil, e
			}
		}
		if e = out.Assign(nil, result); e != nil {
			return nil, e
		}
		return out, nil
	}
	size, err := np.ElementCount(dtype, shape...)
	if err != nil {
		return nil, err
	}
	if size == 0 {
		return finish(np.Zeros(dtype, shape...))
	}
	inner := 1
	for _, dim := range shape[axis+1:] {
		inner *= dim
	}
	outer := size / (outLen * inner)
	if a.DType().IsExtended() || dtype.IsExtended() {
		return finish(transformExtended(a, kind, opt.Norm, n, shape, inLen, outLen, inner, outer, dtype))
	}
	if opt.Out == nil && !a.IsCContiguous() && a.IsFContiguous() {
		// Reverse the axes of a packed F-order input to process it in its
		// native order. Restore the original output layout via finish.
		v, e := a.Transpose()
		if e != nil {
			return nil, e
		}
		opt.Axis = np.Int(len(shape) - 1 - axis)
		v, e = transform(v, kind, []Options{opt})
		if e != nil {
			return nil, e
		}
		return finish(v.Transpose())
	}
	if result, e := transformMixed(a, kind, n, shape, inLen, outLen, inner, outer, dtype, scale, direct); result != nil || e != nil {
		return finish(result, e)
	}
	if result, e := transformSingle(a, kind, n, shape, inLen, outLen, inner, outer, dtype, scale, direct); result != nil || e != nil {
		return finish(result, e)
	}
	if result, e := transformChirp(a, kind, n, shape, inLen, outLen, inner, outer, size, dtype, scale, direct); result != nil || e != nil {
		return finish(result, e)
	}
	if result, e := transformPow2(a, kind, n, shape, inLen, outLen, inner, outer, size, dtype, scale, direct); result != nil || e != nil {
		return finish(result, e)
	}
	if kind == forwardReal {
		input, err := fftRealInput(a)
		if err != nil {
			return nil, err
		}
		output := fftBuffer[complex128](direct, size)
		plan := fourier.NewFFT(n)
		line, spectrum := make([]float64, n), make([]complex128, outLen)
		for batch := 0; batch < outer; batch++ {
			for lane := 0; lane < inner; lane++ {
				seq := line
				if inner == 1 && inLen == n {
					seq = input[batch*n : (batch+1)*n]
				} else {
					gather(line, input, inLen, inner, batch, lane)
				}
				if inner == 1 {
					dest := output[batch*outLen : (batch+1)*outLen]
					plan.Coefficients(dest, seq)
					if scale != 1 {
						for j, z := range dest {
							dest[j] = scaled(z, scale)
						}
					}
					continue
				}
				plan.Coefficients(spectrum, seq)
				for i, z := range spectrum {
					output[(batch*outLen+i)*inner+lane] = scaled(z, scale)
				}
			}
		}
		return finish(complexOutput(output, dtype, shape, direct))
	}
	input, owned, err := fftComplexInput(a)
	if err != nil {
		return nil, err
	}
	if kind == inverseReal {
		output := fftBuffer[float64](direct, size)
		plan := fourier.NewFFT(n)
		spectrum, line := make([]complex128, n/2+1), make([]float64, n)
		for batch := 0; batch < outer; batch++ {
			for lane := 0; lane < inner; lane++ {
				seq := spectrum
				if inner == 1 && inLen == len(spectrum) {
					seq = input[batch*inLen : (batch+1)*inLen]
				} else {
					gather(spectrum, input, inLen, inner, batch, lane)
				}
				if inner == 1 {
					dest := output[batch*n : (batch+1)*n]
					plan.Sequence(dest, seq)
					if scale != 1 {
						for j, x := range dest {
							dest[j] = x * scale
						}
					}
					continue
				}
				plan.Sequence(line, seq)
				for i, x := range line {
					output[(batch*outLen+i)*inner+lane] = x * scale
				}
			}
		}
		return finish(realOutput(output, dtype, shape, direct))
	}
	var output []complex128
	if (direct == nil || dtype == np.Complex64) && owned && n == inLen {
		output = input
	} else {
		output = fftBuffer[complex128](direct, size)
	}
	plan := fourier.NewCmplxFFT(n)
	line, spectrum := make([]complex128, n), make([]complex128, n)
	for batch := 0; batch < outer; batch++ {
		for lane := 0; lane < inner; lane++ {
			seq := line
			if inner == 1 && inLen == n {
				seq = input[batch*n : (batch+1)*n]
			} else {
				gather(line, input, inLen, inner, batch, lane)
			}
			dest := spectrum
			if inner == 1 {
				dest = output[batch*n : (batch+1)*n]
			}
			if kind == inverseComplex {
				plan.Sequence(dest, seq)
			} else {
				plan.Coefficients(dest, seq)
			}
			if inner == 1 {
				if scale != 1 {
					for j, z := range dest {
						dest[j] = scaled(z, scale)
					}
				}
				continue
			}
			for i, z := range dest {
				output[(batch*outLen+i)*inner+lane] = scaled(z, scale)
			}
		}
	}
	return finish(complexOutput(output, dtype, shape, direct))
}

func validateOut(out *np.Array, dtype np.DType, shape []int, axis int) error {
	if out.DType() == 0 {
		return np.ErrDType
	}
	if !out.Writable() {
		return np.ErrReadOnly
	}
	os := out.Shape()
	if len(os) != len(shape) || os[axis] != shape[axis] {
		return np.ErrShape
	}
	for i, d := range shape {
		if i != axis && d != 1 && d != os[i] {
			return np.ErrShape
		}
	}
	ok, err := np.CanCast(dtype, out.DType(), "same_kind")
	if err != nil {
		return err
	}
	if !ok {
		return np.ErrDType
	}
	return nil
}

func complexOutput(v []complex128, d np.DType, shape []int, direct *np.Array) (*np.Array, error) {
	if direct != nil {
		if d == np.Complex64 {
			dst, _ := np.NativeSliceForKernel[complex64](direct, numeric.KernelAccess{}, true)
			numeric.NarrowComplex(dst, v)
		}
		return direct, nil
	}
	out, err := np.WrapComplex128(v, shape...)
	if err != nil || d == np.Complex128 {
		return out, err
	}
	return out.Astype(d)
}

func realOutput(v []float64, d np.DType, shape []int, direct *np.Array) (*np.Array, error) {
	if direct != nil {
		if d == np.Float32 {
			dst, _ := np.NativeSliceForKernel[float32](direct, numeric.KernelAccess{}, true)
			numeric.Narrow(dst, v)
		}
		return direct, nil
	}
	out, e := np.WrapFloat64(v, shape...)
	if e != nil || d == np.Float64 {
		return out, e
	}
	return out.Astype(d)
}

type sample interface {
	float32 | float64 | complex64 | complex128
}

func gather[T sample](dst, src []T, length, inner, batch, lane int) {
	limit := min(length, len(dst))
	if inner == 1 {
		copy(dst, src[batch*length:batch*length+limit])
		clear(dst[limit:])
		return
	}
	for i := 0; i < limit; i++ {
		dst[i] = src[(batch*length+i)*inner+lane]
	}
	clear(dst[limit:])
}

func scaled(z complex128, scale float64) complex128 {
	return complex(real(z)*scale, imag(z)*scale)
}
