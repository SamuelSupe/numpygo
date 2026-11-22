package numeric

import (
	"math"
	"unsafe"
)

// Keep the right operand on equality (including signed zero), and
// propagate NaN from either operand, with the left NaN taking priority.
func extremaValue[T float32 | float64](x, y T, maximum bool) T {
	if x != x || maximum && x > y || !maximum && x < y {
		return x
	}
	return y
}

func Extrema64(dst, x, y []float64, maximum bool) bool {
	n := len(dst) &^ 3
	if n < 16 || !extrema64SIMD(dst[:n], x[:n], y[:n], maximum) {
		n = 0
	}
	for i := n; i < len(dst); i++ {
		dst[i] = extremaValue(x[i], y[i], maximum)
	}
	return true
}

func Extrema32(dst, x, y []float32, maximum bool) bool {
	n := len(dst) &^ 7
	if n < 16 || !extrema32SIMD(dst[:n], x[:n], y[:n], maximum) {
		n = 0
	}
	for i := n; i < len(dst); i++ {
		dst[i] = extremaValue(x[i], y[i], maximum)
	}
	return true
}

// Exp reduces by an integral multiple of ln(2) before evaluating the
// degree-13 Taylor polynomial on [-ln(2)/2, ln(2)/2]. Split ln(2)
// avoids losing the low bits through cancellation. Outside the normal
// scaling range use math.Exp, including underflow, overflow and NaN.
var expConstants = [...]float64{
	1.4426950408889634074, 6755399441055744, 6755399441056767,
	6.93147180369123816490e-1, 1.90821492927058770002e-10,
	1.0 / 6227020800, 1.0 / 479001600, 1.0 / 39916800, 1.0 / 3628800,
	1.0 / 362880, 1.0 / 40320, 1.0 / 5040, 1.0 / 720, 1.0 / 120,
	1.0 / 24, 1.0 / 6, .5, 1, 1,
}

var expConstants32 = [...]float32{
	1.4426950408889634, 12582912, 12583039,
	.693359375, -.00021219444005469057,
	1.0 / 5040, 1.0 / 720, 1.0 / 120, 1.0 / 24, 1.0 / 6, .5, 1, 1,
}

func Exp64(dst, src []float64) {
	if !advancedVectorsAvailable() {
		for i, x := range src {
			dst[i] = math.Exp(x)
		}
		return
	}
	const block = 256
	for start := 0; start < len(dst); start += block {
		end := min(start+block, len(dst))
		n := (end - start) &^ 3
		regular := n >= 16
		for _, x := range src[start : start+n] {
			if !(x >= -700 && x <= 700) {
				regular = false
				break
			}
		}
		if !regular || !exp64SIMD(dst[start:start+n], src[start:start+n], &expConstants) {
			n = 0
		}
		for i := start + n; i < end; i++ {
			dst[i] = math.Exp(src[i])
		}
	}
}

func Exp32(dst, src []float32) {
	if !advancedVectorsAvailable() {
		for i, x := range src {
			dst[i] = float32(math.Exp(float64(x)))
		}
		return
	}
	// The regular float32 range uses native packed arithmetic with an
	// exactly representable high ln(2) term and degree-7 polynomial.
	// Near overflow/subnormals or nonfinite values retain scalar math.Exp.
	const block = 256
	for start := 0; start < len(dst); start += block {
		end := min(start+block, len(dst))
		n := (end - start) &^ 7
		regular := n >= 16
		for _, x := range src[start : start+n] {
			if !(x >= -80 && x <= 80) {
				regular = false
				break
			}
		}
		if !regular || !exp32SIMD(dst[start:start+n], src[start:start+n], &expConstants32) {
			n = 0
		}
		for i := start + n; i < end; i++ {
			dst[i] = float32(math.Exp(float64(src[i])))
		}
	}
}

// ComplexDivide is the scalar Smith ratio used by both the regular and
// vectorized complex division loops. In particular zero denominators
// divide each numerator component by +0.
func ComplexDivide(x, y complex128) complex128 {
	ar, ai, br, bi := real(x), imag(x), real(y), imag(y)
	ab, aiB := math.Abs(br), math.Abs(bi)
	if ab >= aiB {
		if ab == 0 {
			return complex(ar/ab, ai/ab)
		}
		ratio := bi / br
		// Explicit conversions prevent architecture-specific implicit
		// FMA contraction; the SIMD kernel uses separate mul/add.
		scale := 1 / (br + float64(bi*ratio))
		return complex((ar+float64(ai*ratio))*scale, (ai-float64(ar*ratio))*scale)
	}
	ratio := br / bi
	scale := 1 / (bi + float64(br*ratio))
	return complex((float64(ar*ratio)+ai)*scale, (float64(ai*ratio)-ar)*scale)
}

func ComplexBinary128(dst, x, y []complex128, op uint8) bool {
	if len(dst) < 16 {
		return false
	}
	if op <= 1 {
		parts := func(v []complex128) []float64 {
			return unsafe.Slice((*float64)(unsafe.Pointer(unsafe.SliceData(v))), 2*len(v))
		}
		return Binary64(parts(dst), parts(x), parts(y), op)
	}
	n := len(dst) &^ 1
	if !complex128SIMD(dst[:n], x[:n], y[:n], op) {
		return false
	}
	for i := n; i < len(dst); i++ {
		if op == 2 {
			dst[i] = x[i] * y[i]
		} else {
			dst[i] = ComplexDivide(x[i], y[i])
		}
	}
	return true
}

func ComplexBinary64(dst, x, y []complex64, op uint8) bool {
	if len(dst) < 16 {
		return false
	}
	if op <= 1 {
		parts := func(v []complex64) []float32 {
			return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(v))), 2*len(v))
		}
		return Binary32(parts(dst), parts(x), parts(y), op)
	}
	// Widen two complex64 values directly into FP64 registers and
	// narrow the result only at the store. The arithmetic is identical
	// to the existing wide kernel, with no conversion workspace.
	n := len(dst) &^ 1
	if !complex64SIMD(dst[:n], x[:n], y[:n], op) {
		return false
	}
	for i := n; i < len(dst); i++ {
		a, b := complex128(x[i]), complex128(y[i])
		if op == 2 {
			dst[i] = complex64(a * b)
		} else {
			dst[i] = complex64(ComplexDivide(a, b))
		}
	}
	return true
}
