//go:build arm64 && !purego

package numeric

import "unsafe"

func advancedVectorsAvailable() bool { return true }

//go:noescape
func extrema64NEON(dst, x, y []float64, maximum bool)

//go:noescape
func extrema32NEON(dst, x, y []float32, maximum bool)

//go:noescape
func exp64NEON(dst, src []float64, constants *[19]float64)

//go:noescape
func exp32NEON(dst, src []float32, constants *[13]float32)
func exp32SIMD(dst, src []float32, c *[13]float32) bool { exp32NEON(dst, src, c); return true }

//go:noescape
func complex128NEON(dst, x, y []complex128, op uint8)

//go:noescape
func complex64NEON(dst, x, y []complex64, op uint8)
func complex64SIMD(dst, x, y []complex64, op uint8) bool {
	complex64NEON(dst, x, y, op)
	return true
}
func extrema64SIMD(dst, x, y []float64, maximum bool) bool {
	extrema64NEON(dst, x, y, maximum)
	return true
}
func extrema32SIMD(dst, x, y []float32, maximum bool) bool {
	extrema32NEON(dst, x, y, maximum)
	return true
}
func exp64SIMD(dst, src []float64, c *[19]float64) bool { exp64NEON(dst, src, c); return true }
func complex128SIMD(dst, x, y []complex128, op uint8) bool {
	complex128NEON(dst, x, y, op)
	return true
}

//go:noescape
func partial64NEON(dst *[8]float64, src []float64)

//go:noescape
func partial32NEON(dst *[8]float32, src []float32)
func partial64SIMD(dst *[8]float64, src []float64) bool { partial64NEON(dst, src); return true }
func partial32SIMD(dst *[8]float32, src []float32) bool { partial32NEON(dst, src); return true }

//go:noescape
func widenNEON(dst, src unsafe.Pointer, n int)

//go:noescape
func narrowNEON(dst, src unsafe.Pointer, n int)
func widenSIMD(dst, src unsafe.Pointer, n int) bool  { widenNEON(dst, src, n); return true }
func narrowSIMD(dst, src unsafe.Pointer, n int) bool { narrowNEON(dst, src, n); return true }

// Advanced SIMD is part of the ARM64 baseline. No optional extension
// (including the complex-number or fused-multiply-add instruction) is used.
//
//go:noescape
func butterflyNEON(left, right, roots []complex128)

//go:noescape
func radix4NEON(x, w1, w2, w3 []complex128, sign float64)

//go:noescape
func radix3NEON(x, w1, w2 []complex128, sign float64)

//go:noescape
func radix5NEON(x, roots []complex128, sign float64)
func radix3SIMD(x, w1, w2 []complex128, sign float64) bool { radix3NEON(x, w1, w2, sign); return true }
func radix5SIMD(x, roots []complex128, sign float64) bool  { radix5NEON(x, roots, sign); return true }

//go:noescape
func binary64NEON(dst, x, y []float64, op uint8)

//go:noescape
func binary32NEON(dst, x, y []float32, op uint8)

//go:noescape
func broadcast64NEON(dst, x []float64, scalar float64, op uint8)

//go:noescape
func broadcast32NEON(dst, x []float32, scalar float32, op uint8)

func butterflySIMD(left, right, roots []complex128) bool {
	butterflyNEON(left, right, roots)
	return true
}
func radix4SIMD(x, w1, w2, w3 []complex128, sign float64) bool {
	radix4NEON(x, w1, w2, w3, sign)
	return true
}
func binary64SIMD(dst, x, y []float64, op uint8) bool { binary64NEON(dst, x, y, op); return true }
func binary32SIMD(dst, x, y []float32, op uint8) bool { binary32NEON(dst, x, y, op); return true }
func broadcast64SIMD(dst, x []float64, scalar float64, op uint8) bool {
	broadcast64NEON(dst, x, scalar, op)
	return true
}
func broadcast32SIMD(dst, x []float32, scalar float32, op uint8) bool {
	broadcast32NEON(dst, x, scalar, op)
	return true
}
