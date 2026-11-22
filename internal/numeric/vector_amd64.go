//go:build amd64 && !purego

package numeric

import "unsafe"

//go:noescape
func widenAVX(dst, src unsafe.Pointer, n int)

//go:noescape
func narrowAVX(dst, src unsafe.Pointer, n int)

func widenSIMD(dst, src unsafe.Pointer, n int) bool {
	if !avxAvailable {
		return false
	}
	widenAVX(dst, src, n)
	return true
}
func narrowSIMD(dst, src unsafe.Pointer, n int) bool {
	if !avxAvailable {
		return false
	}
	narrowAVX(dst, src, n)
	return true
}

// AVX requires both CPU support and enabled XMM/YMM state in the OS.
//
//go:noescape
func hasAVX() bool

//go:noescape
func butterflyAVX(left, right, roots []complex128)

//go:noescape
func radix4AVX(x, w1, w2, w3 []complex128, sign float64)

//go:noescape
func radix3AVX(x, w1, w2 []complex128, sign float64)

//go:noescape
func radix5AVX(x, roots []complex128, sign float64)
func radix3SIMD(x, w1, w2 []complex128, sign float64) bool {
	if !avxAvailable {
		return false
	}
	radix3AVX(x, w1, w2, sign)
	return true
}
func radix5SIMD(x, roots []complex128, sign float64) bool {
	if !avxAvailable {
		return false
	}
	radix5AVX(x, roots, sign)
	return true
}

//go:noescape
func binary64SSE2(dst, x, y []float64, op uint8)

//go:noescape
func binary32SSE2(dst, x, y []float32, op uint8)

//go:noescape
func binary64AVX(dst, x, y []float64, op uint8)

//go:noescape
func binary32AVX(dst, x, y []float32, op uint8)

//go:noescape
func broadcast64AVX(dst, x []float64, scalar float64, op uint8)

//go:noescape
func broadcast32AVX(dst, x []float32, scalar float32, op uint8)

var avxAvailable = hasAVX()

func advancedVectorsAvailable() bool { return avxAvailable }

//go:noescape
func extrema64AVX(dst, x, y []float64, maximum bool)

//go:noescape
func extrema32AVX(dst, x, y []float32, maximum bool)

//go:noescape
func exp64AVX(dst, src []float64, constants *[19]float64)

//go:noescape
func exp32AVX(dst, src []float32, constants *[13]float32)
func exp32SIMD(dst, src []float32, c *[13]float32) bool {
	if !avxAvailable {
		return false
	}
	exp32AVX(dst, src, c)
	return true
}

//go:noescape
func complex128AVX(dst, x, y []complex128, op uint8)

//go:noescape
func complex64AVX(dst, x, y []complex64, op uint8)

func complex64SIMD(dst, x, y []complex64, op uint8) bool {
	if !avxAvailable {
		return false
	}
	complex64AVX(dst, x, y, op)
	return true
}

func extrema64SIMD(dst, x, y []float64, maximum bool) bool {
	if !avxAvailable {
		return false
	}
	extrema64AVX(dst, x, y, maximum)
	return true
}
func extrema32SIMD(dst, x, y []float32, maximum bool) bool {
	if !avxAvailable {
		return false
	}
	extrema32AVX(dst, x, y, maximum)
	return true
}
func exp64SIMD(dst, src []float64, c *[19]float64) bool {
	if !avxAvailable {
		return false
	}
	exp64AVX(dst, src, c)
	return true
}
func complex128SIMD(dst, x, y []complex128, op uint8) bool {
	if !avxAvailable {
		return false
	}
	complex128AVX(dst, x, y, op)
	return true
}

//go:noescape
func partial64AVX(dst *[8]float64, src []float64)

//go:noescape
func partial32AVX(dst *[8]float32, src []float32)
func partial64SIMD(dst *[8]float64, src []float64) bool {
	if !avxAvailable {
		return false
	}
	partial64AVX(dst, src)
	return true
}
func partial32SIMD(dst *[8]float32, src []float32) bool {
	if !avxAvailable {
		return false
	}
	partial32AVX(dst, src)
	return true
}

func butterflySIMD(left, right, roots []complex128) bool {
	if !avxAvailable {
		return false
	}
	butterflyAVX(left, right, roots)
	return true
}
func radix4SIMD(x, w1, w2, w3 []complex128, sign float64) bool {
	if !avxAvailable {
		return false
	}
	radix4AVX(x, w1, w2, w3, sign)
	return true
}
func binary64SIMD(dst, x, y []float64, op uint8) bool {
	if avxAvailable {
		binary64AVX(dst, x, y, op)
	} else {
		binary64SSE2(dst, x, y, op)
	}
	return true
}
func binary32SIMD(dst, x, y []float32, op uint8) bool {
	if avxAvailable {
		binary32AVX(dst, x, y, op)
	} else {
		binary32SSE2(dst, x, y, op)
	}
	return true
}
func broadcast64SIMD(dst, x []float64, scalar float64, op uint8) bool {
	if !avxAvailable {
		return false
	}
	broadcast64AVX(dst, x, scalar, op)
	return true
}
func broadcast32SIMD(dst, x []float32, scalar float32, op uint8) bool {
	if !avxAvailable {
		return false
	}
	broadcast32AVX(dst, x, scalar, op)
	return true
}
