//go:build (!amd64 && !arm64) || purego

package numeric

import "unsafe"

func advancedVectorsAvailable() bool { return false }

func extrema64SIMD(dst, x, y []float64, maximum bool) bool { return false }
func extrema32SIMD(dst, x, y []float32, maximum bool) bool { return false }
func exp64SIMD(dst, src []float64, c *[19]float64) bool    { return false }
func exp32SIMD(dst, src []float32, c *[13]float32) bool    { return false }
func complex128SIMD(dst, x, y []complex128, op uint8) bool { return false }
func complex64SIMD(dst, x, y []complex64, op uint8) bool   { return false }

func partial64SIMD(dst *[8]float64, src []float64) bool { return false }
func partial32SIMD(dst *[8]float32, src []float32) bool { return false }

func widenSIMD(dst, src unsafe.Pointer, n int) bool  { return false }
func narrowSIMD(dst, src unsafe.Pointer, n int) bool { return false }

func butterflySIMD(left, right, roots []complex128) bool              { return false }
func radix4SIMD(x, w1, w2, w3 []complex128, sign float64) bool        { return false }
func radix3SIMD(x, w1, w2 []complex128, sign float64) bool            { return false }
func radix5SIMD(x, roots []complex128, sign float64) bool             { return false }
func binary64SIMD(dst, x, y []float64, op uint8) bool                 { return false }
func binary32SIMD(dst, x, y []float32, op uint8) bool                 { return false }
func broadcast64SIMD(dst, x []float64, scalar float64, op uint8) bool { return false }
func broadcast32SIMD(dst, x []float32, scalar float32, op uint8) bool { return false }
