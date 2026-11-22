package numeric

import "unsafe"

// KernelAccess authorizes package-internal borrowing of ndarray storage.
// It is deliberately in an internal package, not part of the public API.
type KernelAccess struct{}

// Widen and Narrow convert native components without changing IEEE cast
// semantics. The complex variants use the same interleaved real/imaginary
// component representation as native Go complex values.
func Widen(dst []float64, src []float32) {
	if len(dst) != len(src) {
		panic("cast length mismatch")
	}
	n := len(src) &^ 3
	start := 0
	if n >= 32 && widenSIMD(unsafe.Pointer(unsafe.SliceData(dst)), unsafe.Pointer(unsafe.SliceData(src)), n) {
		start = n
	}
	for i := start; i < len(src); i++ {
		dst[i] = float64(src[i])
	}
}
func Narrow(dst []float32, src []float64) {
	if len(dst) != len(src) {
		panic("cast length mismatch")
	}
	n := len(src) &^ 3
	start := 0
	if n >= 32 && narrowSIMD(unsafe.Pointer(unsafe.SliceData(dst)), unsafe.Pointer(unsafe.SliceData(src)), n) {
		start = n
	}
	for i := start; i < len(src); i++ {
		dst[i] = float32(src[i])
	}
}
func WidenComplex(dst []complex128, src []complex64) {
	if len(dst) != len(src) {
		panic("cast length mismatch")
	}
	n := len(src) &^ 1
	start := 0
	if n >= 16 && widenSIMD(unsafe.Pointer(unsafe.SliceData(dst)), unsafe.Pointer(unsafe.SliceData(src)), 2*n) {
		start = n
	}
	for i := start; i < len(src); i++ {
		dst[i] = complex128(src[i])
	}
}
func NarrowComplex(dst []complex64, src []complex128) {
	if len(dst) != len(src) {
		panic("cast length mismatch")
	}
	n := len(src) &^ 1
	start := 0
	if n >= 16 && narrowSIMD(unsafe.Pointer(unsafe.SliceData(dst)), unsafe.Pointer(unsafe.SliceData(src)), 2*n) {
		start = n
	}
	for i := start; i < len(src); i++ {
		dst[i] = complex64(src[i])
	}
}
