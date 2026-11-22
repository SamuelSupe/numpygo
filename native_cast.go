package np

import numkernel "github.com/SamuelSupe/numpygo/internal/numeric"

// NativeSliceForKernel is restricted to library-internal kernels by the
// access token. A read borrow must not be written or returned as an output;
// a write borrow additionally requires a writable native C-order array.
// Foreign buffers and component views never enter this path.
func NativeSliceForKernel[T Number](a *Array, _ numkernel.KernelAccess, write bool) ([]T, bool) {
	var sample T
	if a == nil || a.DType() != dtypeOf(sample) || !nativeNumericStorage(a) || !a.IsCContiguous() || write && !a.Writable() {
		return nil, false
	}
	v, ok := storageSlice(a.data).([]T)
	if !ok {
		return nil, false
	}
	if a.size == 0 {
		return v[:0], true
	}
	return v[a.offset : a.offset+a.size], true
}

// SIMD precision conversion can operate on an entire C-order storage
// span or on individual contiguous rows, including reversed row order.
func convertNativeLanes[S, D any](out []D, src []S, a *Array, cast func([]D, []S)) bool {
	if a.size == 0 {
		return true
	}
	if a.IsCContiguous() {
		cast(out, src[a.offset:a.offset+a.size])
		return true
	}
	if a.NDim() == 2 && a.strides[1] == 1 {
		m, n := a.shape[0], a.shape[1]
		for i := 0; i < m; i++ {
			off := a.offset + i*a.strides[0]
			cast(out[i*n:(i+1)*n], src[off:off+n])
		}
		return true
	}
	return false
}

// convertNative traverses native storage in logical C order, with no
// per-element dtype dispatch. The destination is independently allocated.
func convertNative[S, D any](out []D, src []S, a *Array, cast func(S) D) {
	if len(out) == 0 {
		return
	}
	if a.IsCContiguous() {
		for i, v := range src[a.offset : a.offset+a.size] {
			out[i] = cast(v)
		}
		return
	}
	if a.NDim() == 2 {
		m, n := a.shape[0], a.shape[1]
		// Keep transposed source tiles cache-resident, whilst writing
		// the independent result in its promised C order.
		if a.strides[0] == 1 || a.strides[0] == -1 {
			const tile = 32
			for r := 0; r < m; r += tile {
				for c := 0; c < n; c += tile {
					for i := r; i < min(r+tile, m); i++ {
						off := a.offset + i*a.strides[0] + c*a.strides[1]
						row := out[i*n+c : i*n+min(c+tile, n)]
						for j := range row {
							row[j] = cast(src[off+j*a.strides[1]])
						}
					}
				}
			}
			return
		}
		for i := 0; i < m; i++ {
			row := out[i*n : (i+1)*n]
			off := a.offset + i*a.strides[0]
			for j := range row {
				row[j] = cast(src[off+j*a.strides[1]])
			}
		}
		return
	}
	a.walk(func(i, off int) { out[i] = cast(src[off]) })
}

// These four native floating dtypes share Go's numerical conversion
// semantics, including signed zero and single-precision component rounding.
// Extended, integer, buffer and special dtypes keep their existing casts.
func nativeFloatingCast(dst, a *Array) bool {
	if a.data.part != nil || a.data.buffer != nil {
		return false
	}
	switch a.DType() {
	case Float32:
		x := a.data.raw.([]float32)
		switch dst.DType() {
		case Float64:
			if !convertNativeLanes(dst.data.real, x, a, numkernel.Widen) {
				convertNative(dst.data.real, x, a, func(v float32) float64 { return float64(v) })
			}
		case Complex64:
			convertNative(dst.data.raw.([]complex64), x, a, func(v float32) complex64 { return complex(v, 0) })
		case Complex128:
			convertNative(dst.data.cplx, x, a, func(v float32) complex128 { return complex(float64(v), 0) })
		default:
			return false
		}
	case Float64:
		x := a.data.real
		switch dst.DType() {
		case Float32:
			out := dst.data.raw.([]float32)
			if !convertNativeLanes(out, x, a, numkernel.Narrow) {
				convertNative(out, x, a, func(v float64) float32 { return float32(v) })
			}
		case Complex64:
			convertNative(dst.data.raw.([]complex64), x, a, func(v float64) complex64 { return complex(float32(v), 0) })
		case Complex128:
			convertNative(dst.data.cplx, x, a, func(v float64) complex128 { return complex(v, 0) })
		default:
			return false
		}
	case Complex64:
		x := a.data.raw.([]complex64)
		switch dst.DType() {
		case Float32:
			convertNative(dst.data.raw.([]float32), x, a, func(v complex64) float32 { return real(v) })
		case Float64:
			convertNative(dst.data.real, x, a, func(v complex64) float64 { return float64(real(v)) })
		case Complex128:
			if !convertNativeLanes(dst.data.cplx, x, a, numkernel.WidenComplex) {
				convertNative(dst.data.cplx, x, a, func(v complex64) complex128 { return complex128(v) })
			}
		default:
			return false
		}
	case Complex128:
		x := a.data.cplx
		switch dst.DType() {
		case Float32:
			convertNative(dst.data.raw.([]float32), x, a, func(v complex128) float32 { return float32(real(v)) })
		case Float64:
			convertNative(dst.data.real, x, a, func(v complex128) float64 { return real(v) })
		case Complex64:
			out := dst.data.raw.([]complex64)
			if !convertNativeLanes(out, x, a, numkernel.NarrowComplex) {
				convertNative(out, x, a, func(v complex128) complex64 { return complex64(v) })
			}
		default:
			return false
		}
	default:
		return false
	}
	return true
}
