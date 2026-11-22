package np

import (
	numkernel "github.com/SamuelSupe/numpygo/internal/numeric"
	"math"
)

// The fast loops are limited to packed or constant operands; arbitrary
// layouts and nonexact aliases keep the snapshot-based implementation.
func nativeExtraOp(name string) uint8 {
	switch name {
	case "positive":
		return 1
	case "conjugate":
		return 23
	case "negative":
		return 2
	case "absolute", "fabs":
		return 3
	case "square":
		return 4
	case "reciprocal":
		return 5
	case "sqrt":
		return 6
	case "exp":
		return 7
	case "log":
		return 8
	case "sin":
		return 9
	case "cos":
		return 10
	case "floor":
		return 11
	case "ceil":
		return 12
	case "trunc":
		return 13
	case "rint":
		return 14
	case "minimum":
		return 15
	case "maximum":
		return 16
	case "fmin":
		return 17
	case "fmax":
		return 18
	}
	return 0
}

func tryMoreNativeOut(u *UFunc, d DType, out, where *Array, args []*Array) bool {
	if d != Float32 && d != Float64 && d != Complex64 && d != Complex128 {
		return false
	}
	code := u.nativeExtra
	binary, arithmetic := nativeFloatOp(u)
	if d.IsComplex() {
		if !arithmetic && !(code == 1 || code == 2 || code == 4 || code == 5 || code == 23) {
			return false
		}
		if arithmetic {
			code = 19 + uint8(binary)
		}
	} else if code == 0 {
		return false
	}
	if !d.IsComplex() && code == 23 {
		code = 1
	}
	for _, a := range args {
		if !a.IsCContiguous() && !constantView(a) {
			return false
		}
	}
	mask, skip, ok := prepareNativeOut(d, out, where, args)
	if !ok || skip {
		return ok
	}
	extraNativeTo(d, out, args, mask, code)
	return true
}

func extraNativeTo(d DType, out *Array, args []*Array, mask []bool, code uint8) {
	switch d {
	case Float64:
		moreRealOut(out, args, out.data.real, mask, code)
	case Float32:
		moreRealOut(out, args, out.data.raw.([]float32), mask, code)
	case Complex64:
		moreComplexOut(out, args, out.data.raw.([]complex64), mask, code)
	case Complex128:
		moreComplexOut(out, args, out.data.cplx, mask, code)
	}
}

// Allocating entry points share exactly the same packed kernels as
// explicit out; the generic implementation retains nonpacked layouts.
func packedBinaryResult(a, b *Array, d DType, code uint8) (*Array, error) {
	if a.DType() != d || b.DType() != d || !nativeNumericStorage(a) || !nativeNumericStorage(b) {
		return nil, nil
	}
	shape, e := BroadcastShapes(a.shape, b.shape)
	if e != nil {
		return nil, e
	}
	x, e := a.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	y, e := b.BroadcastTo(shape...)
	if e != nil {
		return nil, e
	}
	if (!x.IsCContiguous() && !constantView(x)) || (!y.IsCContiguous() && !constantView(y)) {
		return nil, nil
	}
	out, e := Zeros(d, shape...)
	if e != nil || out.size == 0 {
		return out, e
	}
	extraNativeTo(d, out, []*Array{x, y}, nil, code)
	return out, nil
}

func packedOperand[T Number](a *Array) ([]T, int) {
	s := storageSlice(a.data).([]T)
	if a.IsCContiguous() {
		return s[a.offset : a.offset+a.size], 1
	}
	return s[a.offset : a.offset+1], 0
}

func moreRealOut[T float32 | float64](out *Array, args []*Array, storage []T, mask []bool, code uint8) {
	dst := storage[out.offset : out.offset+out.size]
	x, xs := packedOperand[T](args[0])
	var y []T
	var ys int
	if len(args) == 2 {
		y, ys = packedOperand[T](args[1])
	}
	if mask == nil && xs == 1 {
		if code == 1 {
			copy(dst, x)
			return
		}
		if code == 4 && nativeBinary(dst, x, x, 2) {
			return
		}
		if code == 7 {
			switch v := any(dst).(type) {
			case []float64:
				numkernel.Exp64(v, any(x).([]float64))
			case []float32:
				numkernel.Exp32(v, any(x).([]float32))
			}
			return
		}
		if (code == 15 || code == 16) && ys == 1 {
			switch v := any(dst).(type) {
			case []float64:
				if numkernel.Extrema64(v, any(x).([]float64), any(y).([]float64), code == 16) {
					return
				}
			case []float32:
				if numkernel.Extrema32(v, any(x).([]float32), any(y).([]float32), code == 16) {
					return
				}
			}
		}
	}
	// All operation decisions occur after the mask test. False positions,
	// including masked exceptional inputs, are never read or written.
	for i := range dst {
		if mask != nil && !mask[i] {
			continue
		}
		v := x[i*xs]
		switch code {
		case 1:
			dst[i] = v
		case 2:
			dst[i] = -v
		case 3:
			dst[i] = T(math.Abs(float64(v)))
		case 4:
			dst[i] = v * v
		case 5:
			dst[i] = T(1 / float64(v))
		case 6:
			dst[i] = T(math.Sqrt(float64(v)))
		case 7:
			dst[i] = T(math.Exp(float64(v)))
		case 8:
			dst[i] = T(math.Log(float64(v)))
		case 9:
			dst[i] = T(math.Sin(float64(v)))
		case 10:
			dst[i] = T(math.Cos(float64(v)))
		case 11:
			dst[i] = T(math.Floor(float64(v)))
		case 12:
			dst[i] = T(math.Ceil(float64(v)))
		case 13:
			dst[i] = T(math.Trunc(float64(v)))
		case 14:
			dst[i] = T(math.RoundToEven(float64(v)))
		default:
			w := y[i*ys]
			ignore := code >= 17
			maximum := code == 16 || code == 18
			switch {
			case v != v:
				if ignore && w == w {
					v = w
				}
			case w != w:
				if !ignore {
					v = w
				}
			case ignore && v == w, maximum && w < v, !maximum && v < w:
			default:
				v = w
			}
			dst[i] = v
		}
	}
}

func moreComplexOut[T complex64 | complex128](out *Array, args []*Array, storage []T, mask []bool, code uint8) {
	dst := storage[out.offset : out.offset+out.size]
	x, xs := packedOperand[T](args[0])
	var y []T
	var ys int
	if len(args) == 2 {
		y, ys = packedOperand[T](args[1])
	}
	if mask == nil && xs == 1 && code == 1 {
		copy(dst, x)
		return
	}
	if mask == nil && xs == 1 && (code == 4 || ys == 1 && code >= 19 && code <= 22) {
		operation := code - 19
		if code == 4 {
			y, operation = x, 2
		}
		switch v := any(dst).(type) {
		case []complex64:
			if numkernel.ComplexBinary64(v, any(x).([]complex64), any(y).([]complex64), operation) {
				return
			}
		case []complex128:
			if numkernel.ComplexBinary128(v, any(x).([]complex128), any(y).([]complex128), operation) {
				return
			}
		}
		// The no-SIMD fallback is also a packed typed loop; avoid
		// widening whole blocks or choosing the operation per element.
		switch operation {
		case 0:
			for i, v := range x {
				dst[i] = v + y[i]
			}
		case 1:
			for i, v := range x {
				dst[i] = v - y[i]
			}
		case 2:
			for i, v := range x {
				dst[i] = T(complex128(v) * complex128(y[i]))
			}
		case 3:
			for i, v := range x {
				dst[i] = T(complexDivide(complex128(v), complex128(y[i])))
			}
		}
		return
	}
	for i := range dst {
		if mask != nil && !mask[i] {
			continue
		}
		v := complex128(x[i*xs])
		switch code {
		case 1:
		case 23:
			v = complex(real(v), -imag(v))
		case 2:
			v = -v
		case 4:
			v *= v
		case 5:
			if v == 0 {
				v = complex(math.NaN(), math.NaN())
			} else {
				v = complexDivide(1, v)
			}
		case 19:
			v += complex128(y[i*ys])
		case 20:
			v -= complex128(y[i*ys])
		case 21:
			v *= complex128(y[i*ys])
		case 22:
			v = complexDivide(v, complex128(y[i*ys]))
		}
		dst[i] = T(v)
	}
}
