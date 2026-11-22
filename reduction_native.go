package np

import (
	numkernel "github.com/SamuelSupe/numpygo/internal/numeric"
	"math"
)

// A flattened axis group can be traversed by one fixed step when its
// strides are an affine progression in the selected reduction order.
func affineStep(shape, strides []int) (int, bool) {
	step, width := 0, 1
	found := false
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] <= 1 {
			continue
		}
		if !found {
			step = strides[i]
			found = true
		} else if strides[i] != step*width {
			return 0, false
		}
		width *= shape[i]
	}
	return step, true
}

// Native real reductions retain the existing pairwise/sequential summation
// decision, precision, and traversal. Only the element-access mechanism
// changes; mixed dtypes, buffers and nonlinear layouts keep the generic path.
func nativeRealReduction(a, out *Array, r reductionLayout, mean bool) bool {
	d := a.DType()
	if d != out.DType() || (d != Float32 && d != Float64) || !nativeNumericStorage(a) {
		return false
	}
	step, ok := affineStep(r.reduceShape, r.reduceStrides)
	if !ok {
		return false
	}
	outStep, affineOut := affineStep(r.shape, r.outputStrides)
	if d == Float64 {
		reduceReal(a.data.real, out.data.real, a, r, step, outStep, affineOut, mean)
	} else {
		reduceReal(a.data.raw.([]float32), out.data.raw.([]float32), a, r, step, outStep, affineOut, mean)
	}
	return true
}
func reduceReal[T float32 | float64](src, dst []T, a *Array, r reductionLayout, step, outStep int, affineOut, mean bool) {
	if !r.pairwise && affineOut && len(dst) > 1 {
		// Slow-axis NumPy reductions accumulate sequentially for each
		// lane. Interleave those lanes to read spatially adjacent values
		// rather than repeatedly sweeping individual columns.
		for j := 0; j < r.count; j++ {
			base := a.offset + j*step
			if outStep == 1 && nativeBinary(dst, dst, src[base:base+len(dst)], 0) {
				continue
			}
			for i := range dst {
				dst[i] += src[base+i*outStep]
			}
		}
	} else {
		for i := range dst {
			base := r.base(a, i)
			var total T
			if r.pairwise {
				total += pairwiseNative(src, base, step, r.count)
			} else {
				for j := 0; j < r.count; j++ {
					total += src[base+j*step]
				}
			}
			dst[i] = total
		}
	}
	if mean {
		for i, x := range dst {
			dst[i] = T(float64(x) / float64(r.count))
		}
	}
}

// Match pairwiseSum's eight lanes and 128-element recursion boundary,
// but address native storage directly instead of invoking a callback
// for each sample. This order is a NumPy floating-point contract.
func pairwiseNative[T float32 | float64](src []T, base, step, n int) T {
	if step == 1 && n >= 32 {
		switch v := any(src[base : base+n]).(type) {
		case []float64:
			return T(numkernel.Pairwise64(v))
		case []float32:
			return T(numkernel.Pairwise32(v))
		}
	}
	if n < 8 {
		var total T
		for i := 0; i < n; i++ {
			total += src[base+i*step]
		}
		return total
	}
	if n > 128 {
		middle := n / 2
		middle -= middle % 8
		return pairwiseNative(src, base, step, middle) + pairwiseNative(src, base+middle*step, step, n-middle)
	}
	var partial [8]T
	for i := range partial {
		partial[i] = src[base+i*step]
	}
	i := 8
	for ; i+7 < n; i += 8 {
		pos := base + i*step
		for j := range partial {
			partial[j] += src[pos+j*step]
		}
	}
	total := ((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]))
	for ; i < n; i++ {
		total += src[base+i*step]
	}
	return total
}

func nativeBinary[T float32 | float64](dst, x, y []T, op uint8) bool {
	switch v := any(dst).(type) {
	case []float64:
		return numkernel.Binary64(v, any(x).([]float64), any(y).([]float64), op)
	case []float32:
		return numkernel.Binary32(v, any(x).([]float32), any(y).([]float32), op)
	}
	return false
}

func nativeRealVariance(a, out *Array, r reductionLayout, ddof float64, sqrt bool) bool {
	d := a.DType()
	if d != out.DType() || (d != Float32 && d != Float64) || !nativeNumericStorage(a) {
		return false
	}
	step, ok := affineStep(r.reduceShape, r.reduceStrides)
	if !ok {
		return false
	}
	outStep, affineOut := affineStep(r.shape, r.outputStrides)
	if d == Float64 {
		varianceReal(a.data.real, out.data.real, a, r, step, outStep, affineOut, ddof, sqrt)
	} else {
		varianceReal(a.data.raw.([]float32), out.data.raw.([]float32), a, r, step, outStep, affineOut, ddof, sqrt)
	}
	return true
}

func varianceReal[T float32 | float64](src, dst []T, a *Array, r reductionLayout, step, outStep int, affineOut bool, ddof float64, sqrt bool) {
	reduceReal(src, dst, a, r, step, outStep, affineOut, true)
	den := math.Max(float64(r.count)-ddof, 0)
	if !r.pairwise && affineOut && outStep == 1 && len(dst) >= 32 {
		// Sequential accumulation along the slow axis, with independent
		// output columns in vector lanes.
		sums, work := make([]T, len(dst)), make([]T, len(dst))
		for j := 0; j < r.count; j++ {
			base := a.offset + j*step
			if nativeBinary(work, src[base:base+len(dst)], dst, 1) &&
				nativeBinary(work, work, work, 2) && nativeBinary(sums, sums, work, 0) {
				continue
			}
			for i, mu := range dst {
				diff := src[base+i] - mu
				sums[i] += T(diff * diff)
			}
		}
		copy(dst, sums)
	} else {
		var scratch [128]T
		for i, mu := range dst {
			base := r.base(a, i)
			var total T
			if r.pairwise {
				total += sumSquaresNative(src, scratch[:], base, step, r.count, mu)
			} else {
				for j := 0; j < r.count; j++ {
					diff := src[base+j*step] - mu
					total += T(diff * diff)
				}
			}
			dst[i] = total
		}
	}
	for i, v := range dst {
		v = T(float64(v) / den)
		if sqrt {
			v = T(math.Sqrt(float64(v)))
		}
		dst[i] = v
	}
}

func sumSquaresNative[T float32 | float64](src, scratch []T, base, step, n int, mu T) T {
	if n > 128 {
		m := (n / 2) &^ 7
		left := sumSquaresNative(src, scratch, base, step, m, mu)
		return left + sumSquaresNative(src, scratch, base+m*step, step, n-m, mu)
	}
	work := scratch[:n]
	fast := false
	if step == 1 && n >= 32 {
		switch w := any(work).(type) {
		case []float64:
			fast = numkernel.Lane64(w, any(src[base:base+n]).([]float64), []float64{float64(mu)}, 1, 0, 1)
		case []float32:
			fast = numkernel.Lane32(w, any(src[base:base+n]).([]float32), []float32{float32(mu)}, 1, 0, 1)
		}
	}
	if !fast || !nativeBinary(work, work, work, 2) {
		for j := range work {
			diff := src[base+j*step] - mu
			work[j] = diff * diff
		}
	}
	return pairwiseNative(work, 0, 1, n)
}
