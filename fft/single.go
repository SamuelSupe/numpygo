package fft

import (
	"math"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/numeric"

	"gonum.org/v1/gonum/dsp/fourier"
)

// Single-precision arrays are read and written in their native storage.
// A binary64 work lane avoids compounding float32 rounding at each FFT
// stage, particularly at small cancellation frequencies. No full-array
// binary64 cast or output intermediate is required.
func transformSingle(a *np.Array, kind transformKind, n int, shape []int, inLen, outLen, inner, outer int, dtype np.DType, scale float64, direct *np.Array) (*np.Array, error) {
	if a.DType() != np.Float32 && a.DType() != np.Complex64 ||
		kind == inverseReal && dtype != np.Float32 || kind != inverseReal && dtype != np.Complex64 {
		return nil, nil
	}
	var reals []float32
	var complexIn []complex64
	var err error
	if a.DType() == np.Float32 {
		reals, err = singleInput[float32](a)
		for _, v := range reals {
			if !(math.Abs(float64(v)) <= math.MaxFloat32) {
				return nil, nil
			}
		}
	} else {
		complexIn, err = singleInput[complex64](a)
		for _, v := range complexIn {
			if !(math.Abs(float64(real(v))) <= math.MaxFloat32 && math.Abs(float64(imag(v))) <= math.MaxFloat32) {
				return nil, nil
			}
		}
	}
	if err != nil {
		return nil, err
	}
	if direct == nil {
		direct, err = np.Zeros(dtype, shape...)
		if err != nil {
			return nil, err
		}
	}
	// A two-point, unscaled FFT contains exactly one addition or
	// subtraction per component. Native float32 gives the same rounded
	// result as the wide kernel, including when it overflows on output.
	// Longer transforms and nonunit normalization need the wide lane.
	if n == 2 && inLen == 2 && inner == 1 && scale == 1 {
		if kind == inverseReal {
			dst, _ := np.NativeSliceForKernel[float32](direct, numeric.KernelAccess{}, true)
			for i := 0; i < len(dst); i += 2 {
				var x, y float32
				if reals != nil {
					x, y = reals[i], reals[i+1]
				} else {
					x, y = real(complexIn[i]), real(complexIn[i+1])
				}
				dst[i], dst[i+1] = x+y, x-y
			}
		} else {
			dst, _ := np.NativeSliceForKernel[complex64](direct, numeric.KernelAccess{}, true)
			if reals != nil {
				for i := 0; i < len(dst); i += 2 {
					x, y := reals[i], reals[i+1]
					dst[i], dst[i+1] = complex(x+y, 0), complex(x-y, 0)
				}
			} else {
				for i := 0; i < len(dst); i += 2 {
					x, y := complexIn[i], complexIn[i+1]
					dst[i], dst[i+1] = x+y, x-y
				}
			}
		}
		return direct, nil
	}
	pow2 := n >= 64 && n&(n-1) == 0 && (inner*outer > 1 || n >= 4096)
	if kind == forwardReal && pow2 {
		dst, _ := np.NativeSliceForKernel[complex64](direct, numeric.KernelAccess{}, true)
		pow2RealForwardTo(reals, dst, n, inLen, outLen, inner, outer, scale)
		return direct, nil
	}
	if kind == inverseReal && pow2 && complexIn != nil {
		dst, _ := np.NativeSliceForKernel[float32](direct, numeric.KernelAccess{}, true)
		pow2RealInverseTo(complexIn, dst, n, inLen, inner, outer, scale)
		return direct, nil
	}
	if (kind == forwardComplex || kind == inverseComplex) && pow2 {
		dst, _ := np.NativeSliceForKernel[complex64](direct, numeric.KernelAccess{}, true)
		p := newPow2Plan(n, kind == inverseComplex, nil)
		used := min(inLen, n)
		for batch := 0; batch < outer; batch++ {
			for lane := 0; lane < inner; lane++ {
				if used < n {
					clear(p.work)
				}
				base := batch*inLen*inner + lane
				if reals != nil {
					for j := 0; j < used; j++ {
						p.work[p.rev[j]] = complex(float64(reals[base+j*inner]), 0)
					}
				} else {
					for j := 0; j < used; j++ {
						p.work[p.rev[j]] = complex128(complexIn[base+j*inner])
					}
				}
				p.apply()
				base = batch*n*inner + lane
				writeSingleComplex(dst, p.work, base, inner, scale)
			}
		}
		return direct, nil
	}
	// A reusable lane also covers arbitrary lengths, including prime
	// transforms for which Bluestein avoids FFTPACK's quadratic radix.
	var chirp *chirpFFT
	if largeFFTFactor(n) {
		chirp = newChirpFFT(n, kind == inverseReal || kind == inverseComplex, outer*inner)
	}
	if chirp == nil && (kind == forwardReal || kind == inverseReal) {
		plan := fourier.NewFFT(n)
		line, spectrum := make([]float64, n), make([]complex128, n/2+1)
		if kind == forwardReal {
			dst, _ := np.NativeSliceForKernel[complex64](direct, numeric.KernelAccess{}, true)
			for b := 0; b < outer; b++ {
				for k := 0; k < inner; k++ {
					used := min(inLen, n)
					base := b*inLen*inner + k
					if inner == 1 {
						numeric.Widen(line[:used], reals[base:base+used])
					} else {
						for j := 0; j < used; j++ {
							line[j] = float64(reals[base+j*inner])
						}
					}
					clear(line[used:])
					plan.Coefficients(spectrum, line)
					writeSingleComplex(dst, spectrum, b*outLen*inner+k, inner, scale)
				}
			}
		} else {
			dst, _ := np.NativeSliceForKernel[float32](direct, numeric.KernelAccess{}, true)
			for b := 0; b < outer; b++ {
				for k := 0; k < inner; k++ {
					gatherSingle(spectrum, reals, complexIn, inLen, inner, b, k)
					plan.Sequence(line, spectrum)
					base := b*n*inner + k
					for j, v := range line {
						dst[base+j*inner] = float32(v * scale)
					}
				}
			}
		}
		return direct, nil
	}
	var plan *fourier.CmplxFFT
	if chirp == nil {
		plan = fourier.NewCmplxFFT(n)
	}
	line := make([]complex128, n)
	var dst []complex64
	var realDst []float32
	if kind == inverseReal {
		realDst, _ = np.NativeSliceForKernel[float32](direct, numeric.KernelAccess{}, true)
	} else {
		dst, _ = np.NativeSliceForKernel[complex64](direct, numeric.KernelAccess{}, true)
	}
	for b := 0; b < outer; b++ {
		for k := 0; k < inner; k++ {
			if kind == inverseReal {
				gatherSingle(line[:n/2+1], reals, complexIn, inLen, inner, b, k)
				line[0] = complex(real(line[0]), 0)
				for j := 1; j <= n/2; j++ {
					if j*2 == n {
						line[j] = complex(real(line[j]), 0)
					} else {
						line[n-j] = complex(real(line[j]), -imag(line[j]))
					}
				}
			} else {
				gatherSingle(line, reals, complexIn, inLen, inner, b, k)
			}
			if chirp != nil {
				chirp.apply(line)
			} else if kind == inverseComplex {
				plan.Sequence(line, line)
			} else {
				plan.Coefficients(line, line)
			}
			base := b*outLen*inner + k
			if kind == inverseReal {
				for j, z := range line {
					realDst[base+j*inner] = float32(real(z) * scale)
				}
			} else {
				if kind == forwardReal {
					line[0] = complex(real(line[0]), 0)
					if n%2 == 0 {
						line[n/2] = complex(real(line[n/2]), 0)
					}
				}
				writeSingleComplex(dst, line[:outLen], base, inner, scale)
			}
		}
	}
	return direct, nil
}

func singleInput[T float32 | complex64](a *np.Array) ([]T, error) {
	if v, ok := np.NativeSliceForKernel[T](a, numeric.KernelAccess{}, false); ok {
		return v, nil
	}
	return np.Values[T](a)
}

func gatherSingle(dst []complex128, reals []float32, input []complex64, length, inner, batch, lane int) {
	used := min(length, len(dst))
	base := batch*length*inner + lane
	if reals != nil {
		for j := 0; j < used; j++ {
			dst[j] = complex(float64(reals[base+j*inner]), 0)
		}
	} else if inner == 1 {
		numeric.WidenComplex(dst[:used], input[base:base+used])
	} else {
		for j := 0; j < used; j++ {
			dst[j] = complex128(input[base+j*inner])
		}
	}
	clear(dst[used:])
}

func writeSingleComplex(dst []complex64, src []complex128, base, inner int, scale float64) {
	if inner == 1 && scale == 1 {
		numeric.NarrowComplex(dst[base:base+len(src)], src)
		return
	}
	for j, z := range src {
		dst[base+j*inner] = complex64(scaled(z, scale))
	}
}
