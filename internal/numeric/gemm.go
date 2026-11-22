package numeric

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/blas/cblas128"
	"gonum.org/v1/gonum/blas/cblas64"
)

type Floating interface {
	float32 | float64 | complex64 | complex128
}

func finite[T Floating](v []T) bool {
	for _, x := range v {
		if x-x != 0 {
			return false
		}
	}
	return true
}

// Gemm multiplies packed, row-major matrices into zero-initialized z.
// Some BLAS kernels skip zero multiplicands, losing 0*Inf and 0*NaN.
// Retain explicit IEEE arithmetic for matrices with nonfinite entries.
func Gemm[T Floating](m, k, n int, x, y, z []T) {
	if !finite(x) || !finite(y) {
		for i := 0; i < m; i++ {
			for h := 0; h < k; h++ {
				for j := 0; j < n; j++ {
					z[i*n+j] += x[i*k+h] * y[h*n+j]
				}
			}
		}
		return
	}
	switch a := any(x).(type) {
	case []float32:
		blas32.Gemm(blas.NoTrans, blas.NoTrans, 1, blas32.General{Rows: m, Cols: k, Stride: k, Data: a}, blas32.General{Rows: k, Cols: n, Stride: n, Data: any(y).([]float32)}, 0, blas32.General{Rows: m, Cols: n, Stride: n, Data: any(z).([]float32)})
	case []float64:
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: m, Cols: k, Stride: k, Data: a}, blas64.General{Rows: k, Cols: n, Stride: n, Data: any(y).([]float64)}, 0, blas64.General{Rows: m, Cols: n, Stride: n, Data: any(z).([]float64)})
	case []complex64:
		if m >= 32 && k >= 32 && n >= 32 {
			axpyGemm(m, k, n, a, any(y).([]complex64), any(z).([]complex64),
				func(v complex64, x, y []complex64) {
					cblas64.Axpy(v, cblas64.Vector{N: len(x), Inc: 1, Data: x}, cblas64.Vector{N: len(y), Inc: 1, Data: y})
				})
			return
		}
		cblas64.Gemm(blas.NoTrans, blas.NoTrans, 1, cblas64.General{Rows: m, Cols: k, Stride: k, Data: a}, cblas64.General{Rows: k, Cols: n, Stride: n, Data: any(y).([]complex64)}, 0, cblas64.General{Rows: m, Cols: n, Stride: n, Data: any(z).([]complex64)})
	case []complex128:
		if m >= 32 && k >= 32 && n >= 32 {
			axpyGemm(m, k, n, a, any(y).([]complex128), any(z).([]complex128),
				func(v complex128, x, y []complex128) {
					cblas128.Axpy(v, cblas128.Vector{N: len(x), Inc: 1, Data: x}, cblas128.Vector{N: len(y), Inc: 1, Data: y})
				})
			return
		}
		cblas128.Gemm(blas.NoTrans, blas.NoTrans, 1, cblas128.General{Rows: m, Cols: k, Stride: k, Data: a}, cblas128.General{Rows: k, Cols: n, Stride: n, Data: any(y).([]complex128)}, 0, cblas128.General{Rows: m, Cols: n, Stride: n, Data: any(z).([]complex128)})
	}
}

// Reuse Gonum's contiguous vector kernel rather than its scalar complex
// GEMM loop. The reduction order is unchanged and no packing is needed.
// Block the reduction dimension to keep the RHS panel cache-resident.
func axpyGemm[T complex64 | complex128](m, k, n int, a, b, z []T, axpy func(T, []T, []T)) {
	const block = 64
	for h0 := 0; h0 < k; h0 += block {
		end := min(h0+block, k)
		for i := 0; i < m; i++ {
			row := z[i*n : (i+1)*n]
			for h := h0; h < end; h++ {
				axpy(a[i*k+h], b[h*n:(h+1)*n], row)
			}
		}
	}
}
