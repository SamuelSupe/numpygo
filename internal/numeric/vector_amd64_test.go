//go:build amd64 && !purego

package numeric

import (
	"fmt"
	"math"
	"testing"
)

func TestSIMDDispatch(t *testing.T) {
	hardware := avxAvailable
	t.Cleanup(func() { avxAvailable = hardware })
	for _, avx := range []bool{false, true} {
		if avx && !hardware {
			continue
		}
		t.Run(fmt.Sprintf("avx=%v", avx), func(t *testing.T) {
			avxAvailable = avx
			a, b := make([]float64, 69), make([]float64, 69)
			aa, bb := make([]float32, 69), make([]float32, 69)
			for i := range a {
				a[i], b[i] = float64(i-17)/7, float64(i-31)/11
				aa[i], bb[i] = float32(a[i]), float32(b[i])
			}
			for op := uint8(0); op < 4; op++ {
				out, out32 := make([]float64, 69), make([]float32, 69)
				if !Binary64(out, a, b, op) || !Binary32(out32, aa, bb, op) {
					t.Fatal("expected amd64 vector path")
				}
				for i := range a {
					var want float64
					var want32 float32
					switch op {
					case 0:
						want = a[i] + b[i]
						want32 = aa[i] + bb[i]
					case 1:
						want = a[i] - b[i]
						want32 = aa[i] - bb[i]
					case 2:
						want = a[i] * b[i]
						want32 = aa[i] * bb[i]
					case 3:
						want = a[i] / b[i]
						want32 = aa[i] / bb[i]
					}
					if math.Float64bits(out[i]) != math.Float64bits(want) || math.Float32bits(out32[i]) != math.Float32bits(want32) {
						t.Fatalf("op %d index %d: %v %v; want %v %v", op, i, out[i], out32[i], want, want32)
					}
				}
			}
			l, r, w := make([]complex128, 36), make([]complex128, 36), make([]complex128, 36)
			for i := range l {
				l[i], r[i], w[i] = 2+3i, complex(float64(i), -2), 3+4i
			}
			Butterfly(l, r, w)
			for i := range l {
				p := complex(float64(i), -2) * (3 + 4i)
				if l[i] != 2+3i+p || r[i] != 2+3i-p {
					t.Fatalf("butterfly %d: %v %v", i, l[i], r[i])
				}
			}
		})
	}
}

func BenchmarkSIMDVector(b *testing.B) {
	if !avxAvailable {
		b.Skip("AVX unavailable")
	}
	for _, n := range []int{128, 512, 8192} {
		l, r, w := make([]complex128, n), make([]complex128, n), make([]complex128, n)
		for i := range w {
			w[i] = .5 - .5i
		}
		for _, mode := range []string{"scalar", "avx"} {
			b.Run(fmt.Sprintf("butterfly_%d_%s", n, mode), func(b *testing.B) {
				for b.Loop() {
					if mode == "avx" {
						butterflyAVX(l, r, w)
					} else {
						for j, z := range w {
							u, v := l[j], r[j]*z
							l[j], r[j] = u+v, u-v
						}
					}
				}
			})
		}
	}
	const n = 128 * 1024
	x, y, z := make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range x {
		x[i], y[i] = float64(i)/7, 3.
	}
	for _, mode := range []string{"scalar", "sse2", "avx"} {
		b.Run("divide64_"+mode, func(b *testing.B) {
			for b.Loop() {
				switch mode {
				case "avx":
					binary64AVX(z, x, y, 3)
				case "sse2":
					binary64SSE2(z, x, y, 3)
				default:
					for i := range z {
						z[i] = x[i] / y[i]
					}
				}
			}
		})
	}
}
