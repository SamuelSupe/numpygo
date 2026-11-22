package np_test

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/records"
)

var benchmarkResult *np.Array
var benchmarkUnique np.UniqueResult

func BenchmarkTensorContraction(b *testing.B) {
	a, e := np.Full(np.Float64, 1.25, 8, 16, 16)
	if e != nil {
		b.Fatal(e)
	}
	x, e := np.Full(np.Float64, 2.5, 16, 16, 32)
	if e != nil {
		b.Fatal(e)
	}
	p, e := np.EinsumPath("ijk,jkl->il", np.EinsumPathOptions{}, a, x)
	if e != nil {
		b.Fatal(e)
	}
	b.Run("tensordot", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkResult, e = np.Tensordot(a, x, []int{1, 2}, []int{0, 1})
			if e != nil {
				b.Fatal(e)
			}
		}
	})
	b.Run("einsum_path", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkResult, e = np.EinsumWithPath("ijk,jkl->il", p, a, x)
			if e != nil {
				b.Fatal(e)
			}
		}
	})
}

// BenchmarkAuditKernels measures representative workloads for the recently
// added statistical, set, sampling and structured-array APIs. Inputs and
// output sizes are identical across optimization runs.
func BenchmarkAuditKernels(b *testing.B) {
	must := func(a *np.Array, e error) *np.Array {
		if e != nil {
			b.Fatal(e)
		}
		return a
	}
	n := 1 << 16
	ints := make([]int64, n)
	for _, mod := range []int{1024, n} {
		for i := range ints {
			ints[i] = int64((i*40503+19)%mod - mod/2)
		}
		a := must(np.FromSlice(ints, n))
		name := "unique_distinct"
		if mod < n {
			name = "unique_repeated"
		}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				var e error
				benchmarkUnique, e = np.Unique(a)
				if e != nil {
					b.Fatal(e)
				}
			}
		})
	}
	hay := make([]int64, 1<<20)
	for i := range hay {
		hay[i] = int64(i * 2)
	}
	query := make([]int64, 256)
	for i := range query {
		query[i] = int64((i * 7919) % (len(hay) * 2))
	}
	a, q := must(np.WrapSlice(hay, len(hay))), must(np.WrapSlice(query, len(query)))
	b.Run("searchsorted_256_in_1m", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkResult = must(np.SearchSorted(a, q, false))
		}
	})
	x := make([]float64, 1<<17)
	for i := range x {
		x[i] = float64(i%1021) * .125
	}
	g := must(np.WrapFloat64(x, len(x)))
	b.Run("gradient_131k", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkResult = must(np.Gradient(g, 0, .125, 2))
		}
	})
	start := must(np.WrapFloat64([]float64{1, 2, 3, 4, -1, -2, -3, -4}, 8))
	stop := must(np.WrapFloat64([]float64{100, 256, 300, 400, -10, -20, -30, -40}, 8))
	b.Run("geomspace_8x128", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkResult = must(np.GeomspaceArray(start, stop, np.SpaceOptions{Num: 128, Axis: 1}))
		}
	})
	data := make([]float64, 64*512)
	for i := range data {
		data[i] = float64((i*7919)%2047) / 32
	}
	cov := must(np.WrapFloat64(data, 64, 512))
	b.Run("cov_64x512", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkResult = must(np.CovWithOptions(cov, nil, np.CovOptions{}))
		}
	})
	s, e := records.NewSchema([]records.FieldSpec{{Name: "x", Type: records.Numeric(np.Int64, nil)}, {Name: "y", Type: records.Numeric(np.Float64, nil)}}, false, 0)
	if e != nil {
		b.Fatal(e)
	}
	rows := make([][]any, 2048)
	for i := range rows {
		rows[i] = []any{int64(i % 1024), float64(i % 29)}
	}
	r, e := records.FromRows(s, rows, len(rows))
	if e != nil {
		b.Fatal(e)
	}
	other, e := records.FromRows(s, rows[:512], 512)
	if e != nil {
		b.Fatal(e)
	}
	b.Run("record_isin_2048x512", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkResult = must(np.Isin(r.NDArray(), other.NDArray(), false))
		}
	})
}

// BenchmarkDenseArithmetic exercises the same array shape with contiguous
// typed arrays and scalar broadcasting. Allocation is included.
func BenchmarkDenseArithmetic(b *testing.B) {
	for _, d := range []np.DType{np.Float32, np.Float64} {
		a, e := np.Full(d, 1.5, 1<<18)
		if e != nil {
			b.Fatal(e)
		}
		x, e := np.Full(d, 2.25, 1<<18)
		if e != nil {
			b.Fatal(e)
		}
		scalar, e := np.Full(d, 2.25)
		if e != nil {
			b.Fatal(e)
		}
		for _, c := range []struct {
			name  string
			right *np.Array
			fn    binaryFn
		}{
			{"add", x, np.Add}, {"multiply", x, np.Multiply}, {"divide", x, np.Divide}, {"scalar_add", scalar, np.Add},
		} {
			b.Run(d.String()+"/"+c.name, func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(a.NBytes() * 3))
				for b.Loop() {
					benchmarkResult, e = c.fn(a, c.right)
					if e != nil {
						b.Fatal(e)
					}
				}
			})
		}
	}
}
