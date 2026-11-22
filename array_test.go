package np_test

import (
	"errors"
	"math"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestArrayViews(t *testing.T) {
	must := refcheck.Require(t)
	base := refcheck.Input(t, "array_base")
	refcheck.CheckLayout(t, "array_base", base)
	transposed := must(base.Transpose(2, 0, 1))
	refcheck.CheckLayout(t, "array_transpose", transposed)
	refcheck.CheckLayout(t, "array_reverse", must(base.Slice(np.Reverse(), np.Slice{Start: np.Int(1)}, np.Slice{Step: np.Int(-2)})))
	refcheck.CheckLayout(t, "array_empty_slice", must(base.Slice(np.Slice{}, np.Range(1, 1), np.Reverse())))
	refcheck.CheckLayout(t, "array_newaxis", must(base.Slice(np.NewAxis, np.Ellipsis, np.AtIndex(-1))))
	refcheck.CheckLayout(t, "array_clipped_slice", must(base.Slice(np.Strided(-100, 100, 2), np.Strided(2, -4, -1), np.Slice{Start: np.Int(10), Step: np.Int(-2)})))
	refcheck.CheckLayout(t, "array_explicit_negative_stop", must(base.Slice(np.Ellipsis, np.Slice{Stop: np.Int(-1), Step: np.Int(-1)})))
	refcheck.CheckLayout(t, "array_scalar", np.Scalar(3.25))
	refcheck.CheckLayout(t, "array_empty", must(np.Zeros(np.Float64, 2, 0, 3)))
	refcheck.CheckLayout(t, "array_empty_reshape", must(must(np.Zeros(np.Float64, 0)).Reshape(2, 0, 3)))
	if !base.IsCContiguous() || transposed.IsCContiguous() {
		t.Fatal("unexpected C-contiguity flags")
	}
	view := must(transposed.Slice(np.Reverse()))
	if err := view.Set(100, 0, 1, 2); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "array_view_mutation", base)
	value, err := base.At(-1, -1, -1)
	if err != nil || value != 100 {
		t.Fatalf("negative index: %v %v", value, err)
	}
}

func TestReshapeAndCopy(t *testing.T) {
	must := refcheck.Require(t)
	seq := make([]float64, 12)
	for i := range seq {
		seq[i] = float64(i)
	}
	owned := must(np.WrapFloat64(seq, 12))
	stepped := must(must(owned.Slice(np.Slice{Step: np.Int(2)})).Reshape(2, -1))
	refcheck.CheckLayout(t, "array_stepped_reshape", stepped)
	if err := stepped.Set(88, 1, 2); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "array_reshape_mutation", owned)
	if seq[10] != 88 {
		t.Fatal("wrapped slice did not share changes")
	}
	reverse := must(must(np.FromFloat64([]float64{0, 1, 2, 3, 4, 5}, 6)).Slice(np.Reverse()))
	refcheck.CheckLayout(t, "array_reverse_reshape", must(reverse.Reshape(2, 3)))

	base := refcheck.Input(t, "array_base")
	partial := must(must(base.Slice(np.Slice{}, np.Slice{Step: np.Int(2)})).Reshape(2, 2, 2, 2))
	refcheck.CheckLayout(t, "array_partial_reshape", partial)
	if err := partial.Set(41, 1, 1, 1, 0); err != nil {
		t.Fatal(err)
	}
	if v, _ := base.At(1, 2, 2); v != 41 {
		t.Fatal("partial reshape lost view sharing")
	}
	base = refcheck.Input(t, "array_base")
	reshaped := must(must(base.Transpose(1, 0, 2)).Reshape(6, 4))
	refcheck.CheckLayout(t, "array_transpose_reshape", reshaped)
	if err := reshaped.Set(100, 0, 0); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "array_base", base)
	clone := must(base.Copy())
	if err := clone.Set(99, 0, 0, 0); err != nil {
		t.Fatal(err)
	}
	refcheck.Check(t, "array_base", base)

	source := []float64{5}
	a := must(np.FromFloat64(source, 1))
	source[0] = 9
	read, _ := a.Float64s()
	if read[0] != 5 {
		t.Fatal("constructor did not copy input")
	}
	read[0] = 99
	again, _ := a.Float64s()
	if again[0] != 5 {
		t.Fatal("Float64s did not return a detached copy")
	}
	shape, strides := a.Shape(), a.Strides()
	shape[0], strides[0] = 99, 99
	if a.Shape()[0] != 1 || a.Strides()[0] != 8 {
		t.Fatal("metadata escaped array ownership")
	}
}

func TestBroadcastAndErrors(t *testing.T) {
	must := refcheck.Require(t)
	base := refcheck.Input(t, "array_base")
	b := must(np.FromFloat64([]float64{1, 2, 3}, 1, 3))
	broadcast := must(b.BroadcastTo(2, 3))
	refcheck.CheckLayout(t, "array_broadcast", broadcast)
	if broadcast.Writable() {
		t.Fatal("broadcast is writable")
	}
	refcheck.Error(t, "array_readonly", "ValueError", broadcast.Set(9, 0, 0), np.ErrReadOnly)
	if err := b.Set(9, 0, 0); err != nil {
		t.Fatal(err)
	}
	if v, _ := broadcast.At(1, 0); v != 9 {
		t.Fatal("broadcast did not retain underlying storage")
	}
	_, err := base.Reshape(7)
	refcheck.Error(t, "array_bad_reshape", "ValueError", err, np.ErrShape)
	_, err = base.Reshape(-1, -1)
	refcheck.Error(t, "array_double_infer", "ValueError", err, np.ErrShape)
	_, err = must(np.Zeros(np.Float64, 0)).Reshape(0, -1)
	refcheck.Error(t, "array_ambiguous_empty", "ValueError", err, np.ErrShape)
	_, err = base.Transpose(0, 0, 2)
	refcheck.Error(t, "array_axis", "ValueError", err, np.ErrAxis)
	_, err = base.Slice(np.AtIndex(2))
	refcheck.Error(t, "array_index", "IndexError", err, np.ErrIndex)
	_, err = base.Slice(np.Slice{Step: np.Int(0)})
	refcheck.Error(t, "array_slice_step_zero", "ValueError", err, np.ErrIndex)
	_, err = b.BroadcastTo(2, 4)
	refcheck.Error(t, "array_bad_broadcast", "ValueError", err, np.ErrShape)
	if _, err = np.Zeros(np.Float64, -1); !errors.Is(err, np.ErrShape) {
		t.Fatalf("negative extent: %v", err)
	}
	if _, err = np.Zeros(np.Float64, math.MaxInt, 2); !errors.Is(err, np.ErrShape) {
		t.Fatalf("shape overflow: %v", err)
	}
	if _, err = base.Slice(np.Ellipsis, np.Ellipsis); !errors.Is(err, np.ErrIndex) {
		t.Fatalf("duplicate ellipsis: %v", err)
	}
}

func TestArithmetic(t *testing.T) {
	must := refcheck.Require(t)
	a, b := refcheck.Input(t, "real_operands"), refcheck.Input(t, "complex_operands")
	refcheck.Check(t, "array_add", must(np.Add(a, b)))
	refcheck.Check(t, "array_subtract", must(np.Subtract(a, b)))
	refcheck.Check(t, "array_multiply", must(np.Multiply(a, b)))
	refcheck.Check(t, "array_divide", must(np.Divide(a, b)))
	refcheck.Check(t, "array_complex_reverse_divide", must(np.Divide(b, a)))
	base := refcheck.Input(t, "array_base")
	refcheck.Check(t, "array_scalar_broadcast", must(np.Add(must(base.Slice(np.Reverse())), np.Scalar(.5))))
	refcheck.Check(t, "array_real_broadcast", must(np.Multiply(base, must(np.FromFloat64([]float64{1, 2, 3}, 3, 1)))))
	refcheck.Check(t, "array_empty_broadcast", must(np.Add(must(np.Zeros(np.Float64, 2, 0, 3)), must(np.Zeros(np.Float64, 1, 3)))))
	_, err := np.Add(must(np.Zeros(np.Float64, 2)), must(np.Zeros(np.Float64, 3)))
	refcheck.Error(t, "array_arithmetic_shape", "ValueError", err, np.ErrShape)
	refcheck.Check(t, "array_real_divzero", must(np.Divide(
		must(np.FromFloat64([]float64{0, 1, -2}, 3)),
		must(np.FromFloat64([]float64{0, math.Copysign(0, -1), 0}, 3)))))
	refcheck.Check(t, "array_complex_divzero", must(np.Divide(
		must(np.FromComplex128([]complex128{1, 1i, 1 + 1i, 0}, 4)), np.ComplexScalar(0))))
	refcheck.Check(t, "array_complex_nonfinite", must(np.Divide(
		must(np.FromComplex128([]complex128{1 + 1i, complex(math.Inf(1), 0), 1 + 1i, 2 - 1i}, 4)),
		must(np.FromComplex128([]complex128{complex(math.Inf(1), 0), 1 + 1i, complex(math.Inf(1), math.Inf(1)), complex(math.NaN(), 1)}, 4)))))
}

func TestSum(t *testing.T) {
	must := refcheck.Require(t)
	base := refcheck.Input(t, "array_base")
	refcheck.Check(t, "sum_all", must(np.Sum(base)))
	refcheck.Check(t, "sum_last_keepdims", must(np.Sum(base, np.SumOptions{Axes: []int{-1}, KeepDims: true})))
	refcheck.Check(t, "sum_axes", must(np.Sum(must(base.Slice(np.Reverse(), np.Slice{}, np.Reverse())), np.SumOptions{Axes: []int{2, 0}})))
	refcheck.Check(t, "sum_axes_empty", must(np.Sum(base, np.SumOptions{Axes: []int{}})))
	combined := must(np.Add(refcheck.Input(t, "real_operands"), refcheck.Input(t, "complex_operands")))
	refcheck.Check(t, "sum_complex", must(np.Sum(combined, np.SumOptions{Axes: []int{0, 2}})))
	refcheck.Check(t, "sum_empty", must(np.Sum(must(np.Zeros(np.Float64, 2, 0, 3)), np.SumOptions{Axes: []int{1}})))
	refcheck.Check(t, "sum_scalar", must(np.Sum(np.Scalar(2.5))))
	long := make([]float64, 512)
	for i := 0; i < len(long); i += 4 {
		copy(long[i:i+4], []float64{1e16, 1, -1e16, 1})
	}
	array := must(np.WrapFloat64(long, len(long)))
	refcheck.Check(t, "sum_pairwise", must(np.Sum(array)))
	refcheck.Check(t, "sum_pairwise_axis", must(np.Sum(must(array.Reshape(4, 128)), np.SumOptions{Axes: []int{-1}})))
	refcheck.Check(t, "sum_slow_axis", must(np.Sum(must(array.Reshape(128, 4)), np.SumOptions{Axes: []int{0}})))
	_, err := np.Sum(base, np.SumOptions{Axes: []int{3}})
	refcheck.Error(t, "sum_bad_axis", "AxisError", err, np.ErrAxis)
	_, err = np.Sum(base, np.SumOptions{Axes: []int{0, -3}})
	refcheck.Error(t, "sum_duplicate_axis", "ValueError", err, np.ErrAxis)
}
