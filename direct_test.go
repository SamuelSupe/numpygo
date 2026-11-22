package np_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/fft"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestDirectCasts(t *testing.T) {
	for _, tc := range []struct {
		name string
		dst  np.DType
	}{{"float32", np.Float64}, {"float64", np.Float32}, {"complex64", np.Complex128}, {"complex128", np.Complex64}} {
		t.Run(tc.name, func(t *testing.T) {
			m := refcheck.Require(t)
			a := refcheck.Input(t, "direct_cast_input_"+tc.name)
			x := m(a.Slice(np.Range(1, 262)))
			out := m(x.Astype(tc.dst))
			refcheck.CheckExact(t, "direct_cast_offset_"+tc.name, out)
			if np.SharesMemory(x, out) {
				t.Fatal("cast aliased input")
			}
			rows := m(m(m(a.Slice(np.Range(1, 261))).Reshape(4, 65)).Slice(np.Reverse(), np.Slice{}))
			refcheck.CheckExact(t, "direct_cast_reverse_rows_"+tc.name, m(rows.Astype(tc.dst)))
			if tc.name == "complex64" {
				v, e := rows.Complex128s()
				if e != nil {
					t.Fatal(e)
				}
				refcheck.CheckExact(t, "direct_cast_reverse_rows_"+tc.name, m(np.WrapComplex128(v, 4, 65)))
			}
			refcheck.CheckExact(t, "direct_cast_input_"+tc.name, a)
		})
	}
}

func TestDirectUFuncOut(t *testing.T) {
	for _, dtype := range []np.DType{np.Float32, np.Float64} {
		t.Run(dtype.String(), func(t *testing.T) {
			m := refcheck.Require(t)
			lookup := refcheck.RequireValue[*np.UFunc](t)
			d := dtype.String()
			a, b := refcheck.Input(t, "direct_ufunc_a_"+d), refcheck.Input(t, "direct_ufunc_b_"+d)
			mask := refcheck.Input(t, "direct_ufunc_mask_"+d)
			alt := refcheck.Input(t, "direct_ufunc_altmask_"+d)
			var scalar *np.Array
			if dtype == np.Float32 {
				scalar = np.ScalarOf(float32(-1.25))
			} else {
				scalar = np.ScalarOf(float64(-1.25))
			}
			for _, op := range []struct{ tag, name string }{{"add", "add"}, {"sub", "subtract"}, {"mul", "multiply"}, {"div", "divide"}} {
				u := lookup(np.LookupUFunc(op.name))
				base := m(np.Full(dtype, -7, a.Size()+2))
				out := m(m(base.Slice(np.Range(1, a.Size()+1))).Reshape(a.Shape()...))
				opt := np.UFuncOptions{Out: out, Where: mask}
				if m(u.CallWithOptions(opt, a, b)) != out {
					t.Fatal("out identity not preserved")
				}
				refcheck.CheckExact(t, "direct_ufunc_"+op.tag+"_masked_"+d, base)
				base = m(np.Full(dtype, -7, a.Size()+2))
				out = m(m(base.Slice(np.Range(1, a.Size()+1))).Reshape(a.Shape()...))
				m(u.CallWithOptions(np.UFuncOptions{Out: out, Where: alt}, a, b))
				refcheck.CheckExact(t, "direct_ufunc_"+op.tag+"_alt_"+d, base)
				base = m(np.Full(dtype, -7, a.Size()+2))
				out = m(m(base.Slice(np.Range(1, a.Size()+1))).Reshape(a.Shape()...))
				if m(u.CallWithOptions(np.UFuncOptions{Out: out}, scalar, a)) != out {
					t.Fatal("out identity not preserved")
				}
				refcheck.CheckExact(t, "direct_ufunc_"+op.tag+"_left_"+d, base)
				alias := m(a.Copy())
				if m(u.CallWithOptions(np.UFuncOptions{Out: alias, Where: mask}, alias, scalar)) != alias {
					t.Fatal("alias identity")
				}
				refcheck.CheckExact(t, "direct_ufunc_"+op.tag+"_alias_"+d, alias)
			}
			sub, add := lookup(np.LookupUFunc("subtract")), lookup(np.LookupUFunc("add"))
			x := refcheck.Input(t, "direct_overlap_input_"+d)
			in, out := m(x.Slice(np.Range(0, 72))), m(x.Slice(np.Range(1, 73)))
			m(sub.CallWithOptions(np.UFuncOptions{Out: out}, in, scalar))
			refcheck.CheckExact(t, "direct_overlap_shift_"+d, x)
			x = refcheck.Input(t, "direct_broadcast_alias_input_"+d)
			m(add.CallWithOptions(np.UFuncOptions{Out: x}, x, m(x.Slice(np.Range(0, 1)))))
			refcheck.CheckExact(t, "direct_broadcast_alias_"+d, x)
			out = m(m(np.Full(dtype, -7, 69, 3)).Transpose())
			m(add.CallWithOptions(np.UFuncOptions{Out: out, Where: m(mask.Slice(np.Slice{}, np.Reverse()))}, a, b))
			refcheck.CheckExact(t, "direct_noncontig_out_"+d, out)
			out = m(np.Full(dtype, -7, 3, 69))
			m(add.CallWithOptions(np.UFuncOptions{Out: out, Where: np.ScalarOf(false)}, a, b))
			v, _ := out.Complex128s()
			for _, z := range v {
				if z != -7 {
					t.Fatal("false where changed output")
				}
			}
			if e := out.SetWritable(false); e != nil {
				t.Fatal(e)
			}
			if _, e := add.CallWithOptions(np.UFuncOptions{Out: out}, a, b); !errors.Is(e, np.ErrReadOnly) {
				t.Fatalf("readonly: %v", e)
			}
			refcheck.CheckExact(t, "direct_ufunc_a_"+d, a)
			refcheck.CheckExact(t, "direct_ufunc_b_"+d, b)
			refcheck.CheckExact(t, "direct_ufunc_mask_"+d, mask)
		})
	}
	m := refcheck.Require(t)
	c := np.NewErrorContext()
	if _, e := c.SetErr(np.ErrorOptions{Divide: np.Raise}); e != nil {
		t.Fatal(e)
	}
	x, y := m(np.Ones(np.Float64, 69)), m(np.Ones(np.Float64, 69))
	if e := y.Set(0, 4); e != nil {
		t.Fatal(e)
	}
	_, e := c.CallWithOptions("divide", np.UFuncOptions{Out: x}, x, y)
	refcheck.Error(t, "direct_fp_alias_raise", "FloatingPointError", e, np.ErrFloatingPoint)
	var f *np.FloatingPointError
	if !errors.As(e, &f) || f.Kind != np.FPDivide {
		t.Fatalf("incorrect alias FP flag: %v", e)
	}
	refcheck.CheckExact(t, "direct_fp_alias_result", x)
}

func TestDirectFFTDestination(t *testing.T) {
	m := refcheck.Require(t)
	a, c := refcheck.Input(t, "direct_fft_real_input"), refcheck.Input(t, "direct_fft_complex_input")
	ar, cr := m(a.BroadcastTo(a.Shape()...)), m(c.BroadcastTo(c.Shape()...))
	check := func(t *testing.T, key string, d np.DType, shape []int, f func(*np.Array) (*np.Array, error)) {
		t.Helper()
		m := refcheck.Require(t)
		size, e := np.ElementCount(d, shape...)
		if e != nil {
			t.Fatal(e)
		}
		base := m(np.Full(d, -7, size+2))
		out := m(m(base.Slice(np.Range(1, size+1))).Reshape(shape...))
		if m(f(out)) != out {
			t.Fatal("FFT output object changed")
		}
		refcheck.Check(t, key, out)
		v, _ := base.Complex128s()
		if v[0] != -7 || v[len(v)-1] != -7 {
			t.Fatal("FFT wrote outside destination")
		}
	}
	for _, n := range []int{1024, 1000, 509} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			tag := fmt.Sprint(n)
			check(t, "direct_rfft_"+tag, np.Complex128, []int{3, n/2 + 1}, func(out *np.Array) (*np.Array, error) {
				return fft.RFFT(ar, fft.Options{N: np.Int(n), Norm: fft.Ortho, Out: out})
			})
			z := refcheck.Input(t, "direct_rfft_"+tag)
			check(t, "direct_irfft_"+tag, np.Float64, []int{3, n}, func(out *np.Array) (*np.Array, error) {
				return fft.IRFFT(z, fft.Options{N: np.Int(n), Norm: fft.Ortho, Out: out})
			})
			check(t, "direct_fft_"+tag, np.Complex128, []int{3, n}, func(out *np.Array) (*np.Array, error) {
				return fft.FFT(cr, fft.Options{N: np.Int(n), Norm: fft.Forward, Out: out})
			})
			check(t, "direct_ifft_"+tag, np.Complex128, []int{3, n}, func(out *np.Array) (*np.Array, error) {
				return fft.IFFT(cr, fft.Options{N: np.Int(n), Norm: fft.Ortho, Out: out})
			})
			refcheck.CheckExact(t, "direct_rfft_"+tag, z)
		})
	}
	refcheck.CheckExact(t, "direct_fft_real_input", a)
	refcheck.CheckExact(t, "direct_fft_complex_input", c)
	bad := refcheck.Input(t, "direct_fft_nonfinite_input")
	check(t, "direct_rfft_nonfinite", np.Complex128, []int{2, 33}, func(out *np.Array) (*np.Array, error) { return fft.RFFT(bad, fft.Options{Out: out}) })
	x := refcheck.Input(t, "direct_fft_alias_input")
	v, _ := x.Complex128s()
	raw := append(v, 0)
	in, out := m(np.WrapComplex128(raw[:1024], 1024)), m(np.WrapComplex128(raw[1:], 1024))
	if m(fft.FFT(in, fft.Options{Out: out})) != out {
		t.Fatal("overlap identity")
	}
	refcheck.Check(t, "direct_fft_alias_result", out)
	if m(fft.FFT(x, fft.Options{Out: x})) != x {
		t.Fatal("in-place identity")
	}
	refcheck.Check(t, "direct_fft_alias_result", x)
	for _, n := range []int{1000, 509} {
		x = m(m(c.Slice(np.AtIndex(0), np.Range(0, n))).Copy())
		if m(fft.FFT(x, fft.Options{Out: x})) != x {
			t.Fatal("in-place identity")
		}
		refcheck.Check(t, fmt.Sprintf("direct_fft_alias_%d", n), x)
	}
	x = m(m(m(c.Slice(np.Slice{}, np.Range(0, 1024))).Transpose()).Copy())
	if m(fft.FFT(x, fft.Options{Out: x, Norm: fft.Ortho, Axis: np.Int(0)})) != x {
		t.Fatal("axis0 in-place identity")
	}
	refcheck.Check(t, "direct_fft_axis0_alias", x)
	c32, a32 := refcheck.Input(t, "arrayfft_complex32_input"), refcheck.Input(t, "arrayfft_real32_input")
	z32 := refcheck.Input(t, "arrayfft_rfft32")
	check(t, "arrayfft_fft32_oracle", np.Complex64, []int{3, 4096}, func(out *np.Array) (*np.Array, error) { return fft.FFT(c32, fft.Options{Out: out}) })
	check(t, "arrayfft_rfft32_oracle", np.Complex64, []int{3, 2049}, func(out *np.Array) (*np.Array, error) { return fft.RFFT(a32, fft.Options{Out: out}) })
	check(t, "arrayfft_irfft32_oracle", np.Float32, []int{3, 4096}, func(out *np.Array) (*np.Array, error) { return fft.IRFFT(z32, fft.Options{N: np.Int(4096), Out: out}) })
}
