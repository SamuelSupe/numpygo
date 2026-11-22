// Package matrix implements the legacy NumPy matrix semantics over ndarray.
// New scientific code may instead use ndarray and explicit linalg operations.
package matrix

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/linalg"
	"slices"
	"strconv"
	"strings"
)

type Matrix struct{ *np.Array }

func wrap(a *np.Array, e error) (*Matrix, error) {
	if e != nil {
		return nil, e
	}
	if a.NDim() != 2 {
		return nil, np.ErrShape
	}
	return &Matrix{a}, nil
}
func New(a *np.Array, dtype np.DType, copyData bool) (*Matrix, error) {
	return wrap(np.AsMatrix(a, dtype, copyData))
}
func Block(items any) (*Matrix, error) { return wrap(np.BMat(items)) }

// Parse accepts the numeric subset of matrix's semicolon-separated row
// syntax, with comma or whitespace separated real/complex literals.
func Parse(s string, dtype np.DType) (*Matrix, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, np.ErrValue
	}
	rows := strings.Split(s, ";")
	width := -1
	var ints []int64
	var values []complex128
	integer, complexType := true, false
	for _, r := range rows {
		fields := strings.Fields(strings.ReplaceAll(strings.TrimSpace(r), ",", " "))
		if len(fields) == 0 || width >= 0 && len(fields) != width {
			return nil, np.ErrShape
		}
		width = len(fields)
		for _, f := range fields {
			v, e := strconv.ParseInt(f, 10, 64)
			if e == nil {
				ints = append(ints, v)
				values = append(values, complex(float64(v), 0))
				continue
			}
			integer = false
			if strings.ContainsAny(f, "jJ") {
				complexType = true
				z, e := strconv.ParseComplex(strings.ReplaceAll(strings.ToLower(f), "j", "i"), 128)
				if e != nil {
					return nil, np.ErrValue
				}
				values = append(values, z)
			} else {
				x, e := strconv.ParseFloat(f, 64)
				if e != nil {
					return nil, np.ErrValue
				}
				values = append(values, complex(x, 0))
			}
		}
	}
	var a *np.Array
	var e error
	if integer {
		a, e = np.FromSlice(ints, len(rows), width)
	} else if complexType {
		a, e = np.FromSlice(values, len(rows), width)
	} else {
		v := make([]float64, len(values))
		for i, z := range values {
			v[i] = real(z)
		}
		a, e = np.FromSlice(v, len(rows), width)
	}
	if e != nil {
		return nil, e
	}
	return New(a, dtype, false)
}

func (m *Matrix) A() *np.Array           { return m.Array }
func (m *Matrix) A1() (*np.Array, error) { return np.Ravel(m.Array) }
func (m *Matrix) T() (*Matrix, error)    { return wrap(m.Array.Transpose()) }
func (m *Matrix) H() (*Matrix, error) {
	if !m.DType().IsComplex() {
		return m.T()
	}
	a, e := np.Conjugate(m.Array)
	if e != nil {
		return nil, e
	}
	return wrap(a.Transpose())
}
func (m *Matrix) I() (*Matrix, error) {
	s := m.Shape()
	if s[0] == s[1] {
		return wrap(linalg.Inv(m.Array))
	}
	return wrap(linalg.Pinv(m.Array, 1e-15))
}

// IndexMatrix returns a scalar ndarray for a scalar selection, otherwise a
// two-dimensional result. Basic column selections preserve column orientation.
func (m *Matrix) IndexMatrix(ix ...np.Index) (*np.Array, error) {
	a, e := m.Array.Index(ix...)
	if e != nil {
		return nil, e
	}
	if a.NDim() == 0 || a.NDim() == 2 {
		return a, nil
	}
	if a.NDim() != 1 {
		return nil, np.ErrShape
	}
	column := false
	if len(ix) == 2 {
		_, firstInt := ix[0].(np.AtIndex)
		_, lastInt := ix[1].(np.AtIndex)
		column = !firstInt && lastInt
		if x, ok := ix[1].(*np.Array); ok && x.NDim() == 0 && x.DType().IsInteger() {
			column = !firstInt
		}
	}
	if column {
		return a.Reshape(a.Size(), 1)
	}
	return a.Reshape(1, a.Size())
}

// AssignMatrix preserves the legacy row/column selector shape when
// broadcasting a right-hand-side array, then writes through ndarray.
func (m *Matrix) AssignMatrix(ix []np.Index, values *np.Array) error {
	logical, e := m.IndexMatrix(ix...)
	if e != nil {
		return e
	}
	values, e = values.BroadcastTo(logical.Shape()...)
	if e != nil {
		return e
	}
	raw, e := m.Array.Index(ix...)
	if e != nil {
		return e
	}
	values, e = values.Reshape(raw.Shape()...)
	if e != nil {
		return e
	}
	return m.Array.Assign(ix, values)
}
func (m *Matrix) ReshapeMatrix(shape ...int) (*Matrix, error) {
	if len(shape) > 2 {
		return nil, np.ErrShape
	}
	a, e := m.Array.Reshape(shape...)
	if e != nil {
		return nil, e
	}
	return New(a, 0, false)
}
func (m *Matrix) TransposeMatrix(axes ...int) (*Matrix, error) {
	return wrap(m.Array.Transpose(axes...))
}
func (m *Matrix) Ravel() (*Matrix, error) { return m.ReshapeMatrix(1, m.Size()) }
func (m *Matrix) Flatten() (*Matrix, error) {
	a, e := np.Flatten(m.Array)
	if e != nil {
		return nil, e
	}
	return New(a, 0, false)
}
func (m *Matrix) Squeeze(axes ...int) (*Matrix, error) {
	a, e := np.Squeeze(m.Array, axes...)
	if e != nil {
		return nil, e
	}
	return New(a, 0, false)
}
func (m *Matrix) CopyMatrix() (*Matrix, error) { return wrap(m.Array.Copy()) }
func (m *Matrix) CastMatrix(dtype np.DType, opt np.CastOptions) (*Matrix, error) {
	return wrap(m.Array.AstypeWithOptions(dtype, opt))
}
func (m *Matrix) Real() (*Matrix, error) { return wrap(np.Real(m.Array)) }
func (m *Matrix) Imag() (*Matrix, error) { return wrap(np.Imag(m.Array)) }
func (m *Matrix) Diagonal(offset int) (*Matrix, error) {
	a, e := np.Diagonal(m.Array, offset, 0, 1)
	if e != nil {
		return nil, e
	}
	return New(a, 0, false)
}
func (m *Matrix) Take(indices *np.Array, axis *int) (*Matrix, error) {
	a, e := np.Take(m.Array, indices, axis)
	if e != nil {
		return nil, e
	}
	return New(a, 0, false)
}

// Mul implements legacy '*' (matrix product for arrays, elementwise scaling
// for a scalar). Elementwise multiply remains available via np.Multiply.
func (m *Matrix) Mul(b *np.Array, reflected bool) (*Matrix, error) {
	if b.DType() == 0 {
		return nil, np.ErrDType
	}
	if b.NDim() == 0 {
		return wrap(np.Multiply(m.Array, b))
	}
	a, e := np.AsMatrix(b, 0, false)
	if e != nil {
		return nil, e
	}
	if reflected {
		return wrap(linalg.MatMul(a, m.Array))
	}
	return wrap(linalg.MatMul(m.Array, a))
}
func (m *Matrix) MatMul(b *np.Array, reflected bool) (*Matrix, error) {
	if b.NDim() == 0 {
		return nil, np.ErrShape
	}
	// NumPy's @ follows ndarray contraction; the matrix subclass then
	// promotes a 1-D result to a row matrix.
	a, c := m.Array, b
	if reflected {
		a, c = c, a
	}
	v, e := linalg.MatMul(a, c)
	if e != nil {
		return nil, e
	}
	return New(v, 0, false)
}
func (m *Matrix) Add(b *np.Array) (*Matrix, error) { return wrap(np.Add(m.Array, b)) }
func (m *Matrix) Sub(b *np.Array, reflected bool) (*Matrix, error) {
	if reflected {
		return wrap(np.Subtract(b, m.Array))
	}
	return wrap(np.Subtract(m.Array, b))
}
func (m *Matrix) Divide(b *np.Array, reflected bool) (*Matrix, error) {
	if reflected {
		return wrap(np.Divide(b, m.Array))
	}
	return wrap(np.Divide(m.Array, b))
}
func (m *Matrix) Pow(exponent int) (*Matrix, error) {
	return wrap(linalg.MatrixPower(m.Array, exponent))
}

func (m *Matrix) reduce(fn func(*np.Array, ...np.SumOptions) (*np.Array, error), axes []int, dtype np.DType) (*np.Array, error) {
	// axis=None returns a scalar; any explicit axis selection keeps both
	// matrix dimensions, including a tuple reducing both axes.
	return fn(m.Array, np.SumOptions{Axes: slices.Clone(axes), KeepDims: axes != nil, DType: dtype})
}
func (m *Matrix) Sum(axes []int, dtype np.DType) (*np.Array, error) {
	return m.reduce(np.Sum, axes, dtype)
}
func (m *Matrix) Prod(axes []int, dtype np.DType) (*np.Array, error) {
	return m.reduce(np.Prod, axes, dtype)
}
func (m *Matrix) Mean(axes []int, dtype np.DType) (*np.Array, error) {
	return m.reduce(np.Mean, axes, dtype)
}
func (m *Matrix) All(axes []int) (*np.Array, error) { return m.reduce(np.All, axes, 0) }
func (m *Matrix) Any(axes []int) (*np.Array, error) { return m.reduce(np.Any, axes, 0) }
func (m *Matrix) Min(axes []int) (*np.Array, error) { return m.reduce(np.Min, axes, 0) }
func (m *Matrix) Max(axes []int) (*np.Array, error) { return m.reduce(np.Max, axes, 0) }
func (m *Matrix) PTP(axes []int) (*np.Array, error) {
	if axes != nil && len(axes) != 1 {
		return nil, np.ErrValue
	}
	return m.reduce(np.PTP, axes, 0)
}
func (m *Matrix) Var(axes []int, dtype np.DType, ddof float64) (*np.Array, error) {
	return np.Var(m.Array, np.VarOptions{Axes: axes, KeepDims: axes != nil, DType: dtype, DDOF: ddof})
}
func (m *Matrix) Std(axes []int, dtype np.DType, ddof float64) (*np.Array, error) {
	return np.Std(m.Array, np.VarOptions{Axes: axes, KeepDims: axes != nil, DType: dtype, DDOF: ddof})
}
func (m *Matrix) ArgMin(axis *int) (*np.Array, error) {
	return np.ArgMin(m.Array, np.ArgOptions{Axis: axis, KeepDims: axis != nil})
}
func (m *Matrix) ArgMax(axis *int) (*np.Array, error) {
	return np.ArgMax(m.Array, np.ArgOptions{Axis: axis, KeepDims: axis != nil})
}
