package np_test

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func classTextInput(t *testing.T, name string) string {
	t.Helper()
	return string(refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, "class_"+name))))
}
func TestUFuncClassMetadataNumPy(t *testing.T) {
	get := refcheck.RequireValue[*np.UFunc](t)
	for _, line := range strings.Split(classTextInput(t, "ufunc_metadata"), "\n") {
		fields := strings.Split(line, "|")
		t.Run(fields[0], func(t *testing.T) {
			u := get(np.LookupUFunc(fields[0]))
			identity := "None"
			if u.HasIdentity {
				switch v := u.Identity.(type) {
				case bool:
					if v {
						identity = "True"
					} else {
						identity = "False"
					}
				case float64:
					if math.IsInf(v, 1) {
						identity = "inf"
					} else if math.IsInf(v, -1) {
						identity = "-inf"
					} else {
						identity = strconv.FormatFloat(v, 'g', -1, 64)
					}
				default:
					identity = fmt.Sprint(v)
				}
			}
			got := strings.Join([]string{fields[0], u.Name, strconv.Itoa(u.NIn), strconv.Itoa(u.NOut), strconv.Itoa(u.NArgs()), strconv.Itoa(u.NTypes()), identity, u.Signature(), strings.Join(u.Types(), ",")}, "|")
			if got != line {
				t.Errorf("got %q\nNumPy %q", got, line)
			}
		})
	}
	for _, line := range strings.Split(classTextInput(t, "ufunc_resolve"), "\n") {
		fields := strings.Split(line, "|")
		t.Run("resolve/"+fields[0]+"/"+fields[1], func(t *testing.T) {
			u := get(np.LookupUFunc(fields[0]))
			args := make([]np.DType, u.NArgs())
			for i, x := range strings.Split(fields[1], ",") {
				args[i] = refcheck.RequireValue[np.DType](t)(np.ParseDType(x))
			}
			got, e := u.ResolveDTypes(args, np.ResolveDTypesOptions{})
			if e != nil {
				t.Fatal(e)
			}
			names := make([]string, len(got))
			for i, d := range got {
				names[i] = d.String()
			}
			if strings.Join(names, ",") != fields[2] {
				t.Errorf("got %v, want %s", names, fields[2])
			}
		})
	}
	u := get(np.LookupUFunc("add"))
	for _, tc := range []struct {
		name string
		args []np.DType
		opt  np.ResolveDTypesOptions
	}{
		{"weak_int", []np.DType{np.Int8, np.Int64, 0}, np.ResolveDTypesOptions{WeakInputs: []bool{false, true}}},
		{"weak_float", []np.DType{np.Float32, np.Float64, 0}, np.ResolveDTypesOptions{WeakInputs: []bool{false, true}}},
		{"weak_complex", []np.DType{np.Float32, np.Complex128, 0}, np.ResolveDTypesOptions{WeakInputs: []bool{false, true}}},
		{"reduction", []np.DType{0, np.Int8, 0}, np.ResolveDTypesOptions{Reduction: true}},
		{"signature", []np.DType{np.Int16, np.Int16, 0}, np.ResolveDTypesOptions{Signature: []np.DType{0, 0, np.Float32}}},
	} {
		t.Run("resolve/"+tc.name, func(t *testing.T) {
			ds := refcheck.RequireValue[[]np.DType](t)(u.ResolveDTypes(tc.args, tc.opt))
			names := make([]string, len(ds))
			for i, d := range ds {
				names[i] = d.String()
			}
			refcheck.Text(t, "class_ufunc_resolve_"+tc.name, strings.Join(names, ","))
		})
	}
}
func TestUFuncClassOperationsNumPy(t *testing.T) {
	m := refcheck.Require(t)
	get := refcheck.RequireValue[*np.UFunc](t)
	vm := refcheck.RequireValue[[]*np.Array](t)
	x := m(np.FromSlice([]float32{-2.5, 3.25, 0}, 3))
	for _, name := range []string{"frexp", "modf"} {
		v := vm(get(np.LookupUFunc(name)).CallAll(x))
		for i, a := range v {
			refcheck.Check(t, "class_ufunc_"+name+"_"+strconv.Itoa(i), a)
		}
	}
	v := vm(get(np.LookupUFunc("divmod")).CallAll(m(np.FromSlice([]int16{-7, 7, 9}, 3)), m(np.FromSlice([]int16{3}))))
	for i, a := range v {
		refcheck.Check(t, "class_ufunc_divmod_"+strconv.Itoa(i), a)
	}
	out0 := m(m(np.FromSlice([]float32{19, 19, 19}, 3)).Slice(np.Slice{Step: np.Int(-1)}))
	out1 := m(m(np.FromSlice([]float32{29, 29, 29}, 3)).Slice(np.Slice{Step: np.Int(-1)}))
	v = vm(get(np.LookupUFunc("modf")).CallAllWithOptions(np.MultiUFuncOptions{Out: []*np.Array{out0, out1}, Where: m(np.FromSlice([]bool{true, false, true}, 3))}, x))
	if v[0] != out0 || v[1] != out1 {
		t.Fatal("out identity")
	}
	refcheck.CheckLayout(t, "class_ufunc_multi_out0", out0)
	refcheck.CheckLayout(t, "class_ufunc_multi_out1", out1)
	a := m(m(np.ArangeInt64(0, 6, 1)).Reshape(2, 3))
	ix := []np.Index{m(np.FromSlice([]int64{0, 0, 1}, 3)), m(np.FromSlice([]int64{1, 1, 0}, 3))}
	classOK(t, get(np.LookupUFunc("add")).AtIndices(a, ix, m(np.FromSlice([]int64{10, 20, 30}, 3))))
	refcheck.Check(t, "class_ufunc_at_multi", a)
	classOK(t, get(np.LookupUFunc("negative")).AtIndices(a, ix, nil))
	refcheck.Check(t, "class_ufunc_at_unary", a)
	add := get(np.LookupUFunc("add"))
	out := m(m(np.Zeros(np.Float32, 2, 3)).Slice(np.Slice{}, np.Slice{Step: np.Int(-1)}))
	if m(add.AccumulateInto(m(m(m(np.ArangeInt64(0, 6, 1)).Astype(np.Int16)).Reshape(2, 3)), 1, np.Int64, out)) != out {
		t.Fatal("accumulate output identity")
	}
	refcheck.CheckLayout(t, "class_ufunc_accumulate_out", out)
	out = m(m(np.Full(np.Int64, int64(-9), 2, 3)).Slice(np.Slice{}, np.Slice{Step: np.Int(-1)}))
	if m(add.OuterWithOptions(m(np.FromSlice([]int64{1, 2}, 2)), m(np.FromSlice([]int64{10, 20, 30}, 3)), np.UFuncOptions{Out: out, Where: m(np.FromSlice([]bool{true, false, true}, 3))})) != out {
		t.Fatal("outer output identity")
	}
	refcheck.CheckLayout(t, "class_ufunc_outer_out", out)
	mat := m(m(m(np.ArangeInt64(0, 12, 1)).Astype(np.Float64)).Reshape(2, 2, 3))
	vec := m(np.FromSlice([]float64{1, 2, 3}, 3))
	for _, name := range []string{"matmul", "matvec"} {
		refcheck.Check(t, "class_ufunc_"+name, m(get(np.LookupUFunc(name)).Call(mat, vec)))
	}
	refcheck.Check(t, "class_ufunc_vecmat", m(get(np.LookupUFunc("vecmat")).Call(vec, m(mat.Transpose(0, 2, 1)))))
	c := m(np.FromSlice([]complex128{1 + 1i, 2 + 1i, 3 + 1i}, 3))
	refcheck.Check(t, "class_ufunc_vecdot", m(get(np.LookupUFunc("vecdot")).Call(c, mat)))
}
