package lib

import (
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func refText(t *testing.T, n string) string {
	return string(refcheck.RequireValue[[]uint8](t)(np.Values[uint8](refcheck.Input(t, n))))
}
func TestNumpyVersionNumPy(t *testing.T) {
	m := refcheck.RequireValue[NumpyVersion](t)
	ss := strings.Split(refText(t, "lib_version_inputs"), "|")
	v := make([]NumpyVersion, len(ss))
	for i, s := range ss {
		v[i] = m(NewNumpyVersion(s))
		x := v[i]
		dev := 0
		if x.IsDevVersion {
			dev = 1
		}
		refcheck.Text(t, "lib_version_meta_"+strconv.Itoa(i), fmt.Sprintf("%s|%s|%d|%d|%d|%s|%d|%s", x.VString, x.Version, x.Major, x.Minor, x.Bugfix, x.PreRelease, dev, x.String()))
	}
	want := refcheck.RequireValue[[]int64](t)(np.Values[int64](refcheck.Input(t, "lib_version_compare")))
	for i, a := range v {
		t.Run(a.VString, func(t *testing.T) {
			for j, b := range v {
				x, e := a.Compare(b)
				y, f := a.Compare(b.VString)
				if e != nil || f != nil || x != y || int64(x) != want[i*len(v)+j] {
					t.Errorf("%s vs %s: %d %d %v %v; want %d", a, b, x, y, e, f, want[i*len(v)+j])
				}
			}
		})
	}
	_, e := NewNumpyVersion("2.5")
	refcheck.Error(t, "lib_version_invalid", "ValueError", e, np.ErrValue)
	_, e = v[0].Compare(1)
	refcheck.Error(t, "lib_version_bad_compare", "ValueError", e, np.ErrValue)
}
func TestOptFuncInfoNumPy(t *testing.T) {
	m := refcheck.RequireValue[map[string]map[string]DispatchInfo](t)
	a := m(OptFuncInfo("^add$", "float64"))
	keys := []string{}
	for k, v := range a["add"] {
		keys = append(keys, k)
		if v.Current == "" || v.Available == "" {
			t.Fatal("missing backend path")
		}
	}
	slices.Sort(keys)
	refcheck.Text(t, "lib_opt_add_float64", strings.Join(keys, "|"))
	refcheck.Check(t, "lib_opt_empty", np.ScalarOf(int64(len(m(OptFuncInfo("^nobody_had_this_name$", ""))))))
	refcheck.Check(t, "lib_opt_empty_sig", np.ScalarOf(int64(len(m(OptFuncInfo("", "^not_a_dtype$"))))))
	_, e := OptFuncInfo("[", "")
	refcheck.Error(t, "lib_opt_bad_regex", "error", e, np.ErrValue)
}

func TestNDArrayOperatorsMixinNumPy(t *testing.T) {
	m := refcheck.Require(t)
	mm := refcheck.RequireValue[*NDArrayOperatorsMixin](t)
	mv := refcheck.RequireValue[any](t)
	trace := mm(NewNDArrayOperatorsMixin("self", func(name string, inputs, outputs []any) (any, error) {
		labels := func(v []any) string {
			s := make([]string, len(v))
			for i, x := range v {
				s[i] = x.(string)
			}
			return strings.Join(s, ",")
		}
		return name + "|" + labels(inputs) + "|" + labels(outputs), nil
	}))
	calls := []string{}
	for _, op := range strings.Fields("lt le eq ne gt ge add sub mul matmul truediv floordiv mod divmod pow lshift rshift and xor or") {
		calls = append(calls, op+":"+mv(trace.Binary(op, "other")).(string))
		if _, ok := mixinComparisons[op]; !ok {
			calls = append(calls, "r"+op+":"+mv(trace.Reflected(op, "other")).(string))
			if op != "divmod" {
				calls = append(calls, "i"+op+":"+mv(trace.InPlace(op, "other")).(string))
			}
		}
	}
	for _, op := range []string{"neg", "pos", "abs", "invert"} {
		calls = append(calls, op+":"+mv(trace.Unary(op)).(string))
	}
	refcheck.Text(t, "lib_mixin_dispatch", strings.Join(calls, "\n"))
	x := m(np.FromSlice([]int64{1, 2, 3}, 3))
	w := mm(NewNDArrayOperatorsMixin(x, func(name string, inputs, outputs []any) (any, error) {
		if name == "divmod" {
			a, b, e := np.Divmod(inputs[0].(*np.Array), inputs[1].(*np.Array))
			return []*np.Array{a, b}, e
		}
		u, e := np.LookupUFunc(name)
		if e != nil {
			return nil, e
		}
		opt := np.UFuncOptions{}
		if len(outputs) > 0 {
			opt.Out = outputs[0].(*np.Array)
		}
		return u.CallValues(opt, inputs...)
	}))
	one := np.ScalarOf(int64(1))
	refcheck.Check(t, "lib_mixin_binary", mv(w.Binary("sub", one)).(*np.Array))
	refcheck.Check(t, "lib_mixin_reflected", mv(w.Reflected("sub", one)).(*np.Array))
	refcheck.Check(t, "lib_mixin_unary", mv(w.Unary("neg")).(*np.Array))
	two := np.ScalarOf(int64(2))
	refcheck.Check(t, "lib_mixin_inplace", mv(w.InPlace("add", two)).(*np.Array))
	refcheck.Check(t, "lib_mixin_inplace", w.Value().(*np.Array))
	q := mv(w.Binary("divmod", two)).([]*np.Array)
	refcheck.Check(t, "lib_mixin_divmod_q", q[0])
	refcheck.Check(t, "lib_mixin_divmod_r", q[1])
}
