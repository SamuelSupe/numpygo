package lib

import (
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"regexp"
)

type DispatchInfo struct {
	Current, Available string
}

// OptFuncInfo reports actual portable Go dense arithmetic paths, rather
// than NumPy's C/SIMD CPU feature labels. Regex filters match operation
// names, dtype characters, or dtype names like NumPy opt_func_info.
func OptFuncInfo(funcName, signature string) (map[string]map[string]DispatchInfo, error) {
	fn, e := regexp.Compile(funcName)
	if e != nil {
		return nil, fmt.Errorf("%w: %v", np.ErrValue, e)
	}
	sig, e := regexp.Compile(signature)
	if e != nil {
		return nil, fmt.Errorf("%w: %v", np.ErrValue, e)
	}
	out := map[string]map[string]DispatchInfo{}
	// Mirrors the float32/64 denseLoop dispatch in numpygo operations.go.
	for _, name := range []string{"add", "subtract", "multiply", "divide"} {
		if !fn.MatchString(name) {
			continue
		}
		for _, d := range []struct{ code, typ string }{{"f", "float32"}, {"d", "float64"}} {
			if !sig.MatchString(d.code) && !sig.MatchString(d.typ) {
				continue
			}
			if out[name] == nil {
				out[name] = map[string]DispatchInfo{}
			}
			out[name][d.code+d.code+d.code] = DispatchInfo{Current: "go-dense (contiguous or scalar)", Available: "go-dense go-general"}
		}
	}
	return out, nil
}
