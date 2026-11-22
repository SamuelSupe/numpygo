package np_test

import (
	"fmt"
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestLikeShapeAndOrder(t *testing.T) {
	m := refcheck.Require(t)
	base := m(m(np.Arange(0, 24, 1, np.Int64)).Reshape(2, 3, 4))
	for key, a := range map[string]*np.Array{
		"c": base, "f": m(np.AsFortranArray(base)), "transpose": m(base.Transpose(1, 2, 0)),
		"reverse": m(base.Slice(np.Slice{}, np.Reverse())), "scalar": np.ScalarOf(int64(2)),
		"empty": m(np.Zeros(np.Int64, 2, 0, 3)),
	} {
		for _, order := range []string{"C", "F", "A", "K"} {
			for sk, shape := range map[string][]int{"same": a.Shape(), "resize": {3, 5, 2}, "rank": {5, 2}, "scalar": {}} {
				tag := fmt.Sprintf("adv_like_%s_%s_%s", key, order, sk)
				t.Run(tag, func(t *testing.T) {
					m := refcheck.Require(t)
					out := m(np.EmptyLikeWithShape(a, np.Int16, order, shape))
					if out.Base() != nil {
						t.Fatal("empty_like must own its fresh allocation")
					}
					if err := out.Fill(int16(4)); err != nil {
						t.Fatal(err)
					}
					refcheck.CheckLayout(t, tag+"_empty", out)
					refcheck.CheckLayout(t, tag+"_zeros", m(np.ZerosLikeWithShape(a, np.Int16, order, shape)))
					refcheck.CheckLayout(t, tag+"_ones", m(np.OnesLikeWithShape(a, np.Int16, order, shape)))
					refcheck.CheckLayout(t, tag+"_full", m(np.FullLikeWithShape(a, 7, np.Int16, order, shape)))
				})
			}
		}
		refcheck.CheckLayout(t, "adv_like_"+key+"_default_zeros", m(np.ZerosLike(a)))
		refcheck.CheckLayout(t, "adv_like_"+key+"_default_ones", m(np.OnesLike(a)))
		refcheck.CheckLayout(t, "adv_like_"+key+"_default_full", m(np.FullLike(a, 7)))
	}
}
