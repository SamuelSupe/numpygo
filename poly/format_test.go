package poly

import (
	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
	"testing"
)

func TestPrintStyleParity(t *testing.T) {
	m := refcheck.Require(t)
	newSeries := refcheck.RequireValue[*Series](t)
	old := DefaultPrintStyle()
	defer SetDefaultPrintStyle(old)
	for _, style := range []string{"ascii", "unicode"} {
		if e := SetDefaultPrintStyle(style); e != nil {
			t.Fatal(e)
		}
		for _, tc := range []struct {
			b    Basis
			name string
		}{{Power, "Polynomial"}, {Chebyshev, "Chebyshev"}, {Legendre, "Legendre"}, {Hermite, "Hermite"}, {HermiteE, "HermiteE"}, {Laguerre, "Laguerre"}} {
			s := newSeries(NewSeries(tc.b, m(np.WrapSlice([]float64{1, -2, 0, 3}, 4))))
			refcheck.Text(t, "final_poly_"+tc.name+"_"+style, s.String())
		}
		d := [2]float64{0, 2}
		s := newSeries(NewSeries(Power, m(np.WrapSlice([]float64{1, -2, 3}, 3)), SeriesOptions{Domain: &d}))
		refcheck.Text(t, "final_poly_domain_"+style, s.String())
	}
	refcheck.Error(t, "final_poly_invalid_style", "ValueError", SetDefaultPrintStyle("bad"), np.ErrValue)
}
