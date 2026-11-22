package npyformat

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/SamuelSupe/numpygo/dtypes"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestDropMetadataNumPy(t *testing.T) {
	dm := refcheck.RequireValue[dtypes.Descriptor](t)
	base := dm(dtypes.Parse("i2")).WithMetadata(map[string]any{"unit": "ms"})
	d := dm(dtypes.Subarray(base, 2)).WithMetadata(map[string]any{"name": "x"})
	clean := DropMetadata(d)
	b := clean.Base()
	shape := []any{
		d.UserMetadata() != nil, d.Base().UserMetadata() != nil,
		clean.UserMetadata() == nil, b.UserMetadata() == nil,
		[]any{[]any{b.String(), b.ItemSize(), b.Flags(), b.Metadata().HasObject}, clean.Shape(), clean.ItemSize(), clean.Flags(), clean.Metadata().HasObject},
	}
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if e := enc.Encode(shape); e != nil {
		t.Fatal(e)
	}
	refcheck.Text(t, "closure_drop_metadata", strings.TrimSuffix(out.String(), "\n"))
}
