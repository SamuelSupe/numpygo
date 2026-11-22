package lib

import (
	"testing"

	"github.com/SamuelSupe/numpygo"
	"github.com/SamuelSupe/numpygo/internal/refcheck"
)

func TestDocRegistryNumPy(t *testing.T) {
	check := func(e error) {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
	}
	f := &DocTarget{}
	doc := "  direct\n\tindent"
	check(AddDocString(f, doc))
	check(AddDocString(f, doc))
	refcheck.Text(t, "closure_doc_direct", *f.Documentation)
	refcheck.Error(t, "closure_doc_conflict", "RuntimeError", AddDocString(f, "different"), ErrDocConflict)
	refcheck.Error(t, "closure_doc_unsupported", "TypeError", AddDocString(nil, "x"), np.ErrDType)
	main := &DocTarget{SourceDefined: true, Members: map[string]*DocTarget{
		"first": {SourceDefined: true}, "second": {SourceDefined: true}, "last": {SourceDefined: true},
	}}
	reg := DocRegistry{"fixture": {"fn": main}}
	warnCount := int64(0)
	opt := DocOptions{Warn: func(string) { warnCount++ }}
	check(AddNewDoc(reg, "fixture", "fn", "\n    Summary\n      detail\n    final\n", opt))
	check(AddNewDoc(reg, "fixture", "fn", DocEntry{"first", "\tTitle\n\t  detail"}, opt))
	check(AddNewDoc(reg, "fixture", "fn", []DocEntry{{"first", "ignored conflict"}, {"second", "  Second"}}, opt))
	opt.NoWarn = true
	check(AddNewDoc(reg, "fixture", "fn", DocEntry{"last", "Last"}, opt))
	refcheck.Check(t, "closure_doc_warn_count", np.ScalarOf(warnCount))
	refcheck.Text(t, "closure_doc_main", *main.Documentation)
	for _, name := range []string{"first", "second", "last"} {
		refcheck.Text(t, "closure_doc_"+name, *main.Members[name].Documentation)
	}
	refcheck.Error(t, "closure_doc_missing_member", "AttributeError", AddNewDoc(reg, "fixture", "fn", DocEntry{"missing", "x"}, opt), np.ErrIndex)
	refcheck.Error(t, "closure_doc_missing_target", "AttributeError", AddNewDoc(reg, "fixture", "missing", "x", opt), np.ErrIndex)
	ro := &DocTarget{ReadOnly: true}
	reg["fixture"]["ro"] = ro
	check(AddNewDoc(reg, "fixture", "ro", "ignored", opt))
	if ro.Documentation != nil {
		t.Fatal("read-only target was changed")
	}
}
