// Package lib contains NumPy-style numerical library utilities.
package lib

import (
	"cmp"
	"github.com/SamuelSupe/numpygo"
	"regexp"
	"strconv"
	"strings"
)

// NumpyVersion mirrors the NumPy version comparator, including its
// historical one-digit alpha/beta/rc recognition and dev-version ordering.
type NumpyVersion struct {
	VString, Version     string
	Major, Minor, Bugfix int
	PreRelease           string
	IsDevVersion         bool
}

var coreVersion = regexp.MustCompile(`^([0-9]+)\.([0-9]+)\.([0-9]+)`)
var preVersion = regexp.MustCompile(`^(a|b|rc)[0-9]`)
var devVersion = regexp.MustCompile(`.dev`)

func NewNumpyVersion(s string) (NumpyVersion, error) {
	m := coreVersion.FindStringSubmatch(s)
	if m == nil {
		return NumpyVersion{}, np.ErrValue
	}
	v := NumpyVersion{VString: s, Version: m[0], IsDevVersion: devVersion.MatchString(s)}
	parts := []*int{&v.Major, &v.Minor, &v.Bugfix}
	for i, p := range parts {
		x, e := strconv.Atoi(m[i+1])
		if e != nil {
			return NumpyVersion{}, np.ErrValue
		}
		*p = x
	}
	tail := strings.TrimPrefix(s, v.Version)
	if tail == "" {
		v.PreRelease = "final"
	} else {
		v.PreRelease = preVersion.FindString(tail)
	}
	return v, nil
}
func (v NumpyVersion) String() string { return "NumpyVersion(" + v.VString + ")" }
func (v NumpyVersion) Compare(other any) (int, error) {
	var o NumpyVersion
	switch x := other.(type) {
	case string:
		var e error
		o, e = NewNumpyVersion(x)
		if e != nil {
			return 0, e
		}
	case NumpyVersion:
		o = x
	default:
		return 0, np.ErrValue
	}
	for _, x := range []int{cmp.Compare(v.Major, o.Major), cmp.Compare(v.Minor, o.Minor), cmp.Compare(v.Bugfix, o.Bugfix)} {
		if x != 0 {
			return x, nil
		}
	}
	if v.PreRelease != o.PreRelease {
		if v.PreRelease == "final" {
			return 1, nil
		}
		if o.PreRelease == "final" {
			return -1, nil
		}
		return cmp.Compare(v.PreRelease, o.PreRelease), nil
	}
	if v.IsDevVersion == o.IsDevVersion {
		return 0, nil
	}
	if v.IsDevVersion {
		return -1, nil
	}
	return 1, nil
}
