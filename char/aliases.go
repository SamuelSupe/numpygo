package char

import (
	"github.com/SamuelSupe/numpygo"
	s "github.com/SamuelSupe/numpygo/strings"
)

// The legacy namespace shares the modern string kernels wherever NumPy
// exposes the same behavior.
func StrLen(a *np.Array) (*np.Array, error)                   { return s.StrLen(a) }
func Add(a, b *np.Array) (*np.Array, error)                   { return s.Add(a, b) }
func Multiply(a, b *np.Array) (*np.Array, error)              { return s.Multiply(a, b) }
func Mod(a, b *np.Array) (*np.Array, error)                   { return s.Mod(a, b) }
func Upper(a *np.Array) (*np.Array, error)                    { return s.Upper(a) }
func Lower(a *np.Array) (*np.Array, error)                    { return s.Lower(a) }
func Title(a *np.Array) (*np.Array, error)                    { return s.Title(a) }
func Capitalize(a *np.Array) (*np.Array, error)               { return s.Capitalize(a) }
func SwapCase(a *np.Array) (*np.Array, error)                 { return s.SwapCase(a) }
func IsAlpha(a *np.Array) (*np.Array, error)                  { return s.IsAlpha(a) }
func IsAlnum(a *np.Array) (*np.Array, error)                  { return s.IsAlnum(a) }
func IsDigit(a *np.Array) (*np.Array, error)                  { return s.IsDigit(a) }
func IsLower(a *np.Array) (*np.Array, error)                  { return s.IsLower(a) }
func IsUpper(a *np.Array) (*np.Array, error)                  { return s.IsUpper(a) }
func IsTitle(a *np.Array) (*np.Array, error)                  { return s.IsTitle(a) }
func IsSpace(a *np.Array) (*np.Array, error)                  { return s.IsSpace(a) }
func IsNumeric(a *np.Array) (*np.Array, error)                { return s.IsNumeric(a) }
func IsDecimal(a *np.Array) (*np.Array, error)                { return s.IsDecimal(a) }
func Find(a, b *np.Array, bounds ...int) (*np.Array, error)   { return s.Find(a, b, bounds...) }
func RFind(a, b *np.Array, bounds ...int) (*np.Array, error)  { return s.RFind(a, b, bounds...) }
func Index(a, b *np.Array, bounds ...int) (*np.Array, error)  { return s.Index(a, b, bounds...) }
func RIndex(a, b *np.Array, bounds ...int) (*np.Array, error) { return s.RIndex(a, b, bounds...) }
func StartsWith(a, b *np.Array, bounds ...int) (*np.Array, error) {
	return s.StartsWith(a, b, bounds...)
}
func EndsWith(a, b *np.Array, bounds ...int) (*np.Array, error) { return s.EndsWith(a, b, bounds...) }
func Count(a, b *np.Array, bounds ...int) (*np.Array, error)    { return s.Count(a, b, bounds...) }
func Strip(a *np.Array, chars ...string) (*np.Array, error)     { return s.Strip(a, chars...) }
func LStrip(a *np.Array, chars ...string) (*np.Array, error)    { return s.LStrip(a, chars...) }
func RStrip(a *np.Array, chars ...string) (*np.Array, error)    { return s.RStrip(a, chars...) }
func Center(a *np.Array, width int, fill ...string) (*np.Array, error) {
	return s.Center(a, width, fill...)
}
func LJust(a *np.Array, width int, fill ...string) (*np.Array, error) {
	return s.LJust(a, width, fill...)
}
func RJust(a *np.Array, width int, fill ...string) (*np.Array, error) {
	return s.RJust(a, width, fill...)
}
func ZFill(a *np.Array, width int) (*np.Array, error)     { return s.ZFill(a, width) }
func ExpandTabs(a *np.Array, size int) (*np.Array, error) { return s.ExpandTabs(a, size) }
func Replace(a, old, newText *np.Array, count int) (*np.Array, error) {
	return s.Replace(a, old, newText, count)
}
func Encode(a *np.Array, codec, policy string) (*np.Array, error)     { return s.Encode(a, codec, policy) }
func Decode(a *np.Array, codec, policy string) (*np.Array, error)     { return s.Decode(a, codec, policy) }
func Translate(a *np.Array, table map[rune]string) (*np.Array, error) { return s.Translate(a, table) }
func TranslateBytes(a *np.Array, table []byte, deleteChars string) (*np.Array, error) {
	return s.TranslateBytes(a, table, deleteChars)
}
