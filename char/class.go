package char

import "github.com/SamuelSupe/numpygo"

func wrapChar(a *np.Array, err error) (*CharArray, error) {
	if err != nil {
		return nil, err
	}
	if !a.DType().IsCharacter() {
		return nil, np.ErrDType
	}
	return &CharArray{a}, nil
}

// Array-derived methods preserve the fixed-width chararray adapter. The
// underlying ndarray continues to implement layout, copy and writeback.
func (a *CharArray) SliceChar(indices ...np.Index) (*CharArray, error) {
	return wrapChar(a.Array.Slice(indices...))
}
func (a *CharArray) IndexChar(indices ...np.Index) (*CharArray, error) {
	return wrapChar(a.Array.Index(indices...))
}
func (a *CharArray) CopyChar() (*CharArray, error) { return wrapChar(a.Array.Copy()) }
func (a *CharArray) ReshapeChar(shape ...int) (*CharArray, error) {
	return wrapChar(a.Array.Reshape(shape...))
}
func (a *CharArray) TransposeChar(axes ...int) (*CharArray, error) {
	return wrapChar(a.Array.Transpose(axes...))
}
func (a *CharArray) Ravel() (*CharArray, error)   { return wrapChar(np.Ravel(a.Array)) }
func (a *CharArray) Flatten() (*CharArray, error) { return wrapChar(np.Flatten(a.Array)) }
func (a *CharArray) Repeat(n int, axis *int) (*CharArray, error) {
	return wrapChar(np.Repeat(a.Array, n, axis))
}
func (a *CharArray) Take(indices *np.Array, axis *int) (*CharArray, error) {
	return wrapChar(np.Take(a.Array, indices, axis))
}
func (a *CharArray) SwapAxes(x, y int) (*CharArray, error) {
	return wrapChar(np.SwapAxes(a.Array, x, y))
}
func (a *CharArray) Squeeze(axes ...int) (*CharArray, error) {
	return wrapChar(np.Squeeze(a.Array, axes...))
}
func (a *CharArray) Compress(condition *np.Array, axis *int) (*CharArray, error) {
	return wrapChar(np.Compress(condition, a.Array, axis))
}
func (a *CharArray) Diagonal(offset, axis1, axis2 int) (*CharArray, error) {
	return wrapChar(np.Diagonal(a.Array, offset, axis1, axis2))
}
func (a *CharArray) CastChar(d np.FlexibleDType, opt np.CastOptions) (*CharArray, error) {
	return wrapChar(a.Array.CastFlexibleWithOptions(d, opt))
}
func (a *CharArray) ViewChar(d np.FlexibleDType) (*CharArray, error) {
	return wrapChar(a.Array.ViewFlexible(d))
}

// Legacy class character operations use the same kernels as the namespace,
// but retain the character subclass for the string-producing methods.
func (a *CharArray) Capitalize() (*CharArray, error) { return wrapChar(Capitalize(a.Array)) }
func (a *CharArray) Upper() (*CharArray, error)      { return wrapChar(Upper(a.Array)) }
func (a *CharArray) Lower() (*CharArray, error)      { return wrapChar(Lower(a.Array)) }
func (a *CharArray) Title() (*CharArray, error)      { return wrapChar(Title(a.Array)) }
func (a *CharArray) SwapCase() (*CharArray, error)   { return wrapChar(SwapCase(a.Array)) }
func (a *CharArray) Center(w int, fill ...string) (*CharArray, error) {
	return wrapChar(Center(a.Array, w, fill...))
}
func (a *CharArray) LJust(w int, fill ...string) (*CharArray, error) {
	return wrapChar(LJust(a.Array, w, fill...))
}
func (a *CharArray) RJust(w int, fill ...string) (*CharArray, error) {
	return wrapChar(RJust(a.Array, w, fill...))
}
func (a *CharArray) ZFill(w int) (*CharArray, error)      { return wrapChar(ZFill(a.Array, w)) }
func (a *CharArray) ExpandTabs(n int) (*CharArray, error) { return wrapChar(ExpandTabs(a.Array, n)) }
func (a *CharArray) Strip(chars ...string) (*CharArray, error) {
	return wrapChar(Strip(a.Array, chars...))
}
func (a *CharArray) LStrip(chars ...string) (*CharArray, error) {
	return wrapChar(LStrip(a.Array, chars...))
}
func (a *CharArray) RStrip(chars ...string) (*CharArray, error) {
	return wrapChar(RStrip(a.Array, chars...))
}
func (a *CharArray) Replace(old, newText *np.Array, count int) (*CharArray, error) {
	return wrapChar(Replace(a.Array, old, newText, count))
}
func (a *CharArray) Partition(sep *np.Array) (*CharArray, error) {
	return wrapChar(Partition(a.Array, sep))
}
func (a *CharArray) RPartition(sep *np.Array) (*CharArray, error) {
	return wrapChar(RPartition(a.Array, sep))
}
func (a *CharArray) Translate(table map[rune]string) (*CharArray, error) {
	return wrapChar(Translate(a.Array, table))
}
func (a *CharArray) TranslateBytes(table []byte, deleteChars string) (*CharArray, error) {
	return wrapChar(TranslateBytes(a.Array, table, deleteChars))
}
func (a *CharArray) Add(b *np.Array, reflected bool) (*CharArray, error) {
	if reflected {
		return wrapChar(Add(b, a.Array))
	}
	return wrapChar(Add(a.Array, b))
}
func (a *CharArray) Multiply(n *np.Array) (*CharArray, error) { return wrapChar(Multiply(a.Array, n)) }
func (a *CharArray) Mod(b *np.Array) (*CharArray, error)      { return wrapChar(Mod(a.Array, b)) }
func (a *CharArray) Compare(b *np.Array, op string) (*np.Array, error) {
	return CompareCharArrays(a.Array, b, op, true)
}
func (a *CharArray) Argsort(axis *int) (*np.Array, error) { return np.Argsort(a.Array, axis) }

// Unlike string-producing transformations, the NumPy class' predicates,
// split, join and codec methods return ordinary ndarray results.
func (a *CharArray) IsAlpha() (*np.Array, error)   { return IsAlpha(a.Array) }
func (a *CharArray) IsAlnum() (*np.Array, error)   { return IsAlnum(a.Array) }
func (a *CharArray) IsDigit() (*np.Array, error)   { return IsDigit(a.Array) }
func (a *CharArray) IsLower() (*np.Array, error)   { return IsLower(a.Array) }
func (a *CharArray) IsUpper() (*np.Array, error)   { return IsUpper(a.Array) }
func (a *CharArray) IsTitle() (*np.Array, error)   { return IsTitle(a.Array) }
func (a *CharArray) IsSpace() (*np.Array, error)   { return IsSpace(a.Array) }
func (a *CharArray) IsDecimal() (*np.Array, error) { return IsDecimal(a.Array) }
func (a *CharArray) IsNumeric() (*np.Array, error) { return IsNumeric(a.Array) }
func (a *CharArray) Find(b *np.Array, bounds ...int) (*np.Array, error) {
	return Find(a.Array, b, bounds...)
}
func (a *CharArray) RFind(b *np.Array, bounds ...int) (*np.Array, error) {
	return RFind(a.Array, b, bounds...)
}

// FindIndex and RFindIndex are character search methods (ndarray Index is
// reserved for indexing a character array by Go index selectors).
func (a *CharArray) FindIndex(b *np.Array, bounds ...int) (*np.Array, error) {
	return Index(a.Array, b, bounds...)
}
func (a *CharArray) RFindIndex(b *np.Array, bounds ...int) (*np.Array, error) {
	return RIndex(a.Array, b, bounds...)
}
func (a *CharArray) StartsWith(b *np.Array, bounds ...int) (*np.Array, error) {
	return StartsWith(a.Array, b, bounds...)
}
func (a *CharArray) EndsWith(b *np.Array, bounds ...int) (*np.Array, error) {
	return EndsWith(a.Array, b, bounds...)
}
func (a *CharArray) Count(b *np.Array, bounds ...int) (*np.Array, error) {
	return Count(a.Array, b, bounds...)
}
func (a *CharArray) Encode(codec, policy string) (*np.Array, error) {
	return Encode(a.Array, codec, policy)
}
func (a *CharArray) Decode(codec, policy string) (*np.Array, error) {
	return Decode(a.Array, codec, policy)
}
func (a *CharArray) Join(seq *np.Array) (*np.Array, error) { return Join(a.Array, seq) }
func (a *CharArray) Split(sep *string, maxsplit int) (*np.Array, error) {
	return Split(a.Array, sep, maxsplit)
}
func (a *CharArray) RSplit(sep *string, maxsplit int) (*np.Array, error) {
	return RSplit(a.Array, sep, maxsplit)
}
func (a *CharArray) SplitLines(keepEnds bool) (*np.Array, error) {
	return SplitLines(a.Array, keepEnds)
}
