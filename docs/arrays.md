# Arrays and universal functions

The root package, `np`, provides the common array type used by the scientific subpackages.

## Create and access arrays

```go
a, err := np.FromSlice([]int64{10, 20, 30, 40, 50, 60}, 2, 3)
if err != nil {
    return err
}
fmt.Println(a.Shape()) // [2 3]
values, err := np.Values[int64](a)
if err != nil {
    return err
}
fmt.Println(values)
```

- `FromSlice[T]` copies a typed Go slice; `WrapSlice[T]` shares it. Shape must match the element count.
- `Empty`, `Zeros`, `Ones`, `Full`, `Eye`, `Arange`, `Linspace` and their typed/options variants construct arrays.
- `ScalarOf[T]` creates a zero-dimensional array. A reduction returning a single value still returns an array.
- `AtValue`, `SetValue` and `Values[T]` preserve exact typed values. Convenience conversion to complex128 is not lossless for integers above 2^53 or extended-precision data.
- `Astype` creates an independent C-order array. `AstypeWithOptions` provides casting rules, C/F/A/K order and conditional copy elision.

Arrays carry a dtype, shape, signed byte strides and offset into shared storage. The maximum rank is 64. The supported dtype families are described in [Dtypes and data interchange](io.md).

## Views, layout and ownership

Basic slicing (including negative steps), scalar axis selection, axis insertion, transpose and compatible reshape return views of existing storage. Advanced integer or Boolean indexing produces an independent result. `Base()` exposes the owner relationship; an advanced-index result may be a view of its independent copy.

```go
a, err := np.FromSlice([]int64{1, 2, 3, 4}, 2, 2)
if err != nil {
    return err
}
t, err := a.Transpose()
if err != nil {
    return err
}
if err := t.SetValue(int64(99), 1, 0); err != nil {
    return err
}
v, err := a.AtValue(0, 1) // int64(99)
if err != nil {
    return err
}
fmt.Println(v)
```

C-order and Fortran-order arrays are supported. Use `AsContiguousArray` or `AsFortranArray` when a packed layout is required. Broadcasting follows right-aligned shape compatibility; broadcast views are read-only. Mutation of overlapping shared storage is not safe concurrently without caller synchronization.

## Advanced indexing and assignment

```go
idx, err := np.FromSlice([]int64{2, 0}, 2)
if err != nil {
    return err
}
gathered, err := a.Index(np.Slice{}, idx) // a must have at least 3 columns
if err != nil {
    return err
}
mean, err := np.Mean(gathered, np.SumOptions{Axes: []int{1}})
if err != nil {
    return err
}
fmt.Println(mean)
```

Use `np.Slice{}`, `np.Reverse()`, `np.NewAxis`, integer indices, ellipsis, and array indices with the indexing methods. The advanced-axis placement follows NumPy's adjacent/separated-index rules. Assignment broadcasts its right-hand side and snapshots overlapping inputs when necessary. Repeated indices use deterministic logical C-order last-write behavior; use `UFunc.AtIndices` for sequential repeated updates.

## Arithmetic and ufunc options

Functions such as `Add`, `Subtract`, `Multiply`, `Divide`, `Exp`, `Log`, `Sin`, `Minimum`, `Maximum`, `Equal` and `IsClose` perform elementwise operations with dtype promotion and broadcasting.

```go
u, err := np.LookupUFunc("add")
if err != nil {
    return err
}
a, err := np.FromSlice([]float32{1, 2, 3}, 3)
if err != nil {
    return err
}
mask, err := np.FromSlice([]bool{true, false, true}, 3)
if err != nil {
    return err
}
out, err := np.Full(np.Float32, float32(-1), 3)
if err != nil {
    return err
}
result, err := u.CallValues(
    np.UFuncOptions{Out: out, Where: mask},
    a, np.Weak(0.5),
)
if err != nil {
    return err
}
values, err := np.Values[float32](result)
if err != nil {
    return err
}
fmt.Println(values) // [1.5 -1 3.5]
```

`Weak` expresses Python-like weak scalars; explicit Go numeric types express strong scalars. `CallWithOptions` supports dtype/signature selection, casting, order, `Where` and `Out` for supported kernels. Output buffers preserve identity, including non-contiguous writable views. Masked-off positions preserve an existing output value; without an output buffer they are safely zero-initialized.

Ufunc reductions support options for axes, keepdims, dtype, where, initial and output. With `ReduceWithOptions`, nil axes defaults to axis 0; use `AllAxes: true` to reduce every axis. With ordinary `SumOptions`, nil axes means all axes.

## Errors and diagnostics

Functions return Go errors for invalid shapes, indices, dtype combinations and unsupported options. Use `errors.Is` for sentinel error categories rather than matching messages.

`NewErrorContext`, `GetErr`, `SetErr` and callback/writer options provide explicit floating-point error policies for supported operations (divide by zero, overflow, underflow, invalid operation). Supported modes include ignore, warn, raise, call, print and log. An explicit context is preferable for independently configured concurrent work; this is a Go contract, not Python's thread-local error machinery.

`Array2String`, `ArrayRepr`, `ArrayStr`, print options and custom formatters provide NumPy-style display for supported dtypes. `WithPrintOptions` provides per-formatter configuration without changing defaults.
