#!/usr/bin/env python3
"""Generate deterministic reference results using exactly NumPy 2.5.0."""
import gzip
import json
import pathlib
import warnings

import numpy as np

assert np.__version__ == "2.5.0", f"expected NumPy 2.5.0, got {np.__version__}"
arrays = {}
errors = {}


def record(name, value):
    value = np.asarray(value)
    arrays[name] = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "strides": list(value.strides),
        "values": [[str(int(v)), "0"] if value.dtype.kind in "biu" else
                   [repr(float(v.real)), repr(float(v.imag))] for v in value.flat],
    }


def fails(name, operation):
    try:
        operation()
    except Exception as exc:
        errors[name] = type(exc).__name__
    else:
        raise AssertionError(f"{name} unexpectedly succeeded")


base = np.arange(24.0).reshape(2, 3, 4)
record("array_base", base)
record("array_transpose", base.transpose(2, 0, 1))
record("array_reverse", base[::-1, 1:, ::-2])
record("array_empty_slice", base[:, 1:1, ::-1])
record("array_newaxis", base[None, ..., -1])
record("array_clipped_slice", base[-100:100:2, 2:-4:-1, 10::-2])
record("array_explicit_negative_stop", base[:, :, :-1:-1])
record("array_partial_reshape", base[:, ::2, :].reshape(2, 2, 2, 2))
record("array_scalar", np.array(3.25))
record("array_empty", np.zeros((2, 0, 3)))
record("array_empty_reshape", np.zeros((0,)).reshape(2, 0, 3))
record("array_stepped_reshape", np.arange(12.0)[::2].reshape(2, 3))
record("array_reverse_reshape", np.arange(6.0)[::-1].reshape(2, 3))
record("array_transpose_reshape", base.transpose(1, 0, 2).reshape(6, 4))
record("array_broadcast", np.broadcast_to(np.array([[1.0, 2.0, 3.0]]), (2, 3)))
mutated = base.copy()
mutated.transpose(2, 0, 1)[::-1, :, :][0, 1, 2] = 100.0
record("array_view_mutation", mutated)
mutated = np.arange(12.0)
mutated[::2].reshape(2, 3)[1, 2] = 88.0
record("array_reshape_mutation", mutated)
fails("array_bad_reshape", lambda: base.reshape(7))
fails("array_double_infer", lambda: base.reshape(-1, -1))
fails("array_ambiguous_empty", lambda: np.zeros(0).reshape(0, -1))
fails("array_axis", lambda: base.transpose(0, 0, 2))
fails("array_index", lambda: base[2])
fails("array_slice_step_zero", lambda: base[::0])
fails("array_readonly", lambda: np.broadcast_to(np.ones(1), (2,)).__setitem__(0, 1))
fails("array_bad_broadcast", lambda: np.broadcast_to(np.ones(2), (3,)))

real = np.arange(1.0, 7.0).reshape(2, 1, 3)
cplx = np.array([1 + 2j, -2 + .5j, 3 - 1j])
record("real_operands", real)
record("complex_operands", cplx)
record("array_add", np.add(real, cplx))
record("array_subtract", np.subtract(real, cplx))
record("array_multiply", np.multiply(real, cplx))
record("array_divide", np.divide(real, cplx))
record("array_complex_reverse_divide", np.divide(cplx, real))
record("array_scalar_broadcast", np.add(base[::-1], 0.5))
record("array_real_broadcast", np.multiply(base, np.array([[1.0], [2.0], [3.0]])))
record("array_empty_broadcast", np.add(np.zeros((2, 0, 3)), np.ones((1, 3))))
fails("array_arithmetic_shape", lambda: np.add(np.zeros(2), np.zeros(3)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    record("array_real_divzero", np.divide([0.0, 1.0, -2.0], [0.0, -0.0, 0.0]))
    record("array_complex_divzero", np.divide([1 + 0j, 1j, 1 + 1j, 0j], 0j))
    record("array_complex_nonfinite", np.divide(
        [1 + 1j, complex(np.inf, 0), 1 + 1j, 2 - 1j],
        [complex(np.inf, 0), 1 + 1j, complex(np.inf, np.inf), complex(np.nan, 1)],
    ))

record("sum_all", np.sum(base))
record("sum_last_keepdims", np.sum(base, axis=-1, keepdims=True))
record("sum_axes", np.sum(base[::-1, :, ::-1], axis=(2, 0)))
record("sum_axes_empty", np.sum(base, axis=()))
record("sum_complex", np.sum(real + cplx, axis=(0, 2)))
record("sum_empty", np.sum(np.zeros((2, 0, 3)), axis=1))
record("sum_scalar", np.sum(np.array(2.5)))
long = np.tile(np.array([1e16, 1., -1e16, 1.]), 128)
record("sum_pairwise", np.sum(long))
record("sum_pairwise_axis", np.sum(long.reshape(4, 128), axis=-1))
record("sum_slow_axis", np.sum(long.reshape(128, 4), axis=0))
fails("sum_bad_axis", lambda: np.sum(base, axis=3))
fails("sum_duplicate_axis", lambda: np.sum(base, axis=(0, -3)))

a = np.arange(12., dtype=float).reshape(2, 1, 3, 2).swapaxes(-1, -2)
b = (np.arange(24., dtype=float) / 3).reshape(1, 4, 3, 2)
record("matmul_input_a", np.arange(12., dtype=float).reshape(2, 1, 3, 2))
record("matmul_input_b", b)
record("matmul_batched", a @ b)
record("matmul_vv", np.arange(3.) @ np.array([1., 3., -1.]))
record("matmul_vm", np.arange(2.) @ a)
record("matmul_mv", a @ np.arange(3.))
record("matmul_empty_contract", np.empty((2, 0)) @ np.empty((0, 3)))
record("matmul_empty_batch", np.empty((0, 2, 3)) @ np.ones((3, 5)))
fails("matmul_scalar", lambda: np.matmul(np.array(2.0), np.ones((2, 2))))
fails("matmul_mismatch", lambda: np.ones((2, 3)) @ np.ones((2, 3)))

raw = np.array([3., 1., 2., 4., 4., 2., 1., 3.]).reshape(2, 1, 2, 2)
a = raw.swapaxes(-1, -2)
b = np.arange(1., 13.).reshape(1, 3, 2, 2)
record("solve_input_a", raw)
record("solve_input_b", b)
record("solve_matrix", np.linalg.solve(a, b))
record("solve_vector", np.linalg.solve(a, np.array([7., 11.])))
record("solve_rank2_rhs", np.linalg.solve(a, np.array([[1., 2.], [3., 4.]])))
record("solve_empty", np.linalg.solve(np.empty((0, 0)), np.empty((0,))))
record("solve_ill_conditioned", np.linalg.solve(np.diag([1e-20, 1.]), [1e-20, 2.]))
record("solve_pivot", np.linalg.solve(np.array([[0., 2.], [3., 4.]]), [4., 11.]))
record("solve_zero_rhs", np.linalg.solve(np.eye(2), np.empty((2, 0))))
fails("solve_singular", lambda: np.linalg.solve(np.array([[[2., 0.], [0., 2.]], [[1., 2.], [2., 4.]]]), np.ones(2)))
fails("solve_singular_zero_rhs", lambda: np.linalg.solve(np.array([[1., 2.], [2., 4.]]), np.empty((2, 0))))
fails("solve_bad_shape", lambda: np.linalg.solve(np.ones((2, 3)), np.ones(3)))
fails("solve_rhs_shape", lambda: np.linalg.solve(np.eye(2), np.ones(3)))

raw = np.array([1., 2., 3., 4., -2., 1., 1., 2., 2., 4., 3., 6.]).reshape(2, 3, 2)
record("svd_input", raw)
for name, x, full in [
    ("svd_full", raw.swapaxes(-1, -2), True),
    ("svd_reduced", raw, False),
    ("svd_empty_wide", np.empty((0, 3)), True),
    ("svd_empty_tall", np.empty((3, 0)), True),
    ("svd_empty_batch", np.empty((0, 3, 2)), False),
    ("svd_degenerate", np.eye(3), True),
]:
    u, s, vh = np.linalg.svd(x, full_matrices=full)
    record(name + "_u", u)
    record(name + "_s", s)
    record(name + "_vh", vh)
record("svd_values_only", np.linalg.svd(raw, compute_uv=False))
fails("svd_rank", lambda: np.linalg.svd(np.ones(3)))
fails("svd_nan", lambda: np.linalg.svd(np.array([[np.nan, 1.], [1., 1.]])))

signals = np.array([1., -2., 3., .5, 4., 2., 3., -1., 1., 6.]).reshape(2, 5)
record("fft_input", signals)
view = signals.T[::-1, :]
for norm in ("backward", "forward", "ortho"):
    record("fft_" + norm, np.fft.fft(view, n=7, axis=0, norm=norm))
    spec = np.fft.rfft(view, axis=0, norm=norm)
    record("rfft_" + norm, spec)
    record("irfft_" + norm, np.fft.irfft(spec, n=5, axis=0, norm=norm))
    record("ifft_" + norm, np.fft.ifft(np.fft.fft(view, axis=0, norm=norm), axis=0, norm=norm))
record("fft_default", np.fft.fft(signals))
record("fft_cropped_complex", np.fft.fft(np.array([1.+2j, 3.-4j, 5.+1j, -2.+1j, 9.+2j]), n=4))
record("ifft_complex", np.fft.ifft(np.array([1.+2j, 3.-4j, 5.+1j, -2.+1j, 9.+2j])))
record("rfft_even", np.fft.rfft(signals, n=6, axis=-1))
record("irfft_even_default", np.fft.irfft(np.fft.rfft(signals, n=6, axis=-1)))
record("irfft_real_spectrum", np.fft.irfft(np.array([1., 2., 3.]), n=5))
record("irfft_one", np.fft.irfft(np.array([3.+2j]), n=1))
record("fft_one", np.fft.fft(np.array([3.+2j])))
record("rfft_one", np.fft.rfft(np.array([3.])))
record("fft_empty_batch", np.fft.fft(np.ones((0, 4))))
record("fft_empty_padded", np.fft.fft(np.ones((2, 0)), n=4))
record("rfft_prime", np.fft.rfft(np.arange(13., dtype=float)))
record("fft_middle_axis", np.fft.fft(np.arange(24., dtype=float).reshape(2, 3, 4), axis=1, n=5))
record("irfft_padded_spectrum", np.fft.irfft(np.array([1. + 2j, 3. - 1j]), n=7))
record("irfft_cropped_spectrum", np.fft.irfft(np.array([1. + 2j, 3. - 1j, 4. + 5j]), n=2))
fails("fft_axis", lambda: np.fft.fft(signals, axis=2))
fails("fft_rank0", lambda: np.fft.fft(np.array(1.)))
fails("fft_zero_length", lambda: np.fft.fft(signals, n=0))
fails("fft_empty_length", lambda: np.fft.fft(np.zeros(0)))
fails("fft_bad_norm", lambda: np.fft.fft(signals, norm="unknown"))
fails("rfft_complex", lambda: np.fft.rfft(np.array([1.+2j])))
fails("irfft_bad_default", lambda: np.fft.irfft(np.array([1.+2j])))

from extended_reference import populate
populate(record, fails)
from io_reference import populate_io
populate_io(record)
from polynomial_reference import populate_polynomial
populate_polynomial(record, fails)
from core_extra_reference import populate_core_extra
populate_core_extra(record, fails)
from masked_reference import populate_masked
populate_masked(record)
from random_legacy_reference import populate_random_legacy
populate_random_legacy(record, fails)
from core_remaining_reference import populate_remaining_core
populate_remaining_core(record, fails)
from masked_remaining_reference import populate_masked_remaining
populate_masked_remaining(record, fails)
from datetime_reference import populate_datetime
populate_datetime(record, fails)
from final_reference import populate_final
populate_final(record, fails)
from behavior_reference import populate_behavior
populate_behavior(record, fails)
from parameter_reference import populate_parameter
populate_parameter(record, fails)
from parameter_matrix_reference import populate_matrix
populate_matrix(record, fails)
from parameter_cast_reference import populate_cast
populate_cast(record, fails)
from advanced_fft_reference import populate_fft
populate_fft(record, fails)
from advanced_random_reference import populate_random_advanced
populate_random_advanced(record, fails)
from advanced_linalg_reference import populate_linalg_advanced
populate_linalg_advanced(record, fails)
from unified_reference import populate_unified
populate_unified(record, fails)
from extended_math_reference import populate_extended_math
populate_extended_math(record, fails)
from record_interchange_reference import populate_records
populate_records(record, fails)
from flexible_reference import populate_flexible
populate_flexible(record, fails)
from strings_reference import populate_strings
populate_strings(record, fails)
from legacy_character_reference import populate_legacy_character
populate_legacy_character(record,fails)
from descriptor_reference import populate_descriptors
populate_descriptors(record,fails)
from auxiliary_reference import populate_auxiliary
populate_auxiliary(record,fails)
from library_reference import populate_library
populate_library(record,fails)
from class_reference import populate_classes
populate_classes(record,fails)
from series_reference import populate_series
populate_series(record,fails)
from subclass_reference import populate_subclasses
populate_subclasses(record,fails)
from numeric_class_reference import populate_numeric_classes
populate_numeric_classes(record,fails)
from ufunc_options_reference import populate_ufunc_options
populate_ufunc_options(record,fails)
from index_descriptor_reference import populate_index_descriptor
populate_index_descriptor(record,fails)
from final_closure_reference import populate_final_closure
populate_final_closure(record,fails)
from audit_completion_reference import populate_audit_completion
populate_audit_completion(record,fails)
from optimization_reference import populate_optimization
populate_optimization(record,fails)
from hotpaths_reference import populate_hotpaths
populate_hotpaths(record,fails)
from native_reference import populate_native
populate_native(record,fails)
from decomposition_reference import populate_decomposition
populate_decomposition(record,fails)
from spectral_reference import populate_spectral
populate_spectral(record,fails)
from array_fft_reference import populate_array_fft
populate_array_fft(record,fails)
from direct_reference import populate_direct
populate_direct(record,fails)
from precision_reference import populate_precision
populate_precision(record,fails)
from vector_reference import populate_vector
populate_vector(record,fails)
from register_reference import populate_register
populate_register(record,fails)

destination = pathlib.Path(__file__).resolve().parents[1] / "internal/refcheck/numpy-2.5.0.json.gz"
destination.parent.mkdir(parents=True, exist_ok=True)
payload = json.dumps({
    "version": np.__version__,
    "arrays": arrays,
    "errors": errors,
}, indent=2, allow_nan=False) + "\n"
with destination.open("wb") as output:
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as archive:
        archive.write(payload.encode("utf-8"))
print(f"generated {len(arrays)} reference arrays and {len(errors)} reference errors")
