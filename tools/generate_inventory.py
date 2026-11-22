#!/usr/bin/env python3
"""Enumerate NumPy 2.5.0 callable public symbols and link tested Go subsets."""
import csv
import gzip
import importlib
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.0"
ROOT = Path(__file__).resolve().parents[1]
with gzip.open(ROOT / "internal/refcheck/numpy-2.5.0.json.gz", "rt", encoding="utf-8") as f:
    reference = json.load(f)
fixture_names = sorted(set(reference["arrays"]) | set(reference["errors"]))
mapped = {}
adapted = set()


def add(symbol, go, evidence, notes=""):
    evidence = [evidence] if isinstance(evidence, str) else evidence
    fixtures = sorted({name for prefix in evidence for name in fixture_names
                       if name == prefix or name.startswith(prefix + "_")})
    if not fixtures:
        raise ValueError(f"no reference fixture for {symbol}: {evidence}")
    mapped[symbol] = (go, fixtures, notes)


def core(name, go, fixture, note=""):
    add("numpy." + name, "np." + go, fixture, note)

def equivalent(symbol, go, evidence, notes):
    add(symbol, go, evidence, notes)
    adapted.add(symbol)


# Explicit mapping from NumPy public names to reference cases. A mapping means
# the listed numerical behavior is covered, never that every Python option is implemented.
for name, go in {
    "add":"Add", "subtract":"Subtract", "multiply":"Multiply", "divide":"Divide",
    "floor_divide":"FloorDivide", "remainder":"Remainder", "power":"Power",
    "bitwise_and":"BitwiseAnd", "bitwise_or":"BitwiseOr", "bitwise_xor":"BitwiseXor",
}.items():
    core(name,go,"integer_"+("pow" if name=="power" else name),"explicit array operands; fixed-width numeric dtype")
for alias,target in [("true_divide","divide"),("mod","remainder"),("pow","power")]:
    if hasattr(np,alias): mapped["numpy."+alias]=mapped["numpy."+target]

unary = dict(zip(
    "sqrt exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin arccos arctan sinh cosh tanh arcsinh arccosh arctanh cbrt degrees radians rint floor ceil trunc negative positive square reciprocal absolute conjugate sign logical_not isnan isinf isfinite signbit invert".split(),
    "Sqrt Exp Exp2 Expm1 Log Log2 Log10 Log1p Sin Cos Tan Arcsin Arccos Arctan Sinh Cosh Tanh Arcsinh Arccosh Arctanh Cbrt Degrees Radians Rint Floor Ceil Trunc Negative Positive Square Reciprocal Absolute Conjugate Sign LogicalNot IsNaN IsInf IsFinite Signbit Invert".split()))
for name,go in unary.items():
    core(name,go,"math_"+name,"dtype-specific scalar kernel over strided arrays")
for alias,target in [("abs","absolute"),("conj","conjugate"),("bitwise_not","invert"),
                     ("acos","arccos"),("asin","arcsin"),("atan","arctan"),
                     ("acosh","arccosh"),("asinh","arcsinh"),("atanh","arctanh"),
                     ("deg2rad","radians"),("rad2deg","degrees")]:
    if hasattr(np,alias):mapped["numpy."+alias]=mapped["numpy."+target]
for name,go in dict(zip(
    "equal not_equal less less_equal greater greater_equal minimum maximum fmin fmax logical_and logical_or logical_xor hypot arctan2 copysign logaddexp clip isclose allclose".split(),
    "Equal NotEqual Less LessEqual Greater GreaterEqual Minimum Maximum FMin FMax LogicalAnd LogicalOr LogicalXor Hypot Arctan2 Copysign LogAddExp Clip IsClose AllClose".split())).items():
    core(name,go,"math_"+name,"strong array promotion; no out or where ufunc keyword")
if hasattr(np,"atan2"):mapped["numpy.atan2"]=mapped["numpy.arctan2"]

stats = dict(zip(
    "sum nansum prod nanprod mean nanmean var nanvar std nanstd min max nanmin nanmax median nanmedian ptp all any count_nonzero argmin argmax nanargmin nanargmax sort argsort cumsum cumprod nancumsum nancumprod quantile nanquantile percentile nanpercentile cov corrcoef".split(),
    "Sum NanSum Prod NanProd Mean NanMean Var NanVar Std NanStd Min Max NanMin NanMax Median NanMedian PTP All Any CountNonzero ArgMin ArgMax NanArgMin NanArgMax Sort Argsort Cumsum Cumprod NanCumsum NanCumprod Quantile NanQuantile Percentile NanPercentile Cov Corrcoef".split()))
for name,go in stats.items():
    note="axis/keepdims subset; no Python-specific out parameter"
    if name in ("quantile","nanquantile","percentile","nanpercentile"):
        note="one scalar q, linear method"
    if name in ("cov","corrcoef"):note="unweighted variables in rows"
    if name in ("sort","argsort"):note="stable ordering"
    core(name,go,"stats_"+name,note)
for alias,target in [("amin","min"),("amax","max"),("product","prod"),("cumproduct","cumprod")]:
    if hasattr(np,alias):mapped["numpy."+alias]=mapped["numpy."+target]

for name,go,fixture,note in [
    ("array","FromSlice/FromFloat64","array_base","typed Go slices; explicit shape"),
    ("asarray","WrapSlice","array_base","typed Go slice sharing; no Python object inference"),
    ("zeros","Zeros","array_empty","C order"),("ones","Ones","create_ones","C order"),
    ("full","Full","create_full","scalar fill"),("eye","Eye","create_eye","C order"),
    ("identity","Identity","create_identity","C order"),
    ("arange","ArangeInt64/Arange",["create_arange_int","create_arange_float"],"explicit dtype and bounds"),
    ("linspace","Linspace","create_linspace","scalar endpoints"),
    ("logspace","Logspace","create_logspace","scalar endpoints"),
    ("zeros_like","ZerosLike","manip_zeros_like","C order"),
    ("ones_like","OnesLike","manip_ones_like","C order"),
    ("full_like","FullLike","manip_full_like","scalar fill"),
    ("reshape","Array.Reshape","array_stepped_reshape","C order"),
    ("transpose","Array.Transpose","array_transpose","view"),
    ("copy","Array.Copy","array_base","C order copy"),
    ("broadcast_to","Array.BroadcastTo","array_broadcast","read-only view"),
    ("broadcast_shapes","BroadcastShapes","array_add","broadcast shape resolution"),
    ("result_type","PromoteTypes","promote","strong array promotion only"),
    ("promote_types","PromoteTypes","promote","14 supported dtypes"),
    ("astype","Array.Astype","uint64_cast_int64","typed conversion"),
    ("take","Take","advanced_take","raise on out-of-bounds"),
    ("nonzero","Nonzero","advanced_nonzero","integer coordinate arrays"),
    ("flatnonzero","FlatNonzero","advanced_flatnonzero","C order"),
    ("where","Where","advanced_where","three-argument form"),
    ("concatenate","Concatenate","manip_concat","axis is explicit"),
    ("stack","Stack","manip_stack","axis is explicit"),
    ("ravel","Ravel","manip_ravel","C order"),
    ("expand_dims","ExpandDims","manip_expand","single axis"),
    ("squeeze","Squeeze","manip_squeeze","view"),
    ("swapaxes","SwapAxes","manip_swap","view"),
    ("moveaxis","MoveAxis","manip_move","single axis"),
    ("flip","Flip","manip_flip","view"),
    ("roll","Roll","manip_roll","single axis or flatten"),
    ("repeat","Repeat","manip_repeat","uniform repeats"),
    ("tile","Tile","manip_tile","scalar repetitions per axis"),
    ("diff","Diff","manip_diff","no prepend/append"),
    ("diagonal","Diagonal","manip_diagonal","read-only view"),
    ("trace","Trace","manip_trace","explicit axes"),
    ("dot","Dot","contract_dot","typed dense arrays"),
    ("inner","Inner","contract_inner","typed dense arrays"),
    ("outer","Outer","contract_outer","C-order flatten"),
    ("vdot","VDot","contract_vdot","C-order flatten with conjugation"),
    ("kron","Kron","contract_kron","typed dense arrays"),
    ("tensordot","Tensordot","contract_tensordot","explicit axis lists"),
    ("einsum","Einsum","contract_einsum","string subscripts, no optimization/out"),
    ("cross","Cross",["linalg_cross","core_cross2"],"three-component vectors; explicit axes"),
    ("unique","Unique","unique","flattened, sorted, optional equal_nan"),
    ("unique_values","UniqueValues","unique_values_separate","flattened sorted values"),
    ("isin","Isin","set_isin","elementwise membership"),
    ("union1d","Union1D","set_union1d","flattened set operation"),
    ("intersect1d","Intersect1D","set_intersect1d","flattened set operation"),
    ("setdiff1d","Setdiff1D","set_setdiff1d","flattened set operation"),
    ("setxor1d","Setxor1D","set_setxor1d","flattened set operation"),
    ("searchsorted","SearchSorted","search","sorted one-dimensional input"),
    ("digitize","Digitize","digitize","monotonic bins"),
    ("take_along_axis","TakeAlongAxis","along_take","integer indices"),
    ("put_along_axis","PutAlongAxis","along_put","integer indices and broadcast values"),
    ("average","Average/AverageWithWeights","sample_average","matching weights or one reduction axis"),
    ("bincount","Bincount","sample_bincount","nonnegative integers and optional weights"),
    ("histogram","Histogram","sample_hist","numeric bins, weights, density"),
    ("gradient","Gradient","sample_gradient","one axis, uniform spacing"),
    ("trapezoid","Trapezoid","sample_trapezoid","one axis, scalar spacing or coordinates"),
    ("interp","Interp","sample_interp","ascending grid, no period"),
    ("load","LoadNPY/LoadNPZ","io_npy","numeric NPY/NPZ streams, no pickle"),
    ("save","SaveNPY","io_npy","numeric NPY streams"),
    ("savez","SaveNPZ","io_npy","stored named numeric arrays"),
    ("savez_compressed","SaveNPZ","io_npy","compressed named numeric arrays"),
]:
    core(name,go,fixture,note)
for name in ["bool","bool_","int8","int16","int32","int64","uint8","uint16","uint32","uint64","float16","float32","float64","complex64","complex128"]:
    if hasattr(np,name):
        dtype="bool" if name=="bool_" else name
        core(name,"DType/ScalarOf/FromSlice","promote_"+dtype+"_"+dtype,"typed Go scalar/storage; Python scalar object semantics differ")
for name,dtype in [("int_","int64"),("intp","int64"),("uint","uint64"),("uintp","uint64"),("double","float64"),("single","float32"),("half","float16"),("csingle","complex64"),("cdouble","complex128"),("byte","int8"),("ubyte","uint8"),("short","int16"),("ushort","uint16"),("intc","int32"),("uintc","uint32"),("longlong","int64"),("ulonglong","uint64")]:
    if hasattr(np,name):
        core(name,"DType/ScalarOf/FromSlice","promote_"+dtype+"_"+dtype,"Linux/amd64 fixed-width dtype alias")

for name,go,fixture,note in [
    ("matmul","MatMul",["matmul_batched","complex_matmul","integer_matmul","bool_matmul"],"all supported dtypes; batch broadcast"),
    ("solve","Solve",["solve","complex_solve"],"float32/64 and complex64/128; integer input promoted"),
    ("svd","SVD",["svd","complex_svd"],"full/reduced/values-only; finite matrices"),
    ("inv","Inv","complex_inv","batched inverse"),
    ("det","Det","complex_det","finite batched matrix"),
    ("slogdet","Slogdet","complex_slogdet","finite batched matrix"),
    ("cholesky","Cholesky","complex_cholesky","upper/lower Hermitian"),
    ("qr","QR","complex_qr","reduced/complete/R only; no raw format"),
    ("pinv","Pinv","complex_pinv","scalar rcond"),
    ("matrix_rank","MatrixRank","complex_rank","scalar tolerance"),
    ("eig","Eig","eig_complex","finite diagonalizable complex matrices; ordering arbitrary"),
    ("eigvals","Eigvals","eigvals","finite diagonalizable complex matrices"),
    ("eigh","Eigh","eigh_complex","Hermitian eigenvectors; upper/lower"),
    ("eigvalsh","Eigvalsh","eigvalsh","Hermitian eigenvalues"),
    ("matrix_power","MatrixPower","matrix_power","integer exponent"),
    ("lstsq","Lstsq","lstsq","rank-two A, vector/matrix b; scalar rcond"),
    ("norm","Norm","norm","vector/matrix orders and axes"),
    ("cond","Cond","linalg_cond","spectral or standard matrix norms"),
    ("cross","Cross","linalg_cross","three-component vectors"),
    ("matrix_norm","MatrixNorm","linalg_matrix_norm","last two axes"),
    ("vector_norm","VectorNorm","linalg_vector_norm","arbitrary reduction axes"),
    ("matrix_transpose","MatrixTranspose","linalg_matrix_transpose","swap final two axes"),
    ("multi_dot","MultiDot","linalg_multidot","optimal-order chain"),
    ("svdvals","SVDVals","linalg_svdvals","singular values"),
    ("tensorinv","TensorInv","linalg_tensorinv","square reshaped matrix"),
    ("tensorsolve","TensorSolve","linalg_tensorsolve","square reshaped matrix"),
    ("vecdot","Vecdot","linalg_vecdot","complex conjugate dot"),
]:
    add("numpy.linalg."+name,"linalg."+go,fixture,note)
add("numpy.matmul","linalg.MatMul",["matmul_batched","integer_matmul","complex_matmul"],"explicit array operands")
add("numpy.vecdot","linalg.Vecdot","linalg_vecdot","complex conjugate dot")
add("numpy.linalg.LinAlgError","linalg.ErrSingular/ErrConvergence/ErrNotPositiveDefinite",["solve_singular","svd_nan","complex_not_pd"],"Go errors.Is categories")
if hasattr(np,"matrix_transpose"):
    mapped["numpy.matrix_transpose"]=mapped["numpy.linalg.matrix_transpose"]
for name in ("tensordot","outer","diagonal","trace"):
    if hasattr(np.linalg,name): mapped["numpy.linalg."+name]=mapped["numpy."+name]

for name,go,fixture in [
    ("fft","FFT","fft_default"),("ifft","IFFT","ifft_complex"),("rfft","RFFT","rfft_prime"),("irfft","IRFFT","irfft_even_default"),
    ("fftn","FFTN","fft_nd"),("ifftn","IFFTN","ifft_nd"),("fft2","FFT2","fft_2"),("ifft2","IFFT2","ifft_2"),
    ("rfftn","RFFTN","rfft_nd"),("irfftn","IRFFTN","irfft_nd"),("rfft2","RFFT2","rfft_2"),("irfft2","IRFFT2","irfft_2"),
    ("hfft","HFFT","fft_hfft"),("ihfft","IHFFT","fft_ihfft"),("fftfreq","FFTFreq","fft_fftfreq"),
    ("rfftfreq","RFFTFreq","fft_rfftfreq"),("fftshift","FFTShift","fft_shift"),("ifftshift","IFFTShift","fft_ishift")]:
    add("numpy.fft."+name,"fft."+go,fixture,"CPU numeric arrays")

for symbol,go,fixture,note in [
    ("default_rng","DefaultRNG","rng_raw","uint64 seed; PCG64"),
    ("PCG64","NewPCG64","rng_raw","bit generator stream"),
    ("PCG64DXSM","NewPCG64DXSM","rng_dxsm_raw","bit generator stream"),
    ("SeedSequence","Seed/NewSeedSequence","rng_seed_words","uint32 entropy words; default pool"),
    ("SeedSequence.generate_state","SeedSequence.GenerateState32/64","rng_seed32","32/64-bit words"),
    ("SeedSequence.spawn","SeedSequence.Spawn","rng_spawn","child counter persists"),
    ("Generator","Generator","rng_random","PCG64/PCG64DXSM"),
]:
    add("numpy.random."+symbol,"random."+go,fixture,note)
for cls in ("PCG64","PCG64DXSM"):
    for method,go,fixture in [
        ("random_raw","RandomRaw","rng_raw" if cls=="PCG64" else "rng_dxsm_raw"),
        ("advance","Advance128","rng_advance_pcg" if cls=="PCG64" else "rng_advance_dxsm"),
        ("state","State/SetState","rng_cache"),
    ]:
        add("numpy.random."+cls+"."+method,"random.Generator."+go,fixture,"PCG state including cached uint32")
for name,go,fixture in [
    ("random","Random/RandomDType","rng_random"),("integers","IntegersDType","rng_integer"),
    ("normal","Normal","rng_normal"),("standard_normal","StandardNormal/DType","rng_standard_normal"),
    ("standard_exponential","StandardExponential/DType/Inverse","rng_standard_exponential"),
    ("uniform","Uniform","rng_uniform"),("exponential","Exponential","rng_exponential"),
    ("gamma","Gamma","rng_gamma"),("standard_gamma","StandardGamma","rng_standard_gamma"),
    ("beta","Beta","rng_beta"),("chisquare","Chisquare","rng_chisquare"),("f","F","rng_f"),
    ("standard_cauchy","StandardCauchy","rng_standard_cauchy"),("standard_t","StandardT","rng_standard_t"),
    ("pareto","Pareto","rng_pareto"),("weibull","Weibull","rng_weibull"),("power","Power","rng_power"),
    ("laplace","Laplace","rng_laplace"),("gumbel","Gumbel","rng_gumbel"),("logistic","Logistic","rng_logistic"),
    ("lognormal","Lognormal","rng_lognormal"),("rayleigh","Rayleigh","rng_rayleigh"),
    ("poisson","Poisson","rng_poisson"),("negative_binomial","NegativeBinomial","rng_negative_binomial"),
    ("binomial","Binomial","rng_binomial"),("choice","Choice/ChoiceArray","rng_choice"),
    ("permutation","Permutation/PermutationArray","rng_permutation"),("shuffle","Shuffle","rng_shuffle")]:
    add("numpy.random.Generator."+name,"random.Generator."+go,fixture,"seeded PCG streams; scalar distribution parameters")
for name,go,fixture,note in [
    ("bytes","Bytes","rng_bytes","little-endian byte stream"),
    ("spawn","Spawn","rng_generator_spawn","child SeedSequence state"),
    ("bit_generator","State/SetState/Advance128","rng_advance_pcg","Go generator owns its bit generator"),
    ("geometric","Geometric","rng_geometric","scalar p"),
    ("triangular","Triangular","rng_triangular","scalar parameters"),
    ("wald","Wald","rng_wald","scalar parameters"),
    ("logseries","Logseries","rng_logseries","scalar p"),
    ("zipf","Zipf","rng_zipf","scalar exponent"),
    ("noncentral_chisquare","NoncentralChisquare","rng_noncentral_chisquare","scalar parameters"),
    ("noncentral_f","NoncentralF","rng_noncentral_f","scalar parameters"),
    ("vonmises","Vonmises","rng_vonmises","scalar parameters"),
    ("multinomial","Multinomial","rng_multinomial","one probability vector"),
    ("dirichlet","Dirichlet","rng_dirichlet","one concentration vector"),
    ("multivariate_normal","MultivariateNormal","rng_multivariate_normal","Cholesky stream validated; other factorizations may change draws"),
    ("permuted","Permuted","rng_permuted","independent axis permutations"),
    ("hypergeometric","Hypergeometric","rng_hypergeometric","scalar counts; direct and ratio-of-uniforms branches"),
    ("multivariate_hypergeometric","MultivariateHypergeometric","rng_multivariate_hypergeometric","marginals and count methods"),
]:
    add("numpy.random.Generator."+name,"random.Generator."+go,fixture,note)
# Bit generator families share the Go sampling surface but maintain distinct
# state and exact underlying streams.
for cls in ("MT19937","Philox","SFC64","PCG64","PCG64DXSM"):
    tag="bit_"+cls.lower()+"_"
    ns="numpy.random."+cls
    add(ns,"random.New"+cls,tag+"raw","SeedSequence-based constructor; exact raw integer stream")
    for method,go,fixture in [
        ("random_raw","RandomRaw","raw"),("state","State/SetState","state"),
        ("spawn","Spawn","spawn"),("seed_seq","SeedSequence","seed_entropy"),
        ("jumped","Jumped","jumped"),("advance","Advance256","advance"),
        ("lock","WithLock","lock"),
    ]:
        if hasattr(getattr(np.random,cls),method):
            add(ns+"."+method,"random.Generator."+go,tag+fixture,"algorithm-specific state, seeded stream; exact integer comparisons")
for method,go,fixture in [
    ("random_raw","RandomRaw","raw"),("state","State/SetState","state"),
    ("spawn","Spawn","spawn"),("seed_seq","SeedSequence","seed_entropy"),("lock","WithLock","lock"),
]:
    add("numpy.random.BitGenerator."+method,"random.Generator."+go,
        ["bit_"+c+"_"+fixture for c in ("mt19937","philox","sfc64","pcg64","pcg64dxsm")],"concrete Go generators; no C ABI")
for name,go,fixture in [
    ("entropy","Entropy","entropy"),("pool","Pool","pool"),("pool_size","PoolSize","pool_size"),
    ("spawn_key","SpawnKey","spawn_key"),("n_children_spawned","NChildrenSpawned","children"),
    ("state","State","pool"),
]:
    add("numpy.random.SeedSequence."+name,"random.SeedSequence."+go,
        ["bit_seed_parent_"+fixture,"bit_seed_child_"+fixture],"default-size entropy pool; uint32 word representation")
add("numpy.random.Generator","random.Generator",["rng_random","bit_mt19937_double","bit_philox_double","bit_sfc64_double"],"five supported bit generators")

# Legacy RandomState has its own MT seeding, Gaussian cache, integer masking
# and frozen distribution algorithms. Module-level NumPy calls correspond to
# the process singleton, independently seeded in differential tests.
legacy = dict(zip(
    "random random_sample rand randn standard_normal normal uniform standard_exponential exponential gamma standard_gamma beta chisquare f standard_cauchy standard_t pareto weibull power laplace gumbel logistic lognormal rayleigh poisson negative_binomial binomial geometric triangular wald logseries zipf noncentral_chisquare noncentral_f vonmises hypergeometric multinomial dirichlet multivariate_normal randint random_integers tomaxint choice bytes permutation shuffle".split(),
    "Random RandomSample Rand Randn StandardNormal Normal Uniform StandardExponential Exponential Gamma StandardGamma Beta Chisquare F StandardCauchy StandardT Pareto Weibull Power Laplace Gumbel Logistic Lognormal Rayleigh Poisson NegativeBinomial Binomial Geometric Triangular Wald Logseries Zipf NoncentralChisquare NoncentralF Vonmises Hypergeometric Multinomial Dirichlet MultivariateNormal Randint RandomIntegers ToMaxInt Choice Bytes Permutation Shuffle".split()))
for name,go in legacy.items():
    note="legacy MT19937 stream and scalar parameters"
    if name=="multivariate_normal":note="legacy SVD transform; tested diagonal covariance and signs may vary for general SVD"
    if name=="choice":note="integer population, weights and replacement options"
    add("numpy.random.RandomState."+name,"random.RandomState."+go,"legacy_"+name,note)
    if hasattr(np.random,name):
        add("numpy.random."+name,"random.GlobalRandomState()."+go,"legacy_global_"+name,note+"; shared singleton")
for name,go,fixture in [("seed","Seed/SeedArray","reseed"),("get_state","GetState","state"),("set_state","SetState","state_restored")]:
    add("numpy.random.RandomState."+name,"random.RandomState."+go,"legacy_"+fixture,"MT19937 state with Gaussian cache")
    add("numpy.random."+name,"random.GlobalRandomState()."+go,"legacy_global_"+fixture,"shared singleton with Gaussian cache")
for name in ("ranf","sample"):
    add("numpy.random."+name,"random.GlobalRandomState().RandomSample","legacy_global_"+name,"legacy shared stream")
add("numpy.random.RandomState","random.NewRandomState/NewRandomStateArray",["legacy_random_sample","legacy_array_seed"],"legacy initialization and distribution stream")

for name,go,fixture in [
    ("__getitem__","Array.Index/Slice","advanced_adjacent"),("__setitem__","Array.Assign/SetValue","advanced_assignment"),
    ("shape","Array.Shape","array_base"),("strides","Array.Strides","array_reverse"),
    ("ndim","Array.NDim","array_base"),("size","Array.Size","array_base"),("dtype","Array.DType","promote"),
    ("itemsize","DType.ItemSize","array_base"),("flags","Array.IsCContiguous/IsFContiguous/Writable","array_transpose"),
    ("T","Array.Transpose","array_transpose"),
    ("item","Array.AtValue","uint64_input"),("itemset","Array.SetValue","advanced_assignment"),
    ("astype","Array.Astype","uint64_cast_int64"),("reshape","Array.Reshape","array_partial_reshape"),
    ("transpose","Array.Transpose","array_transpose"),("copy","Array.Copy","array_base"),
    ("flatten","Flatten","manip_flatten"),("ravel","Ravel","manip_ravel"),
    ("squeeze","Squeeze","manip_squeeze"),("swapaxes","SwapAxes","manip_swap"),("diagonal","Diagonal","manip_diagonal"),
    ("trace","Trace","manip_trace"),("take","Take","advanced_take"),("nonzero","Nonzero","advanced_nonzero"),
    ("repeat","Repeat","manip_repeat"),("dot","Dot","contract_dot"),("conjugate","Conjugate","math_conjugate"),
    ("conj","Conjugate","math_conjugate"),("clip","Clip","math_clip"),
]:
    if hasattr(np.ndarray,name):
        add("numpy.ndarray."+name,"np."+go,fixture,"Go API adaptation")
for name,go in stats.items():
    if hasattr(np.ndarray,name):
        add("numpy.ndarray."+name,"np."+go,"stats_"+name,"functional Go equivalent; options subset")
core("ndarray","Array","array_base","shared numerical storage and strided views")
core("dtype","DType","promote","14 fixed-width numeric dtypes")

# Core routines, including view behavior, dtype-sensitive kernels, and
# deterministic examples of tuple-returning interfaces.
extra = dict(zip(
    "angle round fabs fix float_power fmod divmod frexp ldexp gcd lcm heaviside i0 sinc logaddexp2 modf nextafter spacing nan_to_num isposinf isneginf iscomplex isreal iscomplexobj isrealobj bitwise_count left_shift right_shift real imag real_if_close ascontiguousarray asfortranarray isfortran array_equal array_equiv copyto shares_memory may_share_memory broadcast_arrays append hstack vstack dstack column_stack split array_split hsplit vsplit dsplit unstack rot90 rollaxis resize compress extract argwhere delete choose select put putmask place partition argpartition lexsort tri tril triu diag diagflat diag_indices diag_indices_from tril_indices triu_indices tril_indices_from triu_indices_from fill_diagonal vander indices meshgrid ix_ ravel_multi_index unravel_index unique_all unique_counts unique_inverse trim_zeros ediff1d bartlett blackman hamming hanning kaiser geomspace convolve correlate packbits unpackbits sort_complex unwrap pad insert".split(),
    "Angle Round Fabs Fix FloatPower Fmod Divmod Frexp Ldexp GCD LCM Heaviside I0 Sinc LogAddExp2 Modf Nextafter Spacing NanToNum IsPosInf IsNegInf IsComplex IsReal IsComplexObj IsRealObj BitwiseCount LeftShift RightShift Real Imag RealIfClose AsContiguousArray AsFortranArray IsFortran ArrayEqual ArrayEquiv CopyTo SharesMemory MayShareMemory BroadcastArrays Append HStack VStack DStack ColumnStack Split/SplitAt ArraySplit HSplit VSplit DSplit Unstack Rot90 RollAxis Resize Compress Extract Argwhere Delete Choose Select Put Putmask Place Partition ArgPartition Lexsort Tri Tril Triu Diag Diagflat DiagIndices DiagIndicesFrom TrilIndices TriuIndices TrilIndicesFrom TriuIndicesFrom FillDiagonal Vander DenseIndices/Indices Meshgrid Ix RavelMultiIndex UnravelIndex UniqueAll UniqueCounts UniqueInverse TrimZeros Ediff1D Bartlett Blackman Hamming Hanning Kaiser Geomspace Convolve Correlate Packbits Unpackbits SortComplex Unwrap Pad Insert".split()))
for name, go in extra.items():
    fixture = "extra_" + ({"ix_":"ix"}.get(name,name))
    notes = {
        "real":"write-through real component view",
        "imag":"write-through complex component view; non-complex gives read-only zeros",
        "broadcast_arrays":"explicit arrays; read-only broadcast views",
        "copyto":"broadcast and where mask; casting options not exposed",
        "partition":"valid partition order; full sorting implementation",
        "argpartition":"valid partition indices; full sorting implementation",
        "pad":"constant, edge, wrap, reflect, symmetric; scalar constant; default even reflection",
        "insert":"scalar and vector integer indices; explicit axis",
        "split":"equal sections and explicit cut indices",
        "nan_to_num":"out-of-place numeric replacement",
        "choose":"raise-mode integer choices",
        "select":"ordered boolean conditions with explicit default",
        "geomspace":"scalar endpoints, explicit dtype",
        "unwrap":"one explicit axis and period",
        "unique_all":"flattened numeric array",
        "unique_counts":"flattened numeric array",
        "unique_inverse":"flattened numeric values and inverse with original shape",
    }.get(name,"numeric array behavior in listed fixtures")
    core(name,go,[fixture,"extra_split_at"] if name=="split" else fixture,notes)
for n in (1,2,3):core(f"atleast_{n}d",f"Atleast{n}D",f"extra_atleast_{n}","one numeric array")
for name,go in [("shape","Array.Shape"),("size","Array.Size/Shape"),("ndim","Array.NDim"),
                ("fliplr","Flip(axis=1)"),("flipud","Flip(axis=0)"),("can_cast","CanCast"),("isdtype","IsDType")]:
    core(name,go,"extra_"+name,"fixed-width numeric array/dtype")
for name,go in [("matvec","MatVec"),("vecmat","VecMat")]:
    add("numpy."+name,"linalg."+go,"extra_"+name,"trailing matrix/vector axes, batched broadcast; vecmat conjugates vector")
add("numpy.ndarray.nbytes","np.Array.NBytes","extra_nbytes","logical element count times dtype itemsize")
add("numpy.ndarray.mT","linalg.MatrixTranspose","linalg_matrix_transpose","swap final two axes")
for alias,target in [("around","round"),("bitwise_left_shift","left_shift"),("bitwise_right_shift","right_shift"),
                     ("bitwise_invert","invert"),("concat","concatenate"),("permute_dims","transpose"),
                     ("cumulative_sum","cumsum"),("cumulative_prod","cumprod")]:
    if hasattr(np,alias):mapped["numpy."+alias]=mapped["numpy."+target]
for name,go,evidence in [
    ("real","Real","extra_real"),("imag","Imag","extra_imag"),("fill","Array.Fill","extra_fill"),
    ("round","Round","extra_round"),("searchsorted","SearchSorted","search_sorted"),
    ("partition","Partition","extra_partition"),("argpartition","ArgPartition","extra_argpartition"),
    ("choose","Choose","extra_choose"),("compress","Compress","extra_compress"),("put","Put","extra_put"),
]:
    if hasattr(np.ndarray,name):
        # Partition/Choose and other ndarray forms are functional Go adaptations.
        if name=="searchsorted":
            mapped["numpy.ndarray."+name]=mapped["numpy.searchsorted"]
        else:add("numpy.ndarray."+name,"np."+go,evidence,"Go functional or explicit mutation API")

# Numeric masked arrays. For masked entries the reference compares the mask
# and canonicalized valid values, not unspecified underlying data.
masked = dict(zip(
    "sqrt log log2 log10 arcsin arccos arccosh arctanh tan exp conjugate sin cos arctan arcsinh sinh cosh tanh absolute fabs negative floor ceil logical_not add subtract multiply divide floor_divide remainder fmod power arctan2 equal not_equal less less_equal greater greater_equal logical_and logical_or logical_xor hypot minimum maximum bitwise_and bitwise_or bitwise_xor left_shift right_shift angle round clip sum prod min max all any ptp mean median var std count count_masked argmin argmax cumsum cumprod anomalies average allclose allequal filled compressed getdata getmask getmaskarray is_masked is_mask make_mask make_mask_none mask_or masked_all masked_all_like masked_where masked_equal masked_not_equal masked_greater masked_greater_equal masked_less masked_less_equal masked_inside masked_outside masked_values masked_invalid fix_invalid reshape transpose ravel expand_dims squeeze swapaxes resize repeat take compress diag diagflat diagonal trace concatenate stack append hstack vstack dstack column_stack hsplit vsplit diff ediff1d where choose sort argsort nonzero dot inner outer clump_masked clump_unmasked compress_rows compress_cols mask_rows mask_cols zeros ones identity arange zeros_like ones_like".split(),
    "Sqrt Log Log2 Log10 Arcsin Arccos Arccosh Arctanh Tan Exp Conjugate Sin Cos Arctan Arcsinh Sinh Cosh Tanh Absolute Fabs Negative Floor Ceil LogicalNot Add Subtract Multiply Divide FloorDivide Remainder Fmod Power Arctan2 Equal NotEqual Less LessEqual Greater GreaterEqual LogicalAnd LogicalOr LogicalXor Hypot Minimum Maximum BitwiseAnd BitwiseOr BitwiseXor LeftShift RightShift Angle Round Clip Sum Prod Min Max All Any PTP Mean Median Var Std Count CountMasked ArgMin ArgMax Cumsum Cumprod Anomalies Average AllClose AllEqual Filled Compressed GetData GetMask GetMaskArray IsMasked IsMask MakeMask MakeMaskNone MaskOr MaskedAll MaskedAllLike MaskedWhere MaskedEqual MaskedNotEqual MaskedGreater MaskedGreaterEqual MaskedLess MaskedLessEqual MaskedInside MaskedOutside MaskedValues MaskedInvalid FixInvalid Reshape Transpose Ravel ExpandDims Squeeze SwapAxes Resize Repeat Take Compress Diag Diagflat Diagonal Trace Concatenate Stack Append HStack VStack DStack ColumnStack HSplit VSplit Diff Ediff1D Where Choose Sort Argsort Nonzero Dot Inner Outer ClumpMasked ClumpUnmasked CompressRows CompressCols MaskRows MaskCols Zeros Ones Identity Arange ZerosLike OnesLike".split()))
for name,go in masked.items():
    add("numpy.ma."+name,"ma."+go,"ma_"+name,"numeric mask and valid data; Go array API")
for n in (1,2,3):
    add(f"numpy.ma.atleast_{n}d",f"ma.Atleast{n}D",f"ma_atleast{n}","one masked numeric array")
for name in ("MaskedArray","array","masked_array","asarray","asanyarray"):
    add("numpy.ma."+name,"ma.New",["ma_g_data","ma_g_input_mask","ma_filled","ma_hardmask","ma_softmask"],"explicit numeric data and boolean mask, fill and hard mask options")
for name, go in (("harden_mask","HardenMask"),("soften_mask","SoftenMask")):
    add("numpy.ma."+name,"ma.Array."+go,"ma_"+("hardmask" if name=="harden_mask" else "softmask"),"masked assignment")
for alias,target in [("abs","absolute"),("alltrue","all"),("amax","max"),("amin","min"),("anom","anomalies"),
                     ("around","round"),("innerproduct","inner"),("mod","remainder"),("product","prod"),
                     ("sometrue","any"),("true_divide","divide")]:
    if hasattr(np.ma,alias):mapped["numpy.ma."+alias]=mapped["numpy.ma."+target]

# All public polynomial-basis routines, with class construction/evaluation
# and domain/window mappings verified for each basis.
poly_ops = {
    "add":"Add","sub":"Sub","mul":"Mul","mulx":"MulX","div":"Div","pow":"Pow",
    "trim":"Trim","line":"Line","fromroots":"FromRoots","roots":"Roots","companion":"Companion",
    "der":"Derivative","int":"Integral","val":"Eval","vander":"Vander","fit":"Fit",
    "val2d":"EvalND","val3d":"EvalND","valnd":"EvalND","grid2d":"EvalND","grid3d":"EvalND",
    "vander2d":"VanderND","vander3d":"VanderND",
}
for name,prefix,clsname,basis in [
    ("polynomial","poly","Polynomial","Power"),("chebyshev","cheb","Chebyshev","Chebyshev"),
    ("legendre","leg","Legendre","Legendre"),("hermite","herm","Hermite","Hermite"),
    ("hermite_e","herme","HermiteE","HermiteE"),("laguerre","lag","Laguerre","Laguerre"),
]:
    ns="numpy.polynomial."+name+"."
    tag="poly_"+name+"_"
    for op,go in poly_ops.items():
        note = "numeric coefficient/evaluation behavior in listed fixtures"
        if op=="roots":note="sorted roots within tolerance; numerically real roots may have different real/complex result dtype"
        add(ns+prefix+op,"poly."+basis+"."+go,[tag+op,tag+"complex_roots"] if op=="roots" else tag+op,note)
    for symbol in (ns+clsname,"numpy.polynomial."+clsname):
        add(symbol,"poly.NewSeries("+basis+")",tag+"series","coefficients, domain/window, evaluation, calculus and roots; other class methods pending")
    if name=="polynomial":
        add(ns+"polyvalfromroots","poly.ValFromRoots",tag+"valfromroots","one root vector")
    else:
        for op,go,fixture in [(prefix+"2poly","Convert(Power)","to_power"),("poly2"+prefix,"Power.Convert("+basis+")","from_power"),
                              (prefix+"gauss","Gauss","gauss"),(prefix+"weight","Weight","weight")]:
            add(ns+op,"poly."+basis+"."+go,tag+fixture,"numeric polynomial basis")
for name,go in [("pts1","ChebPts1"),("pts2","ChebPts2"),("interpolate","ChebInterpolate")]:
    add("numpy.polynomial.chebyshev.cheb"+name,"poly."+go,"poly_chebyshev_"+name,"scalar degree and real interpolation interval")
for name,go,fixture in [
    ("polyadd","LegacyAdd","add"),("polysub","LegacySub","sub"),("polymul","LegacyMul","mul"),
    ("polydiv","LegacyDiv","div"),("polyder","LegacyDerivative","der"),("polyint","LegacyIntegral","int"),
    ("polyval","LegacyEval","val"),("roots","LegacyRoots","roots"),("poly","LegacyFromRoots","poly"),
    ("polyfit","LegacyFit","fit"),("poly1d","NewLegacySeries","series"),
]:
    add("numpy."+name,"poly."+go,"poly_legacy_"+fixture,"legacy descending coefficients; numerical subset")

# Remaining core numeric, raw byte, text I/O and callable/iteration adapters.
remaining = dict(zip(
    "apply_along_axis apply_over_axes asanyarray asarray_chkfinite asmatrix matrix block bmat broadcast empty empty_like finfo iinfo flatiter frombuffer fromfile fromfunction fromiter fromstring genfromtxt histogram2d histogram_bin_edges histogramdd isscalar issubdtype iterable loadtxt long ulong mask_indices min_scalar_type mintypecode ndenumerate ndindex nditer piecewise require savetxt typename vectorize base_repr binary_repr format_float_positional format_float_scientific common_type einsum_path".split(),
    "ApplyAlongAxis ApplyOverAxes AsAnyArray AsArrayChkFinite AsMatrix AsMatrix Block BMat NewBroadcast Empty EmptyLike FInfo IInfo Array.Flat FromBuffer FromFile FromFunction FromIter FromString GenFromText Histogram2D HistogramBinEdges HistogramDD IsScalar IsSubdtype Iterable LoadText ScalarOf[int64] ScalarOf[uint64] MaskIndices MinScalarType MinTypeCode NDEnumerate NDIndex NDIter Piecewise Require SaveText TypeName Vectorize BaseRepr BinaryRepr FormatFloatPositional FormatFloatScientific CommonType EinsumPath/EinsumWithPath".split()))
for name,go in remaining.items():
    note={
        "apply_along_axis":"Go callback, fixed result shape; C-order lanes",
        "apply_over_axes":"Go callback preserving or reducing one axis at a time",
        "asanyarray":"numeric arrays only, no Python subclass dispatch",
        "matrix":"numeric two-dimensional construction only; Python matrix subtype/operators are not implemented",
        "asmatrix":"numeric two-dimensional construction/view only",
        "bmat":"nested array blocks; no string matrix syntax or matrix subtype",
        "broadcast":"broadcast shape and Go scalar iteration",
        "empty":"allocated numeric storage; Go zero-initializes before caller writes",
        "empty_like":"initialized Go allocation with compatible shape/dtype and C/F layout",
        "frombuffer":"sharing []byte, offset/count, explicit endian interpretation",
        "fromfile":"raw binary io.Reader, count/offset, explicit endian",
        "fromiter":"typed Go iterator and count",
        "fromfunction":"Go callback receives numeric coordinate grids",
        "fromstring":"explicit separator, numeric dtype, count",
        "genfromtxt":"numeric text, selected columns, missing values and fill; no structured strings",
        "loadtxt":"numeric text, delimiter, comments, skip/usecols, unpack and ndmin",
        "savetxt":"numeric io.Writer with scalar format, delimiter, header/footer",
        "nditer":"explicit C/F numeric iteration, no Python iterator flags",
        "vectorize":"broadcast numeric Go callback; explicit output dtype",
        "einsum_path":"greedy, none, memory bound, exhaustive optimal up to eight operands; numeric execution",
        "format_float_positional":"float64, unique/fixed precision, sign and trim",
        "format_float_scientific":"float64, unique/fixed precision, sign and trim",
    }.get(name,"supported fixed-width numeric subset in listed fixtures")
    core(name,go,"remaining_"+("flat" if name=="flatiter" else name),note)
for name in ("generic","number","inexact","integer","signedinteger","unsignedinteger","floating","complexfloating"):
    core(name,"IsSubdtype","remaining_issubdtype","numeric abstract type membership; no Python scalar class construction")
for name,go,fixture,note in [
    ("byteswap","ByteSwap","byteswap","endian swap within each scalar component; view mutation"),
    ("view","View","view","shared fixed-width native-order reinterpretation; contiguous last axis for size change"),
    ("tobytes","ToBytes","tobytes","C/F order raw scalar bytes"),
    ("tofile","ToFile","tofile","binary C-order io.Writer"),
    ("getfield","GetField","getfield","fixed-width numeric field at byte offset"),
    ("setfield","SetField","setfield","numeric field writeback"),
    ("data","DataBytes","data_mutation","mutable C-contiguous byte span"),
    ("flat","Flat","flat","numeric Go C-order iterator"),
    ("tolist","ToList","tolist","nested typed Go values"),
    ("resize","ResizeInPlace","resize","owning C-order array; conservatively rejects previously exposed views"),
    ("base","Base","base","original array allocation for derived views; external Go buffer owner is not a Python object"),
    ("device","Device","device","CPU only"),("to_device","ToDevice","to_device","CPU only"),
    ("setflags","SetWritable","setflags","write flag on owning arrays"),
]:
    add("numpy.ndarray."+name,"np.Array."+go,"remaining_"+fixture,note)
core("nested_iters","NestedIters","remaining_nested_iters","two axis groups; yields array views and numeric iteration")
for name,go,fixture,note in [
    ("datetime64","DateTime64/Parse","parse","explicit datetime module tick and unit representation, ISO values"),
    ("timedelta64","TimeDelta64","timedelta","duration ticks and unit"),
    ("datetime_data","Unit.DateTimeData","unit","explicit resolution metadata"),
    ("datetime_as_string","Value.AsString","format","timezone-free ISO and fractional seconds"),
    ("isnat","IsNaT","isnat","datetime arrays with int64 NaT sentinel"),
    ("busdaycalendar","NewBusDayCalendar","calendar","weekmask and normalized holiday ticks"),
    ("is_busday","IsBusDay","is_busday","date array/calendar"),
    ("busday_count","BusdayCount","busday_count","broadcast day arrays; forward and reverse intervals"),
    ("busday_offset","BusdayOffset","busday_offset","broadcast offsets, holidays and all roll modes"),
]:
    add("numpy."+name,"datetime."+go,"datetime_"+fixture,note)

masked_remaining = dict(zip(
    "apply_along_axis apply_over_axes common_fill_value compress_nd compress_rowcols convolve copy corrcoef correlate cov default_fill_value empty empty_like flatnotmasked_contiguous flatnotmasked_edges flatten_mask frombuffer fromfunction in1d indices intersect1d isMA isMaskedArray isarray isin mask_rowcols maximum_fill_value minimum_fill_value ndenumerate ndim notmasked_contiguous notmasked_edges polyfit put putmask set_fill_value setdiff1d setxor1d shape size union1d unique vander bool_ MaskType".split(),
    "ApplyAlongAxis ApplyOverAxes CommonFillValue CompressND CompressRowCols Convolve Array.Copy Corrcoef Correlate Cov DefaultFillValue Empty EmptyLike FlatNotMaskedContiguous FlatNotMaskedEdges FlattenMask FromBuffer FromFunction In1D Indices Intersect1D IsMaskedArray IsMaskedArray IsMaskedArray Isin MaskRowCols MaximumFillValue MinimumFillValue NDEnumerate Array.NDim NotMaskedContiguous NotMaskedEdges Polyfit Put Putmask Array.SetFillValue Setdiff1D Setxor1D Array.Shape Array.Size Union1D Unique Vander np.Bool np.Bool".split()))
for name,go in masked_remaining.items():
    fixture={"default_fill_value":"default_fill","maximum_fill_value":"maximum_fill","minimum_fill_value":"minimum_fill"}.get(name,name)
    add("numpy.ma."+name,go if go.startswith("np.") else "ma."+go,"ma_remaining_"+fixture,
        "numeric masks and valid data; explicit Go representation, no structured or object masks")
for alias,target in [("outerproduct","outer"),("round_","round"),("row_stack","vstack")]:
    if hasattr(np.ma,alias):mapped["numpy.ma."+alias]=mapped["numpy.ma."+target]
add("numpy.ma.MaskError","ma.ErrMask","ma_remaining_mask_error","mask shape mismatch error category")
add("numpy.ma.MAError","ma.ErrMA",["ma_remaining_mask_error","ma_remaining_mask_error_inheritance"],"masked-specific Go error category and inheritance via errors.Is")

# Final compatibility batch: concrete numerical/record contracts versus
# intentionally different Go representations for Python/runtime boundaries.
for name,go,fixture in [
    ("longdouble","extended.LongDouble","final_extended_parse"),
    ("float128","extended.Float128","final_extended_alias"),
    ("clongdouble","extended.CLongDouble","final_extended_complex"),
    ("complex256","extended.Complex256","final_extended_complex_alias"),
]:
    equivalent("numpy."+name,go,[fixture,"final_extended_info","final_extended_add","final_extended_cmul"],
        "portable x87 80-bit value in 16-byte slot (two slots for complex); root ndarray dtype and legacy adapter; selected arithmetic, views and special values")
for name,go,fixture in [
    ("bytes_","records.ByteString","final_record_bytes"),
    ("str_","records.UnicodeString","final_record_unicode"),
    ("void","records.NewVoid","final_record_void"),
    ("character","records.Type.IsCharacter","final_record_hierarchy"),
    ("flexible","records.Type.IsFlexible","final_record_hierarchy"),
    ("recarray","records.Array",["final_record_layout","final_record_aligned_offsets","final_record_column"]),
    ("record","records.Record",["final_record_scalar","final_record_is_record","final_record_column"]),
    ("fromregex","records.FromRegex","final_record_fromregex"),
    ("memmap","np.MemMap","final_mmap"),
    ("array2string","np.Array2String","final_format"),
    ("array_repr","np.ArrayRepr","final_format"),
    ("array_str","np.ArrayStr","final_format"),
    ("get_printoptions","np.GetPrintOptions","final_format_default"),
    ("set_printoptions","np.SetPrintOptions","final_format_set"),
    ("__array_namespace_info__","np.ArrayNamespaceInfo","final_abi_namespace"),
]:
    add("numpy."+name,go,fixture,
        "fixed-width structured fields and scalar representation; RE2 capture subset" if name in ("recarray","record","fromregex","bytes_","str_","void","character","flexible") else
        "Linux OS file mapping; C/F order, arbitrary byte offset, shared/read-only/private modes, flush and explicit close" if name=="memmap" else
        "CPU Array API standard dtype and capability metadata" if name=="__array_namespace_info__" else
        "selected exact numeric formatting, options and dtype/shape representations")
equivalent("numpy.printoptions","np.WithPrintOptions","final_format_scope","explicit scoped Formatter rather than Python context-local state")
for name,go,fixture in [
    ("errstate","np.ErrorContext.ErrState",["final_fp_scope","final_fp_restore","final_fp_raises"]),
    ("geterr","np.ErrorContext.GetErr","final_fp_default"),
    ("seterr","np.ErrorContext.SetErr",["final_fp_callback","final_fp_scope","final_fp_log","final_fp_warn"]),
    ("geterrcall","np.ErrorContext.GetErrCall","final_fp_seterrcall"),
    ("seterrcall","np.ErrorContext.SetErrCall",["final_fp_seterrcall","final_fp_callback","final_fp_log"]),
    ("getbufsize","np.ErrorContext.GetBufSize",["final_fp_buf_default","final_fp_buf_new"]),
    ("setbufsize","np.ErrorContext.SetBufSize",["final_fp_buf_old","final_fp_buf_new","final_fp_invalid_buf"]),
]:
    equivalent("numpy."+name,go,fixture,"explicit Go ErrorContext; tested result-level FP flags, actions and scratch sizing; policy only applies via context.Call/CallUFunc, not hardware fenv")
equivalent("numpy.object_","objects.Array","final_object_type","explicit Go any values; no Python object model")
equivalent("numpy.frompyfunc","objects.FromPyFunc",["final_object_callback","final_object_reduce","final_object_accumulate","final_object_outer","final_object_at"],"explicit Go callbacks; arity, broadcast, multiple outputs, reduce/accumulate/outer/at")
equivalent("numpy.ufunc","np.LookupUFunc","final_ufunc","typed numeric kernel descriptors and selected common ufunc methods; no Python dispatch")
for name,go,fixture in [
    ("flatten_structured_array","ma.FlattenStructuredArray","final_record_flat"),
    ("fromflex","ma.FromFlex","final_record_fromflex"),
    ("make_mask_descr","ma.MakeMaskDescr","final_record_mask_offsets"),
    ("mvoid","ma.MaskedRecord","final_record_mvoid"),
]:
    add("numpy.ma."+name,go,fixture,"numeric and nested fixed-width record and per-field mask subset")
equivalent("numpy.ma.ids","ma.Ids","final_record_ids","opaque alias diagnostic tokens; not Python object addresses")
equivalent("numpy.ma.masked_object","ma.MaskedObject",["final_object_mask","final_object_filled"],"Go object values and equality; no Python rich comparison dispatch")
add("numpy.polynomial.set_default_printstyle","poly.SetDefaultPrintStyle","final_poly","ASCII/Unicode formatting for six polynomial bases and domain mapping")
for symbol,go,fixture,note in [
    ("numpy.from_dlpack","np.FromDLPack","final_abi_dlpack","retained typed native Go tensor handle; shared strided/writeable view, not foreign DLPack capsule"),
    ("numpy.ndarray.ctypes","Array.CTypes","final_abi_ctypes","typed Go handle with shape, strides, identity and safe mutation; no raw C pointer"),
    ("numpy.ndarray.dump","Array.Dump","final_abi_dump","NPY writer round-trip numerical equivalence; no pickle compatibility"),
    ("numpy.ndarray.dumps","Array.Dumps","final_abi_dumps","portable NPY bytes round-trip numerical equivalence; no pickle compatibility"),
    ("numpy.get_include","np.GetInclude","final_abi_include","Go import path and optional installed source path, not C headers"),
    ("numpy.info","np.Info","final_abi_info","Go array metadata diagnostic, not Python introspection"),
    ("numpy.show_config","np.ShowConfig","final_abi_config","Go build/version/dependency metadata instead of NumPy compilation metadata"),
    ("numpy.show_runtime","np.ShowRuntime","final_abi_runtime","Go CPU/OS/runtime metadata instead of Python SIMD/BLAS report"),
    ("numpy.test","np.SelfTest","final_abi_test","Go deployment smoke check; full Go suite is go test, not pytest"),
]:
    equivalent(symbol,go,fixture,note)
algorithms=["MT19937","PCG64","PCG64DXSM","Philox","SFC64"]
equivalent("numpy.random.BitGenerator","random.BitGenerator; random.NewBitGenerator",
        ["final_rng_"+n+"_base" for n in algorithms],"common Go stream interface and typed factory, not Python abstract class")
for cls_name in ["BitGenerator"]+algorithms:
    for interface,method in [("capsule","Capsule"),("ctypes","CTypes"),("cffi","CFFI")]:
        subjects=algorithms if cls_name=="BitGenerator" else [cls_name]
        equivalent(f"numpy.random.{cls_name}.{interface}",f"Generator.{method}",
            ["final_rng_"+n+"_"+interface for n in subjects],
            "retained typed Go raw stream handle; synchronized next_uint64/uint32/double or raw stream; no Python capsule/C pointer")

# Preserve the fixed API denominator and attach the new behavioral grid to
# the existing symbols (including aliases), not to invented Python API names.
from behavior_reference import BINARY, UNARY

def enhance(symbol, prefixes, extra_api="", scope=None):
    if symbol not in mapped:
        return
    go, fixtures, note = mapped[symbol]
    extra = {name for prefix in prefixes for name in fixture_names
             if name == prefix or name.startswith(prefix+"_")}
    if not extra:
        return
    mapped[symbol] = (go + extra_api, sorted(set(fixtures)|extra), scope or note)

for symbol in list(mapped):
    if symbol.count(".") != 1:
        continue
    name = symbol.split(".")[1]
    member = getattr(np,name,None)
    canonical = getattr(member,"__name__",name)
    if canonical not in BINARY+UNARY:
        continue
    prefix = "behavior_binary_" if canonical in BINARY else "behavior_unary_"
    prefixes = [prefix+canonical]
    weak = any(f.startswith("behavior_weak_"+canonical+"_") for f in fixture_names)
    if weak:
        prefixes.append("behavior_weak_"+canonical)
    note = "14 fixed-width dtype grid; reverse-stride broadcast, values/shape/dtype and typed errors"
    api = ""
    if isinstance(member,np.ufunc):
        api = "; UFunc.CallWithOptions"
        note += "; single-output options verified on selected kernels"
    if weak:
        api += "; UFunc.CallValues"
        note += "; weak Python scalar promotion and integer boundaries"
    if canonical in ("add","subtract","multiply","divide"):
        prefixes.append("behavior_dense_"+canonical)
        note += "; contiguous/0-D typed fast path"
    enhance(symbol,prefixes,api,note)

for name in ("sum","prod","mean","var","std","min","max","nansum","nanprod","nanmean",
             "nanvar","nanstd","nanmin","nanmax","all","any","count_nonzero","argmin","argmax","nanargmin","nanargmax"):
    for symbol in ("numpy."+name,"numpy.ndarray."+name):
        enhance(symbol,["behavior_reduce_"+name,"behavior_precision_"+name],
                scope="fixed-width dtype/axis/keepdims grid; accumulator precision and exceptional values where applicable")
for name,go,fixtures,note in [
    ("quantile","; np.QuantileArray/QuantileWithOptions",
     ["behavior_quantile"],"13 sample methods, typed q arrays, axis tuples/keepdims, weighted inverted_cdf and strided out; verified selected dtype combinations"),
    ("nanquantile","; np.NanQuantileArray/NanQuantileWithOptions",
     ["behavior_quantile_nan"],"13 sample methods excluding NaNs, including weighted inverted_cdf and empty input; typed q"),
    ("percentile","; np.PercentileArray",
     ["behavior_quantile_percentile"],"typed q percent arrays and shared 13-method quantile engine; selected axis/method tested"),
    ("nanpercentile","; np.NanPercentileArray",
     ["behavior_quantile_nanpercentile"],"shared nanquantile engine with percentage scale; scalar and typed q axis/method behavior"),
]:
    enhance("numpy."+name,fixtures,go,note)
enhance("numpy.ufunc",["behavior_ufunc","behavior_reduceopt","behavior_accumulate",
                      "behavior_reduceat","behavior_weak_dtype","behavior_weak_compare_mask"],
        "; UFunc.CallWithOptions/CallValues/ReduceWithOptions/Accumulate/ReduceAt",
        "Go numeric descriptor; tested selected single-output dtype/casting/out/where, explicit weak scalars, masked reduction with initial, accumulation and segmented reduction; no Python dispatch")

# Expand behavioral evidence without changing the enumerated API denominator.
for name in ("normal","uniform","exponential","gamma","standard_gamma","beta","chisquare",
             "f","standard_t","pareto","weibull","power","laplace","gumbel","logistic",
             "lognormal","rayleigh","triangular","wald","noncentral_chisquare",
             "noncentral_f","vonmises","poisson","negative_binomial","binomial",
             "geometric","logseries","zipf","hypergeometric"):
    for namespace in ("numpy.random.Generator","numpy.random.RandomState","numpy.random"):
        enhance(namespace+"."+name,["param_dist_"+name,"param_dist_input_"+name],
                "; Generator.Sample",
                "typed array parameter broadcasting, inferred/explicit sizes and next RNG stream; selected finite parameters across five bit generators and legacy RandomState")
for namespace,name in (("numpy.random.Generator","integers"),("numpy.random.RandomState","randint"),
                       ("numpy.random","randint")):
    enhance(namespace+"."+name,["param_int","param_bad_bound"],"; Generator.IntegersArray",
            "nine fixed-width integer/bool dtypes; broadcast bounds, exact uint64/int64 limits, endpoint modes where applicable and next RNG stream")
enhance("numpy.random.Generator.multinomial",["param_multinomial"],"; Generator.MultinomialArray",
        "broadcast modern trial-count and probability arrays, explicit size and next RNG stream")
for name in ("random","standard_normal","standard_exponential","standard_gamma"):
    enhance("numpy.random.Generator."+name,["param_fill"],"; Generator.FillRandom",
            "float32/64 dtype and validated output buffers; C/F physical fill where allowed; gamma shape broadcast and exponential method; next RNG stream")
for name,api,note in [
    ("svd","; SVDOptions.Hermitian","batched Hermitian SVD, values-only, reconstruction and orthonormality across four precision dtypes"),
    ("pinv","; PinvWithOptions","Hermitian and per-matrix rcond/rtol, explicit default rtol=None and empty integer matrix"),
    ("matrix_rank","; MatrixRankWithOptions","Hermitian, absolute/relative array tolerances and expanded batch output"),
    ("qr","; QRRaw","packed Householder reflectors and tau for tall/wide/empty/batched real and complex matrices"),
]:
    prefixes=["param_linalg_raw"] if name=="qr" else ["param_linalg_hermitian","param_linalg_"+name]
    enhance("numpy.linalg."+name,prefixes,api,note)
for symbol in ("numpy.matmul","numpy.linalg.matmul","numpy.dot","numpy.tensordot","numpy.einsum"):
    enhance(symbol,["param_product"],scope="typed BLAS where applicable; tested batched, reversed-stride and nonfinite operands; planned two-operand tensor contractions")
for symbol in ("numpy.astype","numpy.ndarray.astype"):
    enhance(symbol,["param_cast"],"; Array.AstypeWithOptions",
            "14x14 fixed-width cast grid; exact rounding/signed zero, invalid Linux casts where tested, C/F/A/K layout and casting/no-copy checks")

for name in ("fft","ifft","rfft","irfft","hfft","ihfft"):
    enhance("numpy.fft."+name,["adv_fft_"+name,"adv_fft_out_"+name,"adv_fft_wide_"+name],
            "; Options.Out","14 numerical dtypes (where applicable), length/axis/norm grid, K layout, strided/broadcast/alias out, real and complex precision selection")
for name in ("fftn","ifftn","rfftn","irfftn","fft2","ifft2","rfft2","irfft2"):
    enhance("numpy.fft."+name,["adv_fft_"+name,"adv_fft_ndout_"+name],
            "; NDOptions.Out","repeated/negative axes, default/full/explicit lengths, three normalizations and per-stage output buffer geometry")
enhance("numpy.fft.fftfreq",["adv_fft_freq"],scope="odd/even lengths, negative, NaN and infinite spacing")
enhance("numpy.fft.rfftfreq",["adv_fft_rfreq"],scope="odd/even lengths, negative, NaN and infinite spacing")
for name,api in (("empty","Empty"),("zeros","Zeros"),("ones","Ones"),("full","Full")):
    refs=[f for f in fixture_names if f.startswith("adv_like_") and f.endswith("_"+name)]
    enhance("numpy."+name+"_like",refs,"; np."+api+"LikeWithShape",
            "explicit dtype, shape and C/F/A/K layout; scalar, empty and changed rank; empty verified after fill")
for name in ("SeedSequence","SeedSequence.generate_state","SeedSequence.spawn","SeedSequence.state",
             "SeedSequence.entropy","SeedSequence.spawn_key","SeedSequence.pool","SeedSequence.pool_size","SeedSequence.n_children_spawned"):
    enhance("numpy.random."+name,["adv_seed"],"; SeedSequenceFromState/PoolWords",
            "explicit entropy words, restored spawn key/child counter, 4/5/8/16-word pools and nested spawn replay")
for name in ("PCG64","PCG64DXSM","MT19937","Philox","SFC64"):
    enhance("numpy.random."+name,["adv_seed"],
            scope="configurable SeedSequence initialization and child raw streams, in addition to legacy fixtures")
enhance("numpy.random.Generator.permuted",["adv_permuted"],"; Generator.PermutedWithOut",
        "independent logical-axis and flattened permutations, C/F/K/strided/alias output, safe casting and exact following RNG stream")
for ns in ("numpy.random.Generator","numpy.random.RandomState","numpy.random"):
    enhance(ns+".dirichlet",["adv_dirichlet"],scope="finite concentration vectors including zero/tiny/singleton and empty cases; following RNG stream")
    enhance(ns+".multivariate_normal",["adv_mvnormal"],"; Generator.MultivariateNormalWithOptions",
            "covariance check_valid, tolerance and warning callback, diagonal/semidefinite/indefinite matrices, following RNG stream after error; sign equivalence for SVD/eigh")
enhance("numpy.random.Generator.multivariate_hypergeometric",["adv_mvhg"],
        scope="marginals/count algorithms, zero colors and sample, full/complement samples, exact next RNG stream")
for name,suffix in (("svd","_s"),("svdvals","_s"),("pinv","_pinv"),("inv","_inv"),("qr","_raw_h")):
    refs=[f for f in fixture_names if f.startswith("adv_linalg_") and (f.endswith(suffix) or name=="qr" and f.endswith("_raw_tau"))]
    enhance("numpy.linalg."+name,refs,
            scope="existing behavior plus strongly graded/scaled complex matrices; relative-error checks and stable inverse when zero-cutoff full rank" if name!="qr" else
            "raw reflectors/tau at extreme scales and previous batched real/complex tests")
for name in ("norm","vector_norm","matrix_norm"):
    enhance("numpy.linalg."+name,["adv_norm"],scope="axes and keepdims; intermediate dtype overflow/underflow for Euclidean/Frobenius and power norms")

for name in ("longdouble","float128","clongdouble","complex256"):
    enhance("numpy."+name,["unified_ext","unified_npy_extended","unified_npy_complex"],
            "; np.Array/FromSlice/ExtendedFInfo",
            "x87 significant 80-bit words, root dtype, views and exact computation; selected reductions, contraction and NPY; not IEEE binary128")
for name in ("recarray","record","void"):
    enhance("numpy."+name,["unified_record"],"; records.Array.NDArray/np.Array.Field",
            "packed and nested fields, byte-stride views, advanced indexing, native-endian concatenate/where and equality; fixed strings through scalar field API")
enhance("numpy.object_",["unified_obj"],"; np.FromObjects/np.ObjectScalar",
        "root object ndarray with GC-managed slots; shallow copies and selected object arithmetic, comparison and reduction; Go values instead of Python objects")
for name in ("datetime64","timedelta64","datetime_data","isnat"):
    enhance("numpy."+name,["unified_date","unified_npy_datetime","unified_npy_timedelta"],
            "; np.FromDateTicks/DateUnit/CastTemporal",
            "root temporal dtype with unit metadata, shared int64 ticks, selected arithmetic/NaT, sorting, indexing and NPY; supported unit conversions")
for name in ("RandomState","RandomState.get_state","RandomState.set_state","RandomState.seed","RandomState.standard_normal","RandomState.randint"):
    enhance("numpy.random."+name,["unified_random"],"; NewRandomStateFromBitGenerator/NewGeneratorFromBitGenerator",
            "five built-in bit generators and forwarding BitGenerator adapters; shared mutex/raw/uint32 state, independent Gaussian caches, state replay and MT-only reseeding")
for name in ("eig","eigvals"):
    enhance("numpy.linalg."+name,["unified_eigen"],scope="direct scaled complex Hessenberg/Schur QR; repeated and defective eigenvalues, nonnormal, random, extreme scale and complex64 batches; unordered spectrum and normalized residual")
enhance("numpy.linalg.norm",["unified_ext_norm","unified_ext_complex_norm"],scope="previous numeric behavior and extended precision default/axis Euclidean norm")
for name,refs in (("sum",["sum","complex_sum"]),("mean",["mean"]),("var",["var","complex_var"]),
                  ("std",["std"]),("cumsum",["cumsum"]),("cumprod",["cumprod"]),("nextafter",["nextafter"]),
                  ("matmul",["dot","complex_dot"])):
    enhance("numpy."+name,["unified_ext_"+v for v in refs])
for name in ("save","load"):
    enhance("numpy."+name,["unified_npy"],scope="fixed-width numeric, little-endian x87 extended, and datetime/timedelta NPY; C/F order; no pickle")

# Extended dtype numerical comparisons use actual ten-byte x87 values,
# bounded ULP error and scale-relative Fourier residuals, never binary64.
for name in ("sqrt exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin arccos arctan sinh cosh tanh arcsinh arccosh arctanh cbrt degrees radians rint floor ceil trunc fabs sign spacing sinc").split():
    enhance("numpy."+name,["exmath_real_"+name,"exmath_complex_"+name,"exmath_far_"+name,"exmath_nonfinite_"+name,"exmath_bad_complex_"+name])
for name in ("power float_power floor_divide remainder fmod hypot arctan2 copysign logaddexp logaddexp2 heaviside").split():
    enhance("numpy."+name,["exmath_binary_"+name])
for name,refs in (("power",["exmath_cpower"]),("float_power",["exmath_cfloat_power"]),("angle",["exmath_angle"]),("round",["exmath_round","exmath_cround"]),("modf",["exmath_modf"]),("frexp",["exmath_frexp"]),("ldexp",["exmath_ldexp"]),("isclose",["exmath_isclose"]),("i0",["exmath_i0"]),("real_if_close",["exmath_near_real","exmath_not_near_real"]),("nan_to_num",["exmath_nan_to_num","exmath_cnan_to_num"]),("isreal",["exmath_isreal"]),("iscomplex",["exmath_iscomplex"]),("isposinf",["exmath_isposinf"]),("isneginf",["exmath_isneginf"])):
    enhance("numpy."+name,refs)
for name in ("longdouble","float128","clongdouble","complex256"):
    enhance("numpy."+name,["exmath_real","exmath_complex","exmath_binary","exmath_ufunc","exfft"],
            scope="root x87 dtype with guarded-precision transcendental functions, ufunc options, arithmetic and extended Fourier transforms; selected numerical and special-value cases")
for name in ("ufunc","ufunc.reduce","ufunc.accumulate","ufunc.reduceat","ufunc.outer","ufunc.at"):
    enhance("numpy."+name,["exmath_ufunc"])
for name in ("errstate","seterr","seterrcall"):
    enhance("numpy."+name,["exmath_fp"],scope="explicit Go context; result-level FP flags including x87 exponent range and inexact real subnormal arithmetic")
for name in ("fft","ifft","rfft","irfft","hfft","ihfft","fftn","ifftn","rfftn","irfftn","fft2","ifft2","rfft2","irfft2"):
    enhance("numpy.fft."+name,["exfft_"+name,"exfft_native_out_"+name],
            scope="native dtypes plus x87 Float128/Complex256; tested odd/even/prime lengths, crops/padding, axes, normalization, out and multi-axis variants")
for name,go in (("fftfreq","FFTFreqExtended"),("rfftfreq","RFFTFreqExtended")):
    enhance("numpy.fft."+name,["exfft_"+name],"; fft."+go,
            "longdouble spacing and outputs including huge/tiny/negative/zero/infinite spacing")
enhance("numpy.fft.fftshift",["exfft_shift"])
enhance("numpy.fft.ifftshift",["exfft_ishift"])
for name in ("promote_types","result_type","concatenate","where","equal","ndarray.astype"):
    enhance("numpy."+name,["recwide_nested","recwide_text","recwide_aligned","recwide_bad"],"; np.PromoteRecordSchemas/Array.CastRecords",
            "mixed same-name record fields, numeric and character width promotion, nested fields and compatible subarrays; native packing/alignment")
for name in ("save","load"):
    enhance("numpy."+name,["recwide_structured","recwide_overlap_npy"],"; records.LoadNPY",
            "numeric and temporal NPY plus non-object structured descriptors, padding/nesting/subarrays and C/F byte order; no pickle")
for name in ("recarray","record","void"):
    enhance("numpy."+name,["recwide_nested","recwide_text","recwide_aligned","recwide_structured"])
for name in ("eig","eigvals"):
    enhance("numpy.linalg."+name,["unified_eigen_random16","unified_eigen_random24","unified_eigen_random32","unified_eigen_random48","unified_eigen_graded24","unified_eigen_repeated16"],
            scope="scaled complex Schur QR; finite random matrices up to 48x48, repeated/defective spectra, graded nonnormal similarity, extreme scaling, batches; spectrum and residual")

for name,go,prefix in (
    ("bytes_","np.BytesDType/FixedBytes","flex_s"),("str_","np.UnicodeDType/FixedUnicode","flex_u"),
    ("void","np.VoidDType/FixedVoid","flex_void"),("character","np.DType.IsCharacter","flex_s"),
    ("flexible","np.DType.IsFlexible","flex_void"),("dtype","np.ParseFlexibleDType","flex_field"),
):
    enhance("numpy."+name,[prefix,"flex_can_cast"],"; "+go,
            "fixed-width descriptor-sized root ndarray; preserved bytes/UCS-4 order, shared packed fields and selected array operations; not variable-width StringDType")
for name,refs in (
    ("add",["add","ufunc_add_out"]),("equal",["equal"]),("less",["less"]),("sort",["sort"]),
    ("argsort",["argsort"]),("argmin",["argmin"]),("argmax",["argmax"]),("unique",["unique"]),
    ("isin",["isin"]),("union1d",["union"]),("intersect1d",["intersect"]),("setdiff1d",["setdiff"]),
    ("setxor1d",["setxor"]),("searchsorted",["search","searchright"]),("concatenate",["concat"]),
    ("where",["where"]),("count_nonzero",["count"]),("nonzero",["nonzero0","nonzero1"]),
    ("all",["all"]),("any",["any"]),("transpose",["transpose"]),("reshape",["reshape"]),
    ("take",["take"]),("broadcast_to",["broadcast"]),("copy",["copy"]),("zeros_like",["zeros"]),
    ("ones_like",["ones"]),("min",["min_error"]),("sum",["sum_error"]),("logical_not",["logical_not_error"]),
):
    enhance("numpy."+name,["flex_"+family+"_"+v for family in ("s","u","void") for v in refs]+["flex_"+v for v in refs if v.startswith("ufunc")])
for name in ("can_cast","promote_types","result_type","ndarray.astype"):
    enhance("numpy."+name,["flex_can_cast","flex_cast","flex_mixed","flex_field_mixed"],"; np.Array.CastFlexibleWithOptions/CanCastFlexible",
            "descriptor-sized widths, endianness, numeric and character promotions, casting rules and explicit F layout")
for name in ("frombuffer","ndarray.view","ndarray.getfield","ndarray.setfield","ndarray.byteswap"):
    enhance("numpy."+name,["flex_void_view","flex_field","flex_s_byteswap","flex_u_byteswap"],"; np.FromFlexibleBuffer/Array.ViewFlexible/Field")
for name,ref in (("array2string","array2string"),("array_repr","repr"),("array_str","str")):
    enhance("numpy."+name,["flex_fmt_"+d+"_"+ref for d in ("s","u","v")])
enhance("numpy.ufunc",["flex_ufunc"])
for name in ("save","load","savez","savez_compressed"):
    enhance("numpy."+name,["flex_flex_bytes","flex_flex_unicode","flex_flex_void","flex_flex_mixed"],"; records.LoadAnyNPY/LoadNPZ",
            "fixed-width S/U/V NPY, mixed compressed/uncompressed archives with structured, numerical and temporal members; safe descriptor parsing and no pickle")
for name in ("BitGenerator","Generator","RandomState","BitGenerator.random_raw","BitGenerator.state","RandomState.get_state","RandomState.set_state"):
    enhance("numpy.random."+name,["external_rng"],"; random.NewExternalBitGenerator/ExternalBitSource",
            "built-in and independent Go callback engines; NumPy raw-tape differential of shared modern/legacy consumers, caches and saved-state replay; no C ABI or implied custom advance/jump")

for name,go in zip(
    "equal not_equal less less_equal greater greater_equal add multiply isalpha isdigit isspace isalnum islower isupper istitle isdecimal isnumeric str_len find rfind index rindex count startswith endswith lstrip rstrip strip replace expandtabs center ljust rjust zfill partition rpartition slice upper lower swapcase capitalize title mod decode encode translate".split(),
    "Equal NotEqual Less LessEqual Greater GreaterEqual Add Multiply IsAlpha IsDigit IsSpace IsAlnum IsLower IsUpper IsTitle IsDecimal IsNumeric StrLen Find RFind Index RIndex Count StartsWith EndsWith LStrip RStrip Strip Replace ExpandTabs Center LJust RJust ZFill Partition RPartition Slice Upper Lower SwapCase Capitalize Title Mod Decode Encode Translate".split(),
):
    scope = "fixed S/U and default variable StringDType; broadcasted string operands; dtype, shape and logical content"
    if name in ("decode","encode"):
        scope = "UTF-8, ASCII and Latin-1; strict/replace/ignore; scalar codec options"
    elif name in ("partition","rpartition"):
        scope = "three output arrays with independent fixed widths; variable StringDType rejected by NumPy"
    elif name == "mod":
        scope = "typed positional percent formatting and tuple object slots; not mapping placeholders or dynamic widths"
    elif name == "translate":
        scope = "Unicode code-point replacement/deletion and bytes 256-entry table with deletechars"
    elif name in ("center","ljust","rjust","zfill","expandtabs","slice","replace","find","rfind","index","rindex","count","startswith","endswith"):
        scope += "; scalar bounds/count/width/slice options where applicable"
    evidence = ["strings_"+kind+"_"+name for kind in ("s","u","t")]
    if name in ("partition","rpartition"):
        evidence += ["strings_"+kind+"_"+name+str(i) for kind in ("s","u") for i in range(3)]
    if name in ("encode","decode"):
        evidence += ["strings_"+kind+"_"+name+"_"+codec for kind in ("s","u","t") for codec in ("utf-8","ascii","latin-1")]
    add("numpy.strings."+name,"npstrings."+go+("/TranslateBytes" if name=="translate" else ""),evidence,scope)

equivalent("numpy.dtypes.StringDType","np.StringDType/FromStrings/FromStringValues",["varstr_base","varstr_strict","varstr_coerce"],
           "Go-managed variable UTF-8 slots, default and coerce=False; arbitrary na_object and NumPy pointer ABI excluded")
enhance("numpy.dtypes.StringDType",["strna_none_base","strna_nan_base","strna_text_base","strna_int_base","strna_different_sentinel"],
        "; np.StringDescriptor/FromStringDescriptor/StringValues/CastStringDescriptor",
        "Go-managed variable UTF-8 slots, immutable scalar na_object (nil, NaN, string, integer), coerce=False, physical missing mask, promotion and casts; no Python pointer ABI or arbitrary object identity")
for name,refs in {
    "array":["base","coerce","strict"],"astype":["cast_from_num","cast_num","cast_u"],
    "ndarray.astype":["cast_from_num","cast_num","cast_u"],"copy":["copy"],"reshape":["reshape"],
    "transpose":["transpose"],"broadcast_to":["broadcast"],"ndarray.__getitem__":["advanced","reverse"],
    "ndarray.__setitem__":["assign","strict_error"],"sort":["sort"],"argsort":["argsort"],"unique":["unique"],
    "isin":["isin"],"concatenate":["concat"],"where":["where"],"zeros_like":["zero"],"ones_like":["ones"],
    "any":["any"],"count_nonzero":["count"],"isnan":["isnan"],"min":["min"],"max":["max"],"sum":["sum"],
    "array_repr":["repr"],
}.items():
    enhance("numpy."+name,["varstr_"+s for s in refs])
for name in ("add","equal","not_equal","less","less_equal","greater","greater_equal"):
    enhance("numpy."+name,["strings_"+kind+"_"+name for kind in ("s","u","t")])
enhance("numpy.multiply",["strings_t_direct_multiply","strings_u_direct_multiply","strings_t_ufunc_multiply","strings_u_ufunc_multiply"],
        "; np.RepeatStrings/npstrings.Multiply","numeric kernels, direct variable-string multiply; fixed strings require explicit out or numpy.strings.multiply")
enhance("numpy.ufunc",["strings_t_ufunc_add","strings_t_ufunc_multiply","strings_u_ufunc_multiply"])
enhance("numpy.memmap",["descriptor_map"],"; np.MemMapOptions.Flexible/Schema/DateUnit",
        "OS mmap of fixed-width S/U/V, structured packed fields, date/time units and numeric dtypes; file layouts, C/F, read-only and private modes, explicit lifetime")
for name in ("PCG64",):
    for method,prefix in (("advance","advance"),("jumped","jump"),("spawn","spawn0")):
        enhance("numpy.random."+name+"."+method,["external_caps_"+prefix],"; random.External"+{"advance":"Advance","jumped":"Jump","spawn":"Spawn"}[method]+"Source")
enhance("numpy.random.Generator.spawn",["external_caps_spawn0","external_caps_spawn1"],"; random.ExternalSpawnSource",
        "built-in SeedSequence derivation or independently provided external child-stream callback under a single flow lock")

for name,go in zip(
    "equal not_equal less less_equal greater greater_equal add multiply isalpha isdigit isspace isalnum islower isupper istitle isdecimal isnumeric str_len find rfind index rindex count startswith endswith lstrip rstrip strip replace expandtabs center ljust rjust zfill partition rpartition upper lower swapcase capitalize title mod decode encode translate join split rsplit splitlines compare_chararrays".split(),
    "Equal NotEqual Less LessEqual Greater GreaterEqual Add Multiply IsAlpha IsDigit IsSpace IsAlnum IsLower IsUpper IsTitle IsDecimal IsNumeric StrLen Find RFind Index RIndex Count StartsWith EndsWith LStrip RStrip Strip Replace ExpandTabs Center LJust RJust ZFill Partition RPartition Upper Lower SwapCase Capitalize Title Mod Decode Encode Translate Join Split RSplit SplitLines CompareCharArrays".split(),
):
    notes="legacy namespace; shared modern string kernels, tested S/U/T typed subset"
    if name in ("equal","not_equal","less","less_equal","greater","greater_equal","compare_chararrays"):
        notes="fixed S/U strings, legacy right whitespace trimming or explicit rstrip switch; T comparison rejected"
    elif name in ("partition","rpartition"):
        notes="fixed S/U; stacked three-part array, unlike tuple of np.strings; T rejected"
    elif name in ("split","rsplit","splitlines"):
        notes="object ndarray with Go list slots, explicit separator/whitespace, maximum splits and keepends"
    elif name=="join":
        notes="broadcasted separator and Unicode code-point sequence; nonempty bytes are rejected as integer sequence"
    elif name in ("decode","encode"):
        notes="UTF-8, ASCII, Latin-1 typed codec subset"
    add("numpy.char."+name,"char."+go+("/TranslateBytes" if name=="translate" else ""),
        ["char_"+kind+"_"+name for kind in ("s","u","t")],notes)
for name,go in (("array","Array"),("asarray","AsArray"),("chararray","New/CharArray")):
    equivalent("numpy.char."+name,"char."+go,["char_"+kind+"_"+name for kind in ("s","u","t")],
               "explicit Go chararray adapter; selected fixed-width construction, copy vs shared data, scalar whitespace stripping; no Python ndarray subclass ABI")
for name,ref in (("add","add"),("multiply","multiply"),("equal","eq"),("not_equal","ne"),("less","less"),
                 ("isnan","isnan"),("any","any"),("sum","sum"),("min","min"),("max","max"),("sort","sort"),
                 ("unique","unique"),("argmin","argmin"),("argmax","argmax"),("isin","isin"),("take","take"),
                 ("concatenate","concat"),("where","where"),("zeros_like","zeros"),("ones_like","ones"),
                 ("array_repr","repr"),("ndarray.astype","cast_u"),("ndarray.__setitem__","assign")):
    enhance("numpy."+name,["strna_"+k+"_"+ref for k in ("none","nan","text","int")])
for name,ref in (("upper","upper"),("replace","replace"),("strip","strip"),("isalpha","alpha"),("str_len","strlen")):
    enhance("numpy.strings."+name,["strna_"+k+"_"+ref for k in ("none","nan","text","int")])
for name,go in (
    ("find","FindWithBounds"),("rfind","RFindWithBounds"),("index","IndexWithBounds"),("rindex","RIndexWithBounds"),
    ("count","CountWithBounds"),("startswith","StartsWithBounds"),("endswith","EndsWithBounds"),
    ("replace","ReplaceWithCounts"),("center","CenterWithWidths"),("ljust","LJustWithWidths"),("rjust","RJustWithWidths"),
    ("zfill","ZFillWithWidths"),("strip","StripWithChars"),("lstrip","LStripWithChars"),("rstrip","RStripWithChars"),
    ("expandtabs","ExpandTabsWithSizes"),("slice","SliceWithBounds"),
):
    enhance("numpy.strings."+name,["strarray_"+k+"_"+name for k in ("s","u","t")],"; npstrings."+go,
            "S/U/T; character operands and typed integer/string options broadcast together, exact output descriptor; scalar or array parameters")
for name in ("dtype","record","recarray","void","save","load","promote_types","result_type","ndarray.getfield","ndarray.setfield"):
    enhance("numpy."+name,["titles_bytes","titles_speed","titles_itemsize","titles_npy_file","titles_mismatch"],
            "; records.FieldSpec.Title")

for name in np.dtypes.__all__:
    if name in ("StringDType","register_dlpack_dtype"):continue
    entry="numpy.dtypes."+name
    go="dtypes.New("+repr(name)+")"
    refs=["dtype_class_"+name]
    if name in ("VoidDType","DateTime64DType","TimeDelta64DType"):
        equivalent(entry,go,refs,"legacy non-instantiable Python metaclass uses explicit Go width/schema/unit options; metadata and zero arrays")
    else:
        add(entry,go,refs,"NumPy 64-bit dtype class identity metadata, alias type number, endian, alignment, itemsize and zero ndarray")
enhance("numpy.dtypes.StringDType",["dtype_class_StringDType","dtype_nullable","dtype_string_cast"],"; dtypes.New",
        "explicit Go UTF-8 descriptor, sentinel and strict metadata, physical missing slots and casting predicates; no Python pointer ABI")
equivalent("numpy.dtypes.register_dlpack_dtype","dtypes.DLPackRegistry.RegisterDLPackDType",["dtype_dlpack"],
           "isolated Go descriptor registry, code/bits validation, idempotent registration, import/export conflict behavior and built-in precedence; no foreign C capsule")
for name,go,refs in (
    ("record","rec.NewRecord/RecordFromBytes",["rec_record"]),
    ("recarray","rec.RecArray",["rec_recarray","rec_strided","rec_bad_strides"]),
    ("format_parser","rec.FormatParser",["rec_parser"]),
):
    equivalent("numpy.rec."+name,go,refs,"typed Go record/descriptor adapter; field layout, names/titles, bounded raw byte strides and array ownership")
for name,go,refs in (
    ("fromarrays","rec.FromArrays",["rec_fromarrays","rec_subarrays","rec_bad_arrays"]),
    ("fromrecords","rec.FromRecords",["rec_fromrecords"]),
    ("fromstring","rec.FromString",["rec_fromstring","rec_buffer_short"]),
    ("fromfile","rec.FromFile",["rec_fromfile","rec_file_short"]),
    ("array","rec.Array",["rec_array_0","rec_array_1","rec_share_0","rec_share_1"]),
    ("find_duplicate","rec.FindDuplicate",["rec_duplicate"]),
):
    add("numpy.rec."+name,go,refs,"explicit Go typed rows/columns/buffer/reader and selected shape, offsets, constructor inference and copy semantics")
enhance("numpy.promote_types",["dtype_matrix_promote","dtype_matrix_codes","rectime_bad_promotion"],"; dtypes.Promote",
        "descriptor-level 48 by 48 family/width/unit/endian/alias promotions; structured temporal field promotion and exceptions")
enhance("numpy.result_type",["dtype_matrix_promote","rectime_concat","temporal_mixed_concat"],"; dtypes.Promote")
enhance("numpy.can_cast",["dtype_matrix_cast","dtype_struct_cast","dtype_string_cast"],"; dtypes.CanCast",
        "descriptor-level casting predicates for all five rules; 48 by 48 complete grid plus structured and missing-sentinel combinations")
for name in ("astype","ndarray.astype"):
    enhance("numpy."+name,["temporal_cast","temporal_from","dtype_cast_endian","dtype_cast_void"],"; dtypes.Descriptor.Cast",
            "numeric, temporal, explicit character widths and native scalar void casts; descriptor-level casting rules and endian")
for name,ref in (("concatenate","concat"),("where","where"),("equal","equal")):
    enhance("numpy."+name,["rectime_"+ref,"temporal_mixed_"+ref])
for name in ("dtype","record","recarray","save","load","ndarray.getfield","ndarray.setfield"):
    enhance("numpy."+name,["rectime_file","rectime_when","rectime_delta","rectime_stamp","rectime_bytes"],
            "; records.Temporal/np.FromTemporalBuffer")
for name in ("add","negative","absolute","isnan","isinf","isfinite","isnat","ufunc"):
    enhance("numpy."+name,["temporal_ufunc_"+name] if name!="ufunc" else ["temporal_ufunc"])

for name,go,refs in (
    ("append_fields","AppendFields",["rf_append_fields","rf_append_bad_names"]),
    ("assign_fields_by_name","AssignFieldsByName",["rf_assign_keep","rf_assign_zero"]),
    ("drop_fields","DropFields",["rf_drop_fields","rf_drop_all"]),
    ("find_duplicates","FindDuplicates",["rf_dup","rf_dup_index"]),
    ("flatten_descr","FlattenDescr",["rf_flatten_descr"]),
    ("get_fieldstructure","GetFieldStructure",["rf_get_fieldstructure"]),
    ("get_names","GetNames",["rf_get_names"]),
    ("get_names_flat","GetNamesFlat",["rf_get_names_flat"]),
    ("join_by","JoinBy",["rf_join"]),
    ("merge_arrays","MergeArrays",["rf_merge_flat","rf_merge_nested","rf_merge_flat_unequal"]),
    ("rec_append_fields","RecAppendFields",["rf_rec_append_fields"]),
    ("rec_drop_fields","RecDropFields",["rf_rec_drop_fields"]),
    ("rec_join","RecJoin",["rf_rec_join"]),
    ("recursive_fill_fields","RecursiveFillFields",["rf_recursive_fill"]),
    ("rename_fields","RenameFields",["rf_rename_fields","rf_rename_shares","rf_rename_writeback"]),
    ("repack_fields","RepackFields/RepackSchema",["rf_repack"]),
    ("require_fields","RequireFields",["rf_require_fields"]),
    ("stack_arrays","StackArrays",["rf_stack_arrays","rf_stack_filled","rf_stack_bad_dtype"]),
    ("structured_to_unstructured","StructuredToUnstructured",["rf_struct_to_unstruct","rf_uniform","rf_struct_bad_cast","rf_struct_no_fields"]),
    ("unstructured_to_structured","UnstructuredToStructured",["rf_unstruct_to_struct","rf_unstruct_shares","rf_unstruct_writeback","rf_unstruct_nested","rf_unstruct_strided","rf_unstruct_bad_size","rf_unstruct_both"]),
):
    add("numpy.lib.recfunctions."+name,"recfunctions."+go,refs,
        "explicit Go records, field and subarray layouts, selected casting/copy semantics, explicit masks and defaults; joins require unique unmasked keys")
equivalent("numpy.lib.recfunctions.apply_along_fields","recfunctions.ApplyAlongFields",["rf_apply_along_fields"],
           "Go callback receiving the promoted ndarray and last axis instead of Python keyword dispatch")
equivalent("numpy.lib.stride_tricks.as_strided","np.AsStrided",["stride"],
           "signed/zero/overlapping byte strides, alias/writeback and explicit write flag; unsafe out-of-allocation addresses and unaligned pointer slots are rejected")
add("numpy.lib.stride_tricks.sliding_window_view","np.SlidingWindowView",["window"],
    "window dimensions and axes, repeated/negative axes, zero width, readonly by default and storage sharing")
for name,go,refs in (
    ("magic","Magic",["format_magic"]),
    ("read_magic","ReadMagic",["format_read_magic"]),
    ("dtype_to_descr","DTypeToDescr",["format_descr"]),
    ("descr_to_dtype","DescrToDType",["format_descr_roundtrip"]),
    ("header_data_from_array_1_0","HeaderDataFromArray",["format_header_data"]),
    ("read_array_header_1_0","ReadArrayHeader1",["format_header_1","format_header_shape_1","format_header_dtype_1","format_header_limit_1"]),
    ("read_array_header_2_0","ReadArrayHeader2",["format_header_2","format_header_shape_2","format_header_dtype_2","format_header_limit_2","format_long_read"]),
    ("write_array_header_1_0","WriteArrayHeader1",["format_header_1","format_long_v1"]),
    ("write_array_header_2_0","WriteArrayHeader2",["format_header_2","format_long_v2"]),
    ("write_array","WriteArray",["format_write","format_unicode"]),
    ("read_array","ReadArray",["format_read","format_unicode"]),
    ("open_memmap","OpenMemMap",["format_mmap"]),
):
    add("numpy.lib.format."+name,"npyformat."+go,refs,
        "NPY versions 1/2/3, restricted descriptor/header literals, exact byte order, shape/order, bounded header and supported fixed-width payloads; no pickle")
equivalent("numpy.lib.format.drop_metadata","npyformat.DropMetadata",["format_drop_metadata"],
           "Go dtype descriptors do not carry arbitrary Python metadata dictionaries")
equivalent("numpy.lib.format.isfileobj","npyformat.IsFileObj",["format_isfileobj"],
           "Go OS file versus in-memory stream, no Python IO subclass protocol")

for name,go in (("sqrt","Sqrt"),("log","Log"),("log2","Log2"),("log10","Log10"),
                ("arcsin","Arcsin"),("arccos","Arccos"),("arctanh","Arctanh"),("power","Power"),("logn","Logn")):
    if name in ("power","logn"):
        refs=["scimath_"+name]
    else:
        refs=[f"scimath_{i}_{mode}_{name}" for i in range(16) for mode in ("inside","outside","nonreal") if f"scimath_{i}_{mode}_{name}" in fixture_names]
        refs += [f"scimath_edge_{n}_{name}" for n in ("half","extended","extended_positive","signed_min","branch","empty","scalar")]
    for ns in ("numpy.lib.scimath","numpy.emath"):
        add(ns+"."+name,"scimath."+go,refs,
            "16 numeric dtype families, whole-array domain-sensitive complex promotion, principal branches and broadcast binary dtype matrix; selected signed zero/nonfinite cases")
equivalent("numpy.lib.array_utils.byte_bounds","np.ByteBounds",["lib_bounds"],
           "opaque Go allocation address tokens, relative byte spans, empty arrays and pointer-managed slot spans; no foreign pointers")
add("numpy.lib.array_utils.normalize_axis_index","np.NormalizeAxisIndex",["lib_axis_matrix","lib_axis_name"],"rank/axis matrix and named axis errors")
add("numpy.lib.array_utils.normalize_axis_tuple","np.NormalizeAxisTuple",["lib_axis_tuple"],"axis order, negative indices, duplicate policy and invalid-axis errors")
equivalent("numpy.lib.Arrayterator","np.NewArrayterator",["lib_iter"],"explicit shared ndarray/memmap block iterator, bounded buffers and forward slice ranges")
for name,go,refs in (
    ("__getitem__","Slice",["lib_iter_slice","lib_iter_index","lib_iter_nested"]),
    ("__array__","Array",["lib_iter_slice_array","lib_iter_nested_array","lib_iter_scalar"]),
    ("__iter__","Blocks",["lib_iter"]),
    ("flat","Flat",["lib_iter"]),
    ("shape","Shape",["lib_iter_slice_shape","lib_iter_nested_shape","lib_iter_empty_shape"]),
    ("var","Var",["lib_iter_slice_asarray","lib_iter_nested_asarray"]),
    ("buf_size","BufferSize",["lib_iter_7_buffer","lib_iter_0_buffer"]),
    ("start","Start",["lib_iter_slice_indices","lib_iter_nested_indices"]),
    ("stop","Stop",["lib_iter_slice_indices","lib_iter_nested_indices"]),
    ("step","Step",["lib_iter_slice_indices","lib_iter_nested_indices"]),
):
    equivalent("numpy.lib.Arrayterator."+name,"np.Arrayterator."+go,refs,
               "explicit Go view/iterator methods, forward positive slicing and shared blocks; raw source Var versus explicit sliced Array")
add("numpy.lib.NumpyVersion","lib.NewNumpyVersion",["lib_version"],"version metadata and pre/dev/release comparison")
version_fields={"vstring":"VString","version":"Version","major":"Major","minor":"Minor","bugfix":"Bugfix",
                "pre_release":"PreRelease","is_devversion":"IsDevVersion"}
for name,go in version_fields.items():
    add("numpy.lib.NumpyVersion."+name,"lib.NumpyVersion."+go,["lib_version_meta"],"parsed version metadata")
for name in ("__lt__","__le__","__eq__","__ne__","__gt__","__ge__"):
    equivalent("numpy.lib.NumpyVersion."+name,"lib.NumpyVersion.Compare",["lib_version_compare","lib_version_bad_compare"],"three-way Go comparator instead of Python operator overloading")
add("numpy.lib.NumpyVersion.__repr__","lib.NumpyVersion.String",["lib_version_meta"],"version diagnostics")
equivalent("numpy.lib.introspect.opt_func_info","lib.OptFuncInfo",["lib_opt"],
           "regex-filtered actual Go float32/64 dense/general dispatch paths; NumPy platform SIMD target names are not simulated")
equivalent("numpy.lib.mixins.NDArrayOperatorsMixin","lib.NewNDArrayOperatorsMixin",["lib_mixin"],
           "explicit Go ufunc callback, operand order and in-place output dispatch; no language operator overloading")
mixin_methods={}
for op in "lt le eq ne gt ge add sub mul matmul truediv floordiv mod divmod pow lshift rshift and xor or".split():
    mixin_methods["__"+op+"__"]="Binary"
    if op not in "lt le eq ne gt ge".split():
        mixin_methods["__r"+op+"__"]="Reflected"
        if op!="divmod":
            mixin_methods["__i"+op+"__"]="InPlace"
for op in "neg pos abs invert".split():
    mixin_methods["__"+op+"__"]="Unary"
for name,go in mixin_methods.items():
    equivalent("numpy.lib.mixins.NDArrayOperatorsMixin."+name,"lib.NDArrayOperatorsMixin."+go,
               ["lib_mixin_dispatch","lib_mixin_binary","lib_mixin_reflected","lib_mixin_inplace","lib_mixin_divmod_q"],
               "operator name, forward/reflected argument order, in-place out and unary callback")
equivalent("numpy.lib.npyio.DataSource","npyformat.NewDataSource",["lib_data"],"explicit Go cache lifetime, local/plain/gzip/bzip2 reads and HTTP(S) cache; no Python file protocol")
for name,go,refs in (("abspath","AbsPath",["lib_data_abspath","lib_data_http_path"]),
                     ("exists","Exists",["lib_data_exists","lib_data_http_exists","lib_data_http_cached_exists"]),
                     ("open","Open",["lib_data_open","lib_data_append","lib_data_gzip_write","lib_data_missing","lib_data_http","lib_data_http_cached","lib_data_http_write"])):
    equivalent("numpy.lib.npyio.DataSource."+name,"npyformat.DataSource."+go,refs,
               "Go byte streams, existing file modes, compression resolution and explicit HTTP(S) cache; no XZ/LZMA, bzip2 writing or Python text encodings")
equivalent("numpy.lib.npyio.NpzFile","npyformat.OpenNPZ/OpenNPZPath",["lib_npz"],"lazy ordered ZIP mapping, original dtype/endian, raw non-NPY entries and explicit ownership")
npz_methods={
    "close":("Close",["lib_npz_closed","lib_npz_owned_close"]),
    "get":("Get",["lib_npz_a","lib_npz_get_default"]),
    "items":("Items",["lib_npz_items"]),
    "keys":("Keys",["lib_npz_keys"]),
    "values":("Values",["lib_npz_values"]),
    "files":("Files",["lib_npz_keys"]),
    "fid":("FID",["lib_npz_owned","lib_npz_owned_close"]),
    "zip":("Zip",["lib_npz_zip_names","lib_npz_zip_meta"]),
    "f":("At",["lib_npz_attribute"]),
    "__getitem__":("At",["lib_npz_z","lib_npz_z_raw","lib_npz_readme","lib_npz_raw","lib_npz_struct","lib_npz_missing","lib_npz_lazy_bad"]),
    "__contains__":("Contains",["lib_npz_contains"]),
    "__iter__":("Keys",["lib_npz_keys"]),
    "__len__":("Len",["lib_npz_len"]),
    "__repr__":("String",["lib_npz_repr"]),
    "__enter__":("OpenNPZ",["lib_npz_archive","lib_npz_lazy_good"]),
    "__exit__":("Close",["lib_npz_closed","lib_npz_owned_close"]),
}
for name,(go,refs) in npz_methods.items():
    equivalent("numpy.lib.npyio.NpzFile."+name,"npyformat.NpzFile."+go,refs,
               "ordered lazy mapping, raw bytes versus ndarray, archive metadata and explicit Go Close/defer; no pickle")

# Stateful class contracts. Include declared class members and the numerical
# protocol methods explicitly; Python-only object protocol is not enumerated.
def class_members(group, prefix, methods, scope, adaptation=()):
    for name,(go,fixtures) in methods.items():
        refs=[fixtures] if isinstance(fixtures,str) else fixtures
        fn=equivalent if name in adaptation else add
        fn(group+"."+name,prefix+go,refs,scope)

iterator_scope="stateful C-order flat cursor, explicit Go Next/Get/Assign, shared base; tested native integer array subset"
flat_methods={
    "base":("Base","class_flat_base"),"coords":("Coords","class_flat_state"),
    "index":("Index","class_flat_state"),"copy":("Copy","class_flat_copy"),
    "__iter__":("Iter/Next","class_flat_values"),"__next__":("Next","class_flat_state"),
    "__len__":("Len","class_flat_copy_cursor"),"__array__":("Array",["class_flat_array","class_flat_shared"]),
    "__getitem__":("Get",["class_flat_slice","class_flat_scalar","class_flat_gather","class_flat_bad_mask","class_flat_oob"]),
    "__setitem__":("Assign",["class_flat_assign","class_flat_repeated","class_flat_mask","class_flat_empty_rhs"]),
}
class_members("numpy.flatiter","np.FlatIterator.",flat_methods,iterator_scope)
broadcast_methods={
    "index":("Index","class_broadcast_state"),"iters":("Iterators","class_broadcast_state"),
    "numiter":("NumIter","class_broadcast_meta"),"nd":("NDim","class_broadcast_meta"),
    "ndim":("NDim","class_broadcast_meta"),"size":("Size","class_broadcast_meta"),
    "shape":("Shape","class_broadcast_meta"),"reset":("Reset","class_broadcast_reset"),
    "__iter__":("Next","class_broadcast_values"),"__next__":("Next",["class_broadcast_values","class_broadcast_zero"]),
}
class_members("numpy.broadcast","np.BroadcastIterator.",broadcast_methods,"stateful broadcast tuple iteration, independent flat cursors and reset")
class_members("numpy.ndindex","np.IndexIterator.",{"__iter__":("Next","class_ndindex"),"__next__":("Next",["class_ndindex","class_ndindex_empty"])},"retained C-order multidimensional index cursor")
enumerate_methods={"__iter__":("Next","class_ndenumerate"),"__next__":("Next","class_ndenumerate"),"iter":("Iterator","class_ndenumerate_iter")}
class_members("numpy.ndenumerate","np.EnumerateIterator.",enumerate_methods,"retained C-order coordinate/value cursor")

nditer_methods={
    "close":("Close","class_nditer_writeback"),"copy":("Copy","class_nditer_copy"),
    "debug_print":("DebugPrint","class_nditer_empty"),"dtypes":("DTypes","class_nditer_c_dtypes"),
    "enable_external_loop":("EnableExternalLoop","class_nditer_external_enable"),
    "finished":("Finished","class_nditer_empty"),"has_delayed_bufalloc":("HasDelayedBufAlloc","class_nditer_delayed"),
    "has_index":("HasIndex","class_nditer_c_index_meta"),"has_multi_index":("HasMultiIndex","class_nditer_multi_meta"),
    "index":("Index/SetIndex",["class_nditer_c_index_tracked","class_nditer_copy"]),
    "iterationneedsapi":("IterationNeedsAPI","class_nditer_c_meta"),
    "iterindex":("IterIndex/SetIterIndex",["class_nditer_c_steps","class_nditer_seek_iterindex"]),
    "iternext":("IterNext","class_nditer_iternext"),"iterrange":("IterRange/SetIterRange",["class_nditer_range_meta","class_nditer_range"]),
    "itersize":("IterSize","class_nditer_c_meta"),"itviews":("ItViews",["class_nditer_f_view0","class_nditer_broadcast_view0","class_nditer_broadcast_view1"]),
    "multi_index":("MultiIndex/SetMultiIndex",["class_nditer_multi_tracked","class_nditer_seek"]),
    "ndim":("NDim","class_nditer_multi_meta"),"nop":("NOp","class_nditer_broadcast_meta"),
    "operands":("Operands","class_nditer_allocate"),"remove_axis":("RemoveAxis","class_nditer_remove_axis"),
    "remove_multi_index":("RemoveMultiIndex","class_nditer_external_enable"),
    "reset":("Reset","class_nditer_reset"),"shape":("Shape",["class_nditer_c_shape","class_nditer_multi_shape"]),
    "value":("Value",["class_nditer_seek","class_nditer_iternext"]),
    "__iter__":("Iter","class_nditer_range"),"__next__":("Next",["class_nditer_c_values","class_nditer_external_values"]),
    "__getitem__":("Get","class_nditer_iternext"),"__setitem__":("Set",["class_nditer_allocate","class_nditer_writeback"]),
    "__enter__":("Close","class_nditer_writeback"),"__exit__":("Close","class_nditer_writeback"),
}
nditer_scope="explicit flags and native numeric C/F/K traversal, coalesced views, selected range/index combinations, external loop and buffered dtype writeback; no Python reference protocol, op_axes or all flag combinations"
class_members("numpy.nditer","np.NDIterator.",nditer_methods,nditer_scope,["debug_print","__enter__","__exit__","iterationneedsapi"])
equivalent("numpy.nditer.__enter__","np.NewNDIterator","class_nditer_c_values","explicit Go constructor and defer Close manage iterator lifetime")

ufunc_methods={name:(go,"class_ufunc_metadata") for name,go in {
    "__name__":"Name","nin":"NIn","nout":"NOut","nargs":"NArgs","ntypes":"NTypes",
    "identity":"Identity/HasIdentity","signature":"Signature","types":"Types"}.items()}
ufunc_methods.update({
    "resolve_dtypes":("ResolveDTypes","class_ufunc_resolve"),
    "__call__":("Call/CallWithOptions/CallAll/CallAllWithOptions",["class_ufunc_frexp","class_ufunc_modf","class_ufunc_divmod","class_ufunc_multi_out0","class_ufunc_matmul","behavior_ufunc"]),
    "at":("AtIndices",["class_ufunc_at_multi","class_ufunc_at_unary","final_ufunc_at","exmath_ufunc_at"]),
    "outer":("OuterWithOptions",["class_ufunc_outer_out","final_ufunc_outer","exmath_ufunc_outer"]),
    "reduce":("ReduceWithOptions",["behavior_reduceopt","exmath_ufunc_reduce"]),
    "accumulate":("Accumulate/AccumulateInto",["class_ufunc_accumulate_out","behavior_accumulate","exmath_ufunc_accumulate"]),
    "reduceat":("ReduceAt",["behavior_reduceat","exmath_ufunc_reduceat"]),
})
class_members("numpy.ufunc","np.UFunc.",ufunc_methods,"pinned loop metadata for all exported ufunc names; selected executable numeric loops, type-only weak promotion, out/where, reductions and indexed application; not Python dispatch")
poly1d_methods={
    **{n:("Coefficients","class_poly_coeff") for n in ["c","coef","coefficients","coeffs"]},
    **{n:("Order","class_poly_meta") for n in ["o","order"]},
    **{n:("Roots","class_poly_roots") for n in ["r","roots"]},
    "variable":("Variable","class_poly_variable"),"deriv":("Deriv","class_poly_deriv"),"integ":("Integ","class_poly_integ"),
    "__len__":("Len","class_poly_meta"),"__getitem__":("Coefficient","class_poly_meta"),
    "__setitem__":("SetCoefficient",["class_poly_shared","class_poly_expand","class_poly_leading_zero","class_poly_bad_power"]),
    "__call__":("Eval/Compose",["class_poly_eval","class_poly_compose"]), "__repr__":("Repr","class_poly_repr"),
    "__str__":("Format","class_poly_str"),"__add__":("Add","class_poly_add"),
    "__sub__":("Sub","class_poly_sub"),"__mul__":("Mul/Scale",["class_poly_mul","class_poly_scale"]),
    "__truediv__":("Divide/DivideScalar",["class_poly_quotient","class_poly_remainder","class_poly_scalar_div"]),
    "__pow__":("Pow","class_poly_pow"),"__neg__":("Negative","class_poly_negative"),
    "__pos__":("Positive","class_poly_positive"),"__eq__":("Equal","class_poly_equal"),
}
class_members("numpy.poly1d","poly.Poly1D.",poly1d_methods,"legacy descending coefficients, native numeric dtype, view sharing, scalar coefficient mutation, explicit Go methods and selected arithmetic/formatting")
memmap_methods={
    "filename":("MappingInfo","class_memmap_filename"),"offset":("MappingInfo","class_memmap_metadata"),
    "mode":("MappingInfo","class_memmap_mode"),"flush":("FlushMapping",["class_memmap_view","class_memmap_bytes"]),
    "__array_finalize__":("MappingInfo",["class_memmap_metadata","class_memmap_view_filename","class_memmap_copy"]),
}
class_members("numpy.memmap","np.Array.",memmap_methods,"mapping-specific members; inherited ndarray numerical members are counted under numpy.ndarray; original file metadata retained by views, explicit owner Close",["__array_finalize__"])
for symbol,refs,api,scope in [
    ("numpy.flatiter",["class_flat"],"; np.NewFlatIterator","stateful C-order flat cursor, indexed selection and cyclic writeback"),
    ("numpy.ndarray.flat",["class_flat"],"; Array.FlatIterator","shared C-order flat cursor or explicit flat ndarray"),
    ("numpy.broadcast",["class_broadcast"],"; BroadcastIterator.Next/Reset/Iterators","stateful broadcast iterator metadata and values"),
    ("numpy.ndindex",["class_ndindex"],"; np.NewNDIndex","retained or range-based C-order index iterator"),
    ("numpy.ndenumerate",["class_ndenumerate"],"; np.NewNDEnumerate","retained or range-based coordinate/value iterator"),
    ("numpy.nditer",["class_nditer"],"; np.NewNDIterator",nditer_scope),
    ("numpy.ufunc",["class_ufunc"],"; UFunc.CallAll/ResolveDTypes/AtIndices/Types","all pinned exported ufunc descriptors; selected executable kernels and options; no Python dispatch"),
    ("numpy.poly1d",["class_poly"],"; poly.NewPoly1D","legacy typed descending coefficients, shared views, arithmetic, evaluation and mutation"),
    ("numpy.memmap",["class_memmap"],"; Array.MappingInfo/FlushMapping","OS-backed mapping with explicit lifetime and view metadata"),
]:
    enhance(symbol,refs,api,scope)

# The same six classes are re-exported in their basis submodules. Enumerate
# their members once under numpy.polynomial.<Class>, while retaining both
# constructor names already present in the namespace rows.
series_classes=[
    ("polynomial","Polynomial","Power"),("chebyshev","Chebyshev","Chebyshev"),
    ("legendre","Legendre","Legendre"),("hermite","Hermite","Hermite"),
    ("hermite_e","HermiteE","HermiteE"),("laguerre","Laguerre","Laguerre"),
]
series_methods={
    "coef":("Series.CoefficientView",["coef","constructor_copy","copy_mutated","copy_original"]),
    "domain":("Series.Domain",["domain","default_domain"]),"window":("Series.Window",["window","default_window"]),
    "symbol":("Series.Symbol",["symbol","float32_symbol","bad_symbol"]),
    "basis_name":("{basis}.BasisName","basis_name"),"maxpower":("{basis}.MaxPower",["metadata","pow_17","bad_power"]),
    "basis":("{basis}.SeriesBasis",["basis","basis_default","bad_basis"]),
    "cast":("{basis}.Cast","cast"),"convert":("Series.Convert",["convert","float32_convert","complex_convert"]),
    "copy":("Series.Copy",["copy","copy_original","copy_mutated"]),
    "cutdeg":("Series.CutDegree",["cutdeg","cutdeg_large","bad_cutdeg"]),
    "degree":("Series.Degree","metadata"),"deriv":("Series.Derivative","deriv"),
    "fit":("{basis}.FitSeries",["fit","fit_infer","fit_default","fit_constant_domain"]),
    "fromroots":("{basis}.SeriesFromRoots",["fromroots","fromroots_default","fromroots_infer","fromroots_complex"]),
    "has_samecoef":("Series.HasSameCoefficients","comparisons"),"has_samedomain":("Series.HasSameDomain","comparisons"),
    "has_sametype":("Series.HasSameType","comparisons"),"has_samewindow":("Series.HasSameWindow","comparisons"),
    "identity":("{basis}.Identity",["identity","identity_default","identity_eval"]),
    "integ":("Series.Integrate",["integ","integ_default"]),
    "linspace":("Series.Linspace",["linspace_default","linspace_reverse","linspace_empty"]),
    "mapparms":("Series.MapParameters","mapparms"),
    "roots":("Series.Roots","roots"),"trim":("Series.Trim",["trim","trim_tol","trim_all","bad_trim"]),
    "truncate":("Series.Truncate",["truncate","truncate_large","bad_truncate"]),
    "__len__":("Series.Len","metadata"),"__iter__":("Series.Iter","iter"),
    "__call__":("Series.Eval/Compose",["call","compose","complex_eval"]),
    "__repr__":("Series.Repr",["repr","float32_repr"]),
    "__str__":("Series.String","str"),"__format__":("Series.Format",["ascii","unicode","float32_format","bad_format"]),
    "__neg__":("Series.Negative",["neg","float32_neg"]),"__pos__":("Series.Positive",["pos","copy_equal"]),
    "__add__":("Series.Add",["add","add_coef","bad_type","bad_domain","bad_window","different_symbol"]),
    "__sub__":("Series.Sub","sub"),"__mul__":("Series.Mul",["mul","complex_mul"]),
    "__pow__":("Series.Pow",["pow","pow_zero","pow_17","bad_power","bad_power_negative"]),
    "__truediv__":("Series.DivideScalar",["true_div","bad_true_div"]),
    "__floordiv__":("Series.DivMod (quotient)","floor_div"),"__mod__":("Series.DivMod (remainder)","mod"),
    "__divmod__":("Series.DivMod",["divmod_q","divmod_r","bad_division"]),
    "__radd__":("Series.Add","radd"),"__rsub__":("Series.RSub","rsub"),"__rmul__":("Series.Mul","rmul"),
    "__rfloordiv__":("Series.DivMod (reflected quotient)","rfloordiv"),
    "__rmod__":("Series.DivMod (reflected remainder)","rmod"),
    "__rdivmod__":("Series.DivMod (reflected)",["rdivmod_q","rdivmod_r"]),
    "__eq__":("Series.Equal","comparisons"),"__ne__":("Series.Equal (negated)","comparisons"),
}
series_scope="native numerical coefficients, real domain/window, selected scalar/series arithmetic, explicit Go methods; cross-basis conversion and composition; additional extended and complex-interval behavior only where referenced; no Python object protocol"
for mod,clsname,basis in series_classes:
    prefix="series_"+mod+"_"
    group="numpy.polynomial."+clsname
    for name,(go,evidence) in series_methods.items():
        refs=[evidence] if isinstance(evidence,str) else evidence
        add(group+"."+name,"poly."+go.format(basis=basis),[prefix+r for r in refs],series_scope)
    if clsname=="Chebyshev":
        add(group+".interpolate","poly.InterpolateChebyshev",prefix+"interpolate","scalar Go callback and specified real domain")
    for symbol in (group,"numpy.polynomial."+mod+"."+clsname):
        enhance(symbol,[prefix.rstrip("_")],"; poly.Series",scope=series_scope)

for name,go,refs,scope in [
    ("trimseq","TrimSequence","polyutils_trimseq","exact zero trim; typed shared view, empty and all-zero sequences"),
    ("as_series","AsSeries","polyutils_as_series","explicit input array list, native numeric common dtype, independent copies; bool/no common type rejected"),
    ("trimcoef","TrimCoefficients","polyutils_trimcoef","common typed inexact coefficients, magnitude tolerance and NaN behavior"),
    ("getdomain","GetDomain","polyutils_getdomain","real extrema and complex bounding rectangle; native numeric vectors"),
    ("mapparms","MapParameters","polyutils_mapparms","typed two-element real or complex intervals"),
    ("mapdomain","MapDomain","polyutils_mapdomain","typed real/complex affine map, multidimensional input"),
    ("format_float","FormatFloat","polyutils_format_float","float64 coefficient formatting with current print options"),
]:
    add("numpy.polynomial.polyutils."+name,"poly."+go,refs,scope)

# ndarray subclass-specific methods and explicitly tested inherited operations.
# The unchanged ndarray member surface is not duplicated in these class rows.
masked_class_methods={
    "T":("Array.Transpose","transpose"),"mT":("Array.MatrixTranspose","mT"),
    "all":("All","reduce_all"),"any":("Any","reduce_any"),"anom":("Anomalies","anom"),
    "argmax":("ArgMax","argmax"),"argmin":("ArgMin","argmin"),
    "argpartition":("Array.ArgPartition","argpartition"),"argsort":("Argsort","argsort"),
    "baseclass":("Array.BaseClass","baseclass"),"compress":("Compress","compress"),
    "compressed":("Compressed","compressed"),"copy":("Array.Copy","copy"),"count":("Count","count"),
    "cumprod":("Cumprod","cumprod"),"cumsum":("Cumsum","cumsum"),
    "data":("Array.Data",["input_raw","data_view_write"]),"diagonal":("Diagonal","diagonal"),"dot":("Dot","dot"),
    "dtype":("Array.DType","input_raw"),"fill_value":("Array.FillValue/SetFillValue",["input_fill","get_fill"]),
    "filled":("Array.Filled",["filled","filled_explicit"]),"flat":("Array.FlatIterator","flat"),
    "flatten":("Array.Flatten","flatten"),"get_fill_value":("Array.FillValue","get_fill"),
    "get_imag":("Array.Imag","complex_imag"),"get_real":("Array.Real","complex_real"),
    "harden_mask":("Array.HardenMask","harden"),"hardmask":("Array.HardMask","hard_mask_clear"),
    "ids":("Array.IDs","ids"),"imag":("Array.Imag",["imag","complex_imag"]),
    "iscontiguous":("Array.IsContiguous","meta"),"mask":("GetMask/Array.SetMask",["hard_mask_clear","hard_mask_or","soft_mask_clear","nomask","nomask_explicit"]),
    "max":("Max","reduce_max"),"mean":("Mean","reduce_mean"),"min":("Min","reduce_min"),
    "nonzero":("Nonzero","nonzero"),"partition":("Array.PartitionInPlace","partition"),
    "prod":("Prod","reduce_prod"),"product":("Prod","reduce_product"),"ptp":("PTP","reduce_ptp"),
    "put":("Put","put"),"ravel":("Array.Ravel","ravel"),"real":("Array.Real",["real","complex_real"]),
    "recordmask":("Array.RecordMask",["recordmask","shrink_recordmask"]),
    "repeat":("Repeat","repeat"),"reshape":("Array.Reshape","reshape"),
    "resize":("Array.ResizeInPlace","resize"),"round":("Round","round"),
    "set_fill_value":("Array.SetFillValue","get_fill"),"shape":("Array.Shape","meta"),
    "sharedmask":("Array.SharedMask",["input_flags","unshare_view_flags","nomask","nomask_explicit"]),
    "shrink_mask":("Array.ShrinkMask",["shrink_nomask","unshare_shrink"]),
    "soften_mask":("Array.SoftenMask","soft_mask_clear"),"sort":("Array.SortInPlace","sort"),
    "squeeze":("Squeeze","squeeze"),"std":("Std","reduce_std"),"sum":("Sum","reduce_sum"),
    "swapaxes":("SwapAxes","swapped"),"take":("Take","take"),
    "tobytes":("Array.ToBytes",["tobytes","tobytes_f"]),"tofile":("Array.ToFile","tofile"),
    "toflex":("Array.ToFlex","toflex"),"tolist":("Array.ToList",["tolist","tolist_fill"]),
    "torecords":("Array.ToFlex","torecords"),"trace":("Trace","trace"),
    "transpose":("Array.Transpose","transpose"),"unshare_mask":("Array.UnshareMask",["unshare_view","unshare_base"]),
    "var":("Var","reduce_var"),"view":("Array.View",["view","data_view_write"]),
    "astype":("Array.Astype","astype"),"byteswap":("Array.ByteSwap","byteswap"),
    "fill":("Array.Fill","fill_data"),"item":("Array.Item","item"),
    "searchsorted":("Array.SearchSorted","searchsorted"),"clip":("Clip","clip"),
    "conj":("Conjugate","complex_conj"),"conjugate":("Conjugate","complex_conj"),
    **{n:("Array."+go,"meta") for n,go in [("ndim","NDim"),("size","Size"),("strides","Strides"),("itemsize","ItemSize"),("nbytes","NBytes")]},
    "__len__":("Array.Len","meta"),"__iter__":("Array.Iter","iter"),
    "__getitem__":("Array.Index",["slice","advanced","iter"]),"__setitem__":("Array.Assign",["setitem","hard_set_masked_rhs"]),
}
masked_scope="numeric MaskedArray subset, explicit data/mask arrays, fill, hard-mask and view semantics; no Python runtime class dispatch"
for name,(go,refs) in masked_class_methods.items():
    refs=[refs] if isinstance(refs,str) else refs
    fn=equivalent if name in ("baseclass","ids","tofile","__getitem__","flat") else add
    scope=masked_scope
    if name in ("partition","argpartition"):scope="NumPy ignores mask for this operation; valid full ordering instead of unspecified partition order"
    elif name=="put":scope="tested one-dimensional hard mask put; explicit logical flat Go semantics; NumPy multidimensional hard-mask put has first-axis indexing anomaly"
    elif name=="resize":scope="reject in-place resize of masked arrays, use functional ma.Resize"
    elif name=="recordmask":scope="numeric mask; not a structured record-wide reduction"
    fn("numpy.ma.MaskedArray."+name,"ma."+go,["subclass_ma_"+r for r in refs],scope)
masked_iterator_methods={
    "ma":("Base","flat_iterators"),"dataiter":("DataIterator","flat_iterators"),
    "maskiter":("MaskIterator","flat_iterators"),"__iter__":("Next","flat_values"),
    "__next__":("Next",["flat_values","flat_mask"]),"__getitem__":("Get",["flat_slice","flat_gather"]),
    "__setitem__":("Assign",["flat_assign","flat_repeat"]),
}
for name,(go,refs) in masked_iterator_methods.items():
    refs=[refs] if isinstance(refs,str) else refs
    equivalent("numpy.ma.core.MaskedIterator."+name,"ma.MaskedIterator."+go,
               ["subclass_ma_"+r for r in refs],"explicit Go cursor and masked scalar payload/flag, C-order indexed writeback")

char_class_methods={}
for name,go in zip(
    "capitalize center count decode encode endswith expandtabs find index isalnum isalpha isdecimal isdigit islower isnumeric isspace istitle isupper join ljust lower lstrip partition replace rfind rindex rjust rpartition rsplit rstrip split splitlines startswith strip swapcase title translate upper zfill argsort".split(),
    "Capitalize Center Count Decode Encode EndsWith ExpandTabs Find FindIndex IsAlnum IsAlpha IsDecimal IsDigit IsLower IsNumeric IsSpace IsTitle IsUpper Join LJust Lower LStrip Partition Replace RFind RFindIndex RJust RPartition RSplit RStrip Split SplitLines StartsWith Strip SwapCase Title Translate Upper ZFill Argsort".split()):
    char_class_methods[name]=(go,name)
char_class_methods.update({name:(go,fixture) for name,go,fixture in [
    ("__add__","Add","add"),("__radd__","Add (reflected)","radd"),("__mul__","Multiply","multiply"),
    ("__rmul__","Multiply","rmul"),("__mod__","Mod","mod"),("__eq__","Compare","eq"),("__ne__","Compare","ne"),
    ("__lt__","Compare","lt"),("__le__","Compare","le"),("__gt__","Compare","gt"),("__ge__","Compare","ge"),
    ("__getitem__","AtValue/IndexChar","scalar"),("__setitem__","Array.SetValue","view_write"),
    ("copy","CopyChar","copy"),("reshape","ReshapeChar","reshape"),("transpose","TransposeChar","transpose"),
    ("T","TransposeChar","transpose"),("ravel","Ravel","ravel"),("flatten","Flatten","flatten"),
    ("take","Take","take"),("repeat","Repeat","repeat"),("squeeze","Squeeze","squeeze"),
    ("swapaxes","SwapAxes","swapaxes"),("compress","Compress","compress"),("diagonal","Diagonal","diagonal"),
    ("astype","CastChar","cast"),("view","ViewChar","view"),
]})
for name,(go,fixture) in char_class_methods.items():
    refs=["subclass_char_"+kind+"_"+fixture for kind in ("s","u")]
    if name=="__getitem__":refs += ["subclass_char_u_slice","subclass_char_s_gather"]
    if name=="argsort":refs += ["subclass_char_u_argsort_trailing","subclass_char_s_argsort_trailing"]
    add("numpy.char.chararray."+name,"char.CharArray."+go,refs,
        "fixed-width S/U class subset; raw ndarray storage with character-scalar whitespace rules and explicit Go class-preserving transform or ordinary ndarray result")

rec_class_methods={
    "field":("Field/FieldAt/Column.NDArray",["field","nested"]),
    "__getitem__":("Index/Slice/Field/SelectFields",["gather","reverse","multifield","field"]),
    "__setitem__":("Assign/Column.Assign",["assign","multifield_write"]),
    "__getattribute__":("Field/Subrecords",["field_name","field_collision","nested_x","nested_flag"]),
    "__setattr__":("Column.Assign",["scalar_write","multifield_write"]),
    "__array_finalize__":("FromNDArray",["reverse_strides","multifield_layout","gather_id"]),
}
record_class_methods={
    "__getitem__":("Field/FieldAt",["scalar_index","scalar_title","scalar_last"]),
    "__setitem__":("Set/SetAt",["scalar_write_id","scalar_write_shape"]),
    "__getattribute__":("Field",["scalar_title","scalar_last"]),
    "__setattr__":("Set",["scalar_write_nested","scalar_write_id"]),
    "pprint":("PPrint",["pprint","pprint_integral"]),
}
for group,prefix,methods in [("numpy.recarray","records.Array.",rec_class_methods),("numpy.record","records.Record.",record_class_methods)]:
    for name,(go,refs) in methods.items():
        fn=equivalent if name in ("__getattribute__","__setattr__","__array_finalize__") else add
        fn(group+"."+name,"records.FromNDArray" if name=="__array_finalize__" else prefix+go,["subclass_rec_"+r for r in refs],
           "typed explicit field and record view contract, no Python name/attribute dispatch; preserve original field offsets")
for symbol,refs,api,scope in [
    ("numpy.ma.MaskedArray",["subclass_ma"],"; ma.Array/MaskedIterator","numeric masked ndarray with explicit data/mask, class metadata, view/mutation, conversions and numeric class operations"),
    ("numpy.char.chararray",["subclass_char"],"; char.CharArray class methods","legacy S/U character subclass adapter; character methods and class-preserving transformations"),
    ("numpy.recarray",["subclass_rec"],"; records.Array.FieldAt/SelectFields/Index/Assign","structured record class and explicit typed field selection, multi-field storage-sharing views"),
    ("numpy.record",["subclass_rec"],"; records.Record.FieldAt/SetAt/PPrint","structured scalar including ordered field access and selected pretty-printing"),
]:
    enhance(symbol,refs,api,scope)

dtype_class_methods={
    **{n:("Metadata."+go,"dtype_0_meta") for n,go in [("name","Name"),("str","Str"),("kind","Kind"),("char","Char"),("byteorder","ByteOrder")]},
    **{n:("Metadata."+go,"dtype_0_flags") for n,go in [("alignment","Alignment"),("num","Num"),("hasobject","HasObject")]},
    "itemsize":("ItemSize","dtype_sub"),"base":("Base","dtype_sub"),"shape":("Shape","dtype_sub"),
    "ndim":("NDim","dtype_sub"),"subdtype":("Subdtype","dtype_sub"),
    "descr":("Descr",["dtype_struct_descr","dtype_sub","dtype_order"]),
    "fields":("Field/FieldAt",["dtype_field","dtype_title","dtype_last"]),
    "names":("Names/WithNames",["dtype_names","dtype_renamed"]),
    "flags":("Flags",["dtype_struct_flags","dtype_order","dtype_nullable_flags"]),
    "isalignedstruct":("IsAlignedStruct","dtype_struct_flags"),
    "isbuiltin":("IsBuiltin",["dtype_0_flags","dtype_order","dtype_metadata_flags"]),
    "isnative":("IsNative",["dtype_struct_order","dtype_order"]),
    "metadata":("UserMetadata/WithMetadata","dtype_metadata"),
    "type":("ScalarTypeName",["dtype_0_meta","dtype_order","dtype_nullable_meta"]),
    "newbyteorder":("NewByteOrder",["dtype_order","dtype_struct_order","dtype_string_order","dtype_bad_order"]),
    "__getitem__":("Field/FieldAt",["dtype_field","dtype_title","dtype_last","dtype_bad_field"]),
    **{n:("Compare","dtype_compare") for n in ["__eq__","__ne__","__lt__","__le__","__gt__","__ge__"]},
}
for name,(go,refs) in dtype_class_methods.items():
    refs=[refs] if isinstance(refs,str) else refs
    fn=equivalent if name=="type" else add
    fn("numpy.dtype."+name,"dtypes.Descriptor."+go,["numeric_class_"+r for r in refs],
        "pinned Linux descriptor attributes, explicit typed fields and subarray shape, native and nonnative order; no Python type identity")
for name,go in [("coerce","Coerce"),("na_object","NAObject")]:
    equivalent("numpy.dtypes.StringDType."+name,"dtypes.Descriptor."+go,"numeric_class_dtype_nullable_options","explicit Go scalar sentinel and presence flag")

matrix_class_methods={
    "A":("A","A"),"A1":("A1","A1"),"T":("T","T"),"H":("H","H"),"I":("I",["I","rect_I","bad_singular"]),
    "getA":("A","A"),"getA1":("A1","A1"),"getT":("T","T"),"getH":("H","H"),"getI":("I",["I","rect_I"]),
    **{n:(go,n) for n,go in [("sum","Sum"),("prod","Prod"),("mean","Mean"),("min","Min"),("max","Max"),
                               ("ptp","PTP"),("var","Var"),("std","Std"),("all","All"),("any","Any"),("argmin","ArgMin"),("argmax","ArgMax")]},
    "ravel":("Ravel","ravel"),"flatten":("Flatten","flatten"),
    "squeeze":("Squeeze",["row_squeeze","col_squeeze","scalar_squeeze"]),
    "reshape":("ReshapeMatrix",["reshape","reshape1"]),"transpose":("TransposeMatrix","transpose"),
    "tolist":("Array.ToList","tolist"),
    "copy":("CopyMatrix",["copy","copy_original"]),"astype":("CastMatrix","astype"),
    "real":("Real","real"),"imag":("Imag","imag"),"diagonal":("Diagonal","diagonal"),"take":("Take","take"),
    "__getitem__":("IndexMatrix",["row","col","scalar","fancy","bool","rev"]),
    "__setitem__":("AssignMatrix","assign"),
    "__mul__":("Mul",["mul","scale","bad_multiply"]),"__rmul__":("Mul (reflected)","rmul"),
    "__matmul__":("MatMul","matmul"),"__rmatmul__":("MatMul (reflected)","rmatmul"),
    "__add__":("Add","add"),"__sub__":("Sub","sub"),"__rsub__":("Sub (reflected)","rsub"),
    "__truediv__":("Divide","div"),"__rtruediv__":("Divide (reflected)","rdiv"),
    "__pow__":("Pow",["pow0","pow3","powneg"]),
}
for name,(go,refs) in matrix_class_methods.items():
    refs=[refs] if isinstance(refs,str) else refs
    add("numpy.matrix."+name,"matrix.Matrix."+go,["numeric_class_matrix_"+r for r in refs],
        "legacy 2-D matrix shape, explicit scalar-vs-array contraction and axis reduction, typed Go ndarray result")

scalar_class_methods={
    "T":("Transpose","T"),"astype":("Astype","cast"),"byteswap":("ByteSwap","byteswap"),
    "conj":("Conjugate","conj"),"conjugate":("Conjugate","conj"),"copy":("Copy","copy"),
    "dtype":("DType","value"),"item":("Item","item"),"itemsize":("ItemSize","meta"),
    "ndim":("NDim","meta"),"nbytes":("NBytes","meta"),"size":("Size","meta"),"shape":("Shape","meta"),
    "strides":("Strides","meta"),"ravel":("Ravel","ravel"),"flatten":("Flatten","flatten"),
    "reshape":("Reshape","reshape"),"real":("Real","real"),"imag":("Imag","imag"),
    "tobytes":("ToBytes","bytes"),"transpose":("Transpose","transpose"),"squeeze":("Squeeze","squeeze"),
    "device":("Device","device"),"view":("View","view"),"getfield":("GetField","getfield"),
}
for name,(go,r) in scalar_class_methods.items():
    refs=["numeric_class_scalar_"+str(i)+"_"+r for i in range(5)]
    refs.append("numeric_class_scalar_"+r)
    add("numpy.generic."+name,"scalar.Value."+go,refs,"immutable typed numeric scalar, explicit ndarray copy; no Python object or scalar buffer ABI")
for name,go,r in [("numerator","Numerator","integer"),("denominator","Denominator","integer"),("is_integer","IsInteger","integer")]:
    add("numpy.integer."+name,"scalar.Value."+go,"numeric_class_scalar_"+r,"exact integer scalar property")
integer_scalar_classes=["int8","int16","int32","int64","uint8","uint16","uint32","uint64","longlong","ulonglong"]
for i,n in enumerate(integer_scalar_classes):
    add("numpy."+n+".bit_count","scalar.Value.BitCount","numeric_class_scalar_integer_"+str(i),"population count of the exact absolute integer")
float_scalar_classes=["float16","float32","float64","longdouble"]
for i,n in enumerate(float_scalar_classes):
    for name,go in [("as_integer_ratio","AsIntegerRatio"),("is_integer","IsInteger")]:
        p="ratio" if name=="as_integer_ratio" else "is_integer"
        add("numpy."+n+"."+name,"scalar.Value."+go,"numeric_class_scalar_"+p+"_"+str(i),"exact ratio or integral property, including nonfinite errors; arbitrary-size integers")
for name,go in [("hex","Value.Hex"),("fromhex","FromHex")]:
    add("numpy.float64."+name,"scalar."+go,["numeric_class_scalar_hex","numeric_class_scalar_fromhex"],"exact float64 hexadecimal representation and parser")
for symbol,go,refs,scope in [
    ("numpy.matrix","; matrix.New/Parse","numeric_class_matrix","explicit 2D matrix adapter"),
    ("numpy.asmatrix","; matrix.New","numeric_class_matrix_shared","explicit matrix adapter with shared data"),
    ("numpy.bmat","; matrix.Block","numeric_class_matrix_block","typed block assembly"),
    ("numpy.dtype","; dtypes.Descriptor/Subarray","numeric_class_dtype","descriptor class and typed subarray"),
]:
    enhance(symbol,[refs],go,scope)

for name, prefixes in {
    "matmul":["ufopt_matmul","ufopt_mm","ufopt_mv_optional","ufopt_vm_optional","ufopt_vv_optional","ufopt_signature_matmul"],
    "matvec":["ufopt_matvec","ufopt_mv_broadcast_out"],
    "vecmat":["ufopt_vecmat","ufopt_vm_broadcast_out"],
    "vecdot":["ufopt_vecdot","ufopt_vd"],
}.items():
    enhance("numpy."+name,prefixes,"; np.UFunc.CallWithOptions",
            "16 numerical dtypes; nontrailing/reversed core axes, matched core sizes, expanded batch out, C/F/A/K layout; vecdot axis and keepdims")
singleton_shapes=["2_1_3","1_2_3","2_3_1","2_1_1_3"]
for name, prefixes in {
    "add":["ufopt_add","ufopt_signature_add","ufopt_signature_weak","ufopt_lower"]+
        ["ufopt_singleton_"+s+"_"+label for s in singleton_shapes for label in ("scalar","row","cast","mask")],
    "absolute":["ufopt_abs"],"abs":["ufopt_abs"],"less":["ufopt_less"],
    "ldexp":["ufopt_signature_ldexp"],"frexp":["ufopt_frexp_signature","ufopt_frexp_dtype"],
    "modf":["ufopt_modf","ufopt_partial"]+["ufopt_singleton_"+s+"_modf" for s in singleton_shapes],
}.items():
    symbol="numpy."+name
    if symbol in mapped:
        api="CallAllWithOptions" if name in ("frexp","modf") else "CallWithOptions"
        extra_api="" if "UFunc."+api in mapped[symbol][0] else "; np.UFunc."+api
        enhance(symbol,prefixes,extra_api,
                mapped[symbol][2]+"; selected explicit loop signatures and allocation order (where applicable)")
enhance("numpy.ufunc.__call__",["ufopt"],scope=
        "typed native numerical loop signatures, C/F/A/K allocation, out/where; generalized core-axis remapping, optional matmul core axes, vecdot axis/keepdims, expanded batch out; no Python dispatch")
enhance("numpy.ufunc.resolve_dtypes",["ufopt_signature","ufopt_abs","ufopt_less","ufopt_frexp_signature"],scope=
        "pinned numeric loop metadata, selected default/weak/reduction and explicit signature resolutions, complex absolute and mixed multi-output loops; not all dtype/constraint combinations")

enhance("numpy.ndarray.__getitem__",["ixdeep_c","ixdeep_f","ixdeep_perm","ixdeep_reverse","ixdeep_ff","ixdeep_fc","ixdeep_fb","ixdeep_single0","ixdeep_ellipsis0","ixdeep_scalar_and_f","ixdeep_empty_oob","ixdeep_maskempty","ixdeep_singleton","ixdeep_broadcast","ixdeep_index","ixdeep_oob","ixdeep_bad_empty_mask","ixdeep_uint","ixdeep_bad_index_type","ixdeep_two_ellipsis","ixdeep_endian","ixdeep_temporal","ixdeep_unicode","ixdeep_record","ixdeep_nan_payload"],scope=
        "mixed integer, boolean, scalar and slice indexing; adjacent/separated advanced axes, independent copy/base, exact strides, endian and bit-preserving fixed-width gather, empty broadcast and uint64 intp semantics")
enhance("numpy.ndarray.__setitem__",["ixdeep_assign","subdeep_record_assign"],scope=
        "broadcast RHS snapshots and writeback, mixed repeated advanced indices; scalar-to-record replication; Go deterministic last logical position")
enhance("numpy.ufunc.at",["ixdeep_at_repeat"],scope=
        "sequential repeated-index updates over multidimensional, noncontiguous targets")
for symbol,ref in (("numpy.promote_types","subdeep_promote"),("numpy.result_type","subdeep_result_type")):
    enhance(symbol,[ref],scope=mapped[symbol][2]+
            "; equal-shape fixed subarray base promotion, object fallback, incompatible shape checks")
enhance("numpy.can_cast",["subdeep_can_cast","subdeep_record_cast"],scope=
        mapped["numpy.can_cast"][2]+"; fixed subarrays, positional record fields, zero-width and generic void descriptors")
for symbol in ("numpy.astype","numpy.ndarray.astype"):
    enhance(symbol,["subdeep_cast","subdeep_unsafe_cast","subdeep_safe_reject","subdeep_record","subdeep_scalar_record","subdeep_multi_record_scalar","subdeep_bad_field_count"],
            scope=mapped[symbol][2]+"; scalar replication into fixed subarrays, positional/nested record conversion, shape-changing unsafe fields, generic void and single-field extraction")
enhance("numpy.dtype",["subdeep_can_cast","subdeep_promote","subdeep_cast"],scope=
        "explicit dtype descriptors and fixed subarray allocation, native promotion and casting predicates")

equivalent("numpy.lib.add_docstring","lib.AddDocString",["closure_doc_direct","closure_doc_conflict","closure_doc_unsupported"],
           "explicit runtime DocTarget; idempotent same text, conflicts and read-only targets, no Go compiler/Python runtime mutation")
equivalent("numpy.lib.add_newdoc","lib.AddNewDoc",["closure_doc_main","closure_doc_first","closure_doc_second","closure_doc_last","closure_doc_warn_count","closure_doc_missing_member","closure_doc_missing_target"],
           "explicit Go registry, symbol/member and list entries, cleandoc, optional warning callback, ignore attachment failures but reject missing targets")
for symbol,api,refs,scope in [
    ("numpy.copy","; np.Copy",["closure_copy_default","closure_copy"],"K default, all orders, endian-preserving independent copy"),
    ("numpy.ndarray.copy","; Array.CopyWithOrder",["closure_copy","closure_managed_copy"],"C default; C/F/A/K, exact fixed-width bytes and independent object slots"),
    ("numpy.ndarray.__getitem__","",["closure_managed_initial","closure_managed_fancy","closure_managed_projection"],"managed structured fields, detached object-slot gather and field projection"),
    ("numpy.ndarray.__setitem__","",["closure_managed_view_write","closure_managed_projection","closure_managed_copy_detached"],"packed/nested object-slot writeback and view aliasing"),
    ("numpy.dtype","",["closure_nested_dtype","closure_nested_record_descr","closure_object_subdtype","closure_string_sub_dtype","closure_managed_meta","closure_managed_descr"],"nested subarray grouping and GC-managed Object fields; reject unsupported StringDType structured fields"),
    ("numpy.dtype.hasobject","",["closure_managed_meta","closure_object_subdtype","closure_string_sub_dtype"],"recursive object metadata"),
    ("numpy.dtype.flags","",["closure_managed_meta","closure_object_subdtype","closure_string_sub_dtype"],"reference flag bits for managed fields"),
    ("numpy.dtype.subdtype","",["closure_nested_dtype","closure_nested_record_field"],"nested descriptor grouping"),
    ("numpy.dtype.descr","",["closure_nested_record_descr","closure_managed_descr"],"nested dtype literals and object field marker"),
    ("numpy.dtype.newbyteorder","",["closure_nested_record_swapped"],"nested groups and recursive byte order"),
    ("numpy.promote_types","",["closure_nested_promote","closure_nested_promote_semantic"],"nested base/shape promotion; Go recomputes safe allocation widths when upstream retains stale sizes"),
    ("numpy.can_cast","",["closure_nested_cast_predicate","closure_nested_record_predicate"],"nested vs flat descriptor casting predicates"),
    ("numpy.concatenate","",["closure_managed_concat"],"GC-managed structured references in concatenation"),
    ("numpy.equal","",["closure_managed_equal"],"structured object fields by Go value semantics"),
    ("numpy.not_equal","",["closure_managed_neq"],"structured object fields by Go value semantics"),
    ("numpy.lib.format.drop_metadata","",["closure_drop_metadata"],"recursive removal of explicit descriptor metadata"),
    ("numpy.lib.format.dtype_to_descr","",["closure_nested_record_descr"],"nested dtype literal"),
    ("numpy.lib.format.descr_to_dtype","",["closure_nested_record_field","closure_nested_record_values"],"nested tuple descriptor recovery"),
    ("numpy.lib.format.write_array","",["closure_nested_record_values"],"nested structured NPY interchange"),
    ("numpy.lib.format.read_array","",["closure_nested_record_values"],"nested structured NPY interchange"),
    ("numpy.nextafter","",["closure_half_next_pos","closure_half_next_neg","closure_half_next_zero"],"all 65536 binary16 encodings, normalized NaN payloads"),
    ("numpy.spacing","",["closure_half_spacing"],"all 65536 binary16 encodings, including both infinities"),
    ("numpy.float16","",["closure_half_decode","closure_half_roundtrip","closure_half_round_ties"],"exhaustive binary16 conversion and boundary rounding"),
]:
    if symbol in mapped:
        enhance(symbol,refs,api,mapped[symbol][2]+"; "+scope)
for symbol in ("numpy.astype","numpy.ndarray.astype"):
    enhance(symbol,["closure_cast","closure_subcast","closure_nested_cast","closure_nocopy","closure_managed_cast","closure_half_roundtrip","closure_half_round_ties"],
            "; dtypes.Descriptor.CastWithOptions",
            mapped[symbol][2]+"; explicit order/no-copy, nested subarrays, managed record conversion, exhaustive binary16; expanded F subarray values use deterministic logical replication, not unspecified upstream bytes")

# Audit-discovered public members are independently enumerated, even when a
# common ndarray kernel supplies their behavior.
audit_scalar_methods=dict(zip(
    "all any argmax argmin argsort base choose clip compress cumprod cumsum data diagonal dump dumps fill flags flat max mean min nonzero prod put repeat resize round searchsorted setfield setflags sort std sum swapaxes take to_device tofile tolist trace var".split(),
    "All Any ArgMax ArgMin Argsort Base Choose Clip Compress Cumprod Cumsum Data Diagonal Dump Dumps Fill Flags Flat Max Mean Min Nonzero Prod Put Repeat Resize Round SearchSorted SetField SetFlags Sort Std Sum SwapAxes Take ToDevice ToFile ToList Trace Var".split()))
for name,go in audit_scalar_methods.items():
    adapted_scalar=name in ("data","dump","dumps","tofile","flags","base")
    if name=="tofile":
        # Exact raw data output; same bytes as the scalar data reference.
        go="ToFile"
    reference_name=name
    fn=equivalent if adapted_scalar else add
    fn("numpy.generic."+name,"scalar.Value."+go,"audit_scalar_"+reference_name,
       "immutable numeric scalar; ndarray kernel delegation; validated temporary mutation and error behavior; NPY instead of pickle for dump/dumps")
for name,go,refs in [
    ("getfield","GetField",["audit_void_getfield","audit_void_record_getfield"]),
    ("setfield","SetField",["audit_void_setfield","audit_void_record_writeback"]),
    ("flags","Flags",["audit_void_flags"]),
    ("dtype","DType",["audit_void_dtype"]),
    ("base","Base",["audit_void_base","audit_void_record_base"]),
]:
    add("numpy.void."+name,"np.VoidScalar."+go+"; records.Record."+go,refs,
        "opaque scalar bytes and structured record storage sharing; managed references are not exposed as raw pointer fields")
for name,go,refs in [
    ("MaskedRecords","NewStructured/MaskedRecords",["audit_mrecords_data","audit_mrecords_mask","audit_mrecords_recordmask"]),
    ("mrecarray","NewStructured/MaskedRecords",["audit_mrecords_data","audit_mrecords_mask"]),
    ("fromarrays","FromMaskedArrays",["audit_mrecords_data","audit_mrecords_mask"]),
    ("fromrecords","FromMaskedRecords",["audit_mrecords_fromrecords"]),
    ("fromtextfile","FromMaskedText",["audit_mrecords_fromtextfile"]),
    ("addfield","AddField",["audit_mrecords_add"]),
]:
    add("numpy.ma.mrecords."+name,"ma."+go,refs,"explicit structured data and field mask arrays; named numeric text/record construction and field operations")
audit_mrecords_methods={"view":"View","copy":"Copy","harden_mask":"HardenMask","soften_mask":"SoftenMask","tolist":"ToList"}
for name,go in audit_mrecords_methods.items():
    prefix={"harden_mask":"harden","soften_mask":"soften","tolist":"list"}.get(name,name)
    add("numpy.ma.mrecords.MaskedRecords."+name,"ma.StructuredArray."+go,"audit_mrecords_"+prefix,
        "explicit data/mask storage ownership, mask policy and missing scalar list values")

for name,go,refs,note in [
    ("sort","SortWithOptions",["record_sort","record_nested_sort"],"stable ordering and structured field priority, including raw subarray sort keys"),
    ("argsort","ArgsortWithOptions",["record_argsort"],"stable structured order; optional unstable order among equal values"),
    ("unique","Unique",["record_unique","unique_axis","unique_empty_width"],"flattened or axis-wise unique, counts/inverse/first occurrence, structured field equality"),
    ("isin","Isin",["record_isin"],"structured membership by declared field values"),
    ("intersect1d","Intersect1D",["record_intersect"],"structured descriptors and field comparison"),
    ("searchsorted","SearchSortedWithSorter",["search_sorter","search_bad_sorter","record_search"],"sorter index validation and structured ordering"),
    ("gradient","GradientAxes",["gradient","extended_grad"],"multiple axes, scalar or nonuniform coordinates, temporal tick and extended dtype computation"),
    ("fromstring","FromString",["extended_text","extended_complex_fromstring"],"lossless x87 real parsing; NumPy rejects clongdouble fromstring"),
    ("cov","CovWithOptions",["cov_weighted","cov_both_weights","cov_cross"],"row/column variables, optional second dataset, frequency and analytic weights, ddof"),
    ("corrcoef","CorrcoefWithOptions",["corr_cross"],"optional second dataset and variable axis"),
    ("average","AverageWithWeights",["average_multi"],"matched weights or arbitrary reduction axes"),
    ("einsum","EinsumWithOptions",["einsum"],"typed safe casting, output storage, layout and optimized contraction path"),
    ("linspace","LinspaceArray",["linspace","extended_linspace"],"broadcast endpoints, sample axis, retstep, dtype and endpoint"),
    ("logspace","LogspaceArray",["logspace_array"],"broadcast endpoints and base, sample axis"),
    ("geomspace","GeomspaceArray",["geomspace"],"broadcast endpoints, real negative sequences and complex logarithmic path"),
    ("arange","ArangeTyped",["arange_uint64","extended_arange"],"exact typed integer endpoints and extended-precision steps"),
    ("pad","PadWithOptions",["pad"],"per-axis constants/stat lengths/endpoints, statistical padding, linear ramp, odd reflection and Go callback"),
    ("interp","InterpPeriod",["interp_period"],"periodic interpolation, or sorted nonperiodic grid"),
    ("take","TakeWithOptions",["take"],"raise/wrap/clip modes, explicit out"),
    ("choose","ChooseWithOptions",["choose"],"raise/wrap/clip modes and explicit out"),
    ("loadtxt","LoadText; records.LoadText",["text_quotes","text_converter","text_structured","extended_complex_loadtxt"],"quotes, converters, packed structured schemas, numeric columns"),
    ("genfromtxt","GenFromText; records.GenFromText",["text_inferred"],"numeric and inferred named structured text"),
]:
    enhance("numpy."+name,["audit_"+r for r in refs],"; np."+go,note)
for name in ("var","std","nanvar","nanstd"):
    refs={"var":["variance"],"std":["std_where"],"nanvar":["nanvar_where"],"nanstd":["nanstd_where"]}[name]
    enhance("numpy."+name,["audit_"+r for r in refs],scope="axes, keepdims, dtype, ddof/correction, broadcast where, provided mean and out")
for name,refs in [("sort",["record_sort"]),("argsort",["record_argsort"]),("searchsorted",["record_search","search_sorter"]),("var",["variance"]),("std",["std_where"]),("take",["take"]),("choose",["choose"])]:
    enhance("numpy.ndarray."+name,["audit_"+r for r in refs],scope="shared verified ndarray kernel; extended option behavior where referenced")
enhance("numpy.ufunc.reduce",["audit_reduce"],scope="default axis zero, explicit axis=None/tuple, where/out/initial, selected temporal reductions")
enhance("numpy.nditer",["audit_nditer"],scope="retained iterator, C/F/K traversal, op_axes, common dtype, explicit no-broadcast and buffered mask writes; selected flags")
for name in nditer_methods:
    symbol="numpy.nditer."+name
    if symbol in mapped:
        go,refs,note=mapped[symbol]
        mapped[symbol]=(go,refs,note.replace("op_axes or all flag combinations","all flag combinations"))
for name in ("shape","value","operands","dtypes","__iter__","__next__","close"):
    enhance("numpy.nditer."+name,["audit_nditer"])
enhance("numpy.seterr",["audit_fp_where","audit_fp_raise_option"],"; np.ErrorContext.CallWithOptions","explicit context for raw and masked/out ufunc calls; policy-free fast numerical kernels")
enhance("numpy.errstate",["audit_fp_where","audit_fp_raise_option"],"; np.ErrorContext.CallWithOptions","derived context and explicit policy-aware calls, including where and out")
enhance("numpy.linalg.norm",["audit_extended_norm","audit_extended_matrix_norm"],scope="native and extended vector orders, extended Frobenius and row/column sum norms; LAPACK-only longdouble orders rejected")
enhance("numpy.random.RandomState.choice",["audit_legacy_choice"],"; RandomState.ChoiceArray","legacy one-dimensional population, error preserves stream position")
for name, refs, note in [
    ("unique", ["unique"], "exact integer/bool keys, signed/unsigned extremes, first occurrence and inverse in C order, including reversed and endian views"),
    ("isin", ["isin", "record_isin"], "extended precision/signed zero/NaN, temporal tick promotion, Go object equality and structured scalar/subarray field membership"),
    ("searchsorted", ["search"], "strided endian inputs and mixed temporal units, including NaT and incompatible datetime/timedelta kinds"),
    ("gradient", ["gradient"], "uniform floating-point spacing on strided axes, tiny/huge steps and nonfinite inputs"),
    ("geomspace", ["geom"], "default low-precision dtype promotion, rotated complex logarithmic paths and broadcast endpoints"),
    ("cov", ["cov"], "native dot contraction, including strided, weighted complex, float32 and nonfinite input"),
]:
    enhance("numpy."+name,["optimization_"+r for r in refs],scope=note)
enhance("numpy.ndarray.searchsorted",["optimization_search"],scope="shared search kernel for strided endian and temporal data")
for symbol, prefixes in [
    ("numpy.linalg.solve", ["solve"]),
    ("numpy.linalg.cholesky", ["chol"]),
    ("numpy.linalg.det", ["det"]),
    ("numpy.linalg.slogdet", ["det"]),
    ("numpy.fft.fft", ["fft"]),
    ("numpy.fft.ifft", ["ifft"]),
    ("numpy.fft.rfft", ["rfft"]),
    ("numpy.fft.irfft", ["irfft"]),
    ("numpy.add", ["add"]),
    ("numpy.subtract", ["subtract"]),
    ("numpy.multiply", ["multiply"]),
    ("numpy.divide", ["divide"]),
    ("numpy.true_divide", ["divide"]),
    ("numpy.ndarray.astype", ["astype"]),
    ("numpy.ndarray.__setitem__", ["assign"]),
    ("numpy.random.Generator.integers", ["integers", "bounded_after", "integer_array_after"]),
    ("numpy.random.Generator.random", ["random_after"]),
    ("numpy.random.Generator.normal", ["normal", "broadcast_stream_after"]),
    ("numpy.random.Generator.poisson", ["poisson"]),
    ("numpy.random.Generator.binomial", ["binomial"]),
    ("numpy.random.RandomState.randint", ["legacy_integers", "legacy_after"]),
]:
    enhance(symbol,["hot_"+r for r in prefixes])
for name in ("PCG64","PCG64DXSM","MT19937","Philox","SFC64"):
    enhance("numpy.random."+name+".random_raw",["hot_raw_"+name,"hot_raw_cached_"+name,"hot_uint32_after_"+name])
for symbol, refs in [
    ("numpy.ndarray.astype",["copy","assign_f"]),
    ("numpy.ndarray.__setitem__",["assign_f","overlap"]),
    ("numpy.add",["add"]),
    ("numpy.subtract",["sub"]),
    ("numpy.multiply",["mul"]),
    ("numpy.divide",["div"]),
    ("numpy.true_divide",["div"]),
    ("numpy.fft.fft",["fft_512","fft_complex_input"]),
    ("numpy.fft.ifft",["ifft_1024"]),
    ("numpy.fft.rfft",["rfft"]),
    ("numpy.fft.irfft",["irfft"]),
    ("numpy.random.Generator.integers",["rng","philox_cache"]),
    ("numpy.random.Generator.random",["rng_double","rng_float32","rng_tail"]),
    ("numpy.random.PCG64.random_raw",["rng_raw_PCG64"]),
    ("numpy.random.PCG64DXSM.random_raw",["rng_raw_PCG64DXSM"]),
    ("numpy.random.Philox.random_raw",["philox_carry","philox_tail"]),
    ("numpy.matmul",["matmul"]),
    ("numpy.linalg.matmul",["matmul"]),
    ("numpy.tensordot",["tensordot"]),
    ("numpy.linalg.solve",["solve"]),
    ("numpy.linalg.slogdet",["sign","logdet"]),
    ("numpy.linalg.cholesky",["chol"]),
]:
    enhance(symbol,["native_"+r for r in refs])
for symbol,refs in [
    ("numpy.linalg.qr",["qr"]),
    ("numpy.linalg.svd",["svd"]),
    ("numpy.linalg.svdvals",["svd"]),
    ("numpy.linalg.eig",["eig","real_eig_dtype"]),
    ("numpy.linalg.eigvals",["eig_only","real_eig","real_sym_eig_only","eig_nonfinite"]),
    ("numpy.linalg.eigh",["eigh","hermitian_input"]),
    ("numpy.linalg.eigvalsh",["eigvalsh","real_sym_eigvalsh","hermitian_upper"]),
    ("numpy.sum",["sum"]),
    ("numpy.ndarray.sum",["sum"]),
    ("numpy.mean",["mean"]),
    ("numpy.ndarray.mean",["mean"]),
    ("numpy.fft.fft",["fft_prime"]),
    ("numpy.fft.ifft",["ifft_prime"]),
    ("numpy.fft.rfft",["rfft_prime"]),
    ("numpy.fft.irfft",["irfft_prime"]),
]:
    enhance(symbol,["decomp_"+r for r in refs])
for symbol,refs in [
    ("numpy.linalg.eigh",["herm_w","herm_v","upper_w","upper_v","herm_small","herm_large"]),
    ("numpy.linalg.eigvalsh",["herm_vals","upper_vals","herm_small_vals","herm_large_vals"]),
    ("numpy.linalg.svd",["svd","herm_svd"]),
    ("numpy.linalg.svdvals",["svd_tall","svd_wide","svd_small","svd_large","svd_graded"]),
    ("numpy.fft.fft",["fft"]),
    ("numpy.fft.ifft",["ifft"]),
    ("numpy.fft.rfft",["rfft"]),
    ("numpy.fft.irfft",["irfft"]),
    ("numpy.add",["simd_add"]),
    ("numpy.subtract",["simd_sub"]),
    ("numpy.multiply",["simd_mul"]),
    ("numpy.divide",["simd_div"]),
    ("numpy.true_divide",["simd_div"]),
]:
    enhance(symbol,["spectral_"+r for r in refs])
for symbol,refs in [
    ("numpy.ndarray.astype",["cast"]),
    ("numpy.add",["lane_add"]),
    ("numpy.subtract",["lane_sub"]),
    ("numpy.multiply",["lane_mul"]),
    ("numpy.divide",["lane_div"]),
    ("numpy.true_divide",["lane_div"]),
    ("numpy.fft.fft",["fft"]),
    ("numpy.fft.ifft",["ifft"]),
    ("numpy.fft.rfft",["rfft"]),
    ("numpy.fft.irfft",["irfft"]),
]:
    enhance(symbol,["arrayfft_"+r for r in refs])
for symbol,refs in [
    ("numpy.ndarray.astype",["cast"]),
    ("numpy.add",["ufunc_add","broadcast_alias","noncontig_out"]),
    ("numpy.subtract",["ufunc_sub","overlap_shift"]),
    ("numpy.multiply",["ufunc_mul"]),
    ("numpy.divide",["ufunc_div","fp_alias"]),
    ("numpy.true_divide",["ufunc_div","fp_alias"]),
    ("numpy.fft.fft",["fft_","fft_alias"]),
    ("numpy.fft.ifft",["ifft_"]),
    ("numpy.fft.rfft",["rfft_"]),
    ("numpy.fft.irfft",["irfft_"]),
]:
    enhance(symbol,["direct_"+r for r in refs])
for name in ("sum","mean","var","std"):
    enhance("numpy."+name,["precision_"+name])
    enhance("numpy.ndarray."+name,["precision_"+name])
for name in ("positive","negative","absolute","fabs","square","reciprocal","sqrt",
             "exp","log","sin","cos","floor","ceil","trunc","rint","conjugate",
             "minimum","maximum","fmin","fmax","add","subtract","multiply","divide"):
    enhance("numpy."+name,["precision_"+name])
enhance("numpy.true_divide",["precision_divide"])
enhance("numpy.conj",["precision_conjugate"])
enhance("numpy.abs",["precision_absolute"])
for name in ("fft","ifft","rfft","irfft"):
    enhance("numpy.fft."+name,["precision_"+name,"precision_"+name+"2"])
for name in ("exp","minimum","maximum","add","subtract","multiply","divide"):
    enhance("numpy."+name,["vector_"+name])
enhance("numpy.true_divide",["vector_divide"])
for name in ("fft","ifft","rfft","irfft"):
    enhance("numpy.fft."+name,["vector_"+name])
for name in ("multiply","divide"):
    enhance("numpy."+name,["register_"+name])
enhance("numpy.true_divide",["register_divide"])
enhance("numpy.fft.fft",["register_mixed_fft"])
enhance("numpy.fft.rfft",["register_rfft","register_mixed_rfft","register_signal_fft"])
enhance("numpy.fft.irfft",["register_irfft","register_mixed_irfft","register_roundtrip","register_signal_roundtrip","register_wide_irfft","register_wide_native_irfft"])
for i,(mod,clsname,basis) in enumerate(series_classes):
    group="numpy.polynomial."+clsname
    for name,case in [("__add__","add"),("__sub__","sub"),("__mul__","mul"),("__pow__","pow"),
                      ("deriv","deriv"),("integ","integ"),("__truediv__","div"),("__call__","eval"),
                      ("convert","convert"),("roots","linear_roots")]:
        enhance(group+"."+name,["audit_poly_ext_"+str(i)+"_"+case])
    prefix=("poly","cheb","leg","herm","herme","lag")[i]
    for suffix,case,api in [("der","deriv_axis","DerivativeAxis"),("int","integ_axis","IntegralAxis"),
                            ("vander","vander","Vander"),("companion","companion","Companion")]:
        enhance("numpy.polynomial."+mod+"."+prefix+suffix,["audit_poly_ext_"+str(i)+"_"+case],"; poly."+basis+"."+api)
for name,ref,api in [("domain","domain","DomainComplex"),("mapparms","map","MapParametersComplex"),
                     ("__call__","eval","Eval"),("deriv","deriv","Derivative"),("integ","integ","IntegrateComplex"),
                     ("convert","convert","Convert"),("roots","roots","Roots"),("linspace","linspace","Linspace")]:
    enhance("numpy.polynomial.Polynomial."+name,["audit_poly_complex_"+ref],"; poly.Series."+api)

groups = []
def namespace(name, module):
    groups.append((name, module, getattr(module,"__all__",None), False))
def cls(name, object_):
    groups.append((name,object_,None,True))
namespace("numpy",np)
namespace("numpy.strings",np.strings)
namespace("numpy.char",np.char)
namespace("numpy.dtypes",np.dtypes)
namespace("numpy.rec",np.rec)
namespace("numpy.lib.recfunctions",importlib.import_module("numpy.lib.recfunctions"))
namespace("numpy.lib.stride_tricks",importlib.import_module("numpy.lib.stride_tricks"))
namespace("numpy.lib",np.lib)
namespace("numpy.lib.scimath",np.lib.scimath)
namespace("numpy.emath",np.emath)
namespace("numpy.lib.array_utils",np.lib.array_utils)
namespace("numpy.lib.introspect",np.lib.introspect)
namespace("numpy.lib.mixins",np.lib.mixins)
namespace("numpy.lib.npyio",np.lib.npyio)
documented_attrs={
    "numpy.dtype":["type"],
    "numpy.lib.Arrayterator":["var","buf_size","start","stop","step"],
    "numpy.lib.NumpyVersion":list(version_fields),
    "numpy.lib.npyio.NpzFile":["files","f","fid","zip"],
    "numpy.ndenumerate":["iter"],
    "numpy.ufunc":["__name__"],
    "numpy.memmap":["filename","offset","mode"],
    "numpy.ma.core.MaskedIterator":["ma","dataiter","maskiter"],
}
groups.append(("numpy.ma.MaskedArray",np.ma.MaskedArray,list(masked_class_methods),True))
groups.append(("numpy.ma.core.MaskedIterator",np.ma.core.MaskedIterator,list(masked_iterator_methods),True))
groups.append(("numpy.char.chararray",np.char.chararray,list(char_class_methods),True))
groups.append(("numpy.recarray",np.recarray,list(rec_class_methods),True))
groups.append(("numpy.record",np.record,list(record_class_methods),True))
groups.append(("numpy.dtype",np.dtype,list(dtype_class_methods),True))
groups.append(("numpy.dtypes.StringDType",np.dtypes.StringDType,["coerce","na_object"],True))
groups.append(("numpy.matrix",np.matrix,list(matrix_class_methods),True))
groups.append(("numpy.generic",np.generic,list(scalar_class_methods)+list(audit_scalar_methods),True))
groups.append(("numpy.void",np.void,["getfield","setfield","flags","dtype","base"],True))
namespace("numpy.ma.mrecords",importlib.import_module("numpy.ma.mrecords"))
groups.append(("numpy.ma.mrecords.MaskedRecords",importlib.import_module("numpy.ma.mrecords").MaskedRecords,list(audit_mrecords_methods),True))
groups.append(("numpy.integer",np.integer,["numerator","denominator","is_integer"],True))
for n in integer_scalar_classes:
    groups.append(("numpy."+n,getattr(np,n),["bit_count"],True))
for n in float_scalar_classes:
    groups.append(("numpy."+n,getattr(np,n),["as_integer_ratio","is_integer"]+(["hex","fromhex"] if n=="float64" else []),True))
for mod,clsname,basis in series_classes:
    group="numpy.polynomial."+clsname
    documented_attrs[group]=["coef","basis_name","domain","window","maxpower"]
    names=list(series_methods)
    if clsname=="Chebyshev":
        names.append("interpolate")
    groups.append((group,getattr(np.polynomial,clsname),names,True))
namespace("numpy.polynomial.polyutils",np.polynomial.polyutils)
for name,methods in [("flatiter",flat_methods),("broadcast",broadcast_methods),
                     ("nditer",nditer_methods),("ndindex",{"__iter__":None,"__next__":None}),
                     ("ndenumerate",enumerate_methods),("ufunc",ufunc_methods),
                     ("poly1d",poly1d_methods),("memmap",memmap_methods)]:
    groups.append(("numpy."+name,getattr(np,name),list(methods),True))
groups.append(("numpy.lib.Arrayterator",np.lib.Arrayterator,["__getitem__","__array__","__iter__","shape","flat"]+documented_attrs["numpy.lib.Arrayterator"],True))
groups.append(("numpy.lib.NumpyVersion",np.lib.NumpyVersion,["__lt__","__le__","__eq__","__ne__","__gt__","__ge__","__repr__"]+documented_attrs["numpy.lib.NumpyVersion"],True))
groups.append(("numpy.lib.npyio.DataSource",np.lib.npyio.DataSource,["abspath","exists","open"],True))
groups.append(("numpy.lib.npyio.NpzFile",np.lib.npyio.NpzFile,list(npz_methods),True))
groups.append(("numpy.lib.mixins.NDArrayOperatorsMixin",np.lib.mixins.NDArrayOperatorsMixin,list(mixin_methods),True))
# format exposes these public functions as attributes despite an empty __all__.
groups.append(("numpy.lib.format",importlib.import_module("numpy.lib.format"),
               ["descr_to_dtype","drop_metadata","dtype_to_descr","header_data_from_array_1_0","isfileobj","magic",
                "open_memmap","read_array","read_array_header_1_0","read_array_header_2_0","read_magic",
                "write_array","write_array_header_1_0","write_array_header_2_0"],False))
namespace("numpy.linalg",np.linalg)
namespace("numpy.fft",np.fft)
namespace("numpy.random",np.random)
cls("numpy.ndarray",np.ndarray)
for name in ["Generator","RandomState","SeedSequence","BitGenerator","PCG64","PCG64DXSM","MT19937","Philox","SFC64"]:
    cls("numpy.random."+name,getattr(np.random,name))
for name in ["numpy.ma","numpy.polynomial","numpy.polynomial.polynomial","numpy.polynomial.chebyshev",
             "numpy.polynomial.legendre","numpy.polynomial.laguerre","numpy.polynomial.hermite","numpy.polynomial.hermite_e"]:
    namespace(name,importlib.import_module(name))

def pending_reason(symbol):
    name=symbol.rsplit(".",1)[-1]
    if name in ("add_docstring","add_newdoc"):
        return "Python runtime-only documentation mutation; outside numerical behavior"
    if name in ("capsule","ctypes","cffi","from_dlpack","get_include"):
        return "Python/C ABI or foreign-runtime pointer interface"
    if name in ("longdouble","clongdouble","float128","complex256"):
        return "extended precision dtype and platform-specific numerical representation"
    if name in ("errstate","geterr","seterr","geterrcall","seterrcall","getbufsize","setbufsize"):
        return "global/thread-local ufunc error and buffer policy; Go behavior differs"
    if name in ("array2string","array_repr","array_str","printoptions","get_printoptions","set_printoptions","set_default_printstyle"):
        return "full array/polynomial formatting and print settings"
    if name in ("bytes_","character","flexible","object_","str_","void","record","recarray","fromregex","frompyfunc",
                "flatten_structured_array","fromflex","make_mask_descr","masked_object","mvoid"):
        return "Python object/string/structured dtype or callback protocol"
    if name in ("dump","dumps"):
        return "Python pickle serialization"
    if name=="memmap":
        return "OS-backed memory mapping and lifetime semantics"
    if name in ("__array_namespace_info__","info","show_config","show_runtime","test","ids"):
        return "Python runtime metadata, diagnostics or object identity"
    if name in ("BitGenerator","ufunc"):
        return "Python abstract runtime class or dispatch protocol"
    return "no verified Go contract yet"

rows=[]
for group,obj,exported,is_class in groups:
    names=sorted(set(exported if exported is not None else [n for n in dir(obj) if not n.startswith("_")]))
    if group=="numpy.ndarray": names=sorted(set(names)|{"__getitem__","__setitem__"})
    for name in names:
        symbol=group+"."+name
        if name not in documented_attrs.get(group,[]):
            try:member=getattr(obj,name)
            except AttributeError:continue
            if not callable(member) and not (is_class and (inspect.isdatadescriptor(member) or isinstance(member,property))):
                continue
        if symbol in mapped:
            go,fixtures,note=mapped[symbol]
            status="verified-go-equivalent" if symbol in adapted else "verified-subset"
        else:
            go,fixtures,note="",[],pending_reason(symbol)
            status="pending"
        rows.append((group,symbol,status,go,";".join(fixtures),note))

rows.sort(key=lambda row:(row[0],row[1]))
with (ROOT/"API_COVERAGE.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.writer(f,lineterminator="\n");w.writerow(["namespace","numpy_api","status","go_api","reference_fixtures","scope"]);w.writerows(rows)
summary=defaultdict(Counter)
for group,_,status,*_ in rows:summary[group][status]+=1
count=Counter(row[2] for row in rows)
lines=[
    "# NumPy 2.5.0 numerical API inventory",
    "",
    "The inventory enumerates the pinned version's public callable exports, selected public class members and documented instance attributes. It covers the core, linear algebra, FFT, random, masked array and polynomial namespaces (including six Series classes and polyutils), strings, char, dtypes, rec, and selected public numerical utilities in numpy.lib (scimath/emath, array_utils, Arrayterator, NumpyVersion, npyio, introspect, mixins, recfunctions, stride_tricks and format). It also tracks public members and selected numerical protocols of ufunc, flatiter, broadcast, nditer, ndindex, ndenumerate, poly1d, memmap, MaskedArray/MaskedIterator, chararray, recarray/record, matrix, dtype and numeric scalar classes. Unchanged inherited ndarray members are not counted twice; shared Series exports are counted once under numpy.polynomial.<Class>. Shared descriptor and scalar members are grouped by base class. Private APIs, the Python extension C ABI and unlisted namespaces are excluded. Otherwise, aliases and separate public exports are counted independently.",
    "",
    "The inventory includes 40 additional shared numpy.generic members, 5 numpy.void-specific members, 6 public numpy.ma.mrecords exports and 5 MaskedRecords-specific members. Shared inherited implementations are grouped under their base classes.",
    "",
    "verified-subset means that the documented behavior has NumPy differential fixtures. verified-go-equivalent means that a Go contract for a Python/C ABI or platform-specific interface has observable-behavior evidence, without cross-runtime binary compatibility. pending indicates that no verified mapping exists. The CSV records each mapping, reference fixtures and scope; these counts do not represent exhaustive coverage of all arguments or behaviors.",
    "",
    f"- Public symbols/members with verified mappings: {count['verified-subset']+count['verified-go-equivalent']} (behavior subsets: {count['verified-subset']}; Go equivalents: {count['verified-go-equivalent']})",
    f"- Pending public symbols/members in this inventory: {count['pending']}",
    f"- Total tracked public symbols/members: {len(rows)}",
    f"- NumPy reference arrays: {len(reference['arrays'])}; error records: {len(reference['errors'])}.",
    "",
    "| Namespace | Verified subsets | Go equivalents | Pending | Total |",
    "|---|---:|---:|---:|---:|",
]
for group,counts in sorted(summary.items()):
    lines.append(f"| {group} | {counts['verified-subset']} | {counts['verified-go-equivalent']} | {counts['pending']} | {sum(counts.values())} |")
lines.extend(["","To regenerate, run python3 tools/generate_reference.py followed by python3 tools/generate_inventory.py in an environment with NumPy 2.5.0 installed."])
reasons=Counter(row[5] for row in rows if row[2]=="pending")
lines.extend(["","## Remaining entries",""])
if reasons:
    lines.extend(["These entries remain pending; runtime differences do not remove them from the tracked denominator.","",
                  "| Pending reason | Entries |","|---|---:|"])
    for reason,total in sorted(reasons.items(),key=lambda item:(-item[1],item[0])):
        lines.append(f"| {reason} | {total} |")
else:
    lines.append("No entries in the defined inventory are pending. See the CSV and README for Go equivalents and untested arguments, dtypes, namespaces and runtime boundaries.")
(ROOT/"API_COVERAGE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(f"inventory: {len(rows)} symbols, {count['verified-subset']} verified subsets, {count['verified-go-equivalent']} Go equivalents, {count['pending']} pending")
