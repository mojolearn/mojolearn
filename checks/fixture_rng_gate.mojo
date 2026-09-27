# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fixture-RNG gate: every existing copy equals its canonical counterpart
in `checks/fixture_rng.mojo`, bit for bit.

WHAT IT PROVES. Each copy below is imported by name from the file that
defines it and evaluated, next to its canonical counterpart, on every row
of one input table: at least 65,536 rows from a private generator (never a
function under test) plus the full cross product of the adversarial words
(0, 1, all ones, the sign bit, the 32-bit edges and the mixers' own
constants). Outputs compare as BITS (a float by its bit pattern). One row
that differs fails the copy, and one failed copy fails the gate.

VACUITY CONTROL, FIRST. Before any copy is judged, the same comparison runs
a planted splitmix64 with ONE constant changed against the canonical one.
If it cannot tell them apart the gate refuses as VACUOUS: a comparison
that cannot fail proves nothing. `-D FIXTURE_RNG_PLANT=1` swaps the planted
variant in for the canonical splitmix64 everywhere, so the whole splitmix64
group must FAIL; that is the second, end-to-end control.

FINDINGS. A copy passed with `finding=True` would be one that matches NO
canonical behavior and is deliberately left alone: the gate then requires
it to still DIFFER, so a fix cannot leave the list stale. There are none.
The two the first run found (a wrong last multiplier in
`checks/pointwise_resolve_check.mojo::_mix` and in
`checks/estimation_bench.mojo::splitmix`) were fixed at the root by
importing the canonical splitmix64 (DEVIATIONS 5941 and 5942).

NEW COPIES. `tools/fixture_rng_census.py` (run first by `pixi run
check-fixture-rng`) fails when a fixture-RNG definition exists that this
file does not import, so the list below cannot fall behind the tree. New
code imports `checks/fixture_rng.mojo` instead of adding a copy.

GENERATED ONCE from the 2026-09-27 inventory; edit by hand from now on.
"""

from std.memory import bitcast
from std.sys import exit
from std.sys.compile import is_defined

from checks.fixture_rng import (
    splitmix64,
    splitmix64_finalizer,
    splitmix64_fold,
    murmur3_fmix64,
    murmur3_fmix32,
    seeded_fmix32,
    lowbias32,
    golden32_mix,
    splitmix64_of_sum,
    splitmix_pair,
    splitmix_pair_unoffset,
    splitmix_triple,
    u01_triple,
    u01_row,
    u16_row_f32,
    u01_row_feature,
    symmetric_cell_f64,
    splitmix_low31,
    golden_top24,
    mix29_pair,
    mix29_triple,
    modhash_salted,
    modhash,
    xorshift32_of_index,
    xorshift32_of_pair,
    hashed_unit_f64,
    hashed_signed_f32,
    hashed_unit_f32_24,
    hashed_in_range,
    binade_hashed_f32,
)

# ---- every existing copy, by file (the census reads this list) ----
from training.checks.optimizer_fixture import opt_hashed as c_training_checks_optimizer_fixture__opt_hashed
from bench.gemm_excp_ab_main import _mix as c_bench_gemm_excp_ab_main__mix
from gemm.checks.gemm_rtf_boundary_check import _mix32 as c_gemm_checks_gemm_rtf_boundary_check__mix32
from ensemble.checks.atomic_width_probe import _hashed as c_ensemble_checks_atomic_width_probe__hashed
from arima.checks.fixtures import _hashed as c_arima_checks_fixtures__hashed
from checks.multiclass_score_check import hashed as c_checks_multiclass_score_check__hashed
from spectral.checks.spectral_fixture import hashed_unit as c_spectral_checks_spectral_fixture__hashed_unit
from checks.exact_estimation_check import hashed_unit as c_checks_exact_estimation_check__hashed_unit
from checks.multiclass_oracle_check import hashed_unit as c_checks_multiclass_oracle_check__hashed_unit
from checks.multilogit_check import hashed_unit as c_checks_multilogit_check__hashed_unit
from checks.pointwise_target_check import hashed_unit as c_checks_pointwise_target_check__hashed_unit
from bench.speed.forest_grove_kernel import mix32 as c_bench_speed_forest_grove_kernel__mix32
from cholesky.checks.left_update_window_probe import _mix as c_cholesky_checks_left_update_window_probe__mix
from cluster.checks.kmeans_incr_init_check import _mix as c_cluster_checks_kmeans_incr_init_check__mix
from ensemble.checks.rf_perf_candidates_check import h32 as c_ensemble_checks_rf_perf_candidates_check__h32
from gemm.checks.apple_simdgroup_probe import _mix as c_gemm_checks_apple_simdgroup_probe__mix
from neighbors.checks.apple_mma_distance_probe import _mix as c_neighbors_checks_apple_mma_distance_probe__mix
from neighbors.mutex_probe_main import _mix as c_neighbors_mutex_probe_main__mix
from cluster.checks.kmeans_identity_check import _hash64 as c_cluster_checks_kmeans_identity_check__hash64
from core.column_stats_identity_check import _hash64 as c_core_column_stats_identity_check__hash64
from gemm.checks.gemm_backward_check import _hash64 as c_gemm_checks_gemm_backward_check__hash64
from gemm.checks.gemm_device_check import _hash64 as c_gemm_checks_gemm_device_check__hash64
from gemm.checks.gemm_lowbit_check import _hash64 as c_gemm_checks_gemm_lowbit_check__hash64
from gemm.checks.gemm_oracle_check import _hash64 as c_gemm_checks_gemm_oracle_check__hash64
from gemm.checks.gemm_step_arms import _mix as c_gemm_checks_gemm_step_arms__mix
from svm.checks.svc_check import _hash64 as c_svm_checks_svc_check__hash64
from bench.unsupervised_trace_main import _mix as c_bench_unsupervised_trace_main__mix
from checks.one_hot_cardinality_check import _hashed as c_checks_one_hot_cardinality_check__hashed
from checks.ctr_check import _hashed as c_checks_ctr_check__hashed
from checks.ctr_device_check import _hashed as c_checks_ctr_device_check__hashed
from checks.ctr_kernels_check import _h as c_checks_ctr_kernels_check__h
from checks.ctr_train_check import _hashed as c_checks_ctr_train_check__hashed
from checks.nan_mode_check import _hashed as c_checks_nan_mode_check__hashed
from checks.permutation_count_check import _hashed as c_checks_permutation_count_check__hashed
from extratrees.checks.builder_check import mix32 as c_extratrees_checks_builder_check__mix32
from extratrees.checks.leaf_check import mix32 as c_extratrees_checks_leaf_check__mix32
from extratrees.checks.partition_check import mix32 as c_extratrees_checks_partition_check__mix32
from extratrees.checks.partition_leaf_kernel_check import mix32 as c_extratrees_checks_partition_leaf_kernel_check__mix32
from extratrees.checks.partition_multiblock_check import mix32 as c_extratrees_checks_partition_multiblock_check__mix32
from extratrees.checks.pcg_rng import fmix32 as c_extratrees_checks_pcg_rng__fmix32
from extratrees.checks.split_check import mix32 as c_extratrees_checks_split_check__mix32
from training.byte_lm_init import fmix32 as c_training_byte_lm_init__fmix32
from ensemble.checks.criteria_check import _mix as c_ensemble_checks_criteria_check__mix
from ensemble.checks.cuml_oracle_check import _mix as c_ensemble_checks_cuml_oracle_check__mix
from ensemble.checks.fingerprint_probe import _mix as c_ensemble_checks_fingerprint_probe__mix
from ensemble.checks.forest_check import _mix as c_ensemble_checks_forest_check__mix
from ensemble.checks.oob_check import _mix as c_ensemble_checks_oob_check__mix
from ensemble.checks.regression_check import _mix as c_ensemble_checks_regression_check__mix
from ensemble.checks.sample_weight_check import _mix as c_ensemble_checks_sample_weight_check__mix
from ensemble.checks.split_check import _mix as c_ensemble_checks_split_check__mix
from ensemble.checks.train_check import _mix as c_ensemble_checks_train_check__mix
from extratrees.checks.fixed_point_check import mix64 as c_extratrees_checks_fixed_point_check__mix64
from ensemble.checks.builder_kernels_check import h32 as c_ensemble_checks_builder_kernels_check__h32
from checks.division_check import _splitmix as c_checks_division_check__splitmix
from checks.exact_estimation_check import splitmix as c_checks_exact_estimation_check__splitmix
from checks.ieee_arith_check import _splitmix as c_checks_ieee_arith_check__splitmix
from checks.logloss_estimator_check import splitmix as c_checks_logloss_estimator_check__splitmix
from checks.logloss_target_check import splitmix as c_checks_logloss_target_check__splitmix
from checks.logloss_train_check import splitmix as c_checks_logloss_train_check__splitmix
from checks.multiclass_oracle_check import splitmix as c_checks_multiclass_oracle_check__splitmix
from checks.multiclass_score_check import splitmix as c_checks_multiclass_score_check__splitmix
from checks.multiclass_train_check import splitmix as c_checks_multiclass_train_check__splitmix
from checks.multilogit_check import splitmix as c_checks_multilogit_check__splitmix
from checks.pointwise_target_check import splitmix as c_checks_pointwise_target_check__splitmix
from checks.portable_fmax_check import _splitmix as c_checks_portable_fmax_check__splitmix
from checks.portable_gelu_check import _splitmix as c_checks_portable_gelu_check__splitmix
from checks.portable_log2_64_check import _splitmix as c_checks_portable_log2_64_check__splitmix
from checks.portable_log64_check import _splitmix as c_checks_portable_log64_check__splitmix
from checks.portable_nn_check import _splitmix as c_checks_portable_nn_check__splitmix
from checks.portable_pow64_check import _mix as c_checks_portable_pow64_check__mix
from checks.portable_sqrtcos_check import _splitmix as c_checks_portable_sqrtcos_check__splitmix
from checks.portable_translog_check import _splitmix as c_checks_portable_translog_check__splitmix
from checks.portable_trig_check import _splitmix as c_checks_portable_trig_check__splitmix
from checks.second_der_weights_check import splitmix as c_checks_second_der_weights_check__splitmix
from embedding.checks.embedding_fixture import emb_splitmix64 as c_embedding_checks_embedding_fixture__emb_splitmix64
from ensemble.bench.rf_bench import _mix as c_ensemble_bench_rf_bench__mix
from extratrees.checks.fixtures import splitmix64 as c_extratrees_checks_fixtures__splitmix64
from gemm.checks.gemm_rtf_boundary_check import _mix as c_gemm_checks_gemm_rtf_boundary_check__mix
from mamba.checks.mamba_fixture import corpus_splitmix64 as c_mamba_checks_mamba_fixture__corpus_splitmix64
from neighbors.checks.knn_selector_arms_check import splitmix64 as c_neighbors_checks_knn_selector_arms_check__splitmix64
from neighbors.checks.knn_selector_bound_compact_check import splitmix64 as c_neighbors_checks_knn_selector_bound_compact_check__splitmix64
from spectral.checks.spectral_fixture import splitmix64 as c_spectral_checks_spectral_fixture__splitmix64
from training.checks.loss_fixture import ce_splitmix64 as c_training_checks_loss_fixture__ce_splitmix64
from training.checks.optimizer_fixture import opt_splitmix64 as c_training_checks_optimizer_fixture__opt_splitmix64
from training.checks.step_glue_check import _mix as c_training_checks_step_glue_check__mix
from training.checks.train_loop import train_splitmix64 as c_training_checks_train_loop__train_splitmix64
from transformer.checks.transformer_fixture import fixture_splitmix64 as c_transformer_checks_transformer_fixture__fixture_splitmix64
from umap.checks.batch_determinism_check import _mix as c_umap_checks_batch_determinism_check__mix
from umap.checks.batch_epoch_cost_check import _mix as c_umap_checks_batch_epoch_cost_check__mix
from umap.host.umap_oracle import _splitmix64 as c_umap_host_umap_oracle__splitmix64
from umap.optimizer import _splitmix64 as c_umap_optimizer__splitmix64
from umap.optimizer_fast import _mix as c_umap_optimizer_fast__mix
from x_prep.mutual_info import _splitmix as c_x_prep_mutual_info__splitmix
from x_prep.seams.prep_oracle import splitmix as c_x_prep_seams_prep_oracle__splitmix
from extratrees.checks.split_reduce_check import mix64 as c_extratrees_checks_split_reduce_check__mix64
from xtrees.checks.glue_oracle import splitmix as c_xtrees_checks_glue_oracle__splitmix
from xtrees.ops import mix64 as c_xtrees_ops__mix64
from extratrees.checks.bestfirst_check import mix64 as c_extratrees_checks_bestfirst_check__mix64
from extratrees.checks.bestfirst_fingerprint import mix64 as c_extratrees_checks_bestfirst_fingerprint__mix64
from gemm.checks.gemm_tuned_probe import _mix as c_gemm_checks_gemm_tuned_probe__mix
from gemm.checks.gemm_unpinned_price import _mix as c_gemm_checks_gemm_unpinned_price__mix
from ensemble.checks.core_primitives_check import mix as c_ensemble_checks_core_primitives_check__mix
from bench.apple_identical_steps_main import _mix as c_bench_apple_identical_steps_main__mix
from bench.gemm_card_main import _mix as c_bench_gemm_card_main__mix
from bench.gemm_ladder_main import _mix as c_bench_gemm_ladder_main__mix
from bench.gemm_price_main import _mix as c_bench_gemm_price_main__mix
from bench.lanes_price_main import _gemm_mix as c_bench_lanes_price_main__gemm_mix
from bench.linalg_price_main import _mix as c_bench_linalg_price_main__mix
from checks.gram_splitk_check import _mix as c_checks_gram_splitk_check__mix
from checks.partitions_reduce_check import _mix as c_checks_partitions_reduce_check__mix
from checks.pointwise_subsets_check import _mix as c_checks_pointwise_subsets_check__mix
from checks.vendor_correctness_check import _mix as c_checks_vendor_correctness_check__mix
from core.gemm_identity_check import _mix as c_core_gemm_identity_check__mix
from glm.checks.ols_trace import _mix as c_glm_checks_ols_trace__mix
from checks.pointwise_pool_check import _mix as c_checks_pointwise_pool_check__mix
from cholesky.checks.cholesky_fixture import chol_mix64 as c_cholesky_checks_cholesky_fixture__chol_mix64
from holtwinters.checks.hw_fixture import mix64 as c_holtwinters_checks_hw_fixture__mix64
from isolation_forest.checks.if_fixture import mix64 as c_isolation_forest_checks_if_fixture__mix64
from kde.checks.kde_fixture import mix64 as c_kde_checks_kde_fixture__mix64
from metrics.checks.fixtures import splitmix as c_metrics_checks_fixtures__splitmix
from mixture.checks.gmm_fixture import gmm_mix64 as c_mixture_checks_gmm_fixture__gmm_mix64
from resample.checks.resample_fixture import mix64 as c_resample_checks_resample_fixture__mix64
from tsa.checks.fixtures import splitmix as c_tsa_checks_fixtures__splitmix
from decomposition.checks.jacobi_check import _hashed as c_decomposition_checks_jacobi_check__hashed
from bench.bench_main import _u01 as c_bench_bench_main__u01
from bench.dsweep_main import _u01 as c_bench_dsweep_main__u01
from bench.scaling_main import _u01 as c_bench_scaling_main__u01
from bench.speed.classical_ladder_main import _u01 as c_bench_speed_classical_ladder_main__u01
from bench.speed.classical_speed_main import _u01 as c_bench_speed_classical_speed_main__u01
from dbscan.phase_main import _u01 as c_dbscan_phase_main__u01
from decomposition.checks.pca_check import _wide_u01 as c_decomposition_checks_pca_check__wide_u01
from decomposition.checks.svd_full_check import _u01 as c_decomposition_checks_svd_full_check__u01
from glm.checks.logistic_check import _u01 as c_glm_checks_logistic_check__u01
from glm.checks.multinomial_check import _u01 as c_glm_checks_multinomial_check__u01
from glm.checks.ols_check import _u01 as c_glm_checks_ols_check__u01
from glm.checks.ridge_check import _u01 as c_glm_checks_ridge_check__u01
from solver.host.cd_oracle import _u01 as c_solver_host_cd_oracle__u01
from neighbors.checks.ball_cover_check import _hash01 as c_neighbors_checks_ball_cover_check__hash01
from neighbors.checks.ball_cover_knn_check import _hash01 as c_neighbors_checks_ball_cover_knn_check__hash01
from neighbors.checks.radius_check import _hash01 as c_neighbors_checks_radius_check__hash01
from metrics.checks.fixtures import u01 as c_metrics_checks_fixtures__u01
from tsa.checks.fixtures import u01 as c_tsa_checks_fixtures__u01
from bench.identity_price_main import _u01 as c_bench_identity_price_main__u01
from bench.lanes_price_main import _price_u01 as c_bench_lanes_price_main__price_u01
from bench.samba_rms_price_main import _u01 as c_bench_samba_rms_price_main__u01
from checks.radix_sort_check import hashed as c_checks_radix_sort_check__hashed
from checks.segmented_scan_check import hashed as c_checks_segmented_scan_check__hashed
from ensemble.checks.quantiles_check import hashed as c_ensemble_checks_quantiles_check__hashed
from checks.sym_arms_check import mix as c_checks_sym_arms_check__mix

# ---- end of copies ----

comptime HASHED_ROWS = 65536
comptime PLANT = is_defined["FIXTURE_RNG_PLANT"]()


def _input_word(k: UInt64) -> UInt64:
    """The input table's private generator: an xorshift-multiply chain that
    shares no constant with any function under test."""
    var x = k * UInt64(0xD1B54A32D192ED03) + UInt64(0x8CB92BA72F3D8DD7)
    x ^= x >> 32
    x *= UInt64(0xAEF17502108EF2D9)
    x ^= x >> 29
    x *= UInt64(0xDB4F0B9175AE2165)
    x ^= x >> 32
    return x


def _adversarial() -> List[UInt64]:
    return [
        UInt64(0),
        UInt64(1),
        UInt64(2),
        UInt64(0xFFFFFFFFFFFFFFFF),
        UInt64(0xFFFFFFFFFFFFFFFE),
        UInt64(0x8000000000000000),
        UInt64(0x7FFFFFFFFFFFFFFF),
        UInt64(0xFFFFFFFF),
        UInt64(0x100000000),
        UInt64(0x7FFFFFFF),
        UInt64(0x80000000),
        UInt64(0x9E3779B97F4A7C15),
        UInt64(0x61C8864680B583EB),
        UInt64(0xBF58476D1CE4E5B9),
        UInt64(0x94D049BB133111EB),
        UInt64(1000003),
    ]


struct _Inputs(Movable):
    var _a: List[UInt64]
    var _b: List[UInt64]
    var _c: List[UInt64]

    def __init__(out self):
        self._a = List[UInt64]()
        self._b = List[UInt64]()
        self._c = List[UInt64]()
        for k in range(HASHED_ROWS):
            var u = UInt64(k)
            self._a.append(_input_word(u))
            self._b.append(_input_word(u ^ UInt64(0xA5A5A5A5A5A5A5A5)))
            self._c.append(_input_word(u + UInt64(0x5BD1E9955BD1E995)))
        var adv = _adversarial()
        for i in range(len(adv)):
            for j in range(len(adv)):
                for k in range(len(adv)):
                    self._a.append(adv[i])
                    self._b.append(adv[j])
                    self._c.append(adv[k])

    def rows(self) -> Int:
        return len(self._a)

    def a(self, r: Int) -> UInt64:
        return self._a[r]

    def b(self, r: Int) -> UInt64:
        return self._b[r]

    def a32(self, r: Int) -> UInt32:
        return self._a[r].cast[DType.uint32]()

    def ia(self, r: Int) -> Int:
        return Int(self._a[r].cast[DType.int64]())

    def ib(self, r: Int) -> Int:
        return Int(self._b[r].cast[DType.int64]())

    def ic(self, r: Int) -> Int:
        return Int(self._c[r].cast[DType.int64]())

    def id(self, r: Int) -> Int:
        return Int((self._a[r] ^ (self._c[r] >> 7)).cast[DType.int64]())

    def lo(self, r: Int) -> Float64:
        var k = r % 4
        if k == 0:
            return -0.6
        if k == 1:
            return -0.8
        if k == 2:
            return 0.0
        return -3.0

    def hi(self, r: Int) -> Float64:
        var k = r % 4
        if k == 0:
            return 0.6
        if k == 1:
            return 0.8
        if k == 2:
            return 1.0
        return 2.5

    def shift(self, r: Int) -> Int:
        """`binade_hashed_f32` loops `|e|` times; its callers keep the shift
        in `[-100, 100]`, and so does this table."""
        return Int(self._c[r] % UInt64(201)) - 100


def _bits(x: UInt64) -> UInt64:
    return x


def _bits(x: UInt32) -> UInt64:
    return UInt64(x)


def _bits(x: Int) -> UInt64:
    return UInt64(Int64(x).cast[DType.uint64]())


def _bits(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


def _bits(x: Float32) -> UInt64:
    return UInt64(bitcast[DType.uint32](x))


def _planted_splitmix64(x: UInt64) -> UInt64:
    """splitmix64 with ONE constant changed (the last multiplier's low bit
    pair, 0x...11EB -> 0x...11E9). The vacuity control."""
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111E9)
    return z ^ (z >> 31)


def _canonical_splitmix64(x: UInt64) -> UInt64:
    comptime if PLANT:
        return _planted_splitmix64(x)
    return splitmix64(x)


struct _Verdict(Movable):
    var copies: Int
    var agree: Int
    var failed: Int
    var findings: Int

    def __init__(out self):
        self.copies = 0
        self.agree = 0
        self.failed = 0
        self.findings = 0

    def copy(
        mut self, name: String, canon: String, rows: Int, bad: Int, first: Int, finding: Bool
    ):
        self.copies += 1
        if finding:
            if bad > 0:
                self.findings += 1
                print("FINDING ", name, "claims", canon, "but differs on", bad, "of", rows, "rows (known; not changed here)")
            else:
                self.failed += 1
                print("STALE   ", name, "is listed as a finding but now EQUALS", canon, "on all", rows, "rows: move it to the copies")
            return
        if bad == 0:
            self.agree += 1
            print("AGREE   ", name, "==", canon, "on", rows, "rows")
        else:
            self.failed += 1
            print("DIFFER  ", name, "!=", canon, "on", bad, "of", rows, "rows, first at row", first)


def _vacuity_control(t: _Inputs) raises:
    """The comparison must separate two behaviors that differ by one
    constant, or nothing it says afterwards means anything."""
    var bad = 0
    for r in range(t.rows()):
        var A = t.a(r)
        if _bits(_planted_splitmix64(A)) != _bits(splitmix64(A)):
            bad += 1
    print("CONTROL  planted splitmix64 (one constant changed) differs on", bad, "of", t.rows(), "rows")
    if bad == 0:
        raise Error("VACUOUS: the comparison cannot separate a planted variant from splitmix64")


def _check_binade_hashed_f32(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.binade_hashed_f32`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var A = t.a(r)
        var IB = t.ib(r)
        var SHIFT = t.shift(r)
        var want = _bits(binade_hashed_f32(A, IB, SHIFT))
        if _bits(c_training_checks_optimizer_fixture__opt_hashed(A, IB, SHIFT)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("training/checks/optimizer_fixture.mojo::opt_hashed", "binade_hashed_f32", t.rows(), bad[0], first[0], False)


def _check_golden32_mix(t: _Inputs, mut v: _Verdict):
    """2 copies (+0 known findings) against `fixture_rng.golden32_mix`."""
    var bad = List[Int](length=2, fill=0)
    var first = List[Int](length=2, fill=-1)
    for r in range(t.rows()):
        var A32 = t.a32(r)
        var want = _bits(golden32_mix(A32))
        if _bits(c_bench_gemm_excp_ab_main__mix(A32)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_gemm_checks_gemm_rtf_boundary_check__mix32(A32)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
    v.copy("bench/gemm_excp_ab_main.mojo::_mix", "golden32_mix", t.rows(), bad[0], first[0], False)
    v.copy("gemm/checks/gemm_rtf_boundary_check.mojo::_mix32", "golden32_mix", t.rows(), bad[1], first[1], False)


def _check_golden_top24(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.golden_top24`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var want = _bits(golden_top24(IA))
        if _bits(c_ensemble_checks_atomic_width_probe__hashed(IA)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("ensemble/checks/atomic_width_probe.mojo::_hashed", "golden_top24", t.rows(), bad[0], first[0], False)


def _check_hashed_in_range(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.hashed_in_range`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var HI = t.hi(r)
        var IA = t.ia(r)
        var IB = t.ib(r)
        var IC = t.ic(r)
        var ID = t.id(r)
        var LO = t.lo(r)
        var want = _bits(hashed_in_range(IA, IB, IC, ID, LO, HI))
        if _bits(c_arima_checks_fixtures__hashed(IA, IB, IC, ID, LO, HI)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("arima/checks/fixtures.mojo::_hashed", "hashed_in_range", t.rows(), bad[0], first[0], False)


def _check_hashed_signed_f32(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.hashed_signed_f32`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var A = t.a(r)
        var IB = t.ib(r)
        var want = _bits(hashed_signed_f32(A, IB))
        if _bits(c_checks_multiclass_score_check__hashed(A, IB)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("checks/multiclass_score_check.mojo::hashed", "hashed_signed_f32", t.rows(), bad[0], first[0], False)


def _check_hashed_unit_f32_24(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.hashed_unit_f32_24`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var A = t.a(r)
        var IB = t.ib(r)
        var IC = t.ic(r)
        var want = _bits(hashed_unit_f32_24(A, IB, IC))
        if _bits(c_spectral_checks_spectral_fixture__hashed_unit(A, IB, IC)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("spectral/checks/spectral_fixture.mojo::hashed_unit", "hashed_unit_f32_24", t.rows(), bad[0], first[0], False)


def _check_hashed_unit_f64(t: _Inputs, mut v: _Verdict):
    """4 copies (+0 known findings) against `fixture_rng.hashed_unit_f64`."""
    var bad = List[Int](length=4, fill=0)
    var first = List[Int](length=4, fill=-1)
    for r in range(t.rows()):
        var A = t.a(r)
        var IB = t.ib(r)
        var want = _bits(hashed_unit_f64(A, IB))
        if _bits(c_checks_exact_estimation_check__hashed_unit(A, IB)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_checks_multiclass_oracle_check__hashed_unit(A, IB)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_checks_multilogit_check__hashed_unit(A, IB)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_checks_pointwise_target_check__hashed_unit(A, IB)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
    v.copy("checks/exact_estimation_check.mojo::hashed_unit", "hashed_unit_f64", t.rows(), bad[0], first[0], False)
    v.copy("checks/multiclass_oracle_check.mojo::hashed_unit", "hashed_unit_f64", t.rows(), bad[1], first[1], False)
    v.copy("checks/multilogit_check.mojo::hashed_unit", "hashed_unit_f64", t.rows(), bad[2], first[2], False)
    v.copy("checks/pointwise_target_check.mojo::hashed_unit", "hashed_unit_f64", t.rows(), bad[3], first[3], False)


def _check_lowbias32(t: _Inputs, mut v: _Verdict):
    """7 copies (+0 known findings) against `fixture_rng.lowbias32`."""
    var bad = List[Int](length=7, fill=0)
    var first = List[Int](length=7, fill=-1)
    for r in range(t.rows()):
        var A32 = t.a32(r)
        var want = _bits(lowbias32(A32))
        if _bits(c_bench_speed_forest_grove_kernel__mix32(A32)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_cholesky_checks_left_update_window_probe__mix(A32)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_cluster_checks_kmeans_incr_init_check__mix(A32)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_ensemble_checks_rf_perf_candidates_check__h32(A32)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
        if _bits(c_gemm_checks_apple_simdgroup_probe__mix(A32)) != want:
            bad[4] += 1
            if first[4] < 0:
                first[4] = r
        if _bits(c_neighbors_checks_apple_mma_distance_probe__mix(A32)) != want:
            bad[5] += 1
            if first[5] < 0:
                first[5] = r
        if _bits(c_neighbors_mutex_probe_main__mix(A32)) != want:
            bad[6] += 1
            if first[6] < 0:
                first[6] = r
    v.copy("bench/speed/forest_grove_kernel.mojo::mix32", "lowbias32", t.rows(), bad[0], first[0], False)
    v.copy("cholesky/checks/left_update_window_probe.mojo::_mix", "lowbias32", t.rows(), bad[1], first[1], False)
    v.copy("cluster/checks/kmeans_incr_init_check.mojo::_mix", "lowbias32", t.rows(), bad[2], first[2], False)
    v.copy("ensemble/checks/rf_perf_candidates_check.mojo::h32", "lowbias32", t.rows(), bad[3], first[3], False)
    v.copy("gemm/checks/apple_simdgroup_probe.mojo::_mix", "lowbias32", t.rows(), bad[4], first[4], False)
    v.copy("neighbors/checks/apple_mma_distance_probe.mojo::_mix", "lowbias32", t.rows(), bad[5], first[5], False)
    v.copy("neighbors/mutex_probe_main.mojo::_mix", "lowbias32", t.rows(), bad[6], first[6], False)


def _check_mix29_pair(t: _Inputs, mut v: _Verdict):
    """8 copies (+0 known findings) against `fixture_rng.mix29_pair`."""
    var bad = List[Int](length=8, fill=0)
    var first = List[Int](length=8, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var want = _bits(mix29_pair(IA, IB))
        if _bits(c_cluster_checks_kmeans_identity_check__hash64(IA, IB)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_core_column_stats_identity_check__hash64(IA, IB)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_gemm_checks_gemm_backward_check__hash64(IA, IB)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_gemm_checks_gemm_device_check__hash64(IA, IB)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
        if _bits(c_gemm_checks_gemm_lowbit_check__hash64(IA, IB)) != want:
            bad[4] += 1
            if first[4] < 0:
                first[4] = r
        if _bits(c_gemm_checks_gemm_oracle_check__hash64(IA, IB)) != want:
            bad[5] += 1
            if first[5] < 0:
                first[5] = r
        if _bits(c_gemm_checks_gemm_step_arms__mix(IA, IB)) != want:
            bad[6] += 1
            if first[6] < 0:
                first[6] = r
        if _bits(c_svm_checks_svc_check__hash64(IA, IB)) != want:
            bad[7] += 1
            if first[7] < 0:
                first[7] = r
    v.copy("cluster/checks/kmeans_identity_check.mojo::_hash64", "mix29_pair", t.rows(), bad[0], first[0], False)
    v.copy("core/column_stats_identity_check.mojo::_hash64", "mix29_pair", t.rows(), bad[1], first[1], False)
    v.copy("gemm/checks/gemm_backward_check.mojo::_hash64", "mix29_pair", t.rows(), bad[2], first[2], False)
    v.copy("gemm/checks/gemm_device_check.mojo::_hash64", "mix29_pair", t.rows(), bad[3], first[3], False)
    v.copy("gemm/checks/gemm_lowbit_check.mojo::_hash64", "mix29_pair", t.rows(), bad[4], first[4], False)
    v.copy("gemm/checks/gemm_oracle_check.mojo::_hash64", "mix29_pair", t.rows(), bad[5], first[5], False)
    v.copy("gemm/checks/gemm_step_arms.mojo::_mix", "mix29_pair", t.rows(), bad[6], first[6], False)
    v.copy("svm/checks/svc_check.mojo::_hash64", "mix29_pair", t.rows(), bad[7], first[7], False)


def _check_mix29_triple(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.mix29_triple`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var IC = t.ic(r)
        var want = _bits(mix29_triple(IA, IB, IC))
        if _bits(c_bench_unsupervised_trace_main__mix(IA, IB, IC)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("bench/unsupervised_trace_main.mojo::_mix", "mix29_triple", t.rows(), bad[0], first[0], False)


def _check_modhash(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.modhash`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var want = _bits(modhash(IA))
        if _bits(c_checks_one_hot_cardinality_check__hashed(IA)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("checks/one_hot_cardinality_check.mojo::_hashed", "modhash", t.rows(), bad[0], first[0], False)


def _check_modhash_salted(t: _Inputs, mut v: _Verdict):
    """6 copies (+0 known findings) against `fixture_rng.modhash_salted`."""
    var bad = List[Int](length=6, fill=0)
    var first = List[Int](length=6, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var want = _bits(modhash_salted(IA, IB))
        if _bits(c_checks_ctr_check__hashed(IA, IB)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_checks_ctr_device_check__hashed(IA, IB)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_checks_ctr_kernels_check__h(IA, IB)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_checks_ctr_train_check__hashed(IA, IB)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
        if _bits(c_checks_nan_mode_check__hashed(IA, IB)) != want:
            bad[4] += 1
            if first[4] < 0:
                first[4] = r
        if _bits(c_checks_permutation_count_check__hashed(IA, IB)) != want:
            bad[5] += 1
            if first[5] < 0:
                first[5] = r
    v.copy("checks/ctr_check.mojo::_hashed", "modhash_salted", t.rows(), bad[0], first[0], False)
    v.copy("checks/ctr_device_check.mojo::_hashed", "modhash_salted", t.rows(), bad[1], first[1], False)
    v.copy("checks/ctr_kernels_check.mojo::_h", "modhash_salted", t.rows(), bad[2], first[2], False)
    v.copy("checks/ctr_train_check.mojo::_hashed", "modhash_salted", t.rows(), bad[3], first[3], False)
    v.copy("checks/nan_mode_check.mojo::_hashed", "modhash_salted", t.rows(), bad[4], first[4], False)
    v.copy("checks/permutation_count_check.mojo::_hashed", "modhash_salted", t.rows(), bad[5], first[5], False)


def _check_murmur3_fmix32(t: _Inputs, mut v: _Verdict):
    """8 copies (+0 known findings) against `fixture_rng.murmur3_fmix32`."""
    var bad = List[Int](length=8, fill=0)
    var first = List[Int](length=8, fill=-1)
    for r in range(t.rows()):
        var A32 = t.a32(r)
        var want = _bits(murmur3_fmix32(A32))
        if _bits(c_extratrees_checks_builder_check__mix32(A32)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_extratrees_checks_leaf_check__mix32(A32)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_extratrees_checks_partition_check__mix32(A32)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_extratrees_checks_partition_leaf_kernel_check__mix32(A32)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
        if _bits(c_extratrees_checks_partition_multiblock_check__mix32(A32)) != want:
            bad[4] += 1
            if first[4] < 0:
                first[4] = r
        if _bits(c_extratrees_checks_pcg_rng__fmix32(A32)) != want:
            bad[5] += 1
            if first[5] < 0:
                first[5] = r
        if _bits(c_extratrees_checks_split_check__mix32(A32)) != want:
            bad[6] += 1
            if first[6] < 0:
                first[6] = r
        if _bits(c_training_byte_lm_init__fmix32(A32)) != want:
            bad[7] += 1
            if first[7] < 0:
                first[7] = r
    v.copy("extratrees/checks/builder_check.mojo::mix32", "murmur3_fmix32", t.rows(), bad[0], first[0], False)
    v.copy("extratrees/checks/leaf_check.mojo::mix32", "murmur3_fmix32", t.rows(), bad[1], first[1], False)
    v.copy("extratrees/checks/partition_check.mojo::mix32", "murmur3_fmix32", t.rows(), bad[2], first[2], False)
    v.copy("extratrees/checks/partition_leaf_kernel_check.mojo::mix32", "murmur3_fmix32", t.rows(), bad[3], first[3], False)
    v.copy("extratrees/checks/partition_multiblock_check.mojo::mix32", "murmur3_fmix32", t.rows(), bad[4], first[4], False)
    v.copy("extratrees/checks/pcg_rng.mojo::fmix32", "murmur3_fmix32", t.rows(), bad[5], first[5], False)
    v.copy("extratrees/checks/split_check.mojo::mix32", "murmur3_fmix32", t.rows(), bad[6], first[6], False)
    v.copy("training/byte_lm_init.mojo::fmix32", "murmur3_fmix32", t.rows(), bad[7], first[7], False)


def _check_murmur3_fmix64(t: _Inputs, mut v: _Verdict):
    """10 copies (+0 known findings) against `fixture_rng.murmur3_fmix64`."""
    var bad = List[Int](length=10, fill=0)
    var first = List[Int](length=10, fill=-1)
    for r in range(t.rows()):
        var A = t.a(r)
        var want = _bits(murmur3_fmix64(A))
        if _bits(c_ensemble_checks_criteria_check__mix(A)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_ensemble_checks_cuml_oracle_check__mix(A)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_ensemble_checks_fingerprint_probe__mix(A)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_ensemble_checks_forest_check__mix(A)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
        if _bits(c_ensemble_checks_oob_check__mix(A)) != want:
            bad[4] += 1
            if first[4] < 0:
                first[4] = r
        if _bits(c_ensemble_checks_regression_check__mix(A)) != want:
            bad[5] += 1
            if first[5] < 0:
                first[5] = r
        if _bits(c_ensemble_checks_sample_weight_check__mix(A)) != want:
            bad[6] += 1
            if first[6] < 0:
                first[6] = r
        if _bits(c_ensemble_checks_split_check__mix(A)) != want:
            bad[7] += 1
            if first[7] < 0:
                first[7] = r
        if _bits(c_ensemble_checks_train_check__mix(A)) != want:
            bad[8] += 1
            if first[8] < 0:
                first[8] = r
        if _bits(c_extratrees_checks_fixed_point_check__mix64(A)) != want:
            bad[9] += 1
            if first[9] < 0:
                first[9] = r
    v.copy("ensemble/checks/criteria_check.mojo::_mix", "murmur3_fmix64", t.rows(), bad[0], first[0], False)
    v.copy("ensemble/checks/cuml_oracle_check.mojo::_mix", "murmur3_fmix64", t.rows(), bad[1], first[1], False)
    v.copy("ensemble/checks/fingerprint_probe.mojo::_mix", "murmur3_fmix64", t.rows(), bad[2], first[2], False)
    v.copy("ensemble/checks/forest_check.mojo::_mix", "murmur3_fmix64", t.rows(), bad[3], first[3], False)
    v.copy("ensemble/checks/oob_check.mojo::_mix", "murmur3_fmix64", t.rows(), bad[4], first[4], False)
    v.copy("ensemble/checks/regression_check.mojo::_mix", "murmur3_fmix64", t.rows(), bad[5], first[5], False)
    v.copy("ensemble/checks/sample_weight_check.mojo::_mix", "murmur3_fmix64", t.rows(), bad[6], first[6], False)
    v.copy("ensemble/checks/split_check.mojo::_mix", "murmur3_fmix64", t.rows(), bad[7], first[7], False)
    v.copy("ensemble/checks/train_check.mojo::_mix", "murmur3_fmix64", t.rows(), bad[8], first[8], False)
    v.copy("extratrees/checks/fixed_point_check.mojo::mix64", "murmur3_fmix64", t.rows(), bad[9], first[9], False)


def _check_seeded_fmix32(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.seeded_fmix32`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var A32 = t.a32(r)
        var want = _bits(seeded_fmix32(A32))
        if _bits(c_ensemble_checks_builder_kernels_check__h32(A32)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("ensemble/checks/builder_kernels_check.mojo::h32", "seeded_fmix32", t.rows(), bad[0], first[0], False)


def _check_splitmix64(t: _Inputs, mut v: _Verdict):
    """41 copies (+0 known findings) against `fixture_rng.splitmix64`."""
    var bad = List[Int](length=41, fill=0)
    var first = List[Int](length=41, fill=-1)
    for r in range(t.rows()):
        var A = t.a(r)
        var want = _bits(_canonical_splitmix64(A))
        if _bits(c_checks_division_check__splitmix(A)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_checks_exact_estimation_check__splitmix(A)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_checks_ieee_arith_check__splitmix(A)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_checks_logloss_estimator_check__splitmix(A)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
        if _bits(c_checks_logloss_target_check__splitmix(A)) != want:
            bad[4] += 1
            if first[4] < 0:
                first[4] = r
        if _bits(c_checks_logloss_train_check__splitmix(A)) != want:
            bad[5] += 1
            if first[5] < 0:
                first[5] = r
        if _bits(c_checks_multiclass_oracle_check__splitmix(A)) != want:
            bad[6] += 1
            if first[6] < 0:
                first[6] = r
        if _bits(c_checks_multiclass_score_check__splitmix(A)) != want:
            bad[7] += 1
            if first[7] < 0:
                first[7] = r
        if _bits(c_checks_multiclass_train_check__splitmix(A)) != want:
            bad[8] += 1
            if first[8] < 0:
                first[8] = r
        if _bits(c_checks_multilogit_check__splitmix(A)) != want:
            bad[9] += 1
            if first[9] < 0:
                first[9] = r
        if _bits(c_checks_pointwise_target_check__splitmix(A)) != want:
            bad[10] += 1
            if first[10] < 0:
                first[10] = r
        if _bits(c_checks_portable_fmax_check__splitmix(A)) != want:
            bad[11] += 1
            if first[11] < 0:
                first[11] = r
        if _bits(c_checks_portable_gelu_check__splitmix(A)) != want:
            bad[12] += 1
            if first[12] < 0:
                first[12] = r
        if _bits(c_checks_portable_log2_64_check__splitmix(A)) != want:
            bad[13] += 1
            if first[13] < 0:
                first[13] = r
        if _bits(c_checks_portable_log64_check__splitmix(A)) != want:
            bad[14] += 1
            if first[14] < 0:
                first[14] = r
        if _bits(c_checks_portable_nn_check__splitmix(A)) != want:
            bad[15] += 1
            if first[15] < 0:
                first[15] = r
        if _bits(c_checks_portable_pow64_check__mix(A)) != want:
            bad[16] += 1
            if first[16] < 0:
                first[16] = r
        if _bits(c_checks_portable_sqrtcos_check__splitmix(A)) != want:
            bad[17] += 1
            if first[17] < 0:
                first[17] = r
        if _bits(c_checks_portable_translog_check__splitmix(A)) != want:
            bad[18] += 1
            if first[18] < 0:
                first[18] = r
        if _bits(c_checks_portable_trig_check__splitmix(A)) != want:
            bad[19] += 1
            if first[19] < 0:
                first[19] = r
        if _bits(c_checks_second_der_weights_check__splitmix(A)) != want:
            bad[20] += 1
            if first[20] < 0:
                first[20] = r
        if _bits(c_embedding_checks_embedding_fixture__emb_splitmix64(A)) != want:
            bad[21] += 1
            if first[21] < 0:
                first[21] = r
        if _bits(c_ensemble_bench_rf_bench__mix(A)) != want:
            bad[22] += 1
            if first[22] < 0:
                first[22] = r
        if _bits(c_extratrees_checks_fixtures__splitmix64(A)) != want:
            bad[23] += 1
            if first[23] < 0:
                first[23] = r
        if _bits(c_gemm_checks_gemm_rtf_boundary_check__mix(A)) != want:
            bad[24] += 1
            if first[24] < 0:
                first[24] = r
        if _bits(c_mamba_checks_mamba_fixture__corpus_splitmix64(A)) != want:
            bad[25] += 1
            if first[25] < 0:
                first[25] = r
        if _bits(c_neighbors_checks_knn_selector_arms_check__splitmix64(A)) != want:
            bad[26] += 1
            if first[26] < 0:
                first[26] = r
        if _bits(c_neighbors_checks_knn_selector_bound_compact_check__splitmix64(A)) != want:
            bad[27] += 1
            if first[27] < 0:
                first[27] = r
        if _bits(c_spectral_checks_spectral_fixture__splitmix64(A)) != want:
            bad[28] += 1
            if first[28] < 0:
                first[28] = r
        if _bits(c_training_checks_loss_fixture__ce_splitmix64(A)) != want:
            bad[29] += 1
            if first[29] < 0:
                first[29] = r
        if _bits(c_training_checks_optimizer_fixture__opt_splitmix64(A)) != want:
            bad[30] += 1
            if first[30] < 0:
                first[30] = r
        if _bits(c_training_checks_step_glue_check__mix(A)) != want:
            bad[31] += 1
            if first[31] < 0:
                first[31] = r
        if _bits(c_training_checks_train_loop__train_splitmix64(A)) != want:
            bad[32] += 1
            if first[32] < 0:
                first[32] = r
        if _bits(c_transformer_checks_transformer_fixture__fixture_splitmix64(A)) != want:
            bad[33] += 1
            if first[33] < 0:
                first[33] = r
        if _bits(c_umap_checks_batch_determinism_check__mix(A)) != want:
            bad[34] += 1
            if first[34] < 0:
                first[34] = r
        if _bits(c_umap_checks_batch_epoch_cost_check__mix(A)) != want:
            bad[35] += 1
            if first[35] < 0:
                first[35] = r
        if _bits(c_umap_host_umap_oracle__splitmix64(A)) != want:
            bad[36] += 1
            if first[36] < 0:
                first[36] = r
        if _bits(c_umap_optimizer__splitmix64(A)) != want:
            bad[37] += 1
            if first[37] < 0:
                first[37] = r
        if _bits(c_umap_optimizer_fast__mix(A)) != want:
            bad[38] += 1
            if first[38] < 0:
                first[38] = r
        if _bits(c_x_prep_mutual_info__splitmix(A)) != want:
            bad[39] += 1
            if first[39] < 0:
                first[39] = r
        if _bits(c_x_prep_seams_prep_oracle__splitmix(A)) != want:
            bad[40] += 1
            if first[40] < 0:
                first[40] = r
    v.copy("checks/division_check.mojo::_splitmix", "splitmix64", t.rows(), bad[0], first[0], False)
    v.copy("checks/exact_estimation_check.mojo::splitmix", "splitmix64", t.rows(), bad[1], first[1], False)
    v.copy("checks/ieee_arith_check.mojo::_splitmix", "splitmix64", t.rows(), bad[2], first[2], False)
    v.copy("checks/logloss_estimator_check.mojo::splitmix", "splitmix64", t.rows(), bad[3], first[3], False)
    v.copy("checks/logloss_target_check.mojo::splitmix", "splitmix64", t.rows(), bad[4], first[4], False)
    v.copy("checks/logloss_train_check.mojo::splitmix", "splitmix64", t.rows(), bad[5], first[5], False)
    v.copy("checks/multiclass_oracle_check.mojo::splitmix", "splitmix64", t.rows(), bad[6], first[6], False)
    v.copy("checks/multiclass_score_check.mojo::splitmix", "splitmix64", t.rows(), bad[7], first[7], False)
    v.copy("checks/multiclass_train_check.mojo::splitmix", "splitmix64", t.rows(), bad[8], first[8], False)
    v.copy("checks/multilogit_check.mojo::splitmix", "splitmix64", t.rows(), bad[9], first[9], False)
    v.copy("checks/pointwise_target_check.mojo::splitmix", "splitmix64", t.rows(), bad[10], first[10], False)
    v.copy("checks/portable_fmax_check.mojo::_splitmix", "splitmix64", t.rows(), bad[11], first[11], False)
    v.copy("checks/portable_gelu_check.mojo::_splitmix", "splitmix64", t.rows(), bad[12], first[12], False)
    v.copy("checks/portable_log2_64_check.mojo::_splitmix", "splitmix64", t.rows(), bad[13], first[13], False)
    v.copy("checks/portable_log64_check.mojo::_splitmix", "splitmix64", t.rows(), bad[14], first[14], False)
    v.copy("checks/portable_nn_check.mojo::_splitmix", "splitmix64", t.rows(), bad[15], first[15], False)
    v.copy("checks/portable_pow64_check.mojo::_mix", "splitmix64", t.rows(), bad[16], first[16], False)
    v.copy("checks/portable_sqrtcos_check.mojo::_splitmix", "splitmix64", t.rows(), bad[17], first[17], False)
    v.copy("checks/portable_translog_check.mojo::_splitmix", "splitmix64", t.rows(), bad[18], first[18], False)
    v.copy("checks/portable_trig_check.mojo::_splitmix", "splitmix64", t.rows(), bad[19], first[19], False)
    v.copy("checks/second_der_weights_check.mojo::splitmix", "splitmix64", t.rows(), bad[20], first[20], False)
    v.copy("embedding/checks/embedding_fixture.mojo::emb_splitmix64", "splitmix64", t.rows(), bad[21], first[21], False)
    v.copy("ensemble/bench/rf_bench.mojo::_mix", "splitmix64", t.rows(), bad[22], first[22], False)
    v.copy("extratrees/checks/fixtures.mojo::splitmix64", "splitmix64", t.rows(), bad[23], first[23], False)
    v.copy("gemm/checks/gemm_rtf_boundary_check.mojo::_mix", "splitmix64", t.rows(), bad[24], first[24], False)
    v.copy("mamba/checks/mamba_fixture.mojo::corpus_splitmix64", "splitmix64", t.rows(), bad[25], first[25], False)
    v.copy("neighbors/checks/knn_selector_arms_check.mojo::splitmix64", "splitmix64", t.rows(), bad[26], first[26], False)
    v.copy("neighbors/checks/knn_selector_bound_compact_check.mojo::splitmix64", "splitmix64", t.rows(), bad[27], first[27], False)
    v.copy("spectral/checks/spectral_fixture.mojo::splitmix64", "splitmix64", t.rows(), bad[28], first[28], False)
    v.copy("training/checks/loss_fixture.mojo::ce_splitmix64", "splitmix64", t.rows(), bad[29], first[29], False)
    v.copy("training/checks/optimizer_fixture.mojo::opt_splitmix64", "splitmix64", t.rows(), bad[30], first[30], False)
    v.copy("training/checks/step_glue_check.mojo::_mix", "splitmix64", t.rows(), bad[31], first[31], False)
    v.copy("training/checks/train_loop.mojo::train_splitmix64", "splitmix64", t.rows(), bad[32], first[32], False)
    v.copy("transformer/checks/transformer_fixture.mojo::fixture_splitmix64", "splitmix64", t.rows(), bad[33], first[33], False)
    v.copy("umap/checks/batch_determinism_check.mojo::_mix", "splitmix64", t.rows(), bad[34], first[34], False)
    v.copy("umap/checks/batch_epoch_cost_check.mojo::_mix", "splitmix64", t.rows(), bad[35], first[35], False)
    v.copy("umap/host/umap_oracle.mojo::_splitmix64", "splitmix64", t.rows(), bad[36], first[36], False)
    v.copy("umap/optimizer.mojo::_splitmix64", "splitmix64", t.rows(), bad[37], first[37], False)
    v.copy("umap/optimizer_fast.mojo::_mix", "splitmix64", t.rows(), bad[38], first[38], False)
    v.copy("x_prep/mutual_info.mojo::_splitmix", "splitmix64", t.rows(), bad[39], first[39], False)
    v.copy("x_prep/seams/prep_oracle.mojo::splitmix", "splitmix64", t.rows(), bad[40], first[40], False)


def _check_splitmix64_finalizer(t: _Inputs, mut v: _Verdict):
    """3 copies (+0 known findings) against `fixture_rng.splitmix64_finalizer`."""
    var bad = List[Int](length=3, fill=0)
    var first = List[Int](length=3, fill=-1)
    for r in range(t.rows()):
        var A = t.a(r)
        var want = _bits(splitmix64_finalizer(A))
        if _bits(c_extratrees_checks_split_reduce_check__mix64(A)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_xtrees_checks_glue_oracle__splitmix(A)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_xtrees_ops__mix64(A)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
    v.copy("extratrees/checks/split_reduce_check.mojo::mix64", "splitmix64_finalizer", t.rows(), bad[0], first[0], False)
    v.copy("xtrees/checks/glue_oracle.mojo::splitmix", "splitmix64_finalizer", t.rows(), bad[1], first[1], False)
    v.copy("xtrees/ops.mojo::mix64", "splitmix64_finalizer", t.rows(), bad[2], first[2], False)


def _check_splitmix64_fold(t: _Inputs, mut v: _Verdict):
    """2 copies (+0 known findings) against `fixture_rng.splitmix64_fold`."""
    var bad = List[Int](length=2, fill=0)
    var first = List[Int](length=2, fill=-1)
    for r in range(t.rows()):
        var A = t.a(r)
        var B = t.b(r)
        var want = _bits(splitmix64_fold(A, B))
        if _bits(c_extratrees_checks_bestfirst_check__mix64(A, B)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_extratrees_checks_bestfirst_fingerprint__mix64(A, B)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
    v.copy("extratrees/checks/bestfirst_check.mojo::mix64", "splitmix64_fold", t.rows(), bad[0], first[0], False)
    v.copy("extratrees/checks/bestfirst_fingerprint.mojo::mix64", "splitmix64_fold", t.rows(), bad[1], first[1], False)


def _check_splitmix64_of_sum(t: _Inputs, mut v: _Verdict):
    """2 copies (+0 known findings) against `fixture_rng.splitmix64_of_sum`."""
    var bad = List[Int](length=2, fill=0)
    var first = List[Int](length=2, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var want = _bits(splitmix64_of_sum(IA, IB))
        if _bits(c_gemm_checks_gemm_tuned_probe__mix(IA, IB)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_gemm_checks_gemm_unpinned_price__mix(IA, IB)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
    v.copy("gemm/checks/gemm_tuned_probe.mojo::_mix", "splitmix64_of_sum", t.rows(), bad[0], first[0], False)
    v.copy("gemm/checks/gemm_unpinned_price.mojo::_mix", "splitmix64_of_sum", t.rows(), bad[1], first[1], False)


def _check_splitmix_low31(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.splitmix_low31`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var want = _bits(splitmix_low31(IA))
        if _bits(c_ensemble_checks_core_primitives_check__mix(IA)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("ensemble/checks/core_primitives_check.mojo::mix", "splitmix_low31", t.rows(), bad[0], first[0], False)


def _check_splitmix_pair(t: _Inputs, mut v: _Verdict):
    """12 copies (+0 known findings) against `fixture_rng.splitmix_pair`."""
    var bad = List[Int](length=12, fill=0)
    var first = List[Int](length=12, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var want = _bits(splitmix_pair(IA, IB))
        if _bits(c_bench_apple_identical_steps_main__mix(IA, IB)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_bench_gemm_card_main__mix(IA, IB)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_bench_gemm_ladder_main__mix(IA, IB)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_bench_gemm_price_main__mix(IA, IB)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
        if _bits(c_bench_lanes_price_main__gemm_mix(IA, IB)) != want:
            bad[4] += 1
            if first[4] < 0:
                first[4] = r
        if _bits(c_bench_linalg_price_main__mix(IA, IB)) != want:
            bad[5] += 1
            if first[5] < 0:
                first[5] = r
        if _bits(c_checks_gram_splitk_check__mix(IA, IB)) != want:
            bad[6] += 1
            if first[6] < 0:
                first[6] = r
        if _bits(c_checks_partitions_reduce_check__mix(IA, IB)) != want:
            bad[7] += 1
            if first[7] < 0:
                first[7] = r
        if _bits(c_checks_pointwise_subsets_check__mix(IA, IB)) != want:
            bad[8] += 1
            if first[8] < 0:
                first[8] = r
        if _bits(c_checks_vendor_correctness_check__mix(IA, IB)) != want:
            bad[9] += 1
            if first[9] < 0:
                first[9] = r
        if _bits(c_core_gemm_identity_check__mix(IA, IB)) != want:
            bad[10] += 1
            if first[10] < 0:
                first[10] = r
        if _bits(c_glm_checks_ols_trace__mix(IA, IB)) != want:
            bad[11] += 1
            if first[11] < 0:
                first[11] = r
    v.copy("bench/apple_identical_steps_main.mojo::_mix", "splitmix_pair", t.rows(), bad[0], first[0], False)
    v.copy("bench/gemm_card_main.mojo::_mix", "splitmix_pair", t.rows(), bad[1], first[1], False)
    v.copy("bench/gemm_ladder_main.mojo::_mix", "splitmix_pair", t.rows(), bad[2], first[2], False)
    v.copy("bench/gemm_price_main.mojo::_mix", "splitmix_pair", t.rows(), bad[3], first[3], False)
    v.copy("bench/lanes_price_main.mojo::_gemm_mix", "splitmix_pair", t.rows(), bad[4], first[4], False)
    v.copy("bench/linalg_price_main.mojo::_mix", "splitmix_pair", t.rows(), bad[5], first[5], False)
    v.copy("checks/gram_splitk_check.mojo::_mix", "splitmix_pair", t.rows(), bad[6], first[6], False)
    v.copy("checks/partitions_reduce_check.mojo::_mix", "splitmix_pair", t.rows(), bad[7], first[7], False)
    v.copy("checks/pointwise_subsets_check.mojo::_mix", "splitmix_pair", t.rows(), bad[8], first[8], False)
    v.copy("checks/vendor_correctness_check.mojo::_mix", "splitmix_pair", t.rows(), bad[9], first[9], False)
    v.copy("core/gemm_identity_check.mojo::_mix", "splitmix_pair", t.rows(), bad[10], first[10], False)
    v.copy("glm/checks/ols_trace.mojo::_mix", "splitmix_pair", t.rows(), bad[11], first[11], False)


def _check_splitmix_pair_unoffset(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.splitmix_pair_unoffset`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var want = _bits(splitmix_pair_unoffset(IA, IB))
        if _bits(c_checks_pointwise_pool_check__mix(IA, IB)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("checks/pointwise_pool_check.mojo::_mix", "splitmix_pair_unoffset", t.rows(), bad[0], first[0], False)


def _check_splitmix_triple(t: _Inputs, mut v: _Verdict):
    """8 copies (+0 known findings) against `fixture_rng.splitmix_triple`."""
    var bad = List[Int](length=8, fill=0)
    var first = List[Int](length=8, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var IC = t.ic(r)
        var want = _bits(splitmix_triple(IA, IB, IC))
        if _bits(c_cholesky_checks_cholesky_fixture__chol_mix64(IA, IB, IC)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_holtwinters_checks_hw_fixture__mix64(IA, IB, IC)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_isolation_forest_checks_if_fixture__mix64(IA, IB, IC)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_kde_checks_kde_fixture__mix64(IA, IB, IC)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
        if _bits(c_metrics_checks_fixtures__splitmix(IA, IB, IC)) != want:
            bad[4] += 1
            if first[4] < 0:
                first[4] = r
        if _bits(c_mixture_checks_gmm_fixture__gmm_mix64(IA, IB, IC)) != want:
            bad[5] += 1
            if first[5] < 0:
                first[5] = r
        if _bits(c_resample_checks_resample_fixture__mix64(IA, IB, IC)) != want:
            bad[6] += 1
            if first[6] < 0:
                first[6] = r
        if _bits(c_tsa_checks_fixtures__splitmix(IA, IB, IC)) != want:
            bad[7] += 1
            if first[7] < 0:
                first[7] = r
    v.copy("cholesky/checks/cholesky_fixture.mojo::chol_mix64", "splitmix_triple", t.rows(), bad[0], first[0], False)
    v.copy("holtwinters/checks/hw_fixture.mojo::mix64", "splitmix_triple", t.rows(), bad[1], first[1], False)
    v.copy("isolation_forest/checks/if_fixture.mojo::mix64", "splitmix_triple", t.rows(), bad[2], first[2], False)
    v.copy("kde/checks/kde_fixture.mojo::mix64", "splitmix_triple", t.rows(), bad[3], first[3], False)
    v.copy("metrics/checks/fixtures.mojo::splitmix", "splitmix_triple", t.rows(), bad[4], first[4], False)
    v.copy("mixture/checks/gmm_fixture.mojo::gmm_mix64", "splitmix_triple", t.rows(), bad[5], first[5], False)
    v.copy("resample/checks/resample_fixture.mojo::mix64", "splitmix_triple", t.rows(), bad[6], first[6], False)
    v.copy("tsa/checks/fixtures.mojo::splitmix", "splitmix_triple", t.rows(), bad[7], first[7], False)


def _check_symmetric_cell_f64(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.symmetric_cell_f64`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var IC = t.ic(r)
        var want = _bits(symmetric_cell_f64(IA, IB, IC))
        if _bits(c_decomposition_checks_jacobi_check__hashed(IA, IB, IC)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("decomposition/checks/jacobi_check.mojo::_hashed", "symmetric_cell_f64", t.rows(), bad[0], first[0], False)


def _check_u01_row(t: _Inputs, mut v: _Verdict):
    """13 copies (+0 known findings) against `fixture_rng.u01_row`."""
    var bad = List[Int](length=13, fill=0)
    var first = List[Int](length=13, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var IC = t.ic(r)
        var want = _bits(u01_row(IA, IB, IC))
        if _bits(c_bench_bench_main__u01(IA, IB, IC)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_bench_dsweep_main__u01(IA, IB, IC)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_bench_scaling_main__u01(IA, IB, IC)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
        if _bits(c_bench_speed_classical_ladder_main__u01(IA, IB, IC)) != want:
            bad[3] += 1
            if first[3] < 0:
                first[3] = r
        if _bits(c_bench_speed_classical_speed_main__u01(IA, IB, IC)) != want:
            bad[4] += 1
            if first[4] < 0:
                first[4] = r
        if _bits(c_dbscan_phase_main__u01(IA, IB, IC)) != want:
            bad[5] += 1
            if first[5] < 0:
                first[5] = r
        if _bits(c_decomposition_checks_pca_check__wide_u01(IA, IB, IC)) != want:
            bad[6] += 1
            if first[6] < 0:
                first[6] = r
        if _bits(c_decomposition_checks_svd_full_check__u01(IA, IB, IC)) != want:
            bad[7] += 1
            if first[7] < 0:
                first[7] = r
        if _bits(c_glm_checks_logistic_check__u01(IA, IB, IC)) != want:
            bad[8] += 1
            if first[8] < 0:
                first[8] = r
        if _bits(c_glm_checks_multinomial_check__u01(IA, IB, IC)) != want:
            bad[9] += 1
            if first[9] < 0:
                first[9] = r
        if _bits(c_glm_checks_ols_check__u01(IA, IB, IC)) != want:
            bad[10] += 1
            if first[10] < 0:
                first[10] = r
        if _bits(c_glm_checks_ridge_check__u01(IA, IB, IC)) != want:
            bad[11] += 1
            if first[11] < 0:
                first[11] = r
        if _bits(c_solver_host_cd_oracle__u01(IA, IB, IC)) != want:
            bad[12] += 1
            if first[12] < 0:
                first[12] = r
    v.copy("bench/bench_main.mojo::_u01", "u01_row", t.rows(), bad[0], first[0], False)
    v.copy("bench/dsweep_main.mojo::_u01", "u01_row", t.rows(), bad[1], first[1], False)
    v.copy("bench/scaling_main.mojo::_u01", "u01_row", t.rows(), bad[2], first[2], False)
    v.copy("bench/speed/classical_ladder_main.mojo::_u01", "u01_row", t.rows(), bad[3], first[3], False)
    v.copy("bench/speed/classical_speed_main.mojo::_u01", "u01_row", t.rows(), bad[4], first[4], False)
    v.copy("dbscan/phase_main.mojo::_u01", "u01_row", t.rows(), bad[5], first[5], False)
    v.copy("decomposition/checks/pca_check.mojo::_wide_u01", "u01_row", t.rows(), bad[6], first[6], False)
    v.copy("decomposition/checks/svd_full_check.mojo::_u01", "u01_row", t.rows(), bad[7], first[7], False)
    v.copy("glm/checks/logistic_check.mojo::_u01", "u01_row", t.rows(), bad[8], first[8], False)
    v.copy("glm/checks/multinomial_check.mojo::_u01", "u01_row", t.rows(), bad[9], first[9], False)
    v.copy("glm/checks/ols_check.mojo::_u01", "u01_row", t.rows(), bad[10], first[10], False)
    v.copy("glm/checks/ridge_check.mojo::_u01", "u01_row", t.rows(), bad[11], first[11], False)
    v.copy("solver/host/cd_oracle.mojo::_u01", "u01_row", t.rows(), bad[12], first[12], False)


def _check_u01_row_feature(t: _Inputs, mut v: _Verdict):
    """3 copies (+0 known findings) against `fixture_rng.u01_row_feature`."""
    var bad = List[Int](length=3, fill=0)
    var first = List[Int](length=3, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var want = _bits(u01_row_feature(IA, IB))
        if _bits(c_neighbors_checks_ball_cover_check__hash01(IA, IB)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_neighbors_checks_ball_cover_knn_check__hash01(IA, IB)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_neighbors_checks_radius_check__hash01(IA, IB)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
    v.copy("neighbors/checks/ball_cover_check.mojo::_hash01", "u01_row_feature", t.rows(), bad[0], first[0], False)
    v.copy("neighbors/checks/ball_cover_knn_check.mojo::_hash01", "u01_row_feature", t.rows(), bad[1], first[1], False)
    v.copy("neighbors/checks/radius_check.mojo::_hash01", "u01_row_feature", t.rows(), bad[2], first[2], False)


def _check_u01_triple(t: _Inputs, mut v: _Verdict):
    """2 copies (+0 known findings) against `fixture_rng.u01_triple`."""
    var bad = List[Int](length=2, fill=0)
    var first = List[Int](length=2, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var IC = t.ic(r)
        var want = _bits(u01_triple(IA, IB, IC))
        if _bits(c_metrics_checks_fixtures__u01(IA, IB, IC)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_tsa_checks_fixtures__u01(IA, IB, IC)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
    v.copy("metrics/checks/fixtures.mojo::u01", "u01_triple", t.rows(), bad[0], first[0], False)
    v.copy("tsa/checks/fixtures.mojo::u01", "u01_triple", t.rows(), bad[1], first[1], False)


def _check_u16_row_f32(t: _Inputs, mut v: _Verdict):
    """3 copies (+0 known findings) against `fixture_rng.u16_row_f32`."""
    var bad = List[Int](length=3, fill=0)
    var first = List[Int](length=3, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var IC = t.ic(r)
        var want = _bits(u16_row_f32(IA, IB, IC))
        if _bits(c_bench_identity_price_main__u01(IA, IB, IC)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_bench_lanes_price_main__price_u01(IA, IB, IC)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_bench_samba_rms_price_main__u01(IA, IB, IC)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
    v.copy("bench/identity_price_main.mojo::_u01", "u16_row_f32", t.rows(), bad[0], first[0], False)
    v.copy("bench/lanes_price_main.mojo::_price_u01", "u16_row_f32", t.rows(), bad[1], first[1], False)
    v.copy("bench/samba_rms_price_main.mojo::_u01", "u16_row_f32", t.rows(), bad[2], first[2], False)


def _check_xorshift32_of_index(t: _Inputs, mut v: _Verdict):
    """3 copies (+0 known findings) against `fixture_rng.xorshift32_of_index`."""
    var bad = List[Int](length=3, fill=0)
    var first = List[Int](length=3, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var want = _bits(xorshift32_of_index(IA))
        if _bits(c_checks_radix_sort_check__hashed(IA)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
        if _bits(c_checks_segmented_scan_check__hashed(IA)) != want:
            bad[1] += 1
            if first[1] < 0:
                first[1] = r
        if _bits(c_ensemble_checks_quantiles_check__hashed(IA)) != want:
            bad[2] += 1
            if first[2] < 0:
                first[2] = r
    v.copy("checks/radix_sort_check.mojo::hashed", "xorshift32_of_index", t.rows(), bad[0], first[0], False)
    v.copy("checks/segmented_scan_check.mojo::hashed", "xorshift32_of_index", t.rows(), bad[1], first[1], False)
    v.copy("ensemble/checks/quantiles_check.mojo::hashed", "xorshift32_of_index", t.rows(), bad[2], first[2], False)


def _check_xorshift32_of_pair(t: _Inputs, mut v: _Verdict):
    """1 copies (+0 known findings) against `fixture_rng.xorshift32_of_pair`."""
    var bad = List[Int](length=1, fill=0)
    var first = List[Int](length=1, fill=-1)
    for r in range(t.rows()):
        var IA = t.ia(r)
        var IB = t.ib(r)
        var want = _bits(xorshift32_of_pair(IA, IB))
        if _bits(c_checks_sym_arms_check__mix(IA, IB)) != want:
            bad[0] += 1
            if first[0] < 0:
                first[0] = r
    v.copy("checks/sym_arms_check.mojo::mix", "xorshift32_of_pair", t.rows(), bad[0], first[0], False)


def main() raises:
    var t = _Inputs()
    if t.rows() < 65536:
        raise Error("VACUOUS: fewer than 65,536 input rows")
    var saw_zero = False
    var saw_max = False
    for r in range(t.rows()):
        if t.a(r) == 0:
            saw_zero = True
        if t.a(r) == UInt64(0xFFFFFFFFFFFFFFFF):
            saw_max = True
    if not (saw_zero and saw_max):
        raise Error("VACUOUS: the input table lost its 0 or its all-ones word")
    print("fixture-RNG gate:", t.rows(), "input rows (", HASHED_ROWS, "hashed + adversarial cross product )")
    _vacuity_control(t)
    comptime if PLANT:
        print("PLANT    FIXTURE_RNG_PLANT is set: the planted variant stands in for splitmix64; this run MUST fail")
    var v = _Verdict()
    _check_binade_hashed_f32(t, v)
    _check_golden32_mix(t, v)
    _check_golden_top24(t, v)
    _check_hashed_in_range(t, v)
    _check_hashed_signed_f32(t, v)
    _check_hashed_unit_f32_24(t, v)
    _check_hashed_unit_f64(t, v)
    _check_lowbias32(t, v)
    _check_mix29_pair(t, v)
    _check_mix29_triple(t, v)
    _check_modhash(t, v)
    _check_modhash_salted(t, v)
    _check_murmur3_fmix32(t, v)
    _check_murmur3_fmix64(t, v)
    _check_seeded_fmix32(t, v)
    _check_splitmix64(t, v)
    _check_splitmix64_finalizer(t, v)
    _check_splitmix64_fold(t, v)
    _check_splitmix64_of_sum(t, v)
    _check_splitmix_low31(t, v)
    _check_splitmix_pair(t, v)
    _check_splitmix_pair_unoffset(t, v)
    _check_splitmix_triple(t, v)
    _check_symmetric_cell_f64(t, v)
    _check_u01_row(t, v)
    _check_u01_row_feature(t, v)
    _check_u01_triple(t, v)
    _check_u16_row_f32(t, v)
    _check_xorshift32_of_index(t, v)
    _check_xorshift32_of_pair(t, v)
    print(
        "fixture-RNG gate:", v.copies, "copies,", v.agree, "agree,", v.findings,
        "known findings,", v.failed, "failed",
    )
    if v.failed > 0:
        print("FAIL: a fixture-RNG copy differs from its canonical behavior")
        exit(1)
    print("PASS")
