"""Read the exact shared route switches used by GPU and host columns."""
from decomposition.pca_rr_switch import PCA_RR_EIGH, PCA_RR_SWEEPS
from solver.impl.cd_gram_rule import cd_idn_gram_cost_ok, cd_idn_gram_shape
from mixture.chol_order import IDN_GMM_FUSED_CHOL, gmm_idn_chol_applies
from gbdt.models.kernel.add_bin_values import IDN_PREDICT_FOUR
from x_decomp.rr_block import rb_use


def main() raises:
    # TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old-path umbrella, EXP:2438; every IDENTICAL default kept) deleted 2026-11-15 by lane/expdel2-16-decomposition; code recoverable at af37fc8403.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-16-decomposition.patch; record in docs/TOMBSTONES.md.
    # IDN_EIGH_BLOCK and CD_IDN_GRAM_ON are constants since their switches were deleted (2026-11-15,
    # docs/TOMBSTONES.md); the routes are checked through rb_use and cd_idn_gram_shape below.
    if rb_use(4096) or rb_use(513):
        raise Error("default wave must use established rotation eigh on all columns")
    comptime assert PCA_RR_SWEEPS == 60, "PCA budget changed; review quality coverage"
    comptime assert PCA_RR_EIGH
    comptime assert IDN_GMM_FUSED_CHOL
    comptime assert IDN_PREDICT_FOUR
    if not cd_idn_gram_shape(4096, 16):
        raise Error("Gram CD quality fixture does not take expected shape branch")
    # lane/no-bench-tuning-2: the Gram-vs-row-sweep work rule at neighbor
    # shapes around its bounds (tallness n_rows >= 4 n_cols, work
    # n_cols (n_rows + 128) <= 256 n_rows, kernel ceiling 256)
    var gram_cases = [(256, 64, True), (400, 100, True), (400, 101, False),
                      (1024, 227, True), (1024, 228, False),
                      (100000, 255, True), (100000, 256, False), (4096, 65, True)]
    for c in gram_cases:
        if cd_idn_gram_cost_ok(c[0], c[1]) != c[2]:
            raise Error("Gram CD work rule at " + String(c[0]) + " x " + String(c[1]) + " is not " + String(c[2]))
    if not gmm_idn_chol_applies(16):
        raise Error("GMM quality fixture does not take expected fused branch")
    print("WAVE_ROUTES PASS pca_sweeps", PCA_RR_SWEEPS,
          "gmm_fused", IDN_GMM_FUSED_CHOL,
          "gbdt_four", IDN_PREDICT_FOUR)
