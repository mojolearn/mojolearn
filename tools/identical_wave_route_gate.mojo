"""Read the exact shared route switches used by GPU and host columns."""
from std.sys.compile import is_defined
from decomposition.pca_rr_switch import PCA_RR_EIGH, PCA_RR_SWEEPS
from solver.impl.cd_gram_rule import CD_IDN_GRAM_ON, cd_idn_gram_shape
from mixture.chol_order import IDN_GMM_FUSED_CHOL, gmm_idn_chol_applies
from gbdt.models.kernel.add_bin_values import IDN_PREDICT_FOUR
from x_decomp.rr_block import IDN_EIGH_BLOCK, rb_use


def main() raises:
    comptime off = is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    comptime assert not IDN_EIGH_BLOCK, "Unsafe block eigh candidate must remain disabled in the default wave"
    if rb_use(4096) or rb_use(513):
        raise Error("default wave must use established rotation eigh on all columns")
    comptime assert PCA_RR_SWEEPS == 60, "PCA budget changed; review quality coverage"
    comptime assert PCA_RR_EIGH == (not off)
    comptime assert CD_IDN_GRAM_ON == (not off)
    comptime assert IDN_GMM_FUSED_CHOL == (not off)
    comptime assert IDN_PREDICT_FOUR == (not off)
    if cd_idn_gram_shape(4096, 16) != (not off):
        raise Error("Gram CD quality fixture does not take expected shape branch")
    if gmm_idn_chol_applies(16) != (not off):
        raise Error("GMM quality fixture does not take expected fused branch")
    print("WAVE_ROUTES PASS all_off", off, "pca_sweeps", PCA_RR_SWEEPS,
          "gram", CD_IDN_GRAM_ON, "gmm_fused", IDN_GMM_FUSED_CHOL,
          "gbdt_four", IDN_PREDICT_FOUR)
