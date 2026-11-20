#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Harness bounds as a function of the mode policy mask the wheel reports.

The board harness holds our arm to FP32-identity bounds (the host-reference check of the forward lanes,
`grad_max_rel_diff_vs_ours`). Those bounds are right for a wheel that relaxes no row of the mode policy table
(checks/mode_policy.mojo) and wrong for a `fast` wheel that relaxes one: a relaxed build would be marked
`quality_failed` for arithmetic it is allowed. This module reads the mask from the wheel and widens exactly the
bounds a relaxed row moves. Mask 0 returns today's bounds unchanged; a wheel without the mask entry is mask 0.

The mask is read through the export `mojolearn_mode_policy_mask` of the `_mojolearn` binding (bit i = row i).

THE MATRIX-UNITS ROW (bit 4, lane fx-fast-matrix-units, gemm/fast_matrix_units.mojo `matrix_units_cell_bound`).
TF32 keeps 11 significant bits of each operand (relative operand error <= 2^-10), so per cell of K terms
    |c - c_fp32| <= (2 * 2^-10 + 2^-20) * sum|a_k b_k|     plus the FP32 allowance,
and the RMS error over the cells is at most 1.25 * 2^-10 of the cells' scale (truncation gives 1.08; an 8-bit
operand gives at least 1.6, so a silent precision collapse still fails).

The harness sees outputs, never the operands, so `sum|a_k b_k|` of a cell is not available to it. The widened
output checks use the reference's largest magnitude as the cell scale (it is at most the absolute sum of the cell
that produced it), and keep the RMS test, which is the one that catches a collapse. Rows with no bound here
(fold, contraction, flush, math, waits, tile) leave every bound as it is: a build that relaxes one of them and
fails an FP32 bound is reported, not excused, until its lane derives a bound and adds it here.
"""

#: Row names by bit of MODE_POLICY_MASK (checks/mode_policy.mojo).
POLICY_ROWS = ("fold_free", "contract_free", "flush_off", "vendor_math", "matrix_units", "waits_elided", "tile_free", "bf16_units")
MATRIX_UNITS_BIT = 1 << POLICY_ROWS.index("matrix_units")
MASK_BINDING, MASK_ENTRY = "_mojolearn", "mojolearn_mode_policy_mask"

#: Per cell: (2 * 2^-10 + 2^-20) of the cell's absolute sum.
MATRIX_UNITS_CELL_REL = 2.0 * 2.0 ** -10 + 2.0 ** -20
#: RMS over the cells: 1.25 * 2^-10 of the cells' scale.
MATRIX_UNITS_RMS_REL = 1.25 * 2.0 ** -10


def wheel_policy_mask(ml=None):
    """The mode policy mask of the installed wheel's `_mojolearn` binding; 0 when the wheel has no such entry
    (built before the table existed), is not importable here, or the entry fails."""
    try:
        if ml is None:
            import mojolearn as ml
        return int(getattr(_binding(ml), MASK_ENTRY)())
    except Exception:
        return 0


def _binding(ml):
    """The `_mojolearn` binding of the set the package selected: through the selector when the package has one
    (`_backend.binding`, which loads it), else the already loaded extension module."""
    import sys
    backend = getattr(ml, "_backend", None)
    if backend is not None and hasattr(backend, "binding"):
        return backend.binding(MASK_BINDING)
    return sys.modules[ml.__name__ + "." + MASK_BINDING]


def relaxed_rows(mask):
    """Names of the rows set in `mask`."""
    return [row for bit, row in enumerate(POLICY_ROWS) if int(mask or 0) >> bit & 1]


def matrix_units(mask):
    return bool(int(mask or 0) & MATRIX_UNITS_BIT)


def matrix_units_cell_bound(sum_abs, fp32_allowance=0.0):
    """The per-cell bound where the absolute sum of the cell's terms is known."""
    return MATRIX_UNITS_CELL_REL * sum_abs + fp32_allowance


def forward_reference_bound(np, ref, mask, atol, rtol):
    """Per-cell bound of an own arm's forward against the host reference `ref` (float64 array).
    Mask without the matrix-units bit: `atol + rtol * |ref|`, today's bound, unchanged."""
    bound = atol + rtol * np.abs(ref)
    if matrix_units(mask) and ref.size:
        bound = bound + matrix_units_cell_bound(float(np.abs(ref).max()))
    return bound


def forward_reference_rms_passed(np, error, ref, mask):
    """The RMS half of the matrix-units bound: rms(error) <= 1.25 * 2^-10 * rms(ref). True (nothing to check)
    when the matrix-units bit is off. Returns (passed, rms error over rms reference)."""
    if not matrix_units(mask) or not ref.size:
        return True, None
    scale = float(np.sqrt(np.mean(np.square(ref))))
    ratio = float(np.sqrt(np.mean(np.square(error)))) / scale if scale else (0.0 if not float(np.abs(error).max()) else float("inf"))
    return ratio <= MATRIX_UNITS_RMS_REL, ratio


#: Reference-distance metrics of tools/board_quality.py whose FP32 margin a relaxed row widens, with the widening.
#: grad_max_rel_diff_vs_ours is max|g - ours| / max|ours|: already relative to the cell scale.
_REFERENCE_WIDENING = {"grad_max_rel_diff_vs_ours": ((MATRIX_UNITS_BIT, MATRIX_UNITS_CELL_REL),)}


def reference_margin(metric, fp32_margin, mask):
    """The margin of a board_quality REFERENCE metric for a wheel reporting `mask`. Mask 0, or a metric no row
    moves: `fp32_margin` itself."""
    margin = fp32_margin
    for bit, widening in _REFERENCE_WIDENING.get(metric, ()):
        if int(mask or 0) & bit:
            margin = margin + widening
    return margin
