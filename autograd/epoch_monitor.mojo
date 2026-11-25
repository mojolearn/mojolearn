# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE EPOCH MONITOR: per-epoch validation, the best epoch and early stopping, one block for every training loop
(lane rnn-parity, Andrew 2026-11-25: "per-epoch validation, best-epoch restore and early stopping in the GENERIC
training loop, so Sequential, MLP, CNN, transformer and recurrent models all get it").

The decision is scikit-learn's and Keras's, in Mojo (AGENTS.md "No Python in the runtime"): after each epoch the
loop hands the monitored loss (the validation loss, or the training loss without validation data) to `observe`.
An epoch improves when its loss is below the best so far; it resets the stall count when it is below best - tol
(scikit-learn's MLP rule `loss > best - tol` counts a stall); training stops once `n_iter_no_change` epochs in a
row have stalled (0: never). The caller keeps the best epoch's parameters (and optimizer state) when `observe`
says it improved and restores them at the end when asked. Host scalars only: the loop reads the one monitored
word per epoch (one download) and nothing else.

Users: sequence/recurrent.mojo rnn_fit (the recurrent estimators) and autograd/model.mojo model_fit_state (the
generic loop of Sequential: MLP, CNN, transformer and recurrent layers; lane rnn-parity-c). The generic loop carries
the best loss and the stall count across a resumed fit in its state words (FS_BEST, FS_STALL): `EpochMonitor.resumed`
starts the block from them and the loop writes `best` and `stall` back, so the words mean what they meant.
"""


struct EpochMonitor(ImplicitlyCopyable, Movable):
    var tol: Float64
    var patience: Int
    var best: Float64
    var best_epoch: Int
    var stall: Int

    def __init__(out self, tol: Float64, patience: Int):
        """`patience` is n_iter_no_change (0: train every epoch), `tol` the improvement that resets the stall."""
        self.tol = tol
        self.patience = patience
        self.best = Float64(1.0e308)
        self.best_epoch = 0
        self.stall = 0

    @staticmethod
    def resumed(tol: Float64, patience: Int, best: Float64, stall: Int) -> Self:
        """The monitor of a resumed fit: the best loss and the stall count an earlier fit left (the generic loop's
        FS_BEST and FS_STALL words). The best epoch counts this call's epochs only: 0 until one improves."""
        var m = Self(tol, patience)
        m.best = best
        m.stall = stall
        return m

    def restart(mut self):
        """Forget the best loss and the stall count (a fit call that monitors validation data starts afresh)."""
        self.best = Float64(1.0e308)
        self.best_epoch = 0
        self.stall = 0

    def observe(mut self, value: Float64, epoch: Int) raises -> Bool:
        """Record the monitored loss of `epoch` (one-based); True when it is the best so far (the caller then
        keeps this epoch's parameters). Refuses a loss that is not finite."""
        if value != value or abs(value) > Float64(3.4028234663852886e38):
            raise Error("ValueError: fit: the monitored loss is not finite")
        if value > self.best - self.tol:
            self.stall += 1
        else:
            self.stall = 0
        if value < self.best:
            self.best = value
            self.best_epoch = epoch
            return True
        return False

    def stop(self) -> Bool:
        """True once n_iter_no_change epochs in a row have stalled."""
        return self.patience > 0 and self.stall >= self.patience
