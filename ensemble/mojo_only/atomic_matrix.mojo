# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Per-vendor ATOMIC WIDTH rows, for the bins cuML's histograms are made of.

NO CUML COUNTERPART. cuML needs no such table: CUDA gives it a 64-bit
integer `atomicAdd` and a `double` `atomicAdd` on every device it targets,
so `bins.cuh` simply uses them (`:31`, `:114-115`). This file exists because
that is not true of every GPU, and this repository's rule is that vendor
divergence lives in a table as a ROW, never as an inline `if apple`.

WHY IT IS HERE AND NOT IN `mojo_only/kernel_matrix.mojo`. That file is the
repository-wide kernel matrix and is the right long-term home for these
rows. It is also being edited by other sessions right now, and file
convergence -- not delegation -- is what predicts integration pain here. So
these rows are scoped to `ensemble/` for this round and RECONCILING THEM
INTO `kernel_matrix.mojo` IS AN OPEN MERGE-TIME ITEM, recorded in
`ensemble/PLAN.md`. The column constants are IMPORTED from the kernel
matrix rather than restated, so the two files cannot drift on what a column
IS -- only on what it can do.

THE ROWS ARE MEASURED FOR APPLE AND TRANSCRIBED FOR THE REST, and the
difference is marked per row. `ensemble/mojo_only/atomic_width_probe.mojo`
is the measurement.
"""

from mojo_only.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_AMD_RDNA,
    COLUMN_APPLE,
    COLUMN_BIT_IDENTICAL,
    COLUMN_INTEL,
    COLUMN_NVIDIA,
    COLUMN_QUALCOMM,
    COLUMN_SPEC_BASELINE,
    TARGET_COLUMN,
)


def column_has_64bit_int_atomics(column: Int) -> Bool:
    """Whether `atomicAdd` on a 64-bit integer exists at all.

    **THIS ROW DECIDES THE WIDTH OF EVERY HISTOGRAM COUNTER IN `ensemble/`.**
    cuML's `BinCountT` is `unsigned long long int` and its histogram
    increment is a 64-bit integer `atomicAdd` (`bins.cuh:14`, `:29-32`).

    - apple    **False. MEASURED, 2026-08-21**, and measured as a hard
               COMPILE error rather than a silent wrong answer:

                   error: Atomic operation is not supported for this type
                          on Apple GPU
                   error: failed to legalize operation 'pop.atomic.rmw'

               That is the good failure mode. A silent drop would have
               produced a small, plausible histogram.
    - nvidia   True. `atomicAdd(unsigned long long*)` is available from
               compute capability 1.2. TRANSCRIBED, not measured here.
    - amd      True. HIP provides `atomicAdd` on `unsigned long long`.
               TRANSCRIBED.
    - amd-rdna True. Same. TRANSCRIBED.
    - qualcomm/intel/spec-baseline: False, conservatively. OpenCL 1.2 core
               atomics are 32-bit; 64-bit atomics are the optional
               `cl_khr_int64_base_atomics` extension, and Vulkan's
               `shaderBufferInt64Atomics` is likewise a feature bit, not a
               guarantee. Assuming absence costs a width we do not need
               (see `bin_counter_is_exact_at_32_bits`); assuming presence
               would cost a compile failure on a device nobody here can
               test.

    BIT_IDENTICAL returns False: the identity column is the intersection,
    and a counter width that varies by vendor is the one thing a
    bit-identical histogram cannot have.
    """
    if column == COLUMN_NVIDIA:
        return True
    if column == COLUMN_AMD:
        return True
    if column == COLUMN_AMD_RDNA:
        return True
    return False


def bin_counter_bits(column: Int) -> Int:
    """The width `ensemble/` actually accumulates histogram counts in.

    Deliberately 32 on EVERY column, including the ones that have the
    64-bit atomic. This is not a lowest-common-denominator compromise; it
    is what the identity claim requires. A histogram accumulated in 32 bits
    on Apple and 64 bits on NVIDIA would produce the same numbers today and
    would stop being ONE algorithm the moment either width mattered, and
    the whole point of this directory's classification path is that the
    same source produces the same bits everywhere.

    It costs nothing, and `bin_counter_is_exact_at_32_bits` is the
    argument.
    """
    _ = column
    return 32


def bin_counter_is_exact_at_32_bits() -> Bool:
    """Why narrowing their 64-bit counter is EXACT and not a truncation.

    Not a capability row -- a proof obligation, written down where the
    width is chosen so that nobody re-derives it or, worse, doubts it and
    "fixes" it back.

    A histogram cell counts sampled instances, so every cell is bounded by
    `n_sampled_rows`. In cuML's RF, `IdxT` is instantiated as `int`
    throughout: `Dataset::n_sampled_rows` is `IdxT` (`dataset.h:32`), the
    row ids are `rmm::device_uvector<int>` (`randomforest.cuh`'s
    `RowSampler`), and `Builder` is built on that same `IdxT`. So
    `n_sampled_rows <= 2^31 - 1` in any forest cuML itself can index, and
    the sum of all cells in one histogram column equals `n_sampled_rows`.
    A single cell therefore cannot overflow an unsigned 32-bit counter.

    Their 64-bit width buys headroom their own index type never lets them
    reach. Narrowing it changes no value, only the number of bits the value
    is stored in -- which is why this is a spelling deviation and not an
    accuracy one.

    The one place this argument does NOT extend to is a weighted or
    regression bin, whose `label_sum` / `weight` are genuine float64
    accumulators with no integer bound. Those are a different problem and
    are priced separately.
    """
    return True


def column_has_float32_atomics(column: Int) -> Bool:
    """Whether `atomicAdd` on a `Float32` exists, in global memory.

    Needed only by the regression and weighted bins, whose accumulators are
    `double` upstream (`bins.cuh:101`, `:114`) and cannot be that here.

    - apple    True. MEASURED. Note this row was recorded as FALSE in this
               repository for a while and the denial was WRONG -- it came
               from an import path read as missing hardware, with the docs
               agreeing. Probe, do not trust a denial.
    - nvidia   True (global; `atomicAdd(float*)` since compute 1.1).
    - amd / amd-rdna True.
    - others   Conservatively False.

    A float atomic being PRESENT does not make it usable for a histogram
    that has to be reproducible: float addition is not associative, so an
    order-independent result needs fixed point regardless. This row records
    availability, never permission.
    """
    if column == COLUMN_APPLE:
        return True
    if column == COLUMN_NVIDIA:
        return True
    if column == COLUMN_AMD:
        return True
    if column == COLUMN_AMD_RDNA:
        return True
    return False


def struct_in_shared_memory_must_be_trivial(column: Int) -> Bool:
    """Whether a Mojo struct needs `TrivialRegisterPassable` to be stored
    in threadgroup/shared memory.

    A TOOLCHAIN row rather than a hardware one, and it is here because it
    cost a compile cycle and will cost another one for whoever ports the
    split kernel next. Mojo 1.0 refuses, by name:

        error: value of type 'Split[...]' cannot be copied or moved into a
               non-default address space
        error: non-implicitly trivially copyable value cannot be copied
               from a non-default address space

    for a struct declared merely `(Copyable, Movable)`. Declaring it
    `TrivialRegisterPassable` resolves it, and a pointer parameter that may
    receive shared memory must additionally be parameterized on its
    `AddressSpace` -- a plain `MutPointer[T, o]` will not bind one.

    THEIR CODE HAS THE SAME CONSTRAINT AND SOLVES IT THE SAME WAY, which is
    the reason this is a spelling row and not a deviation:
    `builder_kernels_impl.cuh:366-369` does not declare
    `__shared__ Split<DataT,IdxT> split_scratch[...]`. It declares a raw
    `__shared__ __align__(alignof(Split<DataT,IdxT>)) unsigned char
    split_scratch_storage[...]` and `reinterpret_cast`s it -- because a
    non-trivially-constructible type cannot be `__shared__` in CUDA either.
    Two toolchains, one constraint, two spellings of the same workaround.

    True everywhere; recorded per column so the table shape stays uniform
    and so a future column that differs has somewhere to say so.
    """
    _ = column
    return True
