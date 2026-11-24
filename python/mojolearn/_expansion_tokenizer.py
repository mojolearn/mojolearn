# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE TOKENIZER LANE'S DOOR.

Owned by the `tokenizer` expansion lane (`host_surface.EXPANSION_LANES`,
fragment `_surface_tokenizer.py`). Every expansion lane has a door: the
saved-model route merge (`_classical_host._merge_expansion_doors`), the
release smoke and the lane selector import `_expansion_<lane>` for each lane.
The lane's public classes stay in `mojolearn.tokenizer`, so `__all__` is
empty; it carries no saved-model formats and no host binding of its own.
"""
__all__ = ()
