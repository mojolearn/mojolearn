# Mojolearn GP Array label capture

`gp_encode_labels_into` now calls `gp_label_pins`. For exact Mojolearn Array storage, the existing single metadata snapshot feeds native scalar pins directly, without first constructing an outer Python list and per-row Python lists. The public compatibility `gp_label_list` remains for live callers and uses the same capture body with a public-output sink. This is one removed private Array snapshot-container role, not one role per row.

The capture uses the same dtype decoder, row/column order and byte strides. It does not route integer labels through float or alter the existing numeric fast paths. The former rank-zero Array sequence refusal is preserved by the same flatten admission on its boxed scalar. Snapshot storage/owner pins are retained in a finally block through all raw-address reads. No kernel, encoder, provider dispatch, numerical copy or default changes.

Three authored, unrun public cases cover C, Fortran and reversed matrix label storage, checking class ordering and published binary target codes against the original logical input. No product import/build/test/benchmark was run by this lane. Build roots: GP family/parallel/trees consumers of the snapshot helper; standard source closure planning applies.

General native ndarray, pandas and other buffer source conversion remains pending: the shared flatten source can still call the existing `_label_list` conversion. The shared score truth path also remains under review. This producer does not claim full family completion.
