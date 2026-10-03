# gap-lstm: FAST Apple lstm rows vs torch MPS (lane apple-fast-gap-lstm)

Rows (BOARD_M3_FAST.md @18dc09a7a, FAST ms vs best torch ms): lstm-reg synthetic 1,878 vs 695; lstm-reg taxi-hourly
1,879 vs 700; lstm-clf taxi-hourly 1,890 vs 729; lstm-clf synthetic 1,882 vs 729. IDENTICAL is the same time (1,878-1,880).

## Route and shape
- tools/bench_board_algos.py:692-712: hidden 64, 1 layer, Adam, batch 256, 2 epochs, T = 24, D = 1. seqwin block
  (bench_board_algos.py:2001): 64 series x (0.8 x 1,440 - 24) ~ 72,200 fit windows -> 2 x 283 = ~564 optimizer steps.
- python/mojolearn/_x_sequence_rnn.py:192 `_fit` -> one `rnn_fit` binding call (bindings/_mojolearn_x_sequence.mojo:33)
  -> sequence/pyapi.mojo:69 `rnn_fit_py` -> sequence/recurrent.mojo:548 `rnn_fit` on DeviceExec.
- Everything is queued on one Metal stream; one sync at the end (recurrent.mojo `ex.sync()` after the step loop). No host
  step inside the loop except the per-step Adam scalars (recurrent.mojo `opt_scalars`, host arithmetic on launch args).

## Time breakdown hypothesis (~3.3 ms per optimizer step)
1. LAUNCH COUNT (main term). Per optimizer step ~70 launches: gather 1, input GEMM + bias 2, h/c fills 2,
   T = 24 forward `OP_CELL_FWD_H` (recurrent.mojo:334-352), head 2, loss 3, backward head 3 + fill (374),
   dh/dc fills 2, T = 24 `OP_CELL_BWD_H` (410-434), 4 weight/bias folds (446-449), opt 1. ~48 of 70 are the
   recurrence. At ~20-40 us per Metal launch this alone is ~1.4-2.5 ms/step. gru (66 launches) and rnn rows sit at
   1,630 / 1,500 ms with very different FLOPs: the floor is launches, not arithmetic.
2. SERIAL LONG FOLDS. The weight/bias gradients are one K = T x B = 6,144-term chain per output cell
   (recurrent.mojo:446-449 -> ops.mojo `op_gemm` / `op_colsum`:420): dW_ih and both bias sums run on only 256 threads
   (GH = 256), each 6,144 dependent fmas; dW_hh on 16,384 threads x 6,144. Latency-bound, estimated ~0.1-0.3 ms each.
3. RECURRENT GEMM ACCESS. `op_cell_fwd_h` (ops.mojo:554) folds h_prev[b, :] with W_hh row n read by thread u at
   stride H (uncoalesced across the simdgroup); `op_cell_bwd_h` (574) reads dGH_{s+1}[b, :] once per lane. Each lane
   re-reads the same h row / dGH row from device memory.
4. Not the cause: transfers (X 72k x 24 floats uploaded once, params/losses downloaded once), host syncs (one), predict
   (forward only, chunked).

torch MPS runs nn.LSTM as one fused MPSGraph LSTM op per direction, so its step cost is a handful of dispatches.

## Candidates (all FAST + Apple, default OFF; sequence/recurrent_scan.mojo)
- A `-D MOJOLEARN_SEQ_FAST_LSTM_SCAN`: the whole forward and the whole backward recurrence of a layer in ONE launch each,
  one threadgroup per batch row (B blocks x H lanes), `team_barrier` (device-memory ordered) between steps; the h/c and
  dh/dc zero fills folded in. 48 + 4 launches -> 2 per step. Same per-element bodies in the same order: SAME BITS.
  (recurrent_scan.mojo cell_fwd_scan_kernel / cell_bwd_scan_kernel; recurrent.mojo forward / backward SCAN branches;
  exec_device.mojo launch OP_CELL_FWD_SCAN / OP_CELL_BWD_SCAN)
- B `-D MOJOLEARN_SEQ_FAST_LSTM_SCAN -D MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM`: A, plus h_prev (fwd) and dGH_{s+1} (bwd)
  staged in threadgroup memory per step; same fold order: SAME BITS.
- C `-D MOJOLEARN_SEQ_FAST_LSTM_WGRAD`: the 4 K = 6,144 gradient folds split over K (S blocks of >= 512, ~65k
  threads, then an ordered sum of S partials; recurrent.mojo `wgrad_gemm`), bias sums as ones^T dG. Different fold
  order: FAST bits change, quality the same metric within float noise.
- D A + B + C together.
Quality column: r2/rmse (reg), accuracy/logloss (clf); A and B cannot move it (same bits), C moves it by float noise.
