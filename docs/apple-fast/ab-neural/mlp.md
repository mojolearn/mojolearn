# afn-mlp A/B requests (Apple FAST, neural): SmallMLPTrainer, Embedding, CNN

Branch `lane/apple-fast-neural-mlp`; profile and design in `docs/apple-fast/notes/neural-mlp.md`.
Every arm is FAST vs FAST (arm A: no define, arm B: the define), same binding, same batches;
the quality judge is `tools/neural_fast_quality.py` (`mlp` for the trainer). Tags are
`afn-mlp-<define>`; the lines are in `mlp.txt`. The last two lines use afn_ab.sh's `custom`
board-lane slot; if the tier lane provides none, the Python calls to time are given below.

## afn-mlp-fused: `-D MOJOLEARN_AFN_MLP_FUSED_STEP` (binding training, lane mlp-train-step)

Mechanism: the whole step (forward, mean cross-entropy, backward, AdamW) as TWO launches and ONE
wait instead of ~20 launches and 7 waits. One thread per row (64 rows per block, <= 4 blocks),
the 195 weights in threadgroup memory, the block's gradient partials folded in threadgroup memory
into one slot; the second launch sums the slots and updates the parameters. Two allocations per
step (one arena with views, one int32 buffer). Expected effect: the step time drops to transport
(8 uploads, 11 small downloads, one wait); on the M3 the stage times should fall from the
several-hundred-microsecond class to a few launches. Risk: the fold order differs from main
(per-block serial over rows, then over blocks); f32 everywhere so the loss curves agree to
reassociation noise. Touches: mlp-train-step only (modes forward/grads/train all take it).

## afn-mlp-resident: `-D MOJOLEARN_AFN_MLP_RESIDENT` (binding training, lane mlp-train-step)

Mechanism: the fused kernels on a device-resident session: weights, gradients, moments, a
shadow of the state, the loss cells, the block slots and the batch staging in ONE arena for the
trainer's life; a step uploads x and y, runs 3 launches (shadow copy, fused, fold+AdamW), downloads
loss/logits/gradients, waits once; no per-step allocation, the weights never travel per step.
`state_dict()` downloads when asked. A failed step restores the shadow on the device. Expected
effect vs afn-mlp-fused: minus 2 allocations, minus 8 copies (w1, b1, w2, b2, m, v up and down)
per step. Risk: Python keeps the session open across steps (closed on `load_state_dict`,
`apply_gradients`, deletion); the public results are the same arrays. Touches: mlp-train-step.

## afn-mlp-multistep: `-D MOJOLEARN_AFN_MLP_MULTISTEP` (binding training, lane mlp-train-step)

Mechanism: implies RESIDENT and adds `SmallMLPTrainer.train_steps(X, y, k)`: k minibatches
uploaded once, sliced on the device, 2k launches in one command stream, one wait per k steps.
The board lane calls `train_step` per round, so on mlp-train-step this arm measures as RESIDENT
does; the manager times the k-step call directly:

    MOJOLEARN_NUMERIC_MODE=fast python3 -c "
    import numpy as np, time, mojolearn
    rng = np.random.default_rng(0)
    w = [rng.uniform(-.35,.35,(16,8)).astype('f4'), rng.uniform(-.35,.35,16).astype('f4'),
         rng.uniform(-.25,.25,(3,16)).astype('f4'), rng.uniform(-.25,.25,3).astype('f4')]
    m = mojolearn.SmallMLPTrainer(*w, data_schedule={'ab': 'multistep'})
    k, b = 32, 256
    X = rng.standard_normal((k*b, 8)).astype('f4'); y = rng.integers(0, 3, k*b).astype('i4')
    m.train_steps(X, y, k)                              # warm
    t = time.perf_counter(); r = m.train_steps(X, y, k); dt = time.perf_counter() - t
    print('multistep k=%d: %.3f ms/step' % (k, 1e3*dt/k))
    t = time.perf_counter()
    for i in range(k): m.train_step(X[i*b:(i+1)*b], y[i*b:(i+1)*b])
    print('train_step:      %.3f ms/step' % (1e3*(time.perf_counter()-t)/k))"

Expected: the per-step cost falls to the launch pair's device time plus 1/k of a wait. Risk:
the last step's logits and gradients only are returned (documented). Touches: mlp-train-step.

## afn-mlp-all: `-D MOJOLEARN_AFN_MLP_ALL` (binding training, lane mlp-train-step)

The three MLP defines together (on the board lane this equals RESIDENT + the k-step API).

## afn-mlp-emb-atomic: `-D MOJOLEARN_AFN_EMB_ATOMIC_BWD` (binding embedding, no board lane)

Mechanism: `Embedding.backward` as seed + ONE relaxed f32 atomic scatter-add launch + the padding
row instead of counts, run-begin (a one-block scan), permutation and the per-cell run walk; the
device id re-check (a readback, two waits) and the three zero int32 uploads go; uploads stop
waiting. Forward: no-wait uploads and a float4 gather when d % 4 == 0. ~10 waits -> 2 per backward,
5-6 -> 2 per forward. Risk: the dW fold order is free (atomics), so dW differs from main by f32
reassociation; padding rows are +0.0 as the contract states. The Python call to time (V = 50257,
d = 256, T = 65536, torch-shaped):

    MOJOLEARN_NUMERIC_MODE=fast python3 -c "
    import numpy as np, time, mojolearn
    rng = np.random.default_rng(0); V, d, T = 50257, 256, 65536
    e = mojolearn.Embedding.from_pretrained(rng.standard_normal((V, d)).astype('f4'), padding_idx=0)
    ids = rng.integers(0, V, T).astype('i4'); dy = rng.standard_normal((T, d)).astype('f4')
    e.forward(ids); e.backward(ids, dy)
    t = time.perf_counter(); y = e.forward(ids); print('forward  %.2f ms' % (1e3*(time.perf_counter()-t)))
    t = time.perf_counter(); g = e.backward(ids, dy); print('backward %.2f ms' % (1e3*(time.perf_counter()-t)))
    print('dw checksum', float(np.abs(np.asarray(g)).sum()))"

## afn-mlp-cnn-direct: `-D MOJOLEARN_AFN_CNN_DIRECT` (binding x_cnn, no board lane)

Mechanism: for every convolution whose k = C*KH*KW exceeds the one-leaf direct kernel's bound
(every layer but the first on an RGB image), one implicit-GEMM launch (64 positions x 32
channels per block, 32-tap chunks staged in threadgroup memory, f32 registers, bias epilogue,
ReLU fused into the no-pool conv block forward) replaces im2col + GEMM + the NCHW layout launch
(3-5 launches and a rows x k round trip through device memory). Risk: the fold is per-chunk
(free order); the backward is unchanged. The Python call to time (a second-layer shape, N 64,
32 -> 64 channels, 3x3 on 16x16):

    MOJOLEARN_NUMERIC_MODE=fast python3 -c "
    import numpy as np, time, mojolearn
    rng = np.random.default_rng(0)
    c = mojolearn.Conv2d(32, 64, 3, padding=1)
    x = rng.standard_normal((64, 32, 16, 16)).astype('f4')
    c.forward(x)
    t = time.perf_counter(); y = c.forward(x); print('conv2d forward %.2f ms' % (1e3*(time.perf_counter()-t)))
    print('y checksum', float(np.abs(y).sum()))"

and `CNNClassifier` on the digits fixture the lane checks use (`tools/neural_fast_quality.py` has
no cnn subcommand; compare the held-out accuracy of a FAST fit with and without the define).
