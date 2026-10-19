# Pass 5, composed graphs and the step's launches (lane ag-pass5-composed, 2026-10-19)

PASS5_IDEAS ideas 4, 5 and 6, written against the integration tree at cd67d7050. Nothing here was
compiled or run by the lane; the launch counts are read from the code.

## What changed

| define (default on, `_OFF` is arm A) | file | what | bits |
|---|---|---|---|
| `MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF` | `autograd/optim.mojo` (`launch_opt_finish`, `step_lr_finish`), `sequence/mlp_fit.mojo` (`MLP_COMPOSED_OPT_FUSE`, `mlp_grad_finish`) | the composed MLP's export (OP_L2GRAD + OP_DIVS a layer) runs inside the optimizer launch, OP_AGC_OPT_FINISH (2000) | identical by construction: op_l2grad / op_divs on the word, stored, then op_opt on the stored word |
| `MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF` | `autograd/tape.mojo` | backward_from's seed is the root's gradient buffer; a once-consumed add operand (or a subtraction's left one) takes the add's gradient buffer | identical for every kernel (OP_COPY copies bits; the copy VJP's ftz is applied by every reader's ld) |
| `MOJOLEARN_AUTOGRAD_PAIR_OFF` | `autograd/tape.mojo` (`_pair_backward`), `autograd/composed_ops.mojo` | a linear or matmul node's dX, dW and fused db in one OP_AGC_LINEAR_PAIR (2001) launch when neither product takes the 16 x 16 tile | identical by construction: the same chain per word as the tile, thin and cell routes, the same epilogues |
| `MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF` | `autograd/data.mojo` (`gather_epoch`, `gather_epoch_fits`), `sequence/mlp_fit.mojo`, `autograd/model.mojo` | one gather an epoch into a permuted copy (when it fits 1 GiB), each batch a slice | identical (op_gather_rows copies the same words) |

## Launches a batch (HIGGS MLP, D 28, (256,), O 1, b512, adam; read from the code)

| arm | before | after |
|---|---|---|
| fused (default) | 9: gather, Z1, Z2, lb_part, lb_fold, dW2 + db2, dH, dW1 + db1, opt | 7: Z1, Z2, lb_part, lb_fold, dH + dW2 + db2 (paired), dW1 + db1, opt; + 1 gather an epoch |
| composed (`MOJOLEARN_AUTOGRAD_MLP_UNFUSED`) | 25: gather 1, forward 5, loss 2, seed copy 1, backward 11, export 4, opt 1 | 15: forward 5, loss 2, backward 7 (bias folds 2 + 2, paired dX + dW 1, activation VJP 1, dW1 1), opt with the finish 1; + 1 gather an epoch |
| hand (`MOJOLEARN_AUTOGRAD_MLP_OFF`) | 11 | 10 (the gather) |
| Sequential `model_fit` | per step | the gather a batch; one copy VJP per same-shape residual add; dX + dW paired on non-tiled linear layers |

At the profiles' ~3 us (AMD) / ~2.5 us (NVIDIA) a small launch, PASS5_IDEAS expects the composed arm
about 0.89 of today on the average and the fused arm about 0.96 (pair 0.97, gather 0.98).

## Idea 4: the pairing kernel and the road to the 3-launch step

OP_AGC_LINEAR_PAIR runs three index ranges in one launch: [0, rows fi) dX cells, then fi fo dW
cells, then fo bias columns. Each range builds its own Args and runs the body its separate launch
runs one thread a cell (op_gemm, op_agk_gemm_epi with EPI_ACT_BWD or EPI_L2GRAD, op_colsum_div).
The route rule pairs only products that do not take the 16 x 16 tile (`seq_gemm_route`): those are
the thin-edge and short-K products, where every word is one thread's chain with no operand reuse,
so moving them onto cells costs at most the thin tile's staging, while the launch floor is saved.
A few-cell, long-K weight product (the coop GEMM's shape, at most 1024 cells with K >= 32768) is
left alone because one thread a cell would serialize its K.

lb_fold is not in the pair. It is the trainer's loss-block launch (sequence/mlp.mojo, owned by the
loss lane this pass, which also rewrites it under idea 2) and runs before the tape's backward; the
tape cannot see it. The pairing kernel is the "weight program" of idea 8 in its first form. The
remaining steps toward the 3-launch step are:

1. A ready set in `_backward`: after the root's dZ is known, every edge whose inputs are written
   (here dH, dW2, db2) can share a launch; the pair is that set for one linear node.
2. The step program: lb_fold's one thread and OP_AGC_OPT_FINISH's np threads in one launch
   (independent inputs), once the loss block hands its fold arguments to the trainer's step.
3. The row program (idea 8, phase 1), after ideas 1 and 2 land.

## Invariants of the borrowed gradients

- A node borrows only when it is a recorded, tape-owned intermediate with one consumer, not a leaf,
  not linear (its backward applies act' to its own gradient in place), not custom and not in an
  alias group. The lender is an add or subtraction whose own gradient is not caller storage, or the
  seed. Nothing writes a borrowed buffer during the backward that borrows it.
- Every node records its record-time buffer (`_Node.own_grad`). Each backward first returns every
  node to it, the scalar seed fill writes it, and the overlap refusals check it, so a later backward
  never writes a caller's old seed and a seed that overlaps a node's own buffer is still refused.
- `gradient()` of a borrowing node returns the buffer it borrowed (the root's is the seed).

## Checks the orchestrator builds

The autograd suites (cpu, gpu, mlp x3 arms, fused, pass1, order, data, optim) and the sequence
bindings in five builds: default, and each of `MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF`,
`MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF`, `MOJOLEARN_AUTOGRAD_PAIR_OFF`,
`MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF`; plus the Sequential tests. Digests must equal the OFF arms
and today's: hand 2c2662ecb00cceed, composed aee3b17ef66dc060. `MOJOLEARN_AG_PROFILE=1` shows the
counts above (no OP_COPY, OP_L2GRAD, OP_DIVS or per-batch OP_GATHER_ROWS rows on the composed arm).
