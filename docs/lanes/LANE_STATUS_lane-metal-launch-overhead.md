# lane/metal-launch-overhead

**WHAT THIS LANE FOUND.** On the shared M4, a `ctx.synchronize()` that has
pending work costs about **4.1 ms**, and an `enqueue_function` costs about
**0.30 ms**. One host round trip is worth roughly a dozen kernel launches.
Every prior attempt on this problem counted launches. The launch census was
the smaller half of the bill.

The counting mistake is visible in committed evidence that predates this
lane. `bench/results/e1g/2026-09-11_190725-nvidia-h100-80gb-hbm3-step-breakdown/remote/step-breakdown/`
records, for ONE byte LM step, `count.launches` 1324 and
`count.synchronizes` **2021**. There are more waits in a step than there are
kernels, and on Metal a wait is a full submit and completion round trip.

This lane measured the cost model, took the census from source, removed 18
copy launches and 11 waits per step from the shipped byte LM step at one
block, and left the arithmetic untouched.

## 1. The measured cost model

`training/checks/metal_launch_probe.mojo`, `PROBE_N=1000`, three repeats,
alone under the Metal lock on 2026-09-16 at 08:22 ET with the box quiet
(load average 3.95). Microseconds per operation, minimum and median over the
three repeats.

| arm | what one operation is | min us | median us |
| --- | --- | --- | --- |
| `launch_only` | enqueue a 1-thread kernel; ONE synchronize after all 1000 | **295.0** | 335.5 |
| `launch_4096` | enqueue a 4096-thread kernel; ONE synchronize after all 1000 | 784.1 | 836.8 |
| `launch_sync` | enqueue a 1-thread kernel, then synchronize | **4440.9** | 4547.4 |
| `d2h_only` | copy 4 bytes to the host, then synchronize | 4515.1 | 4564.4 |
| `launch_d2h_sync` | enqueue, copy 4 bytes to the host, then synchronize | 5253.1 | 5283.3 |
| `sync_only` | synchronize an ALREADY EMPTY queue | **12.7** | 12.7 |

Four things follow, and the third is the one that matters.

1. An enqueue costs about 0.30 ms and does not depend on the grid. The gap
   between `launch_only` and `launch_4096` is the GPU actually running 32
   blocks instead of one, because both arms drain at the end. Making a
   kernel smaller does not make its launch cheaper. This is why shrinking
   models moved the Apple column only fifteen minutes.
2. `sync_only` on an empty queue costs 12.7 us. The price is not the call.
3. **The price is the submit and wait.** `launch_sync` minus `launch_only`
   is 4.15 ms, and `d2h_only` shows the same 4.5 ms for a four byte download
   with no kernel at all. Anything that makes the host wait on a non-empty
   Metal queue costs about 4 ms, whatever is on it.
4. A four byte download costs the same as a bare wait, so `download_f32` of
   a single loss is a round trip and not a transfer.

**Control for the slot environment.** `mac_slot.sh` exports
`MODULAR_THREAD_BUSY_WAIT_US=0`, which could plausibly have turned a spin
into a scheduler round trip and invented the whole effect. The same probe
run with that variable unset gives `launch_only` 278.9 us, `launch_sync`
4423.8 us, `sync_only` 11.5 us and `d2h_only` 4269.0 us. The busy wait
setting is not the cause.

### 1.1 How much a removed wait is worth

The `batch_k` arm enqueues K kernels per synchronize and varies K. Run at
08:29 ET with the box contended (load average 10 to 15), so read the SHAPE
and not the level. Minimum over three repeats.

| K launches per synchronize | us per launch | wall time for ~1000 launches |
| --- | --- | --- |
| 1 | 4321.5 | 4.32 s |
| 2 | 2792.0 | 2.79 s |
| 4 | 1942.0 | 1.94 s |
| 8 | 1134.9 | 1.13 s |
| 16 | 772.4 | 0.77 s |
| 32 | 511.7 | 0.51 s |

Per launch cost falls 8.4x from K=1 to K=32 and flattens onto the bare
enqueue cost. The wait is a fixed price per submission, so **every wait
removed is worth about one whole round trip** and every launch removed is
worth about a twelfth of one.

## 2. The census of one byte LM step

Line numbers are `training/byte_lm.mojo` as it stood at `596ddb6e9`, before
this lane. `L` is the number of blocks. Nothing in this table is behind
`MOJOLEARN_STEP_PHASE_TIMERS` or `MOJOLEARN_TRANSFORMER_TIMING`; the
`StepPhaseClock.tick` and `timing_tick` calls interleaved with these lines
compile to nothing and wait for nothing unless a timer is switched on, which
is checked in `transformer/impl/llama/modeling_llama.mojo:2740`.

### 2.1 Waits, all on the SHIPPED path

| line | after what | per step | kept, and why |
| --- | --- | --- | --- |
| 861 | two `enqueue_create_host_buffer` | 1 | KEPT. The host stores into `hi` and `ht` on the next line. |
| 870 | two host to device copies | 1 | KEPT. `hi` and `ht` are released right after. |
| 548 | `_unpack_block`, per block | L | REMOVED |
| 881 | the unpack loop and two more copies | 1 | REMOVED |
| 889 | `identical_embedding_forward_into` | 1 | REMOVED |
| 908 | one block forward, per block | L | KEPT. `stages` is popped and reinserted around the call. |
| 920 | the head GEMM | 1 | REMOVED |
| 934 | `identical_ce_forward_into` | 1 | REMOVED |
| 970 | `identical_ce_backward_into` | 1 | REMOVED |
| 986 | the head backward GEMMs | 1 | REMOVED |
| 1014 | one block backward, per block | L | KEPT. Same pop and reinsert. |
| 1023 | `identical_embedding_backward_into` | 1 | REMOVED |
| 564 | `_pack_block`, per block | L | REMOVED |
| 1030 | the pack loop and two more copies | 1 | REMOVED |
| 1077 | the three shadow copies | 1 | REMOVED |

Explicit waits in this file, `3L + 12` before and `L + 1` after, on top of
whatever the composed operations wait for inside themselves. The composed
operations are where the other roughly two thousand waits live, and this
lane did not touch them.

### 2.2 Host round trips that are not waits

| line | what | shipped |
| --- | --- | --- |
| 938 | `download_f32(ce_loss, 1)` | YES. Four bytes, and by section 1 it costs a full round trip. |
| 857, 859 | `enqueue_create_host_buffer` for ids and targets | YES, once per step |
| 866, 868 | two host to device copies of the batch | YES, once per step |
| 700 to 724, 1347 to 1349 | `download_f32` of param, m, v and grad | Stateless path and export only, NOT the resident step |
| `bindings/_mojolearn_byte_lm.mojo:425` to `:427`, `:460` to `:462` | six `download_f32` of the full state | Stateless path only. The resident session keeps the state on the device. |

### 2.3 Launches, the copy half

`_unpack_block` (`training/byte_lm.mojo:535`) issued NINE `_copy_into`
launches per block and `_pack_block` (`:551`) another nine, so `18L` launches
and `2L` waits per step moved nothing but bytes.

## 3. What changed

**The nine became one.** `train_copy_nine_kernel` and `_copy_nine`
(`training/checks/train_loop.mojo`) do one block's nine ranges in a single
launch. The nine ranges of a block are CONSECUTIVE in the flat buffer, so
the concatenated thread index `i` maps to `base_off + i` with no offset
table; the kernel takes the nine element counts and walks them to find which
tensor owns `i`. `_block_counts` derives those counts from consecutive
offsets in one place, so their sum telescopes to the length of the slab and
the counts cannot drift apart from the layout.

The fused kernel moves floats with `unsafe_load` and `unsafe_store` and
contains no arithmetic, exactly like the `train_copy_range_kernel` it
replaces. A copy is a byte move. It cannot move a bit.

**Eleven waits went.** The two inside `_unpack_block` and `_pack_block`, and
the nine in section 2.1 marked REMOVED. Every one of them was followed by
more work queued on the SAME in-order context with no host read in between,
so the wait delayed the host and enforced nothing. Each removal carries its
reason at the site. `training/checks/byte_lm_layer_pool_check.mojo` was the
one caller that read the device straight after `_pack_block` without a wait
of its own, and it now carries that wait at its own call site, so its
behavior is unchanged.

Static call sites in `training/byte_lm.mojo`, counted by grep.

| | before | after |
| --- | --- | --- |
| `ctx.synchronize()` | 23 | 12 |
| `_copy_into(` | 31 | 13 |

Per step at `L` blocks.

| | before | after | saved |
| --- | --- | --- | --- |
| block copy launches | 18L | 2L | 16L |
| waits in this file | 3L + 12 | L + 1 | 2L + 11 |

At `L = 1`, the shape the `byte-lm-resident` identity lane runs, that is 16
launches and 11 waits per step. Priced at section 1, about
`16 * 0.30 + 11 * 4.15 = 50 ms` per step, of which the waits are 91 percent.

**`bindings/build_byte_lm.sh`** hardcoded `-j 2` and `mojo build` refuses a
repeated `-j`, so a shared box could not ask for one compiler worker. It now
reads `MOJOLEARN_COMPILE_JOBS` and still defaults to 2.

**No arithmetic kernel was fused.** Contracting two float operations into an
FMA moves bits. Section 7 lists those as candidates with what they would
save, and they are not this lane's to take.

## 4. Identity evidence

Two bindings of the SAME commit `e1a507c55`, built minutes apart from the
same tree, differing only in this lane's change.

| binding | sha256, first 16 | what it carries |
| --- | --- | --- |
| BEFORE | `5f8c3def3aeb081b` | nine `_copy_into` per block, every wait |
| AFTER | `f680900317a1c865` | one fused copy per block, eleven waits gone |

Everything else was held constant. The base binding and the other seventeen
`.so` files came from one prebuilt identical set and were not rebuilt between
the arms, so the only thing that moved is `_mojolearn_byte_lm.so`.

Both arms ran `tools/identity_break.py --lanes byte-lm,byte-lm-resident
--no-batch`, which is 2 lanes by 9 hostile fixtures, each cell fitted TWICE
in one process.

```
BEFORE   cells=18 stable=18 moved=0 refused=0
AFTER    cells=18 stable=18 moved=0 refused=0
--diff   summary (infer/model): IDENTICAL=36
         summary (rlpair):      IDENTICAL=18
```

**Zero divergent cells and zero moved cells.** Every train, infer, model and
rlpair hash is the same in both arms, on all nine fixtures including
`denormal`, `denormal_ftz` and `ties`.

FILL_SABOTAGE

**The CPU host route was not touched and could not have been.** The host
byte LM lives in `training/byte_lm_host*.mojo` and builds into
`_mojolearn_byte_lm_host.so` through `bindings/build_byte_lm_host.sh`. This
lane's diff touches `training/byte_lm.mojo`,
`training/checks/train_loop.mojo`,
`training/checks/byte_lm_layer_pool_check.mojo` and
`bindings/build_byte_lm.sh`, none of which that binding compiles, so the
host route's bytes cannot have moved.


## 5. Timings

`byte-lm-resident` on the `base` fixture, alone under the Metal lock, same
two bindings, wall clock seconds for the whole lane.

| arm | rep 1 | rep 2 | min |
| --- | --- | --- | --- |
| BEFORE | 264.66 | 59.50 | **59.50** |
| AFTER | 38.97 | 40.10 | **38.97** |

BEFORE rep 1 at 264.66 s is an outlier against its own rep 2 at 59.50 s, so
it is reported and not used. Read the minima, 59.50 s against 38.97 s.

**What this measurement does NOT say.** Both full eighteen cell identity
arms took about the same wall clock, roughly 37 to 39 minutes each. That is
not a contradiction. Those arms are mostly the `byte-lm` stateless lane and
the infer, model and rlpair columns, which download the whole parameter
vector per step and run forward passes this lane did not touch. The isolated
`byte-lm-resident` number above is the honest one for the shipped resident
step.

**The strongest single number on this page is not a ratio.** The BEFORE
identity arm reported `58.06s user 76.83s system 5% cpu 38:39.26 total`. A
byte LM identity arm on Metal spends about **95 percent of its wall clock
blocked**, on a box whose load average was 1.34. It is not computing. It is
waiting for round trips.


## 6. The runtime offers no batching primitive

Checked against the `DeviceContext` reference at
`https://max.modular.com/api/mojo/max/gpu/host/device_context/DeviceContext`.
Its whole method set is `__init__`, `__deinit__`, `__eq__`, `__ne__`,
`enqueue`, `__enter__`, `name`, `api`, `enqueue_create_buffer`,
`create_buffer_sync`, `enqueue_create_host_buffer`, `compile_function`,
`load_function`, `enqueue_function`, `enqueue_cpu_function`,
`enqueue_cpu_range`, `execution_time`, `push_context`, `set_as_current`,
`execution_time_iter`, `enqueue_copy`, `enqueue_copy_no_cross_stream_sync`,
`enqueue_memset`, `create_event`, `stream_priority_range`, `create_stream`,
`create_external_stream`, `synchronize`, `enqueue_wait_for`, `num_streams`,
`select_stream`, `get_api_version`, `get_attribute`, `is_compatible`,
`run_healthcheck`, `id`, `get_memory_info`, `max_single_alloc_size`,
`can_access`, `enable_peer_access`, `supports_multicast`,
`is_host_unified`, `number_of_devices`, `enable_all_peer_access` and
`all_peer_access_enabled`.

**There is no batch or group submit, no command buffer record and replay, no
graph capture, and no completion callback.** So the only levers are FEWER
LAUNCHES and FEWER WAITS. There is nothing to configure.

Two entries are worth naming because they look like levers and are not.
`create_stream` and `select_stream` give more queues, but a second queue does
not reduce the number of submissions or round trips on an in-order dependency
chain, and on Apple the Metal queue budget is per process and already the
scarce resource. `create_event` with `enqueue_wait_for` moves a wait from the
host to the device, which is the right shape for a cross stream dependency
and does nothing for a single in-order context that never needed the host
wait in the first place.

## 7. Remaining levers, ranked

Priced at 4.15 ms per wait and 0.30 ms per launch, per step at `L` blocks.

1. **The waits inside the composed operations, roughly 2000 of them.** The
   H100 census says a step waits 2021 times and this lane removed 11. The
   remaining waits are inside `identical_ce_forward_into` (thirteen stages),
   `identical_optimizer_step`, the GEMMs, the embedding operations and
   `llama_decoder_layer_forward` and `_backward`. Each is the same
   transformation this lane applied, a wait that a later enqueue on the same
   in-order context already orders. Expected saving is the bulk of the
   remaining cost and it is by far the largest number on this page. It needs
   one lane per composed operation, because each has to be read for a host
   read between the wait and the next enqueue.
2. **Views instead of copies, which removes the copies entirely.** `_copy_nine`
   still moves every parameter byte twice per step for nothing.
   `unpack_params` in `training/checks/train_loop.mojo:1314` says it plainly,
   that the per tensor buffers are A VIEW of the flat buffer refreshed at the
   top of every step. `DeviceBuffer.create_sub_buffer` exists and the
   repository already uses it in
   `ensemble/decisiontree/batched_levelalgo/builder.mojo:1277` and
   `metrics/checks/device_io.mojo:43`. If `LlamaDeviceWeights` held
   sub buffers of `tb.param` and the backward stages wrote into sub buffers
   of `tb.grad`, unpack and pack both become nothing at all. Saves the last
   `2L` launches plus `2L * n_total * 4` bytes of device traffic per step,
   and removes the wrong offset defect class by construction. This is the
   change with the best ratio of saving to risk after item 1, and it moves no
   arithmetic.
3. **`download_f32(ce_loss, 1)` at `training/byte_lm.mojo:938`.** One four
   byte read costs a whole round trip, about 4.5 ms, and it happens every
   step to return a loss the caller usually only prints. Deferring it, or
   returning the loss for step `t` at step `t+1`, saves one round trip per
   step. It changes an interface, not a number.
4. **The two `enqueue_create_host_buffer` per step at `:857` and `:859`,
   with the wait at `:861`.** The batch shape is fixed for a session, so
   these two host buffers can be allocated once in `ByteBuffers` and reused.
   Saves one wait and two allocations per step.
5. **Arithmetic kernel fusion, NOT TAKEN HERE.** The thirteen stage cross
   entropy forward, the elementwise chain inside the decoder block and the
   optimizer's scalar passes are all several launches that one kernel could
   do. Fusing them lets the compiler contract a multiply and an add into an
   FMA, which moves bits, so each one needs its own identity gate and its own
   pinned arithmetic. Roughly 30 to 60 launches per block, so about
   `0.30 * 45L` ms, which is worth far less than item 1 and costs far more to
   prove.
