# Parallel training native input admission — 2026-10-20

Internal implementation and qualification record; not a public completion or
performance claim.

`ParallelNeuralTrainer` already delegates construction, training, checkpointing
and cleanup to one native call, including native child-process dispatch. The
remaining boundary flaw was in that native code: `list(shards)`, `tuple(pair)`,
device iteration and `isinstance(model, ...)` could invoke caller Python hooks.
Checkpoint restore also looked up fields before validating the mapping type.

A read-only probe against the immutable stage51 CPU package confirmed that
`ParallelNeuralTrainer.train_step` invoked a supplied `__iter__`. The whole-call
audit recorded that Python frame and the deliberate exception; no workers had
started. This is a product-boundary failure, distinct from the upstream lifetime
blocker. Evidence is retained at
`~/mojolearn-evidence/finish-graph-20261020/parallel-neural-hook-probe.{py,json,log}`.

The repair reuses the native Samba type/layout admission already used by its
public runtime. Import-time records add the exact parallel owner and fixed MLP
class, their native dictionaries, stored metadata and process sockets. Product
operations validate those records before snapshots, locks, process requests or
checkpoint reconstruction. Training accepts native list/tuple shard envelopes;
devices retain list/tuple/range support. Existing native buffers, fixed MLP and
Samba configurations, schedules, ordered sums and checkpoint contents remain.
Python iterator, custom model, conversion, mapping and buffer protocols are
refused before invocation, as required by the owner's native-only instruction.
Cleanup validates only resources it reads, so a rejected model configuration
does not prevent closing native workers.

The dedicated public-operation tests include constructor and subclass controls,
five input locations, four mutated model metadata controls, three checkpoint
restore controls, and a real native training/replay test with independent MLP
calculus. The existing whole-child-loop, Samba dropout/schedule, rollback and
checkpoint suites remain required. Source checks are not numerical qualification;
the new source awaits the parent's frozen tokenizer compile and runtime tests.

## Adjacent scope still pending

- `Sequential.fit` uses the native Sequential/model-fit ABI; it is not the
  generic distributed captured-program training path. Sequential has no public
  `distribute`/`partition` method. Those operations belong to `ops.Program`.
- `Program.distribute` and `Program.partition` and their returned `fit` methods
  already invoke native owners. Their direct wrapper constructors still assign
  two fields in Python, and their native option/device boundaries need the same
  precise protocol audit. This repair does not qualify those unrelated paths.
- Physical two-GPU execution, installed-vendor evidence and numerical identity
  require their own retained witnesses. One CPU or Metal device does not provide
  that evidence.
- Confirmed Mojo std.python reference retention remains an upstream blocker.
  No compensation, toolchain patch or blanket runtime exemption is introduced.
