# `cuda_lib` coverage, file by file

Their `catboost/cuda/cuda_lib/` is 47 headers. This is every one of them and
what happened to it. **Audited by `ls`, not by memory.**

The directory is named `gpu_lib` here, not `cuda_lib`, because nothing in it
is CUDA. That is the only rename; every CLASS keeps its CatBoost name
(`TCudaManager`, `TCudaStream`, `TGpuOneDeviceWorker`) so a reviewer can grep
their tree with the symbol they see in ours.

## Ported

| ours | theirs | status |
|---|---|---|
| `fwd.mojo` | `fwd.h` | transliterated |
| `gpu_base.mojo` | `cuda_base.h` | partial, streams only |
| `slice.mojo` | `slice.h` | transliterated |
| `device_id.mojo` | `device_id.h` | transliterated |
| `task.mojo` | `task.h` | partial+replaced, tagged union for `ICommand` |
| `tasks_queue/single_host_task_queue.mojo` | same | replaced, no thread |
| `gpu_single_worker.mojo` | `gpu_single_worker.h`/`.cpp` | partial, the dispatch switch |
| `gpu_manager.mojo` | `cuda_manager.h` | partial, single device |

## Not ported, and the reason

**Single host, so no MPI.** Their whole serialization dimension exists to
ship a command to another host. `mpi/`, `serialization/`,
`future/mpi_promise_future.h`, `tasks_queue/mpi_task_queue.h`,
`remote_objects.h`, and the `Save`/`Load` half of every command.

**Single device, so no topology or peering.** `peer_devices.h`,
`inter_device_stream_section.h`, `devices_provider.h`, `devices_list.h`,
`hwloc_wrapper.h`, `device_subtasks_helper.h`, `tasks_impl/enable_peers.h`,
`memory_copy_performance.h`, `cpu_reducers.h`.

**Owned by `DeviceContext` instead.** Mojo's context allocates and frees, so
`memory_pool/`, `memory_provider_trait.h`, `tasks_impl/memory_allocation.h`,
`tasks_impl/memory_state_func.h`, `cuda_kernel_buffer.h`, and the
`MemoryAllocation` / `MemoryDeallocation` cases, which `run` RAISES on rather
than silently accepting.

**Real gaps, wanted, not yet written.** These are not excused by the port
being single device. They are simply not done.

| theirs | what it is | why it matters here |
|---|---|---|
| `cuda_buffer.h` (693 lines) | `TCudaBuffer`, the typed device buffer | the driver still passes raw `DeviceBuffer` around |
| `mapping.h` (403 lines) | how a buffer is split across devices | degenerate at one device, but `TSlice` reads want it |
| `cuda_profiler.h` (227) | their per-kernel timer | we count launches and drains only |
| `tasks_impl/kernel_task.h` (341) | `IGpuKernelTask`, `PrepareExec` temp memory | our launches carry no temp-memory contract |
| `tasks_impl/memory_copy_tasks.h` (504) | async copies as stream tasks | our copies are still bare `enqueue_copy` |
| `stream_section_tasks_launcher.h` (138) | groups tasks into stream sections | needs real streams; Metal has none |
| `cuda_events_provider.h` (94) | events, for cross-stream ordering | same, needs real streams |
| `cache.h`, `helpers.h`, `column_aligment_helper.h`, `read_and_write_helpers.h`, `worker_state.h` | small utilities | pulled in as callers need them |

## The one that cannot be ported today

`cuda_events_provider.h` and `stream_section_tasks_launcher.h` both exist to
order work ACROSS streams. On Apple there is one stream:
`DeviceContext.stream()` raises `Metal stream not implemented`
(`device_context.mojo:2172`), measured 2026-08-19. Ordering across streams is
not a thing that can be under-delivered here, it is a thing with no referent.
Both light up on CUDA and HIP with no caller change.
