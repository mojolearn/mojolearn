#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DEVIATION 2495: complete LM training steps with per-step time, memory and bit witnesses.

Runs a few complete IDENTICAL training steps of the generalized decoder LM
through the PUBLIC trainer API only (mojolearn.LanguageModelTrainer with
resident sessions) and records, per step:

  * wall seconds around the public train_step call. The native binding
    synchronizes the device context before it publishes outputs
    (bindings/_mojolearn_byte_lm.mojo, `ctx.synchronize()` before the
    `_write_f32` publication), so the timing boundary already includes
    completion of every enqueued kernel and every host readback. No extra
    device synchronization can be issued from Python without new native
    code; that is recorded, not hidden.
  * tokens per second (batch * length / seconds).
  * peak device memory across the step, sampled by the vendor tool that
    answers on the box (nvidia-smi per-process and device-wide, else
    rocm-smi device-wide, else "unavailable"). Peak device memory from
    INSIDE the process would need new native code; it is recorded as
    unavailable rather than invented.
  * process RSS (VmRSS after the step, VmHWM peak, ru_maxrss) SEPARATELY
    from device memory, each labeled by boundary.
  * sha256 of the loss bits, the flat pre-update gradients, and the updated
    parameters, moments and flags, so a before/after change can be compared
    bit for bit.
  * with --resident-lean (DEVIATION 2514, design section 2.1), the trainer
    is constructed with resident=True, step_result='lean': train_step
    returns loss, step, completed_steps, next_batch_index and flags, and
    the gradient stays on the device. The witnesses then come from
    export_gradients(named=False) and export_state(), taken ONCE after the
    last untimed step (event `final_witness`, result.json `final_witness`)
    or, under --witness-every-step, after every step exactly where the
    'full' run hashes its result and state_dict(). Both exports are outside
    the timed boundary, which stays the public train_step call. The hashes
    are comparable with a 'full' run's per-step `step_end.sha256` records
    (events.jsonl) and `step_witnesses` (result.json) on the same GPU.
  * with --component-timing, one extra step under
    MOJOLEARN_TRANSFORMER_TIMING=1 (DEVIATION 2499): every `timing <name>
    <value> <unit>` line the native step and the Python wrapper print is
    summed by name into result.json as `component_timing_ms` (unit ms) and
    `component_bytes` (unit bytes), with `component_timing_total_ms`, the
    step's own wall `step_seconds`, and the covered fraction, so the
    itemized share of the step is explicit. Envelope lines (`envelope.*`)
    and the `attn.*` sub-phases of `block.attention_total` are kept in the
    dict but out of the total (they would double count).

The configuration is an argument; the default is the 20.45M control shape
(B1 L2048 DM384 H6 KV6 HD64 FF1024, 8 layers, V8192). If setup plus the
first step, or any later step, exceeds --budget-seconds the harness exits 2
and records the limitation. It NEVER substitutes a smaller model.

Synthetic token batches from a seeded generator by default (or a pinned
byte corpus with --corpus, DEVIATIONS 2525 to 2527): a few complete
steps at the named shape are the point, not learning. Nothing here is an
opponent measurement, a default gate, or target-scale qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

CONTROL_SHAPE = [1, 2048, 384, 6, 6, 64, 1024, 8, 8192]
TARGET_SHAPE = [1, 2048, 768, 12, 12, 64, 2048, 12, 50257]
SCHEMA = 'mojolearn.lm-step-memory-probe.v3'
WITNESS_SOURCE = {
    'full': "train_step result flat_gradients + state_dict() after every step",
    'lean': "export_gradients(named=False) + export_state() after every step (outside the timed boundary)",
    'lean-final': "export_gradients(named=False) + export_state() once after the last untimed step",
}
EXIT_LIMIT = 2
# Names summed into component_timing_total_ms exclude these: `envelope.*`
# lines wrap other itemized lines, and `attn.*` are the sub-phases of
# `block.attention_total` (modeling_llama.mojo prints both from one block).
TIMING_TOTAL_EXCLUDED_PREFIXES = ('envelope.', 'attn.')


def _proc_status():
    """VmRSS / VmHWM in bytes from /proc (Linux); None elsewhere."""
    try:
        text = Path('/proc/self/status').read_text()
    except OSError:
        return None
    out = {}
    for line in text.splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value = line.split(':', 1)
            out[key] = int(value.strip().split()[0]) * 1024
    return out or None


def _rss_bytes():
    status = _proc_status()
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss *= 1 if sys.platform == 'darwin' else 1024
    return dict(vm_rss_bytes=status.get('VmRSS') if status else None,
                vm_hwm_bytes=status.get('VmHWM') if status else None,
                ru_maxrss_bytes=maxrss,
                boundary='process host memory (Python + native + pinned staging); not device memory')


class DeviceMemorySampler:
    """Polls the vendor tool on a thread; reports the window maximum.

    nvidia-smi: per-process used memory (`--query-compute-apps`) for THIS pid
    and device-wide `memory.used` for index --gpu-index. rocm-smi: device-wide
    VRAM used only. Anything else: unavailable. Sampling cannot see a peak
    shorter than the interval; the record says so.
    """

    def __init__(self, interval, gpu_index):
        self.interval = interval
        self.gpu_index = gpu_index
        self.pid = os.getpid()
        self.tool = None
        if shutil.which('nvidia-smi'):
            self.tool = 'nvidia-smi'
        elif shutil.which('rocm-smi'):
            self.tool = 'rocm-smi'
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reset()
        self._thread = None
        self.errors = 0
        self.samples = 0

    def _reset(self):
        self.peak_process = None
        self.peak_device = None
        self.device_total = None

    def _read(self):
        if self.tool == 'nvidia-smi':
            dev = subprocess.run(
                ['nvidia-smi', '-i', str(self.gpu_index), '--query-gpu=memory.used,memory.total',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            used, total = [int(x) * 1024 * 1024 for x in dev.stdout.strip().split(',')[:2]]
            apps = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            mine = None
            for line in apps.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == self.pid and parts[1].isdigit():
                    mine = int(parts[1]) * 1024 * 1024
            return mine, used, total
        if self.tool == 'rocm-smi':
            out = subprocess.run(['rocm-smi', '--showmeminfo', 'vram', '--csv'],
                                 capture_output=True, text=True, timeout=5)
            rows = [r for r in out.stdout.strip().splitlines() if r and not r.lower().startswith('device')]
            row = rows[self.gpu_index] if self.gpu_index < len(rows) else rows[0]
            parts = row.split(',')
            total = int(parts[1])
            used = int(parts[2])
            return None, used, total
        raise RuntimeError('no vendor memory tool')

    def _loop(self):
        while not self._stop.is_set():
            try:
                mine, used, total = self._read()
                with self._lock:
                    self.samples += 1
                    self.device_total = total
                    if used is not None and (self.peak_device is None or used > self.peak_device):
                        self.peak_device = used
                    if mine is not None and (self.peak_process is None or mine > self.peak_process):
                        self.peak_process = mine
            except Exception:
                self.errors += 1
            self._stop.wait(self.interval)

    def start(self):
        if self.tool is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def window_reset(self):
        with self._lock:
            self._reset()

    def window_report(self):
        with self._lock:
            if self.tool is None:
                return dict(tool='unavailable', peak_process_bytes='unavailable',
                            peak_device_bytes='unavailable', device_total_bytes='unavailable',
                            boundary='no nvidia-smi or rocm-smi on PATH; in-process device peak needs new native code')
            return dict(tool=self.tool,
                        peak_process_bytes=self.peak_process if self.peak_process is not None else 'unavailable',
                        peak_device_bytes=self.peak_device if self.peak_device is not None else 'unavailable',
                        device_total_bytes=self.device_total,
                        sample_interval_seconds=self.interval, samples_total=self.samples, errors_total=self.errors,
                        boundary='vendor tool polled at the interval; peaks shorter than one interval are missed; '
                                 'device-wide used memory includes other processes; '
                                 'torch.cuda allocator statistics do not cover Mojo allocations and are not used')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class CorpusBatches:
    """Token batches from a pinned byte corpus (DEVIATIONS 2525 to 2527;
    ENGINEERING_RULES section 9: a neural timing claim runs on two
    ORDINARY corpora). `path` is the corpus file; `manifest.json` beside it
    (schema `mojolearn.byte-lm.corpus.v1`) must carry its sha256 and byte
    length, both checked here. Step `k` (zero-based), row `b` reads bytes
    `[(k * batch * length + b * length) % (n - length - 1) : + length + 1]`
    as int32 ids: the byte LM's next-byte schedule at this shape, no
    tokenizer, no normalization. Byte ids are below 256 and valid in any
    vocabulary of 256 or more; the target's 50,257-row embedding is read
    on its first 256 rows, which is the same real training path."""

    def __init__(self, path, batch, length):
        self.path = Path(path)
        manifest_path = self.path.with_name('manifest.json')
        manifest_raw = manifest_path.read_bytes()
        if len(manifest_raw) > 65536:
            raise ValueError('corpus manifest exceeds bound')
        self.manifest = json.loads(manifest_raw)
        if self.manifest.get('schema') != 'mojolearn.byte-lm.corpus.v1':
            raise ValueError('corpus manifest schema is not mojolearn.byte-lm.corpus.v1')
        raw = self.path.read_bytes()
        self.sha256 = _sha(raw)
        if self.sha256 != self.manifest.get('sha256') or len(raw) != self.manifest.get('bytes'):
            raise ValueError('pinned corpus length/SHA mismatch for %s' % self.path)
        self.manifest_sha256 = _sha(manifest_raw)
        import numpy as np  # the worker's local import; the class is module level
        self.data = np.frombuffer(raw, dtype=np.uint8)
        self.batch = batch
        self.length = length
        if len(raw) < length + 2:
            raise ValueError('corpus shorter than one batch row')
        self.modulus = len(raw) - length - 1

    def ids(self, step_index):
        import numpy as np
        rows = []
        for b in range(self.batch):
            start = (step_index * self.batch * self.length + b * self.length) % self.modulus
            rows.append(self.data[start:start + self.length + 1].astype(np.int32))
        return np.stack(rows)

    def describe(self):
        return dict(path=str(self.path), sha256=self.sha256, manifest_sha256=self.manifest_sha256,
                    bytes=int(self.data.size), source_url=self.manifest.get('source_url'),
                    schedule='step k row b: bytes[(k*batch*length + b*length) % (bytes - length - 1) : +length+1] '
                             'as int32; targets shifted one byte')


def _witness(trainer, result, step_result):
    """sha256 of the last step's flat gradient and the committed state.

    'full': from the train_step result and state_dict(), as before.
    'lean': from export_gradients(named=False) and export_state()
    (DEVIATION 2514); the same bytes by design section 2.1, so the hashes
    compare with a 'full' run's. Returns (hashes, completed_steps, seconds
    the exports and hashing took); that time is outside the step boundary.
    """
    start = time.perf_counter()
    if step_result == 'lean':
        gradients = trainer.export_gradients(named=False)['flat_gradients']
        state = trainer.export_state()
    else:
        gradients = result['flat_gradients']
        state = trainer.state_dict()
    hashes = dict(gradients=_sha(gradients.tobytes()),
                  parameters=_sha(state['parameters'].tobytes()),
                  m=_sha(state['m'].tobytes()), v=_sha(state['v'].tobytes()),
                  flags=_sha(state['flags'].tobytes()))
    completed = state['completed_steps']
    del gradients, state
    return hashes, completed, time.perf_counter() - start


def worker(args):
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape

    out = args.out
    events_path = out / 'events.jsonl'

    def emit(record):
        record = dict(record, t=time.time())
        with events_path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)

    deadline = time.monotonic() + args.budget_seconds

    def over_budget(phase):
        if time.monotonic() > deadline:
            emit(dict(event='limitation', phase=phase, budget_seconds=args.budget_seconds,
                      verdict='budget exceeded; no smaller model substituted'))
            return True
        return False

    shape = Shape(*args.shape)
    tokens_per_step = shape.batch * shape.length
    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
    for entry in Trainer.parameter_registry(shape):
        if 'norm' in entry['name']:
            weights[entry['offset']:entry['offset'] + entry['size']] += np.float32(1)
    if args.attention_path:
        os.environ['MOJOLEARN_TRANSFORMER_ATTN_PATH'] = args.attention_path
    resident = not args.no_resident
    step_result = 'lean' if args.resident_lean else 'full'
    corpus = CorpusBatches(args.corpus, shape.batch, shape.length) if args.corpus else None

    def batch_ids(step_index):
        if corpus is not None:
            return corpus.ids(step_index)
        return rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)

    trainer = Trainer(weights, shape=shape, resident=resident, step_result=step_result,
                      data_schedule={'fixture': 'lm step memory probe', 'seed': args.seed,
                                     'batches': ('pinned corpus ' + corpus.sha256) if corpus is not None
                                                else 'synthetic uniform token ids, no corpus'})
    runtime = trainer.run_metadata()
    if runtime['step_result'] != step_result:
        raise RuntimeError('trainer reports step_result=%r, requested %r' % (runtime['step_result'], step_result))
    # 'full' witnesses every step from the result; 'lean' witnesses once
    # after the last untimed step unless --witness-every-step.
    witness_every_step = step_result == 'full' or args.witness_every_step
    mode = dict(resident=resident, step_result=step_result, witness_every_step=witness_every_step,
                witness_source=WITNESS_SOURCE[step_result if witness_every_step else 'lean-final'],
                corpus=corpus.describe() if corpus is not None else None,
                attention_arm=os.environ.get('MOJOLEARN_ATTN_ARM'),
                # DEVIATION 2544: the GEMM step arm this run requested (a
                # trial binding reads it; a shipped binding ignores it).
                gemm_arm=os.environ.get('MOJOLEARN_GEMM_ARM'),
                # DEVIATION 2595: the plan that arm runs, as the leg's build
                # labels it (tools/gemm_step_leg.sh plans.tsv), so `shipped`
                # (the ksplit default where the row is above 0) is never
                # confused with the old plan (`tuned128`). None when unset.
                gemm_plan=os.environ.get('MOJOLEARN_GEMM_PLAN_LABEL'))
    sampler = DeviceMemorySampler(args.sample_interval, args.gpu_index)
    sampler.start()
    emit(dict(event='setup', schema=SCHEMA, shape=shape.to_dict(), profile=shape.profile,
              parameters=shape.n_total, n_tensors=shape.n_tensors, tokens_per_step=tokens_per_step,
              seed=args.seed, budget_seconds=args.budget_seconds, **mode,
              attention_arm_requested=os.environ.get('MOJOLEARN_ATTN_ARM'),
              gemm_arm_requested=os.environ.get('MOJOLEARN_GEMM_ARM'),
              attention_path_requested=os.environ.get('MOJOLEARN_TRANSFORMER_ATTN_PATH'),
              numeric_mode_env=os.environ.get('MOJOLEARN_NUMERIC_MODE'),
              runtime=runtime, initial_parameters_sha256=_sha(weights.tobytes()),
              host_before_first_call=_rss_bytes(), device_tool=sampler.tool or 'unavailable',
              qualification='complete-step probe at the named shape; not an opponent ratio, not a default gate'))
    steps = []
    for index in range(args.steps):
        if over_budget('before step %d' % (index + 1)):
            sampler.stop()
            _write_result(args, shape, steps, limited=True, mode=mode)
            return EXIT_LIMIT
        ids = batch_ids(index)
        sampler.window_reset()
        emit(dict(event='step_start', step=index + 1,
                  first_call_includes_setup=(index == 0),
                  ids_sha256=_sha(ids.tobytes())))
        start = time.perf_counter()
        result = trainer.train_step(ids)
        seconds = time.perf_counter() - start
        sha256 = dict(loss=_sha(np.array([result['loss']], np.float32).tobytes()))
        completed = result['completed_steps']
        export_seconds = None
        if witness_every_step:
            hashes, completed, export_seconds = _witness(trainer, result, step_result)
            sha256.update(hashes)
        record = dict(
            event='step_end', step=index + 1, seconds=seconds,
            tokens_per_second=tokens_per_step / seconds,
            first_call_includes_setup=(index == 0),
            timing_boundary='public train_step call; native binding synchronizes the context before '
                            'publishing outputs; includes host readbacks, validation and gradient copies',
            step_result=step_result, witnessed=witness_every_step,
            witness_export_seconds=export_seconds,
            loss=result['loss'],
            sha256=sha256,
            completed_steps=completed,
            host=_rss_bytes(), device=sampler.window_report())
        emit(record)
        steps.append(record)
        del result
        if over_budget('after step %d' % (index + 1)):
            sampler.stop()
            _write_result(args, shape, steps, limited=True, mode=mode)
            return EXIT_LIMIT
    final_witness = None
    if not witness_every_step:
        # The one export of the lean run: the gradient of the last untimed
        # step and the state after it, BEFORE the timing step below would
        # advance the session (export_gradients is the LAST step's).
        hashes, completed, export_seconds = _witness(trainer, None, step_result)
        final_witness = dict(step=len(steps), completed_steps=completed, sha256=hashes,
                             export_seconds=export_seconds, source=mode['witness_source'])
        emit(dict(event='final_witness', **final_witness))
    timing_step_seconds = None
    if args.component_timing:
        # One extra step with the native phase printer on. Its
        # `timing <phase> <ms>` lines land in this process's stdout (the
        # worker log); the PARENT sums them after this process exits (so
        # every native stdout buffer has been flushed) and writes them into
        # result.json. The step's wall time is recorded ONLY as the
        # denominator of the covered fraction: a timed run is not a timing
        # sample (every tick adds a device wait).
        os.environ['MOJOLEARN_TRANSFORMER_TIMING'] = '1'
        ids = batch_ids(args.steps)
        emit(dict(event='component_timing_step_start', note='MOJOLEARN_TRANSFORMER_TIMING=1; not a timing sample'))
        start = time.perf_counter()
        trainer.train_step(ids)
        timing_step_seconds = time.perf_counter() - start
        sys.stdout.flush()
        emit(dict(event='component_timing_step_end', seconds=timing_step_seconds))
        del os.environ['MOJOLEARN_TRANSFORMER_TIMING']
    sampler.stop()
    trainer.close()
    _write_result(args, shape, steps, limited=False, timing_step_seconds=timing_step_seconds,
                  mode=mode, final_witness=final_witness)
    return 0


def parse_timing_lines(text):
    """Sum every `timing <name> <value> <unit>` line by name.

    Returns (ms_by_name, bytes_by_name, count_by_name). Lines with any
    other unit are ignored. A name printed once per layer (the block
    timers) sums across layers; the count says how many lines fed it.
    """
    ms, nbytes, count = {}, {}, {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 4 or parts[0] != 'timing':
            continue
        name, value, unit = parts[1], parts[2], parts[3]
        try:
            value = float(value)
        except ValueError:
            continue
        count[name] = count.get(name, 0) + 1
        if unit == 'ms':
            ms[name] = ms.get(name, 0.0) + value
        elif unit == 'bytes':
            nbytes[name] = nbytes.get(name, 0) + int(value)
    return ms, nbytes, count


def component_timing_record(text, step_seconds):
    ms, nbytes, count = parse_timing_lines(text)
    total = sum(v for k, v in ms.items() if not k.startswith(TIMING_TOTAL_EXCLUDED_PREFIXES))
    record = dict(
        component_timing_ms={k: ms[k] for k in sorted(ms)},
        component_bytes={k: nbytes[k] for k in sorted(nbytes)},
        component_line_counts={k: count[k] for k in sorted(count)},
        component_timing_total_ms=total,
        component_timing_excluded_from_total=sorted(
            k for k in ms if k.startswith(TIMING_TOTAL_EXCLUDED_PREFIXES)),
        step_seconds=step_seconds,
        component_timing_covered_fraction=(total / 1000.0 / step_seconds) if step_seconds else None,
        component_timing_boundary='one step under MOJOLEARN_TRANSFORMER_TIMING=1; every tick waits on the device, '
                                  'so neither the parts nor step_seconds are a timing sample; the uncovered '
                                  'remainder (1 - covered fraction) is time no timer brackets')
    return record


def _write_result(args, shape, steps, limited, timing_step_seconds=None, mode=None, final_witness=None):
    import statistics
    timed = [s['seconds'] for s in steps[1:]] if len(steps) > 1 else []
    mode = mode or {}
    result = dict(
        schema=SCHEMA, shape=shape.to_dict(), parameters=shape.n_total, steps_completed=len(steps),
        resident=mode.get('resident'), step_result=mode.get('step_result'),
        witness_every_step=mode.get('witness_every_step'), witness_source=mode.get('witness_source'),
        corpus=mode.get('corpus'), attention_arm=mode.get('attention_arm'),
        gemm_arm=mode.get('gemm_arm'), gemm_plan=mode.get('gemm_plan'),
        # Per-step witnesses (loss always; gradients/parameters/m/v/flags
        # when the step was witnessed) so a lean run compares with a full
        # run from result.json alone; the same records are in events.jsonl.
        step_witnesses=[dict(step=s['step'], completed_steps=s['completed_steps'], sha256=s['sha256'])
                        for s in steps],
        # The lean run's one export after the last untimed step (None when
        # every step was witnessed or the run was limited before it).
        final_witness=final_witness,
        first_call_seconds=steps[0]['seconds'] if steps else None,
        steady_step_seconds=timed, steady_median_seconds=statistics.median(timed) if timed else None,
        steady_median_tokens_per_second=(shape.batch * shape.length / statistics.median(timed)) if timed else None,
        peak_device_process_bytes=max([s['device']['peak_process_bytes'] for s in steps
                                       if isinstance(s['device']['peak_process_bytes'], int)] or ['unavailable']),
        peak_device_wide_bytes=max([s['device']['peak_device_bytes'] for s in steps
                                    if isinstance(s['device']['peak_device_bytes'], int)] or ['unavailable']),
        process_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == 'darwin' else 1024),
        budget_seconds=args.budget_seconds, limited=limited,
        component_timing_step_seconds=timing_step_seconds,
        qualification='complete-step probe; device peak from a polled vendor tool; host RSS is a separate boundary; '
                      'no opponent, no default gate, no target-scale qualification claim')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True, help='fresh output directory')
    parser.add_argument('--shape', nargs=9, type=int, default=CONTROL_SHAPE,
                        metavar=('B', 'L', 'DM', 'H', 'KV', 'HD', 'FF', 'LAYERS', 'VOCAB'),
                        help='default: the 20.45M control; --target selects the 162,147,840-parameter shape')
    parser.add_argument('--target', action='store_true', help='use the 12-layer DM768 FF2048 V50257 shape')
    parser.add_argument('--steps', type=int, default=3, help='complete training steps (first includes setup)')
    parser.add_argument('--budget-seconds', type=float, default=300.0,
                        help='deadline for setup plus all steps; exceeding it exits 2')
    parser.add_argument('--seed', type=int, default=93261)
    parser.add_argument('--no-resident', action='store_true', help='reconstruct device state per call')
    parser.add_argument('--resident-lean', action='store_true',
                        help="resident session with step_result='lean' (DEVIATION 2514): train_step returns "
                             "loss/step/flags only; witnesses come from export_gradients()/export_state(), "
                             "once after the last untimed step unless --witness-every-step")
    parser.add_argument('--witness-every-step', action='store_true',
                        help='with --resident-lean, export and hash after EVERY step (outside the timed '
                             'boundary) so the hashes line up with a full run per step; no effect on full')
    parser.add_argument('--corpus', type=Path, default=None,
                        help='a pinned byte corpus (manifest.json beside it, schema mojolearn.byte-lm.corpus.v1, '
                             'sha256 checked); batches are its bytes in step order instead of synthetic ids')
    parser.add_argument('--attention-path', choices=['fused', 'eager'], default=None,
                        help='sets MOJOLEARN_TRANSFORMER_ATTN_PATH for the worker (default: auto = fused)')
    parser.add_argument('--component-timing', action='store_true',
                        help='one extra untimed step with MOJOLEARN_TRANSFORMER_TIMING=1 (phase prints in worker.log)')
    parser.add_argument('--sample-interval', type=float, default=0.2, help='vendor tool polling seconds')
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.target:
        args.shape = TARGET_SHAPE
    if args.steps < 1 or args.budget_seconds <= 0:
        parser.error('need at least one step and a positive budget')
    if args.resident_lean and args.no_resident:
        parser.error("--resident-lean needs a resident session (step_result='lean' refuses resident=False)")
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        parser.error('requires MOJOLEARN_NUMERIC_MODE=identical in the environment')
    if args.worker:
        raise SystemExit(worker(args))
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    # The worker runs under a hard subprocess timeout so a hung native call
    # cannot outlive the budget; the parent records the limitation (exit 2).
    with (args.out / 'worker.log').open('w') as log:
        try:
            proc = subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker'],
                                  stdout=log, stderr=subprocess.STDOUT, timeout=args.budget_seconds + 30)
            status = dict(exit_code=proc.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            status = dict(exit_code=None, timed_out=True)
    status.update(elapsed_seconds=time.monotonic() - start, budget_seconds=args.budget_seconds,
                  shape=args.shape, limitation=(status['timed_out'] or status['exit_code'] == EXIT_LIMIT))
    if args.component_timing:
        # The worker has exited, so its native stdout is flushed into
        # worker.log; sum the timing lines here and fold them into the
        # worker's result.json (written by the worker before it exited).
        result_path = args.out / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            text = (args.out / 'worker.log').read_text(errors='replace')
            result.update(component_timing_record(text, result.get('component_timing_step_seconds')))
            result_path.write_text(json.dumps(result, indent=2) + '\n')
    if status['limitation']:
        status['verdict'] = 'setup or a step exceeded the budget; recorded as a limitation, no smaller model substituted'
    (args.out / 'execution.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))
    if status['limitation']:
        raise SystemExit(EXIT_LIMIT)
    raise SystemExit(0 if status['exit_code'] == 0 else 1)


if __name__ == '__main__':
    main()
