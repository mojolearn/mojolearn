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

The configuration is an argument; the default is the 20.45M control shape
(B1 L2048 DM384 H6 KV6 HD64 FF1024, 8 layers, V8192). If setup plus the
first step, or any later step, exceeds --budget-seconds the harness exits 2
and records the limitation. It NEVER substitutes a smaller model.

Synthetic token batches from a seeded generator, no corpus: a few complete
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
SCHEMA = 'mojolearn.lm-step-memory-probe.v1'
EXIT_LIMIT = 2


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
    trainer = Trainer(weights, shape=shape, resident=not args.no_resident,
                      data_schedule={'fixture': 'lm step memory probe', 'seed': args.seed,
                                     'batches': 'synthetic uniform token ids, no corpus'})
    sampler = DeviceMemorySampler(args.sample_interval, args.gpu_index)
    sampler.start()
    emit(dict(event='setup', schema=SCHEMA, shape=shape.to_dict(), profile=shape.profile,
              parameters=shape.n_total, n_tensors=shape.n_tensors, tokens_per_step=tokens_per_step,
              resident=not args.no_resident, seed=args.seed, budget_seconds=args.budget_seconds,
              attention_path_requested=os.environ.get('MOJOLEARN_TRANSFORMER_ATTN_PATH'),
              numeric_mode_env=os.environ.get('MOJOLEARN_NUMERIC_MODE'),
              runtime=trainer.run_metadata(), initial_parameters_sha256=_sha(weights.tobytes()),
              host_before_first_call=_rss_bytes(), device_tool=sampler.tool or 'unavailable',
              qualification='complete-step probe at the named shape; not an opponent ratio, not a default gate'))
    steps = []
    for index in range(args.steps):
        if over_budget('before step %d' % (index + 1)):
            sampler.stop()
            _write_result(args, shape, steps, limited=True)
            return EXIT_LIMIT
        ids = rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)
        sampler.window_reset()
        emit(dict(event='step_start', step=index + 1,
                  first_call_includes_setup=(index == 0),
                  ids_sha256=_sha(ids.tobytes())))
        start = time.perf_counter()
        result = trainer.train_step(ids)
        seconds = time.perf_counter() - start
        state = trainer.state_dict()
        record = dict(
            event='step_end', step=index + 1, seconds=seconds,
            tokens_per_second=tokens_per_step / seconds,
            first_call_includes_setup=(index == 0),
            timing_boundary='public train_step call; native binding synchronizes the context before '
                            'publishing outputs; includes host readbacks, validation and gradient copies',
            loss=result['loss'],
            sha256=dict(loss=_sha(np.array([result['loss']], np.float32).tobytes()),
                        gradients=_sha(result['flat_gradients'].tobytes()),
                        parameters=_sha(state['parameters'].tobytes()),
                        m=_sha(state['m'].tobytes()), v=_sha(state['v'].tobytes()),
                        flags=_sha(state['flags'].tobytes())),
            completed_steps=state['completed_steps'],
            host=_rss_bytes(), device=sampler.window_report())
        emit(record)
        steps.append(record)
        del result, state
        if over_budget('after step %d' % (index + 1)):
            sampler.stop()
            _write_result(args, shape, steps, limited=True)
            return EXIT_LIMIT
    if args.component_timing:
        # One extra, UNTIMED step with the native phase printer on. Its
        # `timing <phase> <ms>` lines land in this process's stdout (the
        # worker log). A timed run is not a timing sample.
        os.environ['MOJOLEARN_TRANSFORMER_TIMING'] = '1'
        ids = rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)
        emit(dict(event='component_timing_step_start', note='MOJOLEARN_TRANSFORMER_TIMING=1; not a timing sample'))
        trainer.train_step(ids)
        emit(dict(event='component_timing_step_end'))
        del os.environ['MOJOLEARN_TRANSFORMER_TIMING']
    sampler.stop()
    trainer.close()
    _write_result(args, shape, steps, limited=False)
    return 0


def _write_result(args, shape, steps, limited):
    import statistics
    timed = [s['seconds'] for s in steps[1:]] if len(steps) > 1 else []
    result = dict(
        schema=SCHEMA, shape=shape.to_dict(), parameters=shape.n_total, steps_completed=len(steps),
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
    if status['limitation']:
        status['verdict'] = 'setup or a step exceeded the budget; recorded as a limitation, no smaller model substituted'
    (args.out / 'execution.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))
    if status['limitation']:
        raise SystemExit(EXIT_LIMIT)
    raise SystemExit(0 if status['exit_code'] == 0 else 1)


if __name__ == '__main__':
    main()
