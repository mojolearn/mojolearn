"""Conservative CLI thread budgets, applied before native imports in a child."""
import os
import subprocess
import sys


THREAD_VARIABLES = (
    'MOJOLEARN_CPU_THREADS', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'BLIS_NUM_THREADS',
)


def available_cpus():
    """Respect process affinity where available; never assume five CPUs exist."""
    counts = []
    for name in ('process_cpu_count', 'cpu_count'):
        count = getattr(os, name, lambda: None)()
        if count and count > 0:
            counts.append(count)
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    return max(1, min(counts)) if counts else 1


def budget_environment(requested, environ=None, cpus=None):
    if requested < 1:
        raise ValueError('--cpu-threads must be a positive integer')
    available = available_cpus() if cpus is None else max(1, cpus)
    # Leave one available logical CPU as headroom on multi-CPU machines.
    effective = min(requested, max(1, available - 1))
    env = dict(os.environ if environ is None else environ)
    env.update({key: str(effective) for key in THREAD_VARIABLES})
    env['OMP_MAX_ACTIVE_LEVELS'] = '1'
    env['OMP_DYNAMIC'] = 'FALSE'
    env['MOJOLEARN_VERIFY_CPU_THREADS'] = str(effective)
    return effective, env


def run_with_budget(args):
    """Return a child exit code, or None if the caller should dispatch locally.

    Python/native libraries can initialize while importing the package before
    CLI parsing. Set the environment in a fresh interpreter, never pretend a
    late os.environ assignment resized an already initialized native pool.
    This controls supported pools, not OS CPU usage, RAM or GPU memory.
    """
    if args.command != 'verify' or any(getattr(args, flag, None) for flag in (
            'coverage', 'compare', 'commitment', 'commitment_a', 'commitment_b')):
        return None
    effective, env = budget_environment(args.cpu_threads)
    required = (*THREAD_VARIABLES, 'OMP_MAX_ACTIVE_LEVELS', 'OMP_DYNAMIC',
                'MOJOLEARN_VERIFY_CPU_THREADS')
    if all(os.environ.get(key) == env[key] for key in required):
        return None
    print(f'# verifier CPU thread budget: {effective}; one algorithm at a time. '
          'This is not a memory limit.', file=sys.stderr)
    return subprocess.run([sys.executable, '-m', 'mojolearn', *args.argv], env=env).returncode
