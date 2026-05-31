"""
diagnose_numba.py
=================
Tells you *exactly* why numba isn't parallelising on your machine.

Run from the python_ExoREM directory:

    python diagnose_numba.py

Reports:
  * numba version, threading backend, thread count
  * whether the @njit decorator actually compiles or is silently a no-op
  * whether prange runs on multiple threads (timed)
  * the OS-level CPU/thread count visible to Python
  * env vars (NUMBA_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS)

If get_num_threads() == 1 — that's the smoking gun.
"""
import os, sys, time, multiprocessing, platform

print("=" * 72)
print("ENVIRONMENT")
print("=" * 72)
print(f"  Python              : {sys.version.split()[0]}")
print(f"  Platform            : {platform.platform()}")
print(f"  CPU count (os)      : {multiprocessing.cpu_count()}")
print(f"  Env NUMBA_NUM_THREADS : {os.environ.get('NUMBA_NUM_THREADS', '<unset>')}")
print(f"  Env OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS', '<unset>')}")
print(f"  Env MKL_NUM_THREADS   : {os.environ.get('MKL_NUM_THREADS', '<unset>')}")
print(f"  Env NUMBA_THREADING_LAYER : {os.environ.get('NUMBA_THREADING_LAYER', '<unset>')}")
print()

print("=" * 72)
print("NUMBA IMPORT")
print("=" * 72)
try:
    import numba
    print(f"  numba version       : {numba.__version__}")
    print(f"  numba.get_num_threads(): {numba.get_num_threads()}")
    print(f"  numba.config.NUMBA_DEFAULT_NUM_THREADS : "
          f"{numba.config.NUMBA_DEFAULT_NUM_THREADS}")
    print(f"  numba.config.NUMBA_NUM_THREADS         : "
          f"{numba.config.NUMBA_NUM_THREADS}")
    print(f"  THREADING_LAYER preference            : "
          f"{numba.config.THREADING_LAYER}")
    print(f"  THREADING_LAYER_PRIORITY              : "
          f"{numba.config.THREADING_LAYER_PRIORITY}")

    # Threading layer is only known after the first parallel call.
    # Force one now with a tiny @njit(parallel=True) function.
    from numba import njit, prange
    import numpy as np

    @njit(parallel=True, cache=False)
    def _force_threading_init(n):
        s = np.zeros(n)
        for i in prange(n):
            s[i] = i * 2.0
        return s.sum()

    _ = _force_threading_init(1000)
    try:
        print(f"  active threading_layer()              : "
              f"{numba.threading_layer()}")
    except Exception as e:
        print(f"  threading_layer() raised              : {e!r}")

except ImportError as e:
    print(f"  *** numba is NOT installed *** ({e})")
    sys.exit(1)

print()
print("=" * 72)
print("PARALLEL SCALING TEST")
print("=" * 72)
print("If prange is actually parallelising, the elapsed time should drop")
print("as you let it use more threads.  If not, all three rows match —")
print("that's the bug.\n")


@njit(parallel=True, cache=False, fastmath=True)
def _bench_parallel(n_iter, work):
    s = 0.0
    for i in prange(n_iter):
        x = float(i) + 1.0
        for _ in range(work):
            x = (x * 1.0000001 + 1.0) / 1.0000001
        s += x
    return s


@njit(cache=False, fastmath=True)
def _bench_serial(n_iter, work):
    s = 0.0
    for i in range(n_iter):
        x = float(i) + 1.0
        for _ in range(work):
            x = (x * 1.0000001 + 1.0) / 1.0000001
        s += x
    return s


# warmup
_bench_parallel(100, 100); _bench_serial(100, 100)

N_ITER = 4_000
WORK   = 50_000

print(f"  workload: {N_ITER:,} outer × {WORK:,} inner iterations\n")
t0 = time.perf_counter()
_ = _bench_serial(N_ITER, WORK)
t_serial = time.perf_counter() - t0
print(f"  serial   (range, single thread)             : {t_serial:6.3f} s")

t0 = time.perf_counter()
_ = _bench_parallel(N_ITER, WORK)
t_parallel = time.perf_counter() - t0
print(f"  parallel (prange, get_num_threads() threads): {t_parallel:6.3f} s")

ratio = t_serial / t_parallel
n_th  = numba.get_num_threads()
print()
print(f"  Observed speedup    : {ratio:.2f}×  on {n_th} thread(s)")
if ratio < 1.5:
    print()
    print("  ⚠  prange is NOT parallelising effectively.")
    print()
    print("  Likely causes (in order of probability on macOS):")
    print("    1. Numba defaulted to the 'workqueue' backend because TBB or")
    print("       OpenMP weren't installed.  Install one:")
    print("           pip install tbb               (preferred, fast)")
    print("       OR  pip install intel-openmp")
    print("    2. NUMBA_NUM_THREADS=1 was set somewhere.  Check your shell")
    print("       rc files (~/.zshrc, ~/.bash_profile) and conda envs.")
    print("    3. The wrong libomp is being loaded.  On Apple Silicon with")
    print("       brew, sometimes Python's openmp dylib conflicts with the")
    print("       system one.  Run:  otool -L $(python -c 'import numba; ")
    print("       print(numba.__file__)' | sed 's/__init__.py//')/np/ufunc")
    print("       /omppool*.so   to see which OpenMP it's linked against.")
elif ratio < n_th * 0.6:
    print()
    print(f"  ◦ prange IS parallelising, but speedup ({ratio:.1f}×) is less")
    print(f"    than expected for {n_th} threads (would expect ~{n_th*0.7:.0f}×).")
    print(f"    This is normal if other code is running, or if your workload")
    print(f"    has memory-bandwidth limits.  Should still be fine for exorem.")
else:
    print()
    print(f"  ✓ Parallelism is working.  If exorem still takes 47 s per RT")
    print(f"    iteration something else is the bottleneck — most likely the")
    print(f"    per-prange-iteration np.empty allocations on lines 511-514")
    print(f"    of radiative_transfer.py.  See HANDOFF for the fix.")
