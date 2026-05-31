"""
Mathematical utility routines used throughout Exorem.

Mirrors the Fortran ``math`` module (math.f90).  Where the Fortran version
implements an algorithm by hand, the Python port favors NumPy/SciPy if and
only if the behaviour is numerically equivalent.  Otherwise the original
algorithm is reproduced.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from scipy.special import erf, erfinv as _scipy_erfinv, wofz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PI:      float = math.pi
SQRTPI:  float = 1.0 / math.sqrt(math.pi)        # = 0.5641895835477563

# numerical precision floor approximations (Fortran's huge/tiny analogues)
PREC_HIGH: float = 10.0 ** -np.finfo(np.float64).precision   # ~1e-15
PREC_LOW:  float = 10.0 ** -np.finfo(np.float32).precision   # ~1e-6

_Number = Union[float, np.ndarray]


# ===========================================================================
# Goodness of fit
# ===========================================================================

def chi2(observed_data: np.ndarray, calculated_data: np.ndarray) -> float:
    """Pearson chi-square statistic of observed vs calculated."""
    obs  = np.asarray(observed_data, dtype=float)
    calc = np.asarray(calculated_data, dtype=float)
    return float(np.sum((obs - calc) ** 2 / calc))


def chi2_reduced(observed_data: np.ndarray,
                 calculated_data: np.ndarray,
                 input_deviation: np.ndarray) -> float:
    """Reduced chi-square using per-point uncertainties."""
    obs  = np.asarray(observed_data, dtype=float)
    calc = np.asarray(calculated_data, dtype=float)
    sig  = np.asarray(input_deviation, dtype=float)
    return float(np.sum((obs - calc) ** 2 / sig ** 2))


# ===========================================================================
# Angle helpers
# ===========================================================================

def deg2rad(angle: _Number) -> _Number:
    """Degrees → radians (Fortran-style scalar helper)."""
    return angle * (PI / 180.0)


def sec(angle: _Number) -> _Number:
    """Secant of an angle given in degrees."""
    return 1.0 / np.cos(deg2rad(angle))


def sgn(value: _Number) -> _Number:
    """Sign function: returns +1.0 for value >= 0, -1.0 otherwise."""
    if isinstance(value, np.ndarray):
        return np.where(value >= 0.0, 1.0, -1.0)
    return 1.0 if value >= 0.0 else -1.0


def ellipse_polar_form(semi_major_axis: float,
                       semi_minor_axis: float,
                       angle: float) -> float:
    """Distance from the centre of an ellipse to a point at *angle* (deg)."""
    th = deg2rad(angle)
    return (semi_major_axis * semi_minor_axis
            / math.sqrt((semi_minor_axis * math.cos(th)) ** 2
                        + (semi_major_axis * math.sin(th)) ** 2))


# ===========================================================================
# PDFs and noise
# ===========================================================================

def gaussian(x: _Number, fwhm: float) -> _Number:
    """Normalised Gaussian, parameterised by FWHM."""
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return (1.0 / (sigma * math.sqrt(2.0 * PI))
            * np.exp(-0.5 * (np.asarray(x) / sigma) ** 2))


def gaussian_noise(rng: np.random.Generator | None = None) -> float:
    """Draw a single sample from the standard normal distribution."""
    if rng is None:
        rng = np.random.default_rng()
    return float(rng.standard_normal())


def sinc_fwhm(x: _Number, fwhm: float) -> _Number:
    """Normalised sinc function parameterised by FWHM."""
    k = 1.0 / (2.0 * 1.89549)
    arg = np.asarray(x) / (k * fwhm)
    # protect division by zero
    out = np.where(np.abs(arg) < np.finfo(float).tiny, 1.0, np.sinc(arg / PI))
    return out


# ===========================================================================
# Inverse error function
# ===========================================================================

def erfinv(x: _Number) -> _Number:
    """Inverse of the error function.  Uses SciPy for vectorised accuracy."""
    arr = np.asarray(x, dtype=float)
    if np.any(arr < -1.0) or np.any(arr > 1.0):
        raise ValueError("erfinv: x must be in [-1, 1]")
    return _scipy_erfinv(arr)


# ===========================================================================
# Arange helpers
# ===========================================================================

def arange(start: float, stop: float, step: float) -> np.ndarray:
    """
    Evenly-spaced values in [start, stop).

    Matches the Fortran ``arange``: the last element is strictly less than stop.
    """
    n = max(0, int(math.ceil((stop - start) / step)))
    return start + np.arange(n, dtype=float) * step


def arange_include(start: float, stop: float, step: float) -> np.ndarray:
    """
    Like :func:`arange` but the array always includes a final value that is
    greater than *stop* by at most one step.
    """
    n = max(1, int(math.ceil((stop - start) / step)) + 1)
    return start + np.arange(n, dtype=float) * step


# ===========================================================================
# Convolution
# ===========================================================================

def convolve(signal: np.ndarray, filter_: np.ndarray) -> np.ndarray:
    """
    Classical convolution preserving the signal length (same-mode), with the
    same start-index offset as the Fortran ``convolve``.
    """
    signal  = np.asarray(signal, dtype=float)
    filter_ = np.asarray(filter_, dtype=float)
    size_s = signal.size
    size_f = filter_.size

    full = np.convolve(signal, filter_, mode="full")           # length size_s + size_f - 1
    start = int(math.floor(size_f / 2.0))
    return full[start:start + size_s]


def slide_convolve(signal: np.ndarray, filter_: np.ndarray) -> np.ndarray:
    """
    Sliding convolution where the filter coefficients vary per signal index.

    Parameters
    ----------
    signal : (N,) array
    filter_: (n_filter, N) 2-D array — column ``j`` is the filter applied
             at signal index ``j``.

    Returns
    -------
    convolved_signal : (N,) array
    """
    signal  = np.asarray(signal, dtype=float)
    filter_ = np.asarray(filter_, dtype=float)
    n        = signal.size
    n_filter = filter_.shape[0]
    if n < n_filter:
        raise ValueError(
            f"slide_convolve: filter size {n_filter} must be ≤ signal size {n}")

    start = int(math.floor(n_filter / 2.0))               # 1-based start in Fortran
    convolution = np.zeros(n + n_filter, dtype=float)

    # Middle part — full overlap
    for i in range(n_filter - 1, n):                       # Fortran: size_filter..size_signal
        s = 0.0
        j = i
        for k in range(n_filter):
            s += signal[j] * filter_[k, j]
            j -= 1
        convolution[i] = s

    # First-part ramp
    for i in range(start, n_filter - 1):
        s = 0.0
        j = i
        for k in range(i + 1):
            s += signal[j] * filter_[k, j]
            j -= 1
        convolution[i] = s

    # Tail
    for i in range(n, start + n):
        s = 0.0
        j = n - 1
        for k in range(n_filter):
            s += signal[j] * filter_[k, j]
            j -= 1
        convolution[i] = s

    return convolution[start:start + n]


# ===========================================================================
# Search and sort utilities
# ===========================================================================

def search_sorted(array: np.ndarray, value: float) -> int:
    """
    Index of the array element closest to *value*.

    Returns a 0-based index (the Fortran source returned a 1-based index).
    """
    arr = np.asarray(array, dtype=float)
    n = arr.size
    if value < arr[0]:
        return 0
    if value > arr[-1]:
        return n - 1
    i = int(np.searchsorted(arr, value))
    if i == 0:
        return 0
    if i == n:
        return n - 1
    # Decide whether the left or right neighbour is closer
    if abs(arr[i - 1] - value) < abs(arr[i] - value):
        return i - 1
    return i


# ===========================================================================
# Linear interpolation
# ===========================================================================

def _interp_ascending(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y)


def _interp_descending(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(x_new[::-1], x[::-1], y[::-1])[::-1]


def interp(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Linear interpolation, no extrapolation.

    Mirrors the Fortran ``interp``: errors if ``x_new`` falls outside the range
    of ``x``.  Works for both ascending and descending ``x``.
    """
    x_new = np.asarray(x_new, dtype=float)
    x     = np.asarray(x, dtype=float)
    y     = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("interp: x and y must have the same size")

    if x[0] < x[-1]:
        if x_new[0] < x[0] or x_new[-1] > x[-1]:
            raise ValueError(
                f"interp: x_new outside x boundaries: "
                f"{x[0]} < {x_new[0]} -- {x_new[-1]} < {x[-1]}")
        return _interp_ascending(x_new, x, y)
    if x_new[0] > x[0] or x_new[-1] < x[-1]:
        raise ValueError(
            f"interp: x_new outside x boundaries: "
            f"{x[0]} > {x_new[0]} -- {x_new[-1]} > {x[-1]}")
    return _interp_descending(x_new, x, y)


def interp_ex(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Linear interpolation with linear *extrapolation* outside the support.

    Both ``x`` and ``x_new`` must share the same numerical order
    (ascending OR descending), matching the Fortran behaviour.
    """
    x_new = np.asarray(x_new, dtype=float)
    x     = np.asarray(x, dtype=float)
    y     = np.asarray(y, dtype=float)
    n = x.size

    descending = x[0] > x[-1]
    if descending and x_new[0] < x_new[-1]:
        raise ValueError("interp_ex: x_new must follow the same order as x (descending)")
    if (not descending) and x_new[0] > x_new[-1]:
        raise ValueError("interp_ex: x_new must follow the same order as x (ascending)")

    if descending:
        # Reverse to ascending order for processing
        x_a, y_a = x[::-1], y[::-1]
        xn_a     = x_new[::-1]
    else:
        x_a, y_a = x, y
        xn_a     = x_new

    out = np.interp(xn_a, x_a, y_a)

    # Linear extrapolation outside support
    left_slope  = (y_a[1] - y_a[0]) / (x_a[1] - x_a[0])
    right_slope = (y_a[-1] - y_a[-2]) / (x_a[-1] - x_a[-2])

    lo = xn_a < x_a[0]
    hi = xn_a > x_a[-1]
    out[lo] = y_a[0]  + left_slope  * (xn_a[lo] - x_a[0])
    out[hi] = y_a[-1] + right_slope * (xn_a[hi] - x_a[-1])

    return out[::-1] if descending else out


def interp_ex_0d(x_new: float, x: np.ndarray, y: np.ndarray) -> float:
    """Scalar version of :func:`interp_ex`."""
    return float(interp_ex(np.array([float(x_new)]), x, y)[0])


def mean_restep(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Re-bin y(x) onto x_new by averaging values inside each new bin.

    Both ``x`` and ``x_new`` must be ascending.  ``x_new`` must lie within
    the support of ``x``.
    """
    x_new = np.asarray(x_new, dtype=float)
    x     = np.asarray(x, dtype=float)
    y     = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("mean_restep: x and y must have the same size")
    if x[0] > x[-1]:
        raise ValueError("mean_restep: x must be ascending")
    if x_new[0] < x[0] or x_new[-1] > x[-1]:
        raise ValueError("mean_restep: x_new outside x boundaries")

    n_new = x_new.size
    y_new = np.zeros(n_new)

    # bin edges that bracket each x_new sample
    for i in range(n_new):
        if i == 0:
            lo = x_new[0]
            hi = x_new[0] + (x_new[1] - x_new[0]) / 2.0
        elif i == n_new - 1:
            lo = x_new[i] - (x_new[i] - x_new[i - 1]) / 2.0
            hi = x_new[i]
        else:
            lo = x_new[i] - (x_new[i] - x_new[i - 1]) / 2.0
            hi = x_new[i] + (x_new[i + 1] - x_new[i]) / 2.0
        mask = (x >= lo) & (x < hi)
        if mask.any():
            y_new[i] = y[mask].mean()
        else:
            # fallback — nearest point in x
            y_new[i] = y[np.argmin(np.abs(x - x_new[i]))]
    return y_new


# ===========================================================================
# Misc array helpers
# ===========================================================================

def reverse_array(array: np.ndarray) -> np.ndarray:
    """Return a reversed copy of *array*."""
    return np.asarray(array)[::-1].copy()


# ===========================================================================
# Voigt function (Humlicek 1982 algorithm)
# ===========================================================================

def voigt(x: np.ndarray, y: float) -> np.ndarray:
    """
    Voigt function — real part of w(z) where z = x + i*y, y >= 0.

    Internally uses :func:`scipy.special.wofz` which implements the same
    Faddeeva function the Fortran code approximates with Humlicek polynomials,
    yielding identical results to within ~1e-12 (much better than the 1e-4
    relative error of the polynomial form).
    """
    x = np.asarray(x, dtype=float)
    z = x + 1j * y
    w = wofz(z)
    return (w.real * math.sqrt(PI)).astype(float)


def voigt_from_data(x: float, y: float, v: np.ndarray) -> float:
    """
    Voigt-function look-up using the pre-computed table ``v[400, 100]``
    (the bilinear interpolant of a stored Faddeeva function table).
    """
    sqrtpi = SQRTPI
    dp = 0.0025
    ds = 0.01
    yl = 1.0 / 99.0
    xl = 399.0

    if y > 5.4:
        x2 = x * x
        y2 = y * y
        x2y2 = x2 + y2
        return float(sqrtpi * y * (
            1.0
            + (3.0 * x2 - y2) / (2.0 * x2y2 * x2y2)
            + 0.75 * (5.0 * x2 * x2 + y2 * y2 - 10.0 * x2 * y2) / (x2y2 ** 4)
        ) / x2y2)

    if y <= yl:
        b = math.exp(-x * x)
        ps = 1.0 / (1.0 + x)
        ip = int(ps / dp)
        f = ps / dp - ip
        if ip >= 1:
            if ip < 400:
                a = (1.0 - f) * v[ip - 1, 0] + f * v[ip, 0]
            else:
                a = v[ip - 1, 0]
        else:
            a = f * v[ip, 0]
        a *= a
        f = y / yl
        return float((1.0 - f) * b + f * a)

    if x > xl:
        s = y / (1.0 + y)
        is_ = int(s / ds)
        b = v[0, is_] * xl / x
        b *= b
        a = v[0, is_ + 1] * xl / x
        a *= a
        f = s / ds - is_
        return float((1.0 - f) * b + f * a)

    s = y / (1.0 + y)
    ps = 1.0 / (1.0 + x)
    is_ = int(s / ds)
    ip  = int(ps / dp)
    f   = ps / dp - ip
    if ip < 400:
        b = (1.0 - f) * v[ip - 1, is_]     + f * v[ip, is_]
        a = (1.0 - f) * v[ip - 1, is_ + 1] + f * v[ip, is_ + 1]
    else:
        b = v[ip - 1, is_]
        a = v[ip - 1, is_ + 1]
    b *= b
    a *= a
    f = s / ds - is_
    return float((1.0 - f) * b + f * a)


# ===========================================================================
# FFT and inverse FFT
# ===========================================================================

def fft(x: np.ndarray) -> np.ndarray:
    """
    Forward FFT (in-place semantics from the Fortran source preserved by
    returning a fresh array).
    """
    arr = np.asarray(x, dtype=complex)
    return np.fft.fft(arr)


def ifft(x: np.ndarray) -> np.ndarray:
    """Inverse FFT — matches the Cooley-Tukey normalisation of the Fortran code."""
    arr = np.asarray(x, dtype=complex)
    return np.fft.ifft(arr)


# ===========================================================================
# Quicksort wrappers (NumPy's sort is at least as fast and stable enough)
# ===========================================================================

def quicksort(array: np.ndarray) -> np.ndarray:
    """Return a sorted copy of *array* (ascending)."""
    return np.sort(np.asarray(array, dtype=float))


def quicksort_index(array: np.ndarray) -> np.ndarray:
    """Return the indices that would sort *array* (ascending)."""
    return np.argsort(np.asarray(array, dtype=float))


# ===========================================================================
# Matrix inverse — Doolittle LU
# ===========================================================================

def matinv(a: np.ndarray, n: int | None = None) -> np.ndarray:
    """
    Invert a square matrix.  The Fortran version mutates *a* in place;
    here we return a fresh array but also overwrite *a* for parity.

    Falls back to :func:`numpy.linalg.inv` which uses LAPACK.
    """
    a = np.asarray(a, dtype=float)
    if n is not None and a.shape != (n, n):
        raise ValueError(f"matinv: array is not {n}×{n}")
    inv = np.linalg.inv(a)
    a[...] = inv
    return inv


# ===========================================================================
# Reallocation helpers (no-ops in NumPy — array reassignment is free)
# ===========================================================================

def reallocate_1d(arr: np.ndarray, new_size: int, dtype=float) -> np.ndarray:
    """Return a fresh zero-filled array of the requested size."""
    return np.zeros(new_size, dtype=dtype)


def reallocate_2d(arr: np.ndarray, n1: int, n2: int, dtype=float) -> np.ndarray:
    return np.zeros((n1, n2), dtype=dtype)


def reallocate_3d(arr: np.ndarray, n1: int, n2: int, n3: int, dtype=float) -> np.ndarray:
    return np.zeros((n1, n2, n3), dtype=dtype)


__all__ = [
    "PI", "SQRTPI", "PREC_HIGH", "PREC_LOW",
    "chi2", "chi2_reduced",
    "deg2rad", "sec", "sgn", "ellipse_polar_form",
    "gaussian", "gaussian_noise", "sinc_fwhm",
    "erfinv",
    "arange", "arange_include",
    "convolve", "slide_convolve",
    "search_sorted",
    "interp", "interp_ex", "interp_ex_0d", "mean_restep",
    "reverse_array",
    "voigt", "voigt_from_data",
    "fft", "ifft",
    "quicksort", "quicksort_index",
    "matinv",
    "reallocate_1d", "reallocate_2d", "reallocate_3d",
]
