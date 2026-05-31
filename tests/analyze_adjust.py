#!/usr/bin/env python3
"""
analyze_adjust.py — visualise the r45 convective-adjustment dumps.

Usage:
    python analyze_adjust.py <path_outputs>

Reads <path_outputs>/retrieval_adjust_iter*.npz and writes plots to
<path_outputs>/diagnostics/.

The four make-or-break diagnostics for whether convective adjustment is
solving the RC-disconnect:

1. RCB level vs iteration — is the convective zone extending upward to
   absorb the hot-bump region, or stuck?
2. Adjustment dT heatmap — where (and how hard) is convection acting?
3. Hot-bump tracker — T_after at L=27-32 vs iteration. Shrinking = win.
4. Enthalpy flux vs iteration — small & steady = healthy convective flux;
   large & growing = adjustment fighting the retrieval.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(outdir: Path):
    files = sorted(outdir.glob("retrieval_adjust_iter*.npz"))
    if not files:
        sys.exit(f"No retrieval_adjust_iter*.npz in {outdir}. "
                 f"Is CONVECTIVE_ADJUSTMENT = True and the run done?")
    dumps = [np.load(f, allow_pickle=True) for f in files]
    iters = [int(f.stem.split("iter")[-1]) for f in files]
    print(f"Loaded {len(dumps)} adjustment dumps "
          f"(iter {iters[0]}..{iters[-1]})")
    return dumps, iters


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_adjust.py <path_outputs>")
        sys.exit(1)
    outdir = Path(sys.argv[1])
    diag = outdir / "diagnostics"
    diag.mkdir(exist_ok=True)
    dumps, iters = load(outdir)

    n_it = len(dumps)
    n_lev = dumps[0]["T_after"].shape[0]
    P = dumps[0]["pressures"]

    rcb_level = np.array([d["rcb_level"] for d in dumps])
    rcb_P = np.array([float(d["rcb_pressure"]) for d in dumps])
    enth = np.array([float(d["enthalpy_rel_change"]) for d in dumps]) * 100
    n_reg = np.array([int(d["n_regions"]) for d in dumps])
    dT = np.array([d["dT"] for d in dumps])             # (n_it, n_lev)
    T_after = np.array([d["T_after"] for d in dumps])   # (n_it, n_lev)

    # ---- Figure 1: the four key diagnostics ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    ax[0, 0].plot(iters, rcb_level, "-", lw=1.2)
    ax[0, 0].set_xlabel("iteration"); ax[0, 0].set_ylabel("RCB level")
    ax[0, 0].set_title("1. RCB level (top of convective zone) vs iteration")
    ax[0, 0].grid(alpha=0.3)

    vmax = np.percentile(np.abs(dT), 99) or 1.0
    im = ax[0, 1].imshow(dT.T, aspect="auto", origin="lower", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax,
                         extent=[iters[0], iters[-1], 0, n_lev])
    ax[0, 1].set_xlabel("iteration"); ax[0, 1].set_ylabel("level")
    ax[0, 1].set_title("2. Adjustment ΔT (where convection acts)")
    plt.colorbar(im, ax=ax[0, 1], label="ΔT (K)")

    for L in (27, 28, 29, 30, 31, 32):
        if L < n_lev:
            ax[1, 0].plot(iters, T_after[:, L], lw=1.0,
                          label=f"L={L} (P={P[L]:.1e})")
    ax[1, 0].set_xlabel("iteration"); ax[1, 0].set_ylabel("T after adj (K)")
    ax[1, 0].set_title("3. Hot-bump tracker — shrinking = win")
    ax[1, 0].legend(fontsize=7); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(iters, enth, lw=1.0)
    ax[1, 1].axhline(0, color="gray", ls=":")
    ax[1, 1].set_xlabel("iteration")
    ax[1, 1].set_ylabel("enthalpy Δ per call (%)")
    ax[1, 1].set_title("4. Convective heat flux (small+steady = healthy)")
    ax[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(diag / "adjust_overview.png", dpi=90)
    plt.close()
    print("  wrote adjust_overview.png")

    # ---- Figure 2: before/after profiles at selected iterations ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    picks = [iters[0],
             iters[min(len(iters) - 1, len(iters) // 4)],
             iters[min(len(iters) - 1, len(iters) // 2)],
             iters[-1]]
    colours = plt.cm.viridis(np.linspace(0, 1, len(picks)))
    for c, it in zip(colours, picks):
        k = iters.index(it)
        axes[0].semilogy(dumps[k]["T_before"], P, "--", color=c, alpha=0.5)
        axes[0].semilogy(dumps[k]["T_after"], P, "-", color=c, lw=1.5,
                         label=f"iter {it}")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("T (K)"); axes[0].set_ylabel("P (Pa)")
    axes[0].set_title("Before (dashed) vs after (solid) adjustment")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    k = -1
    axes[1].semilogy(dumps[k]["T_before"], P, "b--", label="before")
    axes[1].semilogy(dumps[k]["T_after"], P, "r-", lw=1.5, label="after")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("T (K)"); axes[1].set_ylabel("P (Pa)")
    axes[1].set_title(f"Final iteration ({iters[-1]}): RCB region detail")
    axes[1].set_ylim(3e4, 5e3)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(diag / "adjust_profiles.png", dpi=90)
    plt.close()
    print("  wrote adjust_profiles.png")

    # ---- summary ----
    print("\n=== Summary ===")
    print(f"RCB level: start {rcb_level[0]}, end {rcb_level[-1]}, "
          f"max {rcb_level.max()}")
    print(f"Convective regions per iter: mean {n_reg.mean():.1f}, "
          f"max {n_reg.max()}")
    print(f"Enthalpy Δ per call: mean {enth.mean():+.3f}%, "
          f"last {enth[-1]:+.3f}%")
    if n_lev > 30:
        print(f"T(L=30) after adj: start {T_after[0,30]:.0f} K, "
              f"end {T_after[-1,30]:.0f} K  "
              f"({'SHRINKING' if T_after[-1,30] < T_after[0,30] else 'growing/stable'})")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# r46 addendum: radiative-smoothing dumps (retrieval_smooth_iter*.npz)
# Run:  python analyze_adjust.py <path_outputs>   (auto-detects smooth dumps)
# ---------------------------------------------------------------------------
def analyze_smoothing(outdir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    outdir = Path(outdir)
    files = sorted(outdir.glob("retrieval_smooth_iter*.npz"))
    if not files:
        print("No retrieval_smooth_iter*.npz found (RADIATIVE_SMOOTHING off?).")
        return
    diag = outdir / "diagnostics"; diag.mkdir(exist_ok=True)
    dumps = [np.load(f, allow_pickle=True) for f in files]
    iters = [int(f.stem.split("iter")[-1]) for f in files]
    P = dumps[0]["pressures"]
    print(f"\nLoaded {len(dumps)} smoothing dumps (iter {iters[0]}..{iters[-1]})")

    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    # max jump before/after each smoothing call (is the dipole reforming?)
    mjb = [float(d["max_jump_before"]) for d in dumps]
    mja = [float(d["max_jump_after"]) for d in dumps]
    ax[0].plot(iters, mjb, "o-", label="max jump BEFORE smoothing")
    ax[0].plot(iters, mja, "s-", label="max jump AFTER smoothing")
    ax[0].axhline(150, color="gray", ls=":", label="threshold")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("max |ΔT| above RCB (K)")
    ax[0].set_title("Is the dipole reforming between smoothings?\n"
                    "(BEFORE staying high = retrieval rebuilds it)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # profiles right after each smoothing, colour by iteration
    cols = plt.cm.viridis(np.linspace(0, 1, len(dumps)))
    for c, d, it in zip(cols, dumps, iters):
        ax[1].semilogy(d["T_after"], P, color=c, lw=1, alpha=0.7)
    ax[1].semilogy(dumps[-1]["T_before"], P, "r--", lw=1.5,
                   label=f"before final smoothing (iter {iters[-1]})")
    ax[1].semilogy(dumps[-1]["T_after"], P, "k-", lw=2,
                   label="after final smoothing")
    ax[1].invert_yaxis(); ax[1].set_xlabel("T (K)"); ax[1].set_ylabel("P (Pa)")
    ax[1].set_title("Post-smoothing profiles (dark→late)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(diag / "smoothing_overview.png", dpi=90)
    plt.close()
    print("  wrote smoothing_overview.png")
    print(f"  max jump BEFORE smoothing: first {mjb[0]:.0f} K, last {mjb[-1]:.0f} K")
    if mjb[-1] > 0.6 * mjb[0]:
        print("  -> dipole is REFORMING between smoothings. Try SMOOTH_APRIORI_BLEND>0.")
    else:
        print("  -> jumps shrinking over time: smoothing is winning.")


if __name__ == "__main__":
    # also run smoothing analysis if those dumps exist
    try:
        analyze_smoothing(sys.argv[1])
    except Exception as _e:
        pass


# ---------------------------------------------------------------------------
# r47 addendum: skin-temperature dumps (retrieval_skin_iter*.npz)
# ---------------------------------------------------------------------------
def analyze_skin(outdir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    outdir = Path(outdir)
    files = sorted(outdir.glob("retrieval_skin_iter*.npz"))
    if not files:
        print("No retrieval_skin_iter*.npz (RADIATIVE_EQUILIBRIUM_CAP off?).")
        return
    diag = outdir / "diagnostics"; diag.mkdir(exist_ok=True)
    dumps = [np.load(f, allow_pickle=True) for f in files]
    iters = [int(f.stem.split("iter")[-1]) for f in files]
    P = dumps[0]["pressures"]
    print(f"\nLoaded {len(dumps)} skin-temperature dumps "
          f"(iter {iters[0]}..{iters[-1]})")

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: per-layer Jacobian sensitivity at selected iterations
    picks = [iters[0], iters[len(iters)//4], iters[len(iters)//2], iters[-1]]
    cols = plt.cm.viridis(np.linspace(0, 1, len(picks)))
    for c, it in zip(cols, picks):
        k = iters.index(it)
        s = np.array(dumps[k]["sensitivity"])
        s_norm = s / (s.max() if s.max() > 0 else 1.0)
        ax[0, 0].semilogx(s_norm + 1e-10, P, color=c, lw=1.5,
                          label=f"iter {it}")
    thr = float(dumps[-1]["threshold"]) / (max(float(dumps[-1]["sensitivity"].max()), 1e-30))
    ax[0, 0].axvline(thr, color="red", ls="--", alpha=0.7,
                     label=f"threshold ({thr:.2f})")
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_xlabel("Jacobian column sensitivity (normalized)")
    ax[0, 0].set_ylabel("P (Pa)")
    ax[0, 0].set_title("1. Where the OE has information\n(left of red = data-empty, pinned)")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    # Panel 2: number of data-empty layers over time
    n_emp = [int(d["n_affected"]) for d in dumps]
    ax[0, 1].plot(iters, n_emp, "b-", lw=1)
    ax[0, 1].set_xlabel("iteration"); ax[0, 1].set_ylabel("n data-empty layers")
    ax[0, 1].set_title("2. How much of the column r47 is holding")
    ax[0, 1].grid(alpha=0.3)

    # Panel 3: T at selected upper-atmosphere levels — does it converge to T_skin?
    n_lev = dumps[0]["T_after"].shape[0]
    T_skin = float(dumps[0]["T_skin"])
    for L in [40, 50, 60, 70, 80]:
        if L < n_lev:
            T_L = [float(d["T_after"][L]) for d in dumps]
            ax[1, 0].plot(iters, T_L, lw=1.0,
                          label=f"L={L} (P={P[L]:.1e})")
    ax[1, 0].axhline(T_skin, color="red", ls="--",
                     label=f"T_skin = {T_skin:.0f} K")
    ax[1, 0].set_xlabel("iteration"); ax[1, 0].set_ylabel("T after r47 (K)")
    ax[1, 0].set_title("3. Upper-atmosphere convergence to T_skin")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # Panel 4: max |ΔT| applied by r47 per iter
    mdt = [float(d["T_after"].max() - d["T_before"].max())  # crude proxy
           for d in dumps]
    actual_max_dT = [float(np.max(np.abs(np.array(d["T_after"]) - np.array(d["T_before"]))))
                     for d in dumps]
    ax[1, 1].plot(iters, actual_max_dT, "b-", lw=1)
    ax[1, 1].set_xlabel("iteration")
    ax[1, 1].set_ylabel("max |ΔT| applied by r47 (K)")
    ax[1, 1].set_title("4. r47 activity per iter\n(should decay toward 0)")
    ax[1, 1].grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(diag / "skin_overview.png", dpi=90)
    plt.close()
    print("  wrote skin_overview.png")
    print(f"\n=== r47 summary ===")
    print(f"  T_skin = {T_skin:.1f} K (= T_int·2^(-1/4) for non-irradiated)")
    print(f"  data-empty layers at end: {n_emp[-1]}")
    if n_lev > 60:
        print(f"  T(L=60) at end: {float(dumps[-1]['T_after'][60]):.0f} K  "
              f"(want ≈ {T_skin:.0f})")
    if actual_max_dT and actual_max_dT[-1] < 0.5:
        print(f"  CONVERGED — r47 making sub-Kelvin adjustments per iter at end.")
    elif actual_max_dT and actual_max_dT[-1] < 5:
        print(f"  near-converged ({actual_max_dT[-1]:.1f} K/iter at end).")
    else:
        print(f"  NOT CONVERGED yet ({actual_max_dT[-1]:.1f} K/iter at end) — "
              f"OE may be fighting the cap. Consider raising REQ_RELAX.")


# extend the main block
_orig_main = None
try:
    _orig_main = locals().get("main", None)
except Exception:
    pass

if __name__ == "__main__":
    try:
        analyze_skin(sys.argv[1])
    except Exception as _e:
        pass
