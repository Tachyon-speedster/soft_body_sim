"""
Publication-style Force(Z) vs Time plotting, matched to the style of the
supervisor's real suture-pad readings (force_vs_time_both.png).

IMPORTANT: the real readings' deep ~-40N dip is the blade cutting all the
way through the pad and then pressing on the rigid table underneath --
that's a rig artifact, not suture-pad tissue resistance (see
force_sensor.find_tissue_only_region). This module therefore:
  - for REAL readings: plots the full trace for context, shades the
    post-cliff region red and labels it as excluded, and annotates the
    real tissue-only peak (small, ~-3 to -4N) rather than the table dip.
  - for the SIM trace: annotates the plain peak, since the sim never
    models probe-vs-table contact in the first place (see
    softbody_core.py's collide_capsule_sensed -- it only ever measures
    probe-vs-tissue-particle force), so there's no equivalent artifact
    to strip out there.

Usable two ways:
  1. From inside the running extension (UI_builder.py calls
     save_force_plot() when you click "Export Trace").
  2. Standalone, after the fact, to build a sim-vs-real comparison figure
     for your report -- run this file directly:

         python3 plot_export.py sim_trace.csv reading_1.txt reading_2.txt

     which saves sim_trace_vs_real.png next to the CSV.
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .force_sensor import load_real_reading, find_tissue_only_region
except ImportError:
    from force_sensor import load_real_reading, find_tissue_only_region


def _annotate_sim(ax, t, f, title):
    """Sim traces never contain the table-contact artifact by construction
    (see module docstring), so just annotate the plain peak/baseline."""
    baseline_mask = f > -3.0
    baseline = f[baseline_mask].mean() if baseline_mask.any() else f.mean()
    i_peak = int(np.argmin(f))
    ax.plot(t, f, linewidth=1.2)
    ax.axhline(baseline, linestyle="--", color="gray",
               label=f"baseline mean \u2248 {baseline:.2f} N")
    ax.annotate(
        f"peak {f[i_peak]:.1f} N\n@ t={t[i_peak]:.2f}s",
        xy=(t[i_peak], f[i_peak]),
        xytext=(t[i_peak] - 0.3 * max(t[-1] - t[0], 1e-6), f[i_peak] - 3),
        arrowprops=dict(arrowstyle="->"),
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force$_Z$ (N)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def _annotate_real(ax, t, f, title):
    """Real readings: shade/label the table-contact artifact and annotate
    the genuine tissue-only peak instead of the ~-40N table dip."""
    region = find_tissue_only_region(t, f)
    onset_idx = region["onset_idx"]

    ax.plot(t, f, linewidth=1.0, color="#1f77b4", alpha=0.45,
             label="full sensor reading")
    if onset_idx is not None and onset_idx < len(t) - 1:
        ax.axvspan(t[onset_idx], t[-1], color="red", alpha=0.08)
        ymin = min(f.min(), region["tissue_peak_n"] - 5)
        ax.text(0.5 * (t[onset_idx] + t[-1]), ymin,
                 "table contact\n(excluded)", ha="center", va="bottom",
                 fontsize=8, color="darkred")
        ax.plot(region["t_tissue"], region["f_tissue"], linewidth=1.8,
                 color="#1f77b4", label="tissue-only (real suture-pad signal)")

    ax.axhline(region["baseline_n"], linestyle="--", color="gray",
               label=f"baseline mean \u2248 {region['baseline_n']:.2f} N")
    ax.annotate(
        f"tissue peak {region['tissue_peak_n']:.1f} N\n@ t={region['tissue_peak_t']:.2f}s",
        xy=(region["tissue_peak_t"], region["tissue_peak_n"]),
        xytext=(region["tissue_peak_t"] - 0.3 * max(t[-1] - t[0], 1e-6),
                 region["tissue_peak_n"] - 4),
        arrowprops=dict(arrowstyle="->"),
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force$_Z$ (N)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return region


def save_force_plot(t, f, out_path, title="Simulated Force (Z) vs. Time"):
    """t, f: 1D arrays for a SIM trace. Saves a PNG at out_path."""
    t = np.asarray(t); f = np.asarray(f)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if t.size > 1:
        _annotate_sim(ax, t, f, title)
    else:
        ax.set_title(title + " (no data yet)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_sim_vs_real(sim_t, sim_f, real_readings, out_path,
                      sim_title="Simulated (Warp soft body)"):
    """real_readings: list of (label, t_array, f_array) for one or more
    real reference readings, stacked as subplots above the sim result.
    Real subplots shade/exclude the table-contact artifact automatically
    -- see _annotate_real."""
    n = 1 + len(real_readings)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    _annotate_sim(axes[0], np.asarray(sim_t), np.asarray(sim_f), sim_title)
    for ax, (label, t, f) in zip(axes[1:], real_readings):
        _annotate_real(ax, np.asarray(t), np.asarray(f), label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: plot_export.py sim_trace.csv [real_reading.txt ...]")
        raise SystemExit(1)

    sim_csv = sys.argv[1]
    sim_data = np.genfromtxt(sim_csv, delimiter=",", names=True)
    sim_t = sim_data["time_s"]
    sim_f = sim_data["force_z_n"]

    real_readings = []
    for path in sys.argv[2:]:
        t, f = load_real_reading(path)
        real_readings.append((f"Real: {path.split('/')[-1]}", t, f))

    out_path = sim_csv.rsplit(".", 1)[0] + "_vs_real.png"
    save_sim_vs_real(sim_t, sim_f, real_readings, out_path)
    print(f"saved {out_path}")
