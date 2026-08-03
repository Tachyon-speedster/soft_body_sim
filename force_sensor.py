"""
Force-feedback sensor model for the Warp soft-body suture-pad simulation.

Physics basis
-------------
The soft body is solved with position-based dynamics (XPBD-style): contact
against a rigid collider (the probe/scalpel) is enforced by directly
projecting any penetrating particle back onto the collider surface each
substep (see collide_capsule_sensed in softbody_core.py). That projection
displacement, dx, is exactly the correction the constraint solver applied
to stop overlap over one substep of length h. Treating that correction as
though it came from a constant force acting over the substep --

    x = x0 + (F / m) * h^2   =>   F = m * dx / h^2

-- gives a dimensionally-correct, Newton-scale estimate of the contact
force on that particle. This is the standard "force recovery" identity
used in PBD/XPBD-based haptic-rendering work: you don't get a force
variable for free out of a position solver, but you can recover one from
how far the solver had to move something to satisfy the contact
constraint. Newton's third law then gives the reaction the pad exerts
back on the probe:

    F_probe = -sum_i F_particle_i

over every particle the probe capsule touched that substep. That's what
collide_capsule_sensed() accumulates, averaged across the substeps in one
step() call (see softbody_core.py).

What this IS and ISN'T
-----------------------
- IS a live force estimate driven by your actual mesh resolution, your
  per-layer tissue stiffness (k_edge/k_vol), and the probe's real
  simulated motion -- not a scripted or hand-drawn curve. Its *shape*
  (a gentle ramp as the blade presses through skin/fat/muscle, then
  back toward baseline on withdrawal) comes directly from the sim.
- ISN'T an absolute physical prediction out of the box: total_mass and
  k_edge/k_vol are XPBD convergence parameters tuned for numerical
  stability, not measured Pa or kg/m^3 of real tissue. So the RAW signal
  has the right shape and timing but an arbitrary absolute scale.
  CALIBRATION_GAIN is the one scalar you fit against your supervisor's
  real readings to fix that scale -- see calibrate_gain() below, and
  fit_gain_least_squares() for a slightly better multi-sample fit.

  IMPORTANT -- what "real" means here: in both supplied readings, the
  deep ~-40N dip is the blade cutting all the way through the pad and
  then pressing on the rigid table underneath it -- that's a rig
  artifact, not suture-pad tissue resistance. The real tissue-only
  signal is the gentle drift from baseline down to about -3 to -4N
  *before* that cliff. See find_tissue_only_region()/
  real_tissue_peak_from_reading() below -- calibrate against THAT peak,
  not the table dip.

This split (physically-derived shape + one fitted scale factor) is the
same pattern used any time a compliance/PBD sim is matched to a real F/T
sensor for haptics, so it's an honest thing to show your professor: the
maths connecting the simulation to Newtons is real, and the one thing
that's fitted (a single scalar gain) is clearly labeled as fitted.
"""

from __future__ import annotations

import csv
from collections import deque

import numpy as np


class ForceFeedbackSensor:
    """Turns the sim's raw recovered force into a sensor-like reading and
    keeps a rolling (time, force) history for plotting/export."""

    def __init__(
        self,
        calibration_gain: float = 1.0,
        tare_n: float = 0.0,
        noise_std_n: float = 0.0,
        smoothing_alpha: float = 0.5,
        history_seconds: float = 15.0,
        sample_hz: float = 60.0,
    ):
        # Fitted scale from raw recovered-Newtons -> the real sensor's
        # scale. Start at 1.0 and call calibrate_gain()/fit it yourself
        # (see bottom of this file) once you have one real puncture run.
        self.calibration_gain = calibration_gain
        # Baseline offset: real readings sit at a small negative value at
        # rest (tool weight / mounting bias on the real F/T sensor), not
        # exactly 0N -- see reading_1/reading_2 baseline means (~-1 to -2N).
        # Defaults to 0 here: this class is now used for the LIVE, in-sim
        # feedback signal, and a nonzero tare with no context just reads
        # as "phantom force when nothing is touching." Set tare_n=-1.5
        # explicitly (see for_report_matching() below) only when you
        # deliberately want the sim trace to visually match a real F/T
        # sensor's rest offset for a report figure.
        self.tare_n = tare_n
        # Sensor noise floor. Measured from the two supplied readings:
        # std ~= 0.46-0.58N on the resting baseline. Defaults to 0 here --
        # this noise was being added to EVERY reading unconditionally,
        # including the live in-viewport feedback used while actually
        # testing the probe. Since the real recovered contact signal at
        # this rod radius/mesh resolution is itself only ~0.1-0.5N
        # (similar order of magnitude to the old 0.5N noise std), that
        # noise was completely swamping the real touch signal: it made
        # the reading look random at rest AND made pressing into the pad
        # look like it "didn't do anything" once you're used to the
        # jitter. Only turn this back on (see for_report_matching()) once
        # you've confirmed the raw signal responds to touch and you're
        # building a final sim-vs-real comparison figure.
        self.noise_std_n = noise_std_n
        # EMA smoothing on top of the noise -- real F/T sensors are
        # internally filtered too; without this the raw PBD correction
        # is spikier frame-to-frame than either real reading. Bumped up
        # from 0.35 -> 0.5 as a fairer default now that noise is off by
        # default (0.35 was tuned to fight noise that no longer exists
        # here by default).
        self.smoothing_alpha = smoothing_alpha

        maxlen = int(history_seconds * sample_hz) + 32
        self._t_hist = deque(maxlen=maxlen)
        self._f_hist = deque(maxlen=maxlen)
        self._raw_hist = deque(maxlen=maxlen)
        self._t0 = None
        self._ema = None
        self._rng = np.random.default_rng()

    def reset(self):
        self._t_hist.clear()
        self._f_hist.clear()
        self._raw_hist.clear()
        self._t0 = None
        self._ema = None

    def record(self, sim_time: float, raw_force_z: float):
        """raw_force_z: the uncalibrated Newton-scale estimate from
        WarpSoftBodySim.get_probe_force_raw()[2] -- already real Newtons
        via the F = m*dx/h^2 recovery, just not yet scale-matched to a
        real sensor (see module docstring)."""
        if self._t0 is None:
            self._t0 = sim_time
        t = sim_time - self._t0

        calibrated = self.tare_n + self.calibration_gain * raw_force_z
        if self.noise_std_n > 0.0:
            calibrated += float(self._rng.normal(0.0, self.noise_std_n))

        if self._ema is None:
            self._ema = calibrated
        else:
            a = self.smoothing_alpha
            self._ema = a * calibrated + (1.0 - a) * self._ema

        self._t_hist.append(t)
        self._f_hist.append(self._ema)
        self._raw_hist.append(raw_force_z)
        return t, self._ema

    def history_arrays(self):
        return np.array(self._t_hist), np.array(self._f_hist)

    def raw_history_arrays(self):
        return np.array(self._t_hist), np.array(self._raw_hist)

    def latest(self):
        if not self._f_hist:
            return None
        return self._t_hist[-1], self._f_hist[-1]

    def peak(self):
        """(peak_force_n, t_of_peak) -- most negative point, matching how
        the reference plots annotate their puncture peak."""
        if not self._f_hist:
            return None
        f = np.array(self._f_hist)
        i = int(np.argmin(f))
        return float(f[i]), float(np.array(self._t_hist)[i])

    def raw_peak(self):
        """(raw_peak_force_n, t_of_peak) -- peak of the UNCALIBRATED
        signal (before tare/gain/noise), i.e. straight out of
        WarpSoftBodySim.get_probe_force_raw(). This is the number you
        feed into calibrate_gain()/fit_gain_least_squares() as
        sim_raw_peak_n -- calibrating against peak() (the already-tared,
        already-gained value) would double-apply tare/gain."""
        if not self._raw_hist:
            return None
        r = np.array(self._raw_hist)
        i = int(np.argmin(r))
        return float(r[i]), float(np.array(self._t_hist)[i])

    @classmethod
    def for_report_matching(cls, **overrides):
        """Explicit opt-in to the OLD defaults (tare_n=-1.5,
        noise_std_n=0.5, smoothing_alpha=0.35) that make a sim trace look
        visually comparable to a real F/T sensor's noise floor and rest
        offset (see reading_1.txt/reading_2.txt). Use this only when
        building a final sim-vs-real comparison figure for your report --
        not for the live in-sim probe feedback, where this noise is
        larger than the real signal you're trying to see (see __init__
        docstring above)."""
        kwargs = dict(tare_n=-1.5, noise_std_n=0.5, smoothing_alpha=0.35)
        kwargs.update(overrides)
        return cls(**kwargs)

    def baseline_mean(self, threshold_n: float = -3.0):
        f = np.array(self._f_hist)
        if f.size == 0:
            return None
        mask = f > threshold_n
        if not mask.any():
            return float(f.mean())
        return float(f[mask].mean())

    def export_csv(self, path: str):
        t, f = self.history_arrays()
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s", "force_z_n"])
            for ti, fi in zip(t, f):
                w.writerow([f"{ti:.6f}", f"{fi:.6f}"])


def load_real_reading(path: str):
    """Parses the supervisor's reading files: rows of
    sec, nsec, unix_timestamp_float, force_z -- with a stray header
    (hardware_timestamp_sec,...) sometimes trailing the data. Returns
    (t_seconds_from_start, force_z_newtons)."""
    ts, fs = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or not line[0].lstrip("-").isdigit():
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            ts.append(float(parts[2]))
            fs.append(float(parts[3]))
    t = np.array(ts)
    f = np.array(fs)
    t -= t[0]
    return t, f


def find_tissue_only_region(t, f, smooth_win: int = 9, slope_thresh_n_per_s: float = -6.0,
                             sustain_samples: int = 8):
    """Splits a real reading into "real suture-pad resistance" (before the
    blade cuts all the way through) vs. a rig artifact: once the blade has
    fully penetrated, it presses directly on the rigid table underneath the
    pad, and the sensor's huge dip (~-40N) from that point on is the
    table's reaction, not the pad's. That portion is NOT real tissue force
    feedback and should be excluded from both calibration and any
    sim-vs-real comparison.

    Detects the onset of that dip as the first point where a smoothed
    version of the signal sustains a steep decline (default: < -6 N/s for
    at least `sustain_samples` samples in a row) -- smoothing + a sustain
    requirement is needed because raw sensor noise alone (std ~0.5N at
    125Hz) produces single-sample slopes far steeper than that by chance.

    Returns a dict with:
      onset_idx        -- index where the table-contact artifact begins
                           (None if no such cliff was found in this trace)
      t_tissue, f_tissue -- the trimmed, artifact-free prefix
      tissue_peak_n, tissue_peak_t -- the real suture-pad's own peak
                           resistance (small, e.g. ~-3 to -4N -- NOT the
                           ~-40N table dip)
      baseline_n        -- resting mean over the artifact-free prefix
    """
    t = np.asarray(t); f = np.asarray(f)
    if smooth_win > 1 and f.size >= smooth_win:
        kernel = np.ones(smooth_win) / smooth_win
        f_smooth = np.convolve(f, kernel, mode="same")
    else:
        f_smooth = f
    slope = np.gradient(f_smooth, t)
    steep = slope < slope_thresh_n_per_s

    onset_idx = None
    for i in range(len(steep) - sustain_samples):
        if steep[i:i + sustain_samples].all():
            onset_idx = i
            break

    if onset_idx is None:
        # No table-contact cliff detected -- whole trace is tissue-only.
        t_tissue, f_tissue = t, f
    else:
        t_tissue, f_tissue = t[:onset_idx + 1], f[:onset_idx + 1]

    i_peak = int(np.argmin(f_tissue)) if f_tissue.size else 0
    baseline_mask = f_tissue > -3.0
    baseline_n = float(f_tissue[baseline_mask].mean()) if baseline_mask.any() else (
        float(f_tissue.mean()) if f_tissue.size else 0.0)

    return {
        "onset_idx": onset_idx,
        "t_tissue": t_tissue,
        "f_tissue": f_tissue,
        "tissue_peak_n": float(f_tissue[i_peak]) if f_tissue.size else 0.0,
        "tissue_peak_t": float(t_tissue[i_peak]) if f_tissue.size else 0.0,
        "baseline_n": baseline_n,
    }


def real_tissue_peak_from_reading(path: str, **kwargs):
    """Convenience: load a reading file and return just
    (tissue_peak_n, baseline_n) -- the two numbers you actually want to
    calibrate the sim against, with the table-contact artifact excluded."""
    t, f = load_real_reading(path)
    region = find_tissue_only_region(t, f, **kwargs)
    return region["tissue_peak_n"], region["baseline_n"]


def calibrate_gain(sim_raw_peak_n: float, tare_n: float, real_peak_n: float = -4.0):
    """One-point gain fit: solve calibration_gain so that
    tare_n + gain*sim_raw_peak_n == real_peak_n.

    IMPORTANT: real_peak_n should be the TISSUE-ONLY peak (see
    find_tissue_only_region/real_tissue_peak_from_reading above), which is
    small (reading_1.txt: ~-4.2N, reading_2.txt: ~-3.5N) -- NOT the ~-40N
    dip, which is the probe hitting the rigid table once it has cut all
    the way through the pad, not a property of the suture-pad tissue
    itself. The default here (-4.0) reflects that; do not fit against the
    ~-40N number.

    Drive one full press-through-the-pad stroke with calibration_gain=1.0
    and noise/smoothing off, read ForceFeedbackSensor.peak() for the raw
    (pre-gain) peak, and pass it here."""
    denom = sim_raw_peak_n
    if abs(denom) < 1e-9:
        raise ValueError("raw peak is ~0 -- no contact signal to calibrate against")
    return (real_peak_n - tare_n) / denom


def fit_gain_least_squares(sim_raw_peaks, real_peaks):
    """Fit a single scalar gain across several (sim_raw_peak, real_peak)
    pairs (e.g. a few strokes at different press depths) by least
    squares instead of a single point -- more robust if your one-point
    fit above is noisy."""
    sim_raw_peaks = np.asarray(sim_raw_peaks, dtype=np.float64)
    real_peaks = np.asarray(real_peaks, dtype=np.float64)
    gain = float(np.dot(sim_raw_peaks, real_peaks) / np.dot(sim_raw_peaks, sim_raw_peaks))
    return gain
