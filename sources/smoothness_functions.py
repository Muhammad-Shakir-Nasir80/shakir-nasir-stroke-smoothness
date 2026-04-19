"""
smoothness_functions.py
=======================
Reusable functions for movement smoothness analysis.
Used by main.ipynb — place this file in the sources/ folder.

Author : Shakir nasir
Course : Python-R-Git — Engineering and Ergonomics of Physical Activity
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def butter_lowpass_filter(signal, cutoff=10, fs=100, order=4):
    """
    Apply a zero-phase Butterworth low-pass filter to a signal.

    Why Butterworth: maximally flat frequency response in the passband.
    Why zero-phase (filtfilt): avoids time-shifting the signal.
    Why 10 Hz: human arm movement energy is below 10 Hz (Winter, 2009).

    Parameters
    ----------
    signal : array-like   raw signal (e.g. acceleration)
    cutoff : float        cutoff frequency in Hz (default 10)
    fs     : int          sampling frequency in Hz (default 100)
    order  : int          filter order (default 4)

    Returns
    -------
    filtered : np.array   filtered signal, same length as input
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# SMOOTHNESS METRIC 1 — NVP (Number of Velocity Peaks)
# ─────────────────────────────────────────────────────────────────────────────

def compute_nvp(velocity, threshold_ratio=0.1):
    """
    Compute the Number of Velocity Peaks (NVP).

    A smooth healthy movement has exactly 1 velocity peak (bell-shaped curve).
    A stroke patient's fragmented movement has multiple peaks.
    More peaks = less smooth movement.

    Parameters
    ----------
    velocity       : np.array   velocity signal (already segmented)
    threshold_ratio: float      minimum peak height as fraction of max velocity
                                (default 0.1 = 10% of peak)

    Returns
    -------
    nvp : int   number of velocity peaks above threshold
    """
    if velocity is None or len(velocity) < 5:
        return np.nan

    # Find local maxima with NO height constraint first
    # Then filter by height after
    min_distance = max(int(len(velocity) * 0.1), 5)  # At least 10% of segment apart or 5 samples
    peaks, properties = find_peaks(velocity, distance=min_distance)
    
    if len(peaks) == 0:
        return 1  # At least 1 peak (the movement itself)
    
    # Now filter peaks by height threshold
    min_height = threshold_ratio * np.max(velocity)
    peak_heights = velocity[peaks]
    significant_peaks = peak_heights >= min_height
    
    nvp = np.sum(significant_peaks)
    
    # If no significant peaks, return 1 (smooth single-peak movement)
    if nvp == 0:
        nvp = 1
    
    return int(nvp)


# ─────────────────────────────────────────────────────────────────────────────
# SMOOTHNESS METRIC 2 — LDLJ (Log Dimensionless Jerk)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ldlj(velocity, dt):
    """
    Compute the Log Dimensionless Jerk (LDLJ).

    Formula:
        LDLJ = -log( (T^5 / D^2) * integral(jerk^2) * dt )

    Where:
        T    = movement duration (seconds)
        D    = movement amplitude = integral of velocity (displacement)
        jerk = first derivative of acceleration = second derivative of velocity

    The result is always negative. More negative = LESS smooth.

    Reference: Balasubramanian et al. (2012), J NeuroEng Rehabil.

    Parameters
    ----------
    velocity : np.array   velocity signal (already segmented)
    dt       : float      time step = 1/sampling_rate

    Returns
    -------
    ldlj : float   log dimensionless jerk value (negative number)
    """
    if velocity is None or len(velocity) < 5:
        return np.nan

    # Movement duration
    T = len(velocity) * dt

    # Movement amplitude (displacement = integral of velocity)
    D = np.sum(velocity) * dt
    if D == 0:
        return np.nan

    # Jerk = derivative of acceleration = second derivative of velocity
    # We use np.diff twice: velocity -> acceleration -> jerk
    acceleration = np.diff(velocity) / dt
    jerk         = np.diff(acceleration) / dt

    # Mean squared jerk (integral approximated by sum * dt)
    mean_sq_jerk = np.sum(jerk**2) * dt

    # Dimensionless jerk (normalized by T and D to remove units)
    dimensionless_jerk = (T**5 / D**2) * mean_sq_jerk

    # Log to compress the scale — result is always negative for valid signals
    if dimensionless_jerk <= 0:
        return np.nan

    ldlj = -np.log(dimensionless_jerk)

    return float(ldlj)
