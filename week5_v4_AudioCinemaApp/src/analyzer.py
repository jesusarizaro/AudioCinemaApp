#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import sounddevice as sd

# ===================== Utilidades de audio =====================

def normalize_mono(x: np.ndarray) -> np.ndarray:
    """Convierte a mono (promedio) y normaliza si el pico excede 1.0."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)
    m = np.max(np.abs(x)) if x.size else 0.0
    return x / (m + 1e-12) if m > 1.0 else x

def record_audio(duration_sec: float, fs: int = 48000, channels: int = 1,
                 device: Optional[int] = None) -> np.ndarray:
    duration_sec = max(0.5, float(duration_sec))
    kwargs = dict(samplerate=fs, channels=channels, dtype="float32")
    if device is not None:
        kwargs["device"] = device
    rec = sd.rec(int(duration_sec * fs), **kwargs)
    sd.wait()
    return normalize_mono(rec.squeeze())

def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return 20.0 * np.log10(np.sqrt(np.mean(x**2) + 1e-20) + 1e-20)

def crest_factor_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) + 1e-20
    rms  = np.sqrt(np.mean(x**2) + 1e-20)
    return 20.0 * np.log10(peak / (rms + 1e-20))

# ===================== PSD tipo Welch (sin SciPy) =====================

def _frame_signal(x: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    """Divide en ventanas con solape."""
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap debe ser menor a nperseg")
    n = len(x)
    nwin = 1 + max(0, (n - nperseg) // step)
    if nwin <= 0:
        return np.empty((0, nperseg), dtype=np.float32)
    out = np.empty((nwin, nperseg), dtype=np.float32)
    k = 0
    for i in range(nwin):
        s = i * step
        out[i, :] = x[s:s+nperseg]
        if len(out[i, :]) < nperseg:
            # zero-pad
            pad = np.zeros(nperseg, dtype=np.float32)
            pad[:len(x)-s] = x[s:]
            out[i, :] = pad
        k += 1
    return out

def welch_db(x: np.ndarray, fs: int, nperseg: int = 4096, window: str = "hann"):
    """PSD en dB/Hz estilo Welch usando numpy."""
    x = np.asarray(x, dtype=np.float32)
    noverlap = nperseg // 2
    frames = _frame_signal(x, nperseg, noverlap)
    if frames.size == 0:
        f = np.linspace(0, fs/2, nperseg//2 + 1)
        return f, np.full_like(f, -300.0, dtype=np.float32)

    if window == "hann":
        win = np.hanning(nperseg).astype(np.float32)
    else:
        win = np.ones(nperseg, dtype=np.float32)

    U = (win**2).sum()  # normalización de potencia
    frames = (frames * win[None, :]).astype(np.float32)

    # FFT real
    X = np.fft.rfft(frames, n=nperseg, axis=1)
    Pxx = (np.abs(X)**2) / (fs * U)
    Pxx = Pxx.mean(axis=0)  # promedio de ventanas

    f = np.fft.rfftfreq(nperseg, d=1.0/fs)
    Pxx = np.maximum(Pxx, 1e-30)
    Pxx_db = 10.0 * np.log10(Pxx)
    return f.astype(np.float32), Pxx_db.astype(np.float32)

# ===================== Filtro pasa-altos simple (sin SciPy) =====================

def highpass_first_order(x: np.ndarray, fs: int, cutoff: float = 1000.0) -> np.ndarray:
    """
    Pasa-altos de 1er orden (dif. recursiva): y[n] = a*(y[n-1] + x[n] - x[n-1])
    a = RC/(RC + dt), RC = 1/(2*pi*fc)
    """
    x = np.asarray(x, dtype=np.float32)
    dt = 1.0 / fs
    RC = 1.0 / (2.0 * np.pi * max(1.0, float(cutoff)))
    a = RC / (RC + dt)
    y = np.empty_like(x)
    y_prev = 0.0
    x_prev = 0.0
    for n in range(len(x)):
        y_n = a * (y_prev + x[n] - x_prev)
        y[n] = y_n
        y_prev = y_n
        x_prev = x[n]
    return y

# ===================== Detección de beeps y segmentos =====================

def short_time_rms(x: np.ndarray, fs: int, win_s: float = 0.02, hop_s: float = 0.01):
    win = max(1, int(round(win_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    n = len(x)
    frames = 1 + max(0, (n - win) // hop)
    rms_vals = np.zeros(frames, dtype=np.float32)
    times = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        s = i * hop
        e = s + win
        seg = x[s:e]
        rms_vals[i] = np.sqrt(np.mean(seg**2) + 1e-20)
        times[i] = (s + win/2) / fs
    return times, rms_vals

def detect_beeps(x: np.ndarray, fs: int, use_hpf: bool = True, cutoff_hz: float = 1000.0,
                 thr_db_over_median: float = 10.0, min_sep_s: float = 0.6) -> List[int]:
    y = highpass_first_order(x, fs, cutoff_hz) if use_hpf else x
    _, r = short_time_rms(y, fs, 0.02, 0.01)
    r_db = 20.0 * np.log10(r + 1e-20)
    med = np.median(r_db)
    thr = med + float(thr_db_over_median)
    above = r_db > thr
    beeps_frames = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            k = i + int(np.argmax(r_db[i:j]))
            beeps_frames.append(k)
            i = j
        else:
            i += 1
    markers_s = np.array(beeps_frames, dtype=float) * 0.01
    markers = (markers_s * fs).astype(int)
    markers.sort()
    final = []
    last = -1e9
    for m in markers:
        t = m / fs
        if t - last >= float(min_sep_s):
            final.append(int(m))
            last = t
    return final

def build_segments(x: np.ndarray, fs: int, markers: List[int], guard_ms: int = 60,
                   min_len_s: float = 0.25) -> List[Tuple[int,int]]:
    if len(markers) < 2:
        return []
    guard = int(round(guard_ms * 1e-3 * fs))
    segs = []
    for i in range(len(markers) - 1):
        a = max(0, markers[i] + guard)
        b = max(0, markers[i+1] - guard)
        if b > a and (b - a)/fs >= min_len_s:
            segs.append((a, b))
    return segs

def crop_same_length(x: np.ndarray, y: np.ndarray):
    n = min(len(x), len(y))
    return x[:n], y[:n]

# ===================== Análisis =====================

BANDS = {
    "LFE": (30.0, 100.0),
    "LF":  (30.0, 120.0),
    "MF":  (120.0, 2000.0),
    "HF":  (2000.0, 8000.0),
}

def band_energy_db(f: np.ndarray, psd_db: np.ndarray, band: tuple) -> float:
    f1, f2 = band
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return -120.0
    p_lin = 10 ** (psd_db[mask] / 10.0)
    return 10.0 * np.log10(np.mean(p_lin) + 1e-30)

def relative_spectrum_db(f_ref, psd_ref_db, f_x, psd_x_db):
    psd_x_i = np.interp(f_ref, f_x, psd_x_db)
    return f_ref, psd_x_i - psd_ref_db

def analyze_pair(x_ref: np.ndarray, x_cur: np.ndarray, fs: int) -> dict:
    rms_ref, rms_cur = rms_db(x_ref), rms_db(x_cur)
    crest_ref, crest_cur = crest_factor_db(x_ref), crest_factor_db(x_cur)
    f_ref, psd_ref_db = welch_db(x_ref, fs)
    f_cur, psd_cur_db = welch_db(x_cur, fs)
    f_rel, rel_db = relative_spectrum_db(f_ref, psd_ref_db, f_cur, psd_cur_db)
    bands_ref = {k: band_energy_db(f_ref, psd_ref_db, v) for k, v in BANDS.items()}
    bands_cur = {k: band_energy_db(f_cur, psd_cur_db, v) for k, v in BANDS.items()}
    diff_rms = rms_cur - rms_ref
    diff_crest = crest_cur - crest_ref
    diff_bands = {k: (bands_cur[k] - bands_ref[k]) for k in BANDS}
    dead_channel = (rms_cur < (rms_ref - 10.0))
    mask = (f_rel >= 50.0) & (f_rel <= 8000.0)
    rel_abs = np.abs(rel_db[mask]) if np.any(mask) else np.abs(rel_db)
    spec_dev95 = float(np.percentile(rel_abs, 95)) if rel_abs.size else 0.0
    band_fail  = any(abs(v) > 6.0 for v in diff_bands.values())
    crest_fail = abs(diff_crest) > 4.0
    spec_fail  = spec_dev95 > 12.0
    rms_fail   = diff_rms < -10.0
    any_fail = dead_channel or band_fail or crest_fail or spec_fail or rms_fail
    return {
        "fs": fs,
        "rms_ref": rms_ref, "rms_cur": rms_cur,
        "crest_ref": crest_ref, "crest_cur": crest_cur,
        "diff_rms": diff_rms, "diff_crest": diff_crest,
        "bands_ref": bands_ref, "bands_cur": bands_cur, "diff_bands": diff_bands,
        "f_ref": f_ref, "psd_ref_db": psd_ref_db,
        "f_cur": f_cur, "psd_cur_db": psd_cur_db,
        "f_rel": f_rel, "rel_db": rel_db,
        "spec_dev95": spec_dev95,
        "dead_channel": dead_channel,
        "overall": ("PASSED" if not any_fail else "FAILED"),
    }

# ===================== JSON =====================

def _round(v, nd=3):
    return None if v is None else float(np.round(v, nd))

def _summarize_result(res: dict) -> dict:
    return {
        "overall": res["overall"],
        "dead_channel": bool(res["dead_channel"]),
        "spec_dev95_db": _round(res["spec_dev95"]),
        "rms": {"ref_db": _round(res["rms_ref"]),
                "cin_db": _round(res["rms_cur"]),
                "diff_db": _round(res["diff_rms"])},
        "crest": {"ref_db": _round(res["crest_ref"]),
                  "cin_db": _round(res["crest_cur"]),
                  "diff_db": _round(res["diff_crest"])},
        "bands_diff_db": {k: _round(v) for k, v in res["diff_bands"].items()},
        "bands_ref_db": {k: _round(v) for k, v in res["bands_ref"].items()},
        "bands_cin_db": {k: _round(v) for k, v in res["bands_cur"].items()},
    }

def build_json_payload(fs: int, global_result: dict | None, channel_results: List[dict],
                       ref_markers: List[int], cur_markers: List[int],
                       ref_segments: List[Tuple[int,int]], cur_segments: List[Tuple[int,int]],
                       ref_wav: str | None, cin_wav: str | None) -> dict:
    def markers_to_s(mks: List[int]) -> List[float]:
        return [float(np.round(m/fs, 6)) for m in mks]
    def segs_to_s(segs: List[Tuple[int,int]]) -> List[dict]:
        out = []
        for a, b in segs:
            s = a / fs; e = b / fs
            out.append({"start_s": float(np.round(s, 6)),
                        "end_s": float(np.round(e, 6)),
                        "dur_s": float(np.round(e - s, 6))})
        return out
    return {
        "app": "AudioCinema",
        "version": "2.0",
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "fs_hz": int(fs),
        "reference_file": ref_wav,
        "cinema_file": cin_wav,
        "beeps": {
            "reference": {"count": len(ref_markers), "markers_s": markers_to_s(ref_markers), "segments": segs_to_s(ref_segments)},
            "cinema":    {"count": len(cur_markers), "markers_s": markers_to_s(cur_markers), "segments": segs_to_s(cur_segments)},
        },
        "summary": (_summarize_result(global_result) if global_result else None),
        "channels": [{"index": i+1, **_summarize_result(cr)} for i, cr in enumerate(channel_results)],
        "channels_detected": min(len(ref_segments), len(cur_segments)),
    }
