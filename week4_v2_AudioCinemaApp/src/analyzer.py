#!/usr/bin/env python3
from __future__ import annotations
import os, json
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

# ---------- Utilidades generales ----------
APP_NAME = "AudioCinema"

def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)

def normalize_peak(x: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(x)) if x.size else 0.0
    return x / (m + 1e-12) if m > 1.0 else x

def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return 20.0*np.log10(np.sqrt(np.mean(x**2) + 1e-20) + 1e-20)

def crest_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) + 1e-20
    rms = np.sqrt(np.mean(x**2) + 1e-20)
    return 20.0*np.log10(peak/(rms+1e-20))

# ---------- Welch sin SciPy ----------
def welch_psd_db(x: np.ndarray, fs: int, nperseg: int=4096, noverlap: Optional[int]=None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if noverlap is None: noverlap = nperseg//2
    step = max(1, nperseg - noverlap)
    if len(x) < nperseg:
        nperseg = max(16, len(x))
        if nperseg <= 16:
            f = np.linspace(0, fs/2, 32)
            return f, np.full_like(f, -120.0)
    win = np.hanning(nperseg)
    scale = (1.0/(fs*np.sum(win**2)))
    segs = []
    for start in range(0, len(x)-nperseg+1, step):
        seg = x[start:start+nperseg] * win
        X = np.fft.rfft(seg)
        Pxx = (np.abs(X)**2) * scale
        segs.append(Pxx)
    if not segs:
        f = np.fft.rfftfreq(nperseg, 1/fs)
        return f, np.full_like(f, -120.0)
    P = np.mean(segs, axis=0)
    P = np.maximum(P, 1e-30)
    f = np.fft.rfftfreq(nperseg, 1/fs)
    return f, 10.0*np.log10(P)

# ---------- Detección de beeps (HPF 1° orden + RMS corto) ----------
def hpf_first_order(x: np.ndarray, fs: int, fc: float=1000.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    dt = 1.0/fs; rc = 1.0/(2*np.pi*fc)
    a = rc/(rc+dt)
    y = np.zeros_like(x)
    for n in range(1, len(x)):
        y[n] = a*(y[n-1] + x[n] - x[n-1])
    return y

def short_time_rms(x: np.ndarray, fs: int, win_s: float=0.02, hop_s: float=0.01):
    win = max(1, int(round(win_s*fs)))
    hop = max(1, int(round(hop_s*fs)))
    n = len(x); frames = 1 + max(0, (n-win)//hop)
    r = np.zeros(frames, dtype=np.float32); t = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        s = i*hop; e = s+win
        seg = x[s:e]
        r[i] = np.sqrt(np.mean(seg**2) + 1e-20)
        t[i] = (s+win*0.5)/fs
    return t, r

def detect_beeps(x: np.ndarray, fs: int, use_hpf=True, cutoff_hz=1000.0, thr_db_over_med=10.0, min_sep_s=0.6) -> List[int]:
    y = hpf_first_order(x, fs, cutoff_hz) if use_hpf else x
    _, r = short_time_rms(y, fs, 0.02, 0.01)
    r_db = 20.0*np.log10(r + 1e-20)
    med = np.median(r_db); thr = med + float(thr_db_over_med)
    above = r_db > thr
    beeps = []
    i=0; n=len(above)
    while i<n:
        if above[i]:
            j=i
            while j<n and above[j]: j+=1
            k=i+int(np.argmax(r_db[i:j]))
            beeps.append(int(k))
            i=j
        else:
            i+=1
    times = np.array(beeps, dtype=float) * 0.01
    samples = (times*fs).astype(int)
    out=[]; last=-1e9
    for m in samples:
        t=m/fs
        if t-last >= float(min_sep_s):
            out.append(int(m)); last=t
    return out

def build_segments(x: np.ndarray, fs: int, markers: List[int], guard_ms: int=60, min_len_s: float=0.25) -> List[Tuple[int,int]]:
    if len(markers) < 2: return []
    guard = int(round(guard_ms*1e-3*fs))
    segs=[]
    for i in range(len(markers)-1):
        a=max(0, markers[i]+guard); b=max(0, markers[i+1]-guard)
        if b>a and (b-a)/fs >= min_len_s:
            segs.append((a,b))
    return segs

# ---------- Bandas / relativos ----------
BANDS = {"LFE": (30.0,100.0), "LF": (30.0,120.0), "MF": (120.0,2000.0), "HF": (2000.0,8000.0)}

def band_energy_db(f: np.ndarray, psd_db: np.ndarray, band: tuple) -> float:
    f1,f2 = band
    mask = (f>=f1) & (f<=f2)
    if not np.any(mask): return -120.0
    p_lin = 10.0**(psd_db[mask]/10.0)
    return 10.0*np.log10(np.mean(p_lin) + 1e-30)

def relative_spectrum_db(f_ref, psd_ref_db, f_x, psd_x_db):
    psd_x_i = np.interp(f_ref, f_x, psd_x_db)
    return f_ref, psd_x_i - psd_ref_db

def crop_same_length(x: np.ndarray, y: np.ndarray):
    n = min(len(x), len(y))
    return x[:n], y[:n]

# ---------- Análisis de par ----------
def analyze_pair(x_ref: np.ndarray, x_cur: np.ndarray, fs: int) -> dict:
    rms_ref, rms_cur = rms_db(x_ref), rms_db(x_cur)
    crest_ref, crest_cur = crest_db(x_ref), crest_db(x_cur)

    f_ref, psd_ref_db = welch_psd_db(x_ref, fs)
    f_cur, psd_cur_db = welch_psd_db(x_cur, fs)
    f_rel, rel_db     = relative_spectrum_db(f_ref, psd_ref_db, f_cur, psd_cur_db)

    bands_ref = {k: band_energy_db(f_ref, psd_ref_db, v) for k,v in BANDS.items()}
    bands_cur = {k: band_energy_db(f_cur, psd_cur_db, v) for k,v in BANDS.items()}

    diff_rms   = rms_cur - rms_ref
    diff_crest = crest_cur - crest_ref
    diff_bands = {k: (bands_cur[k]-bands_ref[k]) for k in BANDS}

    mask    = (f_rel>=50.0) & (f_rel<=8000.0)
    rel_abs = np.abs(rel_db[mask]) if np.any(mask) else np.abs(rel_db)
    spec95  = float(np.percentile(rel_abs, 95)) if rel_abs.size else 0.0

    dead_ch    = (rms_cur < (rms_ref - 10.0))
    band_fail  = any(abs(v) > 6.0 for v in diff_bands.values())
    crest_fail = abs(diff_crest) > 4.0
    spec_fail  = spec95 > 12.0
    rms_fail   = diff_rms < -10.0
    any_fail   = dead_ch or band_fail or crest_fail or spec_fail or rms_fail

    return {
        "fs": fs,
        "rms_ref": rms_ref, "rms_cur": rms_cur, "diff_rms": diff_rms,
        "crest_ref": crest_ref, "crest_cur": crest_cur, "diff_crest": diff_crest,
        "f_ref": f_ref, "psd_ref_db": psd_ref_db,
        "f_cur": f_cur, "psd_cur_db": psd_cur_db,
        "f_rel": f_rel, "rel_db": rel_db,
        "bands_ref": bands_ref, "bands_cur": bands_cur, "diff_bands": diff_bands,
        "spec_dev95": spec95,
        "dead_channel": dead_ch,
        "overall": ("PASSED" if not any_fail else "FAILED")
    }

# ---------- JSON resumen ----------
def _round(v, nd=3): return None if v is None else float(np.round(v, nd))

def summarize(res: dict) -> dict:
    return {
        "overall": res["overall"],
        "dead_channel": bool(res["dead_channel"]),
        "spec_dev95_db": _round(res["spec_dev95"]),
        "rms":   {"ref_db": _round(res["rms_ref"]),   "cin_db": _round(res["rms_cur"]),   "diff_db": _round(res["diff_rms"])},
        "crest": {"ref_db": _round(res["crest_ref"]), "cin_db": _round(res["crest_cur"]), "diff_db": _round(res["diff_crest"])},
        "bands_diff_db": {k: _round(v) for k,v in res["diff_bands"].items()},
        "bands_ref_db":  {k: _round(v) for k,v in res["bands_ref"].items()},
        "bands_cin_db":  {k: _round(v) for k,v in res["bands_cur"].items()},
    }

def build_json(fs: int, global_res: dict, ref_wav: str, cin_wav: str, test_name: str) -> dict:
    return {
        "app": APP_NAME,
        "version": "1.0",
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "test_name": test_name,
        "fs_hz": int(fs),
        "reference_file": ref_wav,
        "cinema_file": cin_wav,
        "summary": summarize(global_res)
    }

# ---------- Grabación ----------
def pick_input_device(preferred_name_substr: Optional[str]) -> Optional[int]:
    try: devices = sd.query_devices()
    except Exception: return None
    if preferred_name_substr:
        s = preferred_name_substr.lower()
        for i,d in enumerate(devices):
            if s in str(d.get("name","")).lower() and d.get("max_input_channels",0) > 0: return i
    for i,d in enumerate(devices):
        if d.get("max_input_channels",0) > 0: return i
    return None

def record(duration_sec: float, fs: int, channels: int=1, device: Optional[int]=None) -> np.ndarray:
    kwargs=dict(samplerate=fs, channels=channels, dtype="float32")
    if device is not None: kwargs["device"]=device
    rec = sd.rec(int(duration_sec*fs), **kwargs); sd.wait()
    return to_mono(rec.squeeze())

def load_ref(path: str, target_fs: int) -> np.ndarray:
    x, fs = sf.read(path, dtype="float32", always_2d=False)
    x = to_mono(x)
    if fs != target_fs:
        # re-muestreo mínimo (FFT) para evitar SciPy
        ratio = target_fs / float(fs)
        n_new = int(round(len(x)*ratio))
        X = np.fft.rfft(x)
        pad = max(0, n_new - len(x))
        # fallback simple: interpolación temporal
        t_old = np.linspace(0, 1, len(x), endpoint=False)
        t_new = np.linspace(0, 1, n_new, endpoint=False)
        x = np.interp(t_new, t_old, x).astype(np.float32)
    return normalize_peak(x)
