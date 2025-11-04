#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np
import sounddevice as sd

def normalize_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)
    m = float(np.max(np.abs(x))) if x.size else 1.0
    if m > 1.0:
        x = x / (m + 1e-12)
    return x

def record_audio(duration_sec: float, fs: int = 48000, channels: int = 1, device: Optional[int] = None) -> np.ndarray:
    duration_sec = max(0.5, float(duration_sec))
    kw = dict(samplerate=int(fs), channels=int(channels), dtype="float32")
    if device is not None:
        kw["device"] = int(device)
    rec = sd.rec(int(duration_sec * fs), **kw)
    sd.wait()
    return normalize_mono(rec.squeeze())

def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return 20.0 * np.log10(np.sqrt(np.mean(x**2) + 1e-20) + 1e-20)

def crest_factor_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) + 1e-20
    rms = np.sqrt(np.mean(x**2) + 1e-20)
    return 20.0 * np.log10(peak / (rms + 1e-20))

def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)

def welch_db(x: np.ndarray, fs: int, nperseg: int = 4096, noverlap: Optional[int] = None):
    x = np.asarray(x, dtype=np.float64)
    fs = int(fs)
    n = len(x)
    if n <= 1:
        f = np.linspace(0, fs/2, 1)
        return f, np.full_like(f, -300.0, dtype=np.float64)
    nperseg = int(min(max(256, nperseg), n))
    if noverlap is None:
        noverlap = nperseg // 2
    step = max(1, nperseg - int(noverlap))
    w = _hann(nperseg); w2 = (w**2).sum(); scale = 1.0 / (w2 * fs)
    acc = None
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start+nperseg] * w
        spec = np.fft.rfft(seg)
        pxx = (np.abs(spec)**2) * scale * 2.0
        acc = pxx if acc is None else (acc + pxx)
    if acc is None:
        acc = np.zeros(nperseg//2 + 1)
    Pxx = np.maximum(acc / max(1, ((n - nperseg) // step + 1)), 1e-30)
    f = np.fft.rfftfreq(nperseg, 1.0/fs)
    return f, 10.0 * np.log10(Pxx)

BANDS = {"LFE": (30.0,100.0), "LF": (30.0,120.0), "MF": (120.0,2000.0), "HF": (2000.0,8000.0)}

def band_energy_db(f: np.ndarray, psd_db: np.ndarray, band: tuple) -> float:
    f1,f2 = band
    m = (f>=f1)&(f<=f2)
    if not np.any(m): return -120.0
    p = 10.0**(psd_db[m]/10.0)
    return 10.0*np.log10(np.mean(p)+1e-30)

def relative_spectrum_db(fr, psd_ref_db, fx, psd_x_db):
    px = np.interp(fr, fx, psd_x_db)
    return fr, px - psd_ref_db

def _one_pole_highpass(x: np.ndarray, fs: int, cutoff: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if cutoff <= 1.0: return x
    rc = 1.0 / (2.0*np.pi*float(cutoff))
    a = rc / (rc + 1.0/float(fs))
    y = np.zeros_like(x, dtype=np.float32)
    prev_x = x[0] if x.size else 0.0
    prev_y = 0.0
    for i in range(len(x)):
        xi = x[i]; yi = a*(prev_y + xi - prev_x)
        y[i] = yi; prev_y = yi; prev_x = xi
    return y

def short_time_rms(x: np.ndarray, fs: int, win_s=0.02, hop_s=0.01):
    win = max(1, int(round(win_s*fs)))
    hop = max(1, int(round(hop_s*fs)))
    n = len(x)
    frames = 1 + max(0,(n-win)//hop)
    rms = np.zeros(frames, dtype=np.float32)
    times = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        s = i*hop; e = s+win
        seg = x[s:e]
        rms[i] = np.sqrt(np.mean(seg**2)+1e-20)
        times[i] = (s + win/2)/fs
    return times, rms

def detect_beeps(x: np.ndarray, fs: int, use_hpf=True, cutoff_hz=1000.0, thr_db_over_median=10.0, min_sep_s=0.6) -> List[int]:
    y = _one_pole_highpass(x, fs, cutoff_hz) if use_hpf else x
    _, r = short_time_rms(y, fs, 0.02, 0.01)
    r_db = 20.0*np.log10(r+1e-20); med = float(np.median(r_db)); thr = med + float(thr_db_over_median)
    above = r_db > thr
    beeps = []; i=0; n=len(above)
    while i<n:
        if above[i]:
            j=i
            while j<n and above[j]: j+=1
            k = i + int(np.argmax(r_db[i:j])); beeps.append(k); i=j
        else:
            i+=1
    markers = (np.array(beeps, dtype=float)*0.01*fs).astype(int); markers.sort()
    final=[]; last=-1e9
    for m in markers:
        t=m/fs
        if t-last>=float(min_sep_s):
            final.append(int(m)); last=t
    return final

def build_segments(x: np.ndarray, fs: int, markers: List[int], guard_ms=60, min_len_s=0.25)->List[tuple]:
    if len(markers)<2: return []
    g = int(round(guard_ms*1e-3*fs))
    out=[]
    for i in range(len(markers)-1):
        a=max(0,markers[i]+g); b=max(0,markers[i+1]-g)
        if b>a and (b-a)/fs>=min_len_s:
            out.append((a,b))
    return out

def crop_same_length(x: np.ndarray, y: np.ndarray):
    n=min(len(x),len(y)); return x[:n], y[:n]

def analyze_pair(x_ref: np.ndarray, x_cur: np.ndarray, fs: int)->dict:
    x_ref = np.asarray(x_ref, dtype=np.float32)
    x_cur = np.asarray(x_cur, dtype=np.float32)
    fs = int(fs)
    rms_ref, rms_cur = rms_db(x_ref), rms_db(x_cur)
    crest_ref, crest_cur = crest_factor_db(x_ref), crest_factor_db(x_cur)
    f_ref, psd_ref_db = welch_db(x_ref, fs); f_cur, psd_cur_db = welch_db(x_cur, fs)
    f_rel, rel_db = relative_spectrum_db(f_ref, psd_ref_db, f_cur, psd_cur_db)
    bands_ref = {k: band_energy_db(f_ref, psd_ref_db, v) for k,v in BANDS.items()}
    bands_cur = {k: band_energy_db(f_cur, psd_cur_db, v) for k,v in BANDS.items()}
    diff_rms = rms_cur - rms_ref; diff_crest = crest_cur - crest_ref
    diff_bands = {k: (bands_cur[k]-bands_ref[k]) for k in BANDS}
    dead_channel = (rms_cur < (rms_ref - 10.0))
    mask = (f_rel>=50.0)&(f_rel<=8000.0); rel_abs = np.abs(rel_db[mask]) if np.any(mask) else np.abs(rel_db)
    spec_dev95 = float(np.percentile(rel_abs,95)) if rel_abs.size else 0.0
    band_fail = any(abs(v)>6.0 for v in diff_bands.values())
    crest_fail = abs(diff_crest)>4.0; spec_fail = spec_dev95>12.0; rms_fail = diff_rms < -10.0
    any_fail = dead_channel or band_fail or crest_fail or spec_fail or rms_fail
    return {
        "fs": fs, "rms_ref": float(rms_ref), "rms_cur": float(rms_cur),
        "crest_ref": float(crest_ref), "crest_cur": float(crest_cur),
        "diff_rms": float(diff_rms), "diff_crest": float(diff_crest),
        "bands_ref": {k:float(v) for k,v in bands_ref.items()},
        "bands_cur": {k:float(v) for k,v in bands_cur.items()},
        "diff_bands": {k:float(v) for k,v in diff_bands.items()},
        "f_ref": f_ref.astype(np.float32), "psd_ref_db": psd_ref_db.astype(np.float32),
        "f_cur": f_cur.astype(np.float32), "psd_cur_db": psd_cur_db.astype(np.float32),
        "f_rel": f_rel.astype(np.float32), "rel_db": np.asarray(rel_db, dtype=np.float32),
        "spec_dev95": float(spec_dev95), "dead_channel": bool(dead_channel),
        "overall": ("PASSED" if not any_fail else "FAILED")
    }

def _round(v, nd=3): return None if v is None else float(np.round(v, nd))

def _summarize_result(res: dict) -> dict:
    return {
        "overall": res["overall"],
        "dead_channel": bool(res["dead_channel"]),
        "spec_dev95_db": _round(res["spec_dev95"]),
        "rms": {"ref_db": _round(res["rms_ref"]), "cin_db": _round(res["rms_cur"]), "diff_db": _round(res["diff_rms"])},
        "crest": {"ref_db": _round(res["crest_ref"]), "cin_db": _round(res["crest_cur"]), "diff_db": _round(res["diff_crest"])},
        "bands_diff_db": {k:_round(v) for k,v in res["diff_bands"].items()},
        "bands_ref_db": {k:_round(v) for k,v in res["bands_ref"].items()},
        "bands_cin_db": {k:_round(v) for k,v in res["bands_cur"].items()},
    }

def build_json_payload(fs:int, global_result:dict|None, channel_results:List[dict],
                       ref_markers:List[int], cur_markers:List[int],
                       ref_segments:List[tuple], cur_segments:List[tuple],
                       ref_wav:str|None, cin_wav:str|None)->dict:
    def markers_to_s(mks): return [float(np.round(m/fs,6)) for m in mks]
    def segs_to_s(segs):
        out=[]
        for a,b in segs:
            s=a/fs; e=b/fs
            out.append({"start_s":float(np.round(s,6)),"end_s":float(np.round(e,6)),"dur_s":float(np.round(e-s,6))})
        return out
    return {
        "app":"AudioCinema","version":"2.0",
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "fs_hz": int(fs), "reference_file": ref_wav, "cinema_file": cin_wav,
        "beeps": {
            "reference":{"count":len(ref_markers),"markers_s":markers_to_s(ref_markers),"segments":segs_to_s(ref_segments)},
            "cinema":   {"count":len(cur_markers),"markers_s":markers_to_s(cur_markers),"segments":segs_to_s(cur_segments)}
        },
        "summary": (_summarize_result(global_result) if global_result else None),
        "channels":[{"index":i+1, **_summarize_result(cr)} for i,cr in enumerate(channel_results)],
        "channels_detected": min(len(ref_segments),len(cur_segments))
    }
