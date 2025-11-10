#!/usr/bin/env python3
import time, math, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import welch

# Bandas de tercio de octava (centros aproximados Hz)
BANDS = np.array([31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000], dtype=np.float32)

def normalize_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim>1: x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)
    m = np.max(np.abs(x)) if x.size else 1.0
    return x/(m+1e-7)

def record_audio(duration_s: float, fs: int=44100, channels: int=1, device: Optional[int]=None) -> np.ndarray:
    duration_s = float(duration_s)
    assert duration_s>0 and fs>0
    rec = sd.rec(int(duration_s*fs), samplerate=fs, channels=channels, dtype='float32', device=device)
    sd.wait()
    if channels>1: rec = rec.mean(axis=1)
    return normalize_mono(rec.flatten())

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)+1e-12)))

def _crest(x: np.ndarray) -> float:
    r = _rms(x)
    p = float(np.max(np.abs(x)) + 1e-12)
    return float(20.0*math.log10(p/(r+1e-12)))

def welch_db(x: np.ndarray, fs: int) -> Tuple[np.ndarray,np.ndarray]:
    f, Pxx = welch(x, fs=fs, nperseg=min(4096, max(256, 2**int(np.floor(np.log2(len(x)))))))
    Pxx = np.maximum(Pxx, 1e-20)
    db = 10.0*np.log10(Pxx)
    return f.astype(np.float32), db.astype(np.float32)

def _band_levels_db(x: np.ndarray, fs: int) -> Dict[float,float]:
    f, db = welch_db(x, fs)
    out={}
    for c in BANDS:
        lo = c/(2**(1/6)); hi = c*(2**(1/6))
        m = (f>=lo)&(f<=hi)
        if not np.any(m):
            out[float(c)] = -120.0
        else:
            out[float(c)] = float(np.mean(db[m]))
    return out

def crop_same_length(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]

def detect_beeps(x: np.ndarray, fs: int, thr: float=0.6, min_gap_s: float=0.3) -> List[int]:
    # Picos de alta amplitud (beeps guía)
    x = np.abs(x)
    idx = np.where(x>=thr)[0]
    if idx.size==0: return []
    picks=[int(idx[0])]
    min_gap = int(min_gap_s*fs)
    last=idx[0]
    for i in idx[1:]:
        if i-last >= min_gap:
            picks.append(int(i))
            last = i
    return picks

def build_segments(x: np.ndarray, fs: int, markers: List[int], seg_len_s: float=1.0) -> List[Tuple[int,int]]:
    out=[]
    L=int(seg_len_s*fs)
    for m in markers:
        a=max(0, m-L//2); b=min(len(x), a+L)
        out.append((a,b))
    return out

def _compare_rms_crest(ref: np.ndarray, cur: np.ndarray) -> Dict[str,Any]:
    ref_rms=_rms(ref); cur_rms=_rms(cur)
    ref_crest=_crest(ref); cur_crest=_crest(cur)
    rms_diff_db = 20*math.log10((cur_rms+1e-9)/(ref_rms+1e-9))
    crest_diff = cur_crest - ref_crest
    return {
        "ref_rms": ref_rms, "cur_rms": cur_rms, "rms_diff_db": rms_diff_db,
        "ref_crest_db": ref_crest, "cur_crest_db": cur_crest, "crest_diff_db": crest_diff
    }

def _compare_bands(ref: np.ndarray, cur: np.ndarray, fs: int) -> Dict[str,Any]:
    ref_b = _band_levels_db(ref, fs); cur_b = _band_levels_db(cur, fs)
    diffs = {str(int(k)): float(cur_b[k]-ref_b[k]) for k in ref_b}
    return {"ref_bands_db": ref_b, "cur_bands_db": cur_b, "diff_bands_db": diffs}

def analyze_pair(x_ref: np.ndarray, x_cur: np.ndarray, fs: int) -> Dict[str,Any]:
    x_ref = normalize_mono(x_ref); x_cur = normalize_mono(x_cur)
    x_ref, x_cur = crop_same_length(x_ref, x_cur)
    r1=_compare_rms_crest(x_ref, x_cur)
    r2=_compare_bands(x_ref, x_cur, fs)

    # decisión simple: pasa si |rms_diff|<=3 dB y bandas dentro de ±6 dB (P95 simplificado)
    rms_ok = abs(r1["rms_diff_db"]) <= 3.0
    bands_ok = np.all(np.abs(list(r2["diff_bands_db"].values())) <= 6.0)
    overall = "PASSED" if (rms_ok and bands_ok) else "FAILED"

    return {"overall": overall, "rms_crest": r1, "bands": r2}

def build_json_payload(
    fs: int, result: Dict[str,Any],
    bands_order: List[float],
    ref_markers: List[int], cur_markers: List[int],
    ref_segments: List[Tuple[int,int]], cur_segments: List[Tuple[int,int]],
    ref_extra: Optional[Dict[str,Any]], cur_extra: Optional[Dict[str,Any]]
) -> Dict[str,Any]:
    return {
        "ts_utc": int(time.time()*1000),
        "fs": fs,
        "overall": result.get("overall","UNKNOWN"),
        "metrics": result,
        "markers": {"ref": ref_markers, "cur": cur_markers},
        "segments": {"ref": ref_segments, "cur": cur_segments},
        "extra": {"ref": ref_extra, "cur": cur_extra}
    }
