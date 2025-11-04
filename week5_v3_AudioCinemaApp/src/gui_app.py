#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
import soundfile as sf

from app_platform import APP_DIR, ASSETS_DIR, ensure_dirs
from configio import load_config, save_config
from analyzer import (
    normalize_mono, record_audio, analyze_pair, detect_beeps, build_segments,
    build_json_payload, BANDS
)
from iot_tb import send_json_to_thingsboard

APP_NAME = "AudioCinema"
SAVE_DIR = (APP_DIR / "data" / "captures").absolute()
EXPORT_DIR = (APP_DIR / "data" / "reports").absolute()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_INPUT_NAME = os.environ.get("AUDIOCINEMA_INPUT_NAME","")
ENV_INPUT_INDEX = os.environ.get("AUDIOCINEMA_INPUT_INDEX")

def pick_input_device(preferred_name_substr: Optional[str] = None) -> Optional[int]:
    import sounddevice as sd
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    if ENV_INPUT_INDEX:
        try:
            idx = int(ENV_INPUT_INDEX)
            if 0 <= idx < len(devices) and devices[idx].get("max_input_channels",0) > 0:
                return idx
        except Exception:
            pass
    if preferred_name_substr:
        s = preferred_name_substr.lower()
        for i,d in enumerate(devices):
            if s in str(d.get("name","")).lower() and d.get("max_input_channels",0) > 0:
                return i
    for i,d in enumerate(devices):
        if d.get("max_input_channels",0) > 0:
            return i
    return None

class AudioCinemaGUI:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title(APP_NAME)

        # Tema claro y fondo gris
        tb.Style(theme="flatly")
        try:
            self.root.configure(bg="#e6e6e6")
        except Exception:
            pass

        # Icono y WM_CLASS
        try:
            self._icon_img = tk.PhotoImage(file=str(ASSETS_DIR / "audiocinema.png"))
            self.root.iconphoto(True, self._icon_img)
        except Exception:
            self._icon_img = None
        try:
            self.root.wm_class(APP_NAME, APP_NAME)
        except Exception:
            pass

        ensure_dirs()
        self.cfg = load_config()

        self.fs = tk.IntVar(value=int(self.cfg["audio"]["fs"]))
        self.duration = tk.DoubleVar(value=float(self.cfg["audio"]["duration_s"]))
        self.use_hpf = tk.BooleanVar(value=True)
        self.hpf_cut = tk.DoubleVar(value=1000.0)
        self.beep_thr = tk.DoubleVar(value=10.0)
        self.beep_minsep = tk.DoubleVar(value=0.6)
        self.beep_guard = tk.IntVar(value=60)

        self.global_result: Optional[dict] = None
        self.ref_markers: List[int] = []
        self.cur_markers: List[int] = []
        self.ref_segments: List[Tuple[int,int]] = []
        self.cur_segments: List[Tuple[int,int]] = []
        self.input_device_index: Optional[int] = None

        self.test_name = tk.StringVar(value="—")

        self._build_ui()
        self._auto_select_input_device()

    def _build_ui(self):
        # Frame raíz
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=BOTH, expand=True)

        paned = ttk.Panedwindow(root_frame, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True)

        # -------- Izquierda (card) --------
        left = ttk.Frame(paned, padding=(8,8))
        paned.add(left, weight=1)

        card = ttk.Frame(left, padding=12)
        card.pack(fill=BOTH, expand=False)

        # Logo
        if self._icon_img is not None:
            ttk.Label(card, image=self._icon_img).grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0,10))
        else:
            ttk.Label(card, text="[Logo]", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="nw")

        ttk.Label(card, text="AudioCinema", font=("Segoe UI", 16, "bold")).grid(row=0, column=1, sticky="w")

        sub = ttk.Frame(card)
        sub.grid(row=1, column=1, sticky="w", pady=(4,6))
        ttk.Label(sub, text="Prueba:").grid(row=0, column=0, sticky="w")
        ttk.Label(sub, textvariable=self.test_name).grid(row=0, column=1, sticky="w", padx=(6,0))
        self.status_badge = tb.Label(sub, text="SIN ANÁLISIS", bootstyle=SECONDARY, font=("Segoe UI", 12, "bold"))
        self.status_badge.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6,2))

        ctrl = ttk.Frame(card)
        ctrl.grid(row=2, column=0, columnspan=2, sticky="we", pady=(8,0))
        ctrl.columnconfigure(3, weight=1)
        ttk.Label(ctrl, text="Sample Rate:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(ctrl, from_=16000, to=96000, increment=1000, textvariable=self.fs, width=8).grid(row=0, column=1, sticky="w", padx=(6,18))
        ttk.Label(ctrl, text="Duración (s):").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(ctrl, from_=3.0, to=180.0, increment=1.0, textvariable=self.duration, width=8).grid(row=0, column=3, sticky="w", padx=(6,0))

        btns = ttk.Frame(card)
        btns.grid(row=3, column=0, columnspan=2, sticky="we", pady=(10,0))
        for c in range(2): btns.columnconfigure(c, weight=1)

        tb.Button(btns, text="Configurar (Setup)", bootstyle=SECONDARY, command=self._popup_config).grid(row=0, column=0, sticky="we", padx=(0,6))
        tb.Button(btns, text="Ejecutar ahora", bootstyle=PRIMARY, command=self._run_once).grid(row=0, column=1, sticky="we")
        tb.Button(btns, text="Doctor", bootstyle=INFO, command=self._doctor).grid(row=1, column=0, sticky="we", pady=(6,0), padx=(0,6))
        tb.Button(btns, text="Instalar programación", bootstyle=SECONDARY, command=self._install_timer).grid(row=1, column=1, sticky="we", pady=(6,0))

        # -------- Derecha (3 gráficas) --------
        right = ttk.Frame(paned, padding=(8,8))
        paned.add(right, weight=3)

        graphs_card = ttk.Frame(right, padding=8)
        graphs_card.pack(fill=BOTH, expand=True)

        self.fig = Figure(figsize=(7,6), dpi=100)
        self.ax1 = self.fig.add_subplot(3,1,1)  # PSD
        self.ax2 = self.fig.add_subplot(3,1,2)  # Relativo
        self.ax3 = self.fig.add_subplot(3,1,3)  # Δ Bandas
        self.canvas = FigureCanvasTkAgg(self.fig, master=graphs_card)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.fig.tight_layout()

    def _auto_select_input_device(self):
        self.input_device_index = pick_input_device(self.cfg["audio"].get("preferred_input_name",""))

    # ----------- Acciones -----------
    def _doctor(self):
        messagebox.showinfo(APP_NAME, "OK: Config/Carpetas listas. Puedes ejecutar una prueba.")

    def _install_timer(self):
        script = APP_DIR / "install_systemd.sh"
        if script.exists():
            os.system(f'"{script}"')
            messagebox.showinfo(APP_NAME, "Programación instalada/actualizada.")
        else:
            messagebox.showwarning(APP_NAME, "No se encontró install_systemd.sh")

    def _run_once(self):
        fs = int(self.fs.get()); dur = float(self.duration.get())

        # 1) cargar referencia
        ref_path = Path(self.cfg["reference"]["wav_path"])
        if not ref_path.exists():
            messagebox.showerror(APP_NAME, f"No existe archivo de referencia:\n{ref_path}"); return
        try:
            x_ref, fs_ref = sf.read(ref_path, dtype="float32", always_2d=False)
            if x_ref.ndim == 2: x_ref = x_ref.mean(axis=1)
            x_ref = normalize_mono(x_ref)
            if fs_ref != fs:
                n_new = int(round(len(x_ref) * fs / fs_ref))
                x_idx = np.linspace(0, 1, len(x_ref)); new_idx = np.linspace(0, 1, n_new)
                x_ref = np.interp(new_idx, x_idx, x_ref).astype(np.float32)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"No se pudo leer referencia: {e}"); return

        # 2) grabar actual
        try:
            x_cur = record_audio(dur, fs=fs, channels=1, device=self.input_device_index)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"Error grabando micrófono: {e}"); return

        # 3) analizar
        self.global_result = analyze_pair(x_ref, x_cur, fs)

        # 4) beeps → segments (para JSON)
        self.ref_markers = detect_beeps(x_ref, fs, bool(self.use_hpf.get()), float(self.hpf_cut.get()), float(self.beep_thr.get()), float(self.beep_minsep.get()))
        self.cur_markers = detect_beeps(x_cur, fs, bool(self.use_hpf.get()), float(self.hpf_cut.get()), float(self.beep_thr.get()), float(self.beep_minsep.get()))
        self.ref_segments = build_segments(x_ref, fs, self.ref_markers, int(self.beep_guard.get()))
        self.cur_segments = build_segments(x_cur, fs, self.cur_markers, int(self.beep_guard.get()))

        # 5) actualizar UI
        self._update_status(self.global_result)
        self._update_plots(self.global_result)
        self.test_name.set(datetime.now().strftime("Test_%Y-%m-%d_%H-%M-%S"))

        # 6) export JSON
        payload = build_json_payload(fs, self.global_result, [],
                                     self.ref_markers, self.cur_markers, self.ref_segments, self.cur_segments,
                                     None, None)
        out = EXPORT_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(out, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"No se pudo escribir JSON: {e}"); return

        # 7) enviar a TB (opcional)
        tb = self.cfg["thingsboard"]
        if tb.get("token") and tb["token"] != "REEMPLAZA_TU_TOKEN":
            send_json_to_thingsboard(payload, tb["host"], int(tb["port"]), tb["token"], bool(tb["use_tls"]))

        messagebox.showinfo(APP_NAME, f"Análisis terminado.\nJSON: {out}")

    def _popup_config(self):
        w = tk.Toplevel(self.root); w.title("Configuración")
        if self._icon_img is not None:
            try: w.iconphoto(True, self._icon_img); w.wm_class(APP_NAME, APP_NAME)
            except Exception: pass

        frm = ttk.Frame(w, padding=10); frm.pack(fill=BOTH, expand=True)
        nb = ttk.Notebook(frm); nb.pack(fill=BOTH, expand=True)

        # General
        g = ttk.Frame(nb); nb.add(g, text="General")
        ref_var = tk.StringVar(value=self.cfg["reference"]["wav_path"])
        oncal_var = tk.StringVar(value=self.cfg["oncalendar"])
        ttk.Label(g, text="Archivo de referencia (.wav):").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(g, textvariable=ref_var, width=50).grid(row=0, column=1, sticky="we", pady=(6,2))
        ttk.Label(g, text="OnCalendar (systemd):").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(g, textvariable=oncal_var, width=30).grid(row=1, column=1, sticky="w", pady=(6,2))

        # Audio
        a = ttk.Frame(nb); nb.add(a, text="Audio")
        fs_var = tk.IntVar(value=int(self.cfg["audio"]["fs"]))
        dur_var = tk.DoubleVar(value=float(self.cfg["audio"]["duration_s"]))
        pref_in = tk.StringVar(value=self.cfg["audio"].get("preferred_input_name",""))
        ttk.Label(a, text="Sample Rate (Hz):").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=fs_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(a, text="Duración (s):").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=dur_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(a, text="Preferir dispositivo:").grid(row=2, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=pref_in, width=28).grid(row=2, column=1, sticky="w")

        # ThingsBoard
        t = ttk.Frame(nb); nb.add(t, text="ThingsBoard")
        host_var = tk.StringVar(value=self.cfg["thingsboard"]["host"])
        port_var = tk.IntVar(value=int(self.cfg["thingsboard"]["port"]))
        tls_var  = tk.BooleanVar(value=bool(self.cfg["thingsboard"]["use_tls"]))
        token_var = tk.StringVar(value=self.cfg["thingsboard"]["token"])
        ttk.Label(t, text="Host:").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=host_var, width=24).grid(row=0, column=1, sticky="w")
        ttk.Label(t, text="Port:").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=port_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(t, text="Usar TLS (8883)", variable=tls_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,2))
        ttk.Label(t, text="Token:").grid(row=3, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=token_var, width=40).grid(row=3, column=1, sticky="w")

        # Botonera
        btns = ttk.Frame(frm); btns.pack(fill=X, pady=(10,0))
        def on_save():
            self.cfg["reference"]["wav_path"] = ref_var.get().strip()
            self.cfg["oncalendar"] = oncal_var.get().strip()
            self.cfg["audio"]["fs"] = int(fs_var.get())
            self.cfg["audio"]["duration_s"] = float(dur_var.get())
            self.cfg["audio"]["preferred_input_name"] = pref_in.get().strip()
            self.cfg["thingsboard"]["host"] = host_var.get().strip()
            self.cfg["thingsboard"]["port"] = int(port_var.get())
            self.cfg["thingsboard"]["use_tls"] = bool(tls_var.get())
            self.cfg["thingsboard"]["token"] = token_var.get().strip()
            save_config(self.cfg)
            messagebox.showinfo(APP_NAME, "Configuración guardada."); w.destroy()
        tb.Button(btns, text="Guardar", bootstyle=PRIMARY, command=on_save).pack(side=RIGHT)
        tb.Button(btns, text="Cancelar", bootstyle=SECONDARY, command=w.destroy).pack(side=RIGHT, padx=(0,6))

    def _update_status(self, res: dict):
        style = SUCCESS if res["overall"] == "PASSED" else DANGER
        self.status_badge.config(text=f"OVERALL: {res['overall']}", bootstyle=style)

    def _update_plots(self, res: dict):
        for ax in (self.ax1, self.ax2, self.ax3): ax.clear()
        self.ax1.set_title("PSD (dB/Hz)")
        self.ax1.semilogx(res["f_ref"], res["psd_ref_db"], label="Ref")
        self.ax1.semilogx(res["f_cur"], res["psd_cur_db"], label="Cinema", alpha=0.9)
        self.ax1.set_xlabel("Hz"); self.ax1.set_ylabel("dB/Hz"); self.ax1.grid(True, which='both', ls=':'); self.ax1.legend()

        self.ax2.set_title("Relativo (Cinema - Ref)")
        self.ax2.semilogx(res["f_rel"], res["rel_db"])
        self.ax2.axhline(0, ls='--', lw=1); self.ax2.axhline(6, ls=':', lw=1); self.ax2.axhline(-6, ls=':', lw=1)
        self.ax2.set_xlabel("Hz"); self.ax2.set_ylabel("dB"); self.ax2.grid(True, which='both', ls=':')

        bands = list(BANDS.keys()); diffs = [res["diff_bands"][k] for k in bands]
        self.ax3.set_title("Δ Energía por bandas (dB)")
        self.ax3.bar(bands, diffs); self.ax3.axhline(0, ls='--', lw=1); self.ax3.axhline(3, ls=':', lw=1); self.ax3.axhline(-3, ls=':', lw=1)
        self.ax3.set_ylabel("dB"); self.ax3.grid(True, axis='y', ls=':')

        self.fig.tight_layout(); self.canvas.draw_idle()

def main():
    root = tb.Window(themename="flatly")
    app = AudioCinemaGUI(root)
    root.geometry("980x680"); root.minsize(820,600)
    root.mainloop()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: pass
