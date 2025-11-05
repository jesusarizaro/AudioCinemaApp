#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# GUI
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Matplotlib (forzamos TkAgg para compatibilidad)
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Audio / análisis
import numpy as np
import sounddevice as sd
import soundfile as sf

from app_platform import APP_DIR, ensure_dirs
from configio import load_config, save_config
from analyzer import (
    normalize_mono, record_audio, analyze_pair, detect_beeps, build_segments,
    crop_same_length, build_json_payload, BANDS, welch_db
)
from iot_tb import send_json_to_thingsboard

APP_NAME = "AudioCinema"
SAVE_DIR = (APP_DIR / "data" / "captures").absolute()
EXPORT_DIR = (APP_DIR / "data" / "reports").absolute()
ASSETS_DIR = (APP_DIR / "assets").absolute()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Preferencias de audio -------- #
PREFERRED_INPUT_NAME = os.environ.get("AUDIOCINEMA_INPUT_NAME", "")
ENV_INPUT_INDEX = os.environ.get("AUDIOCINEMA_INPUT_INDEX")

# --------------------------- Utilidades internas --------------------------- #
def pick_input_device(preferred_name_substr: Optional[str] = None) -> Optional[int]:
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

def save_wav(x: np.ndarray, fs: int, stem: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SAVE_DIR / f"{stem}_{ts}.wav"
    sf.write(path, x, fs)
    return str(path)

# --------------------------- GUI --------------------------- #
class AudioCinemaGUI:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title(APP_NAME)

        # --- Icono consistente en ventana (barra superior) ---
        icon_png = ASSETS_DIR / "audiocinema.png"
        try:
            # Sugerido por Tk 8.6: iconphoto con PNG
            self._icon_img = tk.PhotoImage(file=str(icon_png))
            self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass
        # Establecer WM_CLASS para gestores de ventana de Raspberry Pi
        try:
            self.root.wm_class(APP_NAME, APP_NAME)
        except Exception:
            pass

        # --- Tema claro con fondo gris ---
        # 'flatly' es claro; forzamos un frame de fondo gris para la app
        self.root.style = tb.Style(theme="flatly")
        self.root.configure(bg="#e6e6e6")
        self.style = ttk.Style()
        self.style.configure("Gray.TFrame", background="#e6e6e6")
        self.style.configure("Card.TFrame", background="#f5f5f5")
        self.style.configure("H1.TLabel", font=("Segoe UI", 16, "bold"))

        ensure_dirs()
        self.cfg = load_config()

        # Variables básicas
        self.fs = tk.IntVar(value=int(self.cfg["audio"]["fs"]))
        self.duration = tk.DoubleVar(value=float(self.cfg["audio"]["duration_s"]))
        self.use_hpf = tk.BooleanVar(value=True)
        self.hpf_cut = tk.DoubleVar(value=1000.0)
        self.beep_thr = tk.DoubleVar(value=10.0)
        self.beep_minsep = tk.DoubleVar(value=0.6)
        self.beep_guard = tk.IntVar(value=60)

        # Estado de corrida
        self.x_ref: Optional[np.ndarray] = None
        self.x_cur: Optional[np.ndarray] = None
        self.ref_last_wav: Optional[str] = None
        self.cin_last_wav: Optional[str] = None
        self.global_result: Optional[dict] = None
        self.channel_results: List[dict] = []
        self.ref_markers: List[int] = []
        self.cur_markers: List[int] = []
        self.ref_segments: List[Tuple[int,int]] = []
        self.cur_segments: List[Tuple[int,int]] = []
        self.input_device_index: Optional[int] = None

        # Nombre de prueba (solo lectura; se asigna al terminar)
        self.test_name = tk.StringVar(value="—")

        self._build_ui()
        self._auto_select_input_device()

    # ---------------- UI layout ----------------
    def _build_ui(self):
        # Contenedor raíz con fondo gris
        root_frame = ttk.Frame(self.root, style="Gray.TFrame", padding=10)
        root_frame.pack(fill=BOTH, expand=True)

        # Panel principal dividido (izq info/acciones, der gráficas)
        paned = ttk.Panedwindow(root_frame, orient=HORIZONTAL, style="Gray.TFrame")
        paned.pack(fill=BOTH, expand=True)

        # ----- Izquierda (tarjeta) -----
        left = ttk.Frame(paned, style="Gray.TFrame", padding=(8,8))
        paned.add(left, weight=1)

        card = ttk.Frame(left, style="Card.TFrame", padding=12)
        card.pack(fill=BOTH, expand=False)

        # Logo
        try:
            logo_img = tk.PhotoImage(file=str(ASSETS_DIR/"audiocinema.png"))
            self.logo_label = ttk.Label(card, image=logo_img, style="Card.TFrame")
            self.logo_label.image = logo_img
            self.logo_label.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0,10))
        except Exception:
            ttk.Label(card, text="[Logo]", style="H1.TLabel").grid(row=0, column=0, sticky="nw")

        # Título
        ttk.Label(card, text="AudioCinema", style="H1.TLabel").grid(row=0, column=1, sticky="w")

        # Estado + nombre de prueba (solo lectura)
        sub = ttk.Frame(card, style="Card.TFrame")
        sub.grid(row=1, column=1, sticky="w", pady=(4,6))

        ttk.Label(sub, text="Prueba:", style="Card.TFrame").grid(row=0, column=0, sticky="w")
        self.lbl_testname = ttk.Label(sub, textvariable=self.test_name, style="Card.TFrame")
        self.lbl_testname.grid(row=0, column=1, sticky="w", padx=(6,0))

        self.status_badge = tb.Label(sub, text="SIN ANÁLISIS",
                                     bootstyle=SECONDARY,
                                     font=("Segoe UI", 12, "bold"))
        self.status_badge.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6,2))

        # Controles básicos (fila superior)
        ctrl = ttk.Frame(card, style="Card.TFrame")
        ctrl.grid(row=2, column=0, columnspan=2, sticky="we", pady=(8,0))
        ctrl.columnconfigure(3, weight=1)

        ttk.Label(ctrl, text="Sample Rate:", style="Card.TFrame").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(ctrl, from_=16000, to=96000, increment=1000,
                    textvariable=self.fs, width=8).grid(row=0, column=1, sticky="w", padx=(6,18))

        ttk.Label(ctrl, text="Duración (s):", style="Card.TFrame").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(ctrl, from_=3.0, to=180.0, increment=1.0,
                    textvariable=self.duration, width=8).grid(row=0, column=3, sticky="w", padx=(6,0))

        # Botones principales
        btns = ttk.Frame(card, style="Card.TFrame")
        btns.grid(row=3, column=0, columnspan=2, sticky="we", pady=(10,0))
        for c in range(2):
            btns.columnconfigure(c, weight=1)

        self.btn_cfg   = tb.Button(btns, text="Configurar (Setup)", bootstyle=SECONDARY, command=self._popup_config)
        self.btn_run   = tb.Button(btns, text="Ejecutar ahora",      bootstyle=PRIMARY,   command=self._run_once)
        self.btn_doc   = tb.Button(btns, text="Doctor",              bootstyle=INFO,      command=self._doctor)
        self.btn_timer = tb.Button(btns, text="Instalar programación", bootstyle=SECONDARY, command=self._install_timer)

        self.btn_cfg.grid(row=0, column=0, sticky="we", padx=(0,6))
        self.btn_run.grid(row=0, column=1, sticky="we")
        self.btn_doc.grid(row=1, column=0, sticky="we", pady=(6,0), padx=(0,6))
        self.btn_timer.grid(row=1, column=1, sticky="we", pady=(6,0))

        # ----- Derecha (gráficas) -----
        right = ttk.Frame(paned, style="Gray.TFrame", padding=(8,8))
        paned.add(right, weight=3)

        graphs_card = ttk.Frame(right, style="Card.TFrame", padding=8)
        graphs_card.pack(fill=BOTH, expand=True)

        # 3 gráficas (quitamos RMS/Crest)
        self.fig = Figure(figsize=(7,6), dpi=100)
        self.ax1 = self.fig.add_subplot(3,1,1)  # PSD
        self.ax2 = self.fig.add_subplot(3,1,2)  # Relativo
        self.ax3 = self.fig.add_subplot(3,1,3)  # ΔBandas
        self.canvas = FigureCanvasTkAgg(self.fig, master=graphs_card)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.fig.tight_layout()

    def _auto_select_input_device(self):
        idx = pick_input_device(self.cfg["audio"].get("preferred_input_name",""))
        self.input_device_index = idx

    # ------------------- Acciones -------------------
    def _doctor(self):
        messagebox.showinfo(APP_NAME, "Revisión básica:\n- Config cargada\n- Carpetas listas\n- Lista para ejecutar")

    def _install_timer(self):
        script = APP_DIR / "install_systemd.sh"
        if script.exists():
            os.system(f'"{script}"')
            messagebox.showinfo(APP_NAME, "Programación instalada/actualizada.")
        else:
            messagebox.showwarning(APP_NAME, "No se encontró install_systemd.sh")

    def _run_once(self):
        """Ejecutar flujo completo: cargar ref, grabar actual, analizar, graficar, exportar y (opcional) ThingsBoard."""
        fs = int(self.fs.get())
        dur = float(self.duration.get())

        # Cargar referencia desde config
        ref_path = Path(self.cfg["reference"]["wav_path"])
        if not ref_path.exists():
            messagebox.showerror(APP_NAME, f"No existe archivo de referencia:\n{ref_path}")
            return
        try:
            x_ref, fs_ref = sf.read(ref_path, dtype="float32", always_2d=False)
            if x_ref.ndim == 2:
                x_ref = x_ref.mean(axis=1)
            x_ref = normalize_mono(x_ref)
            # si difiere el fs, re-sample simples (linear)
            if fs_ref != fs:
                n_new = int(round(len(x_ref) * fs / fs_ref))
                x_idx = np.linspace(0, 1, len(x_ref))
                new_idx = np.linspace(0, 1, n_new)
                x_ref = np.interp(new_idx, x_idx, x_ref).astype(np.float32)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"No se pudo leer referencia: {e}")
            return

        # Grabar actual
        try:
            x_cur = record_audio(dur, fs=fs, channels=1, device=self.input_device_index)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"Error grabando micrófono: {e}")
            return

        # Guardar capturas
        self.ref_last_wav = save_wav(x_ref, fs, "reference")
        self.cin_last_wav = save_wav(x_cur, fs, "cinema")

        # Analizar (global + canales por beeps)
        self.global_result = analyze_pair(x_ref, x_cur, fs)

        # Beeps → segmentos
        self.ref_markers = detect_beeps(x_ref, fs, bool(self.use_hpf.get()),
                                        float(self.hpf_cut.get()), float(self.beep_thr.get()), float(self.beep_minsep.get()))
        self.cur_markers = detect_beeps(x_cur, fs, bool(self.use_hpf.get()),
                                        float(self.hpf_cut.get()), float(self.beep_thr.get()), float(self.beep_minsep.get()))
        self.ref_segments = build_segments(x_ref, fs, self.ref_markers, int(self.beep_guard.get()))
        self.cur_segments = build_segments(x_cur, fs, self.cur_markers, int(self.beep_guard.get()))

        # (Solo global mostrado en la UI resumida)
        self._update_status(self.global_result)
        self._update_plots(self.global_result)

        # Generar nombre de prueba ahora (no editable)
        tn = datetime.now().strftime("Test_%Y-%m-%d_%H-%M-%S")
        self.test_name.set(tn)

        # Exportar JSON
        payload = build_json_payload(
            fs, self.global_result, [],  # channel_results vacío en UI simplificada
            self.ref_markers, self.cur_markers, self.ref_segments, self.cur_segments,
            self.ref_last_wav, self.cin_last_wav
        )
        out = EXPORT_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"No se pudo escribir JSON: {e}")
            return

        # Enviar a ThingsBoard si token está configurado
        tb = self.cfg["thingsboard"]
        tok = tb.get("token", "REEMPLAZA_TU_TOKEN")
        if tok and tok != "REEMPLAZA_TU_TOKEN":
            ok = send_json_to_thingsboard(payload, tb["host"], int(tb["port"]), tok, bool(tb["use_tls"]))
            if not ok:
                messagebox.showwarning(APP_NAME, "No se pudo enviar telemetría a ThingsBoard.")

        messagebox.showinfo(APP_NAME, f"Análisis terminado.\nNombre: {tn}\nJSON: {out}")

    # ------------------- Config popup -------------------
    def _popup_config(self):
        w = tk.Toplevel(self.root)
        w.title("Configuración")
        try:
            w.iconphoto(True, self._icon_img)
            w.wm_class(APP_NAME, APP_NAME)
        except Exception:
            pass
        frm = ttk.Frame(w, padding=10)
        frm.pack(fill=BOTH, expand=True)

        nb = ttk.Notebook(frm)
        nb.pack(fill=BOTH, expand=True)

        # --- Pestaña General ---
        g = ttk.Frame(nb)
        nb.add(g, text="General")

        ref_var = tk.StringVar(value=self.cfg["reference"]["wav_path"])
        oncal_var = tk.StringVar(value=self.cfg["oncalendar"])

        ttk.Label(g, text="Archivo de referencia (.wav):").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(g, textvariable=ref_var, width=50).grid(row=0, column=1, sticky="we", pady=(6,2))
        ttk.Label(g, text="OnCalendar (systemd):").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(g, textvariable=oncal_var, width=30).grid(row=1, column=1, sticky="w", pady=(6,2))

        # --- Pestaña Audio ---
        a = ttk.Frame(nb)
        nb.add(a, text="Audio")
        fs_var = tk.IntVar(value=int(self.cfg["audio"]["fs"]))
        dur_var = tk.DoubleVar(value=float(self.cfg["audio"]["duration_s"]))
        pref_in_var = tk.StringVar(value=self.cfg["audio"].get("preferred_input_name",""))

        ttk.Label(a, text="Sample Rate (Hz):").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=fs_var, width=10).grid(row=0, column=1, sticky="w", pady=(6,2))
        ttk.Label(a, text="Duración (s):").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=dur_var, width=10).grid(row=1, column=1, sticky="w", pady=(6,2))
        ttk.Label(a, text="Preferir dispositivo con nombre:").grid(row=2, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=pref_in_var, width=28).grid(row=2, column=1, sticky="w", pady=(6,2))

        # --- Pestaña ThingsBoard ---
        t = ttk.Frame(nb)
        nb.add(t, text="ThingsBoard")
        host_var = tk.StringVar(value=self.cfg["thingsboard"]["host"])
        port_var = tk.IntVar(value=int(self.cfg["thingsboard"]["port"]))
        tls_var  = tk.BooleanVar(value=bool(self.cfg["thingsboard"]["use_tls"]))
        token_var = tk.StringVar(value=self.cfg["thingsboard"]["token"])  # visible (sin show="*")

        ttk.Label(t, text="Host:").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=host_var, width=24).grid(row=0, column=1, sticky="w", pady=(6,2))
        ttk.Label(t, text="Port:").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=port_var, width=10).grid(row=1, column=1, sticky="w", pady=(6,2))
        ttk.Checkbutton(t, text="Usar TLS (8883)", variable=tls_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,2))
        ttk.Label(t, text="Token:").grid(row=3, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=token_var, width=40).grid(row=3, column=1, sticky="w", pady=(6,2))  # << visible

        # Botones guardar/cerrar
        btns = ttk.Frame(frm)
        btns.pack(fill=X, pady=(10,0))
        def on_save():
            self.cfg["reference"]["wav_path"] = ref_var.get().strip()
            self.cfg["oncalendar"] = oncal_var.get().strip()
            self.cfg["audio"]["fs"] = int(fs_var.get())
            self.cfg["audio"]["duration_s"] = float(dur_var.get())
            self.cfg["audio"]["preferred_input_name"] = pref_in_var.get().strip()
            self.cfg["thingsboard"]["host"] = host_var.get().strip()
            self.cfg["thingsboard"]["port"] = int(port_var.get())
            self.cfg["thingsboard"]["use_tls"] = bool(tls_var.get())
            self.cfg["thingsboard"]["token"] = token_var.get().strip()
            save_config(self.cfg)
            messagebox.showinfo(APP_NAME, "Configuración guardada.")
            w.destroy()
        tb.Button(btns, text="Guardar", bootstyle=PRIMARY, command=on_save).pack(side=RIGHT)
        tb.Button(btns, text="Cancelar", bootstyle=SECONDARY, command=w.destroy).pack(side=RIGHT, padx=(0,6))

    # ------------------- Plots y estado -------------------
    def _update_status(self, res: dict):
        style = SUCCESS if res["overall"] == "PASSED" else DANGER
        self.status_badge.config(text=f"OVERALL: {res['overall']}", bootstyle=style)

    def _update_plots(self, res: dict):
        # Limpia 3 ejes
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.clear()

        # PSD
        self.ax1.set_title("PSD (dB/Hz)")
        self.ax1.semilogx(res["f_ref"], res["psd_ref_db"], label="Ref")
        self.ax1.semilogx(res["f_cur"], res["psd_cur_db"], label="Cinema", alpha=0.9)
        self.ax1.set_xlabel("Hz"); self.ax1.set_ylabel("dB/Hz")
        self.ax1.grid(True, which='both', ls=':')
        self.ax1.legend()

        # Relativo
        self.ax2.set_title("Relativo (Cinema - Ref)")
        self.ax2.semilogx(res["f_rel"], res["rel_db"])
        self.ax2.axhline(0, ls='--', lw=1)
        self.ax2.axhline(6, ls=':', lw=1); self.ax2.axhline(-6, ls=':', lw=1)
        self.ax2.set_xlabel("Hz"); self.ax2.set_ylabel("dB")
        self.ax2.grid(True, which='both', ls=':')

        # Δ Bandas
        bands = list(BANDS.keys())
        diffs = [res["diff_bands"][k] for k in bands]
        self.ax3.set_title("Δ Energía por bandas (dB)")
        self.ax3.bar(bands, diffs)
        self.ax3.axhline(0, ls='--', lw=1)
        self.ax3.axhline(3, ls=':', lw=1); self.ax3.axhline(-3, ls=':', lw=1)
        self.ax3.set_ylabel("dB")
        self.ax3.grid(True, axis='y', ls=':')

        self.fig.tight_layout()
        self.canvas.draw_idle()

# --------------------------- main --------------------------- #
def main():
    root = tb.Window(themename="flatly")  # tema claro
    app = AudioCinemaGUI(root)
    root.geometry("980x680")
    root.minsize(820, 600)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
