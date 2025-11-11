#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app_platform import APP_DIR, ASSETS_DIR, ensure_dirs
from configio import load_config, save_config
from analyzer import (
    normalize_mono, record_audio, analyze_pair,
    detect_beeps, build_segments, crop_same_length,
    build_json_payload
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

INFO_TEXT = (
    "AudioCinema\n\n"
    "Esta aplicación graba, evalúa y compara una pista de PRUEBA con una "
    "pista de REFERENCIA.\n\n"
    "• Graba la pista de prueba con el micrófono.\n"
    "• Compara vs. la referencia (RMS, crest, bandas, espectro relativo, P95).\n"
    "• Muestra la forma de onda de ambas pistas.\n"
    "• Exporta un JSON y puede enviarlo a ThingsBoard.\n"
)

class AudioCinemaGUI:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title(APP_NAME)

        tb.Style(theme="flatly")
        try: self.root.configure(bg="#e6e6e6")
        except Exception: pass

        try:
            self._icon_img = tk.PhotoImage(file=str(ASSETS_DIR / "audiocinema.png"))
            self.root.iconphoto(True, self._icon_img)
            self.root.wm_class(APP_NAME, APP_NAME)
        except Exception:
            self._icon_img = None

        ensure_dirs()
        self.cfg = load_config()

        self.fs = tk.IntVar(value=int(self.cfg["audio"]["fs"]))
        self.duration = tk.DoubleVar(value=float(self.cfg["audio"]["duration_s"]))

        self.input_device_index: Optional[int] = None
        self.test_name = tk.StringVar(value="—")
        self.eval_text = tk.StringVar(value="—")

        self.last_ref: Optional[np.ndarray] = None
        self.last_cur: Optional[np.ndarray] = None
        self.last_fs: int = int(self.fs.get())

        self.ref_markers: List[int] = []
        self.cur_markers: List[int] = []
        self.ref_segments: List[Tuple[int,int]] = []
        self.cur_segments: List[Tuple[int,int]] = []

        self._build_ui()
        self._auto_select_input_device()

    # --------------------- UI ---------------------
    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=8)
        root_frame.pack(fill=BOTH, expand=True)

        paned = ttk.Panedwindow(root_frame, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True)

        # izquierda
        left = ttk.Frame(paned, padding=(6,6)); paned.add(left, weight=1)
        card = ttk.Frame(left, padding=6); card.pack(fill=Y, expand=False)

        if self._icon_img is not None:
            ttk.Label(card, image=self._icon_img).pack(anchor="n", pady=(0,4))
        ttk.Label(card, text="AudioCinema", font=("Segoe UI", 18, "bold")).pack(anchor="n")

        desc = "Graba, evalúa y analiza tu sistema de audio."
        ttk.Label(card, text=desc, wraplength=220, justify="center").pack(anchor="n", pady=(6,10))

        btn_style = {"bootstyle": PRIMARY, "width": 20}
        tb.Button(card, text="Información",   command=self._show_info, **btn_style).pack(pady=6, fill=X)
        tb.Button(card, text="Configuración", command=self._open_config, **btn_style).pack(pady=6, fill=X)
        tb.Button(card, text="Confirmación",  command=self._popup_confirm, **btn_style).pack(pady=6, fill=X)
        tb.Button(card, text="Prueba ahora",  command=self._run_once, **btn_style).pack(pady=(6,0), fill=X)

        paned.add(ttk.Separator(root_frame, orient=VERTICAL))

        # derecha
        right = ttk.Frame(paned, padding=(8,6)); paned.add(right, weight=4)
        header = ttk.Frame(right); header.pack(fill=X, pady=(0,8))
        ttk.Label(header, text="PRUEBA:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0,6))
        ttk.Entry(header, textvariable=self.test_name, width=32, state="readonly", justify="center").grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="EVALUACIÓN:", font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(0,6), pady=(6,0))
        self.eval_lbl = ttk.Label(header, textvariable=self.eval_text, font=("Segoe UI", 11, "bold"), foreground="#333333")
        self.eval_lbl.grid(row=1, column=1, sticky="w", pady=(6,0))

        fig_card = ttk.Frame(right, padding=4); fig_card.pack(fill=BOTH, expand=True)
        self.fig = Figure(figsize=(5,4), dpi=100)
        self.ax_ref = self.fig.add_subplot(2,1,1)
        self.ax_cur = self.fig.add_subplot(2,1,2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_card)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self._clear_waves(); self.fig.tight_layout()

        msg_card = ttk.Frame(right, padding=4); msg_card.pack(fill=BOTH, expand=False, pady=(6,0))
        ttk.Label(msg_card, text="Mensajes", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.msg_text = tk.Text(msg_card, height=6, wrap="word")
        self.msg_text.pack(fill=BOTH, expand=True)
        self._set_messages(["Listo. Presiona «Prueba ahora» para iniciar."])

    def _clear_waves(self):
        for ax, title in ((self.ax_ref, "Pista de referencia"), (self.ax_cur, "Pista de prueba")):
            ax.clear(); ax.set_title(title); ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Amplitud"); ax.grid(True, axis='x', ls=':')
        self.canvas.draw_idle()

    def _plot_wave(self, ax, x: np.ndarray, fs: int):
        n = len(x); 
        if n == 0: return
        t = np.arange(n, dtype=np.float32) / float(fs)
        ax.plot(t, x, linewidth=0.8); ax.set_xlim(0.0, float(t[-1]))

    def _set_eval(self, passed: Optional[bool]):
        if passed is None:
            self.eval_text.set("—"); self.eval_lbl.configure(foreground="#333333")
        elif passed:
            self.eval_text.set("PASSED"); self.eval_lbl.configure(foreground="#0d8a00")
        else:
            self.eval_text.set("FAILED"); self.eval_lbl.configure(foreground="#cc0000")

    def _set_messages(self, lines: List[str]):
        self.msg_text.delete("1.0", tk.END)
        for ln in lines: self.msg_text.insert(tk.END, "• " + ln + "\n")
        self.msg_text.see(tk.END)

    # acciones
    def _auto_select_input_device(self):
        self.input_device_index = pick_input_device(self.cfg["audio"].get("prefer_input_name",""))

    def _show_info(self):
        messagebox.showinfo(APP_NAME, INFO_TEXT)

    def _popup_confirm(self):
        tb_cfg = self.cfg["thingsboard"]
        txt = (
            f"Archivo de referencia:\n  {self.cfg['reference']['file']}\n\n"
            f"Audio:\n  fs={self.cfg['audio']['fs']}  duración={self.cfg['audio']['duration_s']} s\n"
            f"  preferir dispositivo='{self.cfg['audio'].get('prefer_input_name','')}'\n\n"
            f"ThingsBoard:\n  host={tb_cfg['host']}  port={tb_cfg['port']}  TLS={tb_cfg['use_tls']}\n"
            f"  token={tb_cfg['token']}\n\n"
            f"Programación (systemd):\n  {self.cfg['general']['oncalendar']}\n"
        )
        messagebox.showinfo("Confirmación", txt)

    def _open_config(self):
        dlg = ConfigDialog(self.root, self.cfg)
        self.root.wait_window(dlg)
        self.cfg = load_config()  # recargar desde disco

    # ejecución de prueba
    def _run_once(self):
        fs = int(self.fs.get()); dur = float(self.duration.get())
        ref_path = Path(self.cfg["reference"]["file"])
        if not ref_path.exists():
            messagebox.showerror(APP_NAME, f"No existe archivo de referencia:\n{ref_path}")
            return
        try:
            x_ref, fs_ref = sf.read(ref_path, dtype="float32", always_2d=False)
            if x_ref.ndim == 2: x_ref = x_ref.mean(axis=1)
            x_ref = normalize_mono(x_ref)
            if fs_ref != fs:
                n_new = int(round(len(x_ref) * fs / fs_ref))
                x_idx = np.linspace(0, 1, len(x_ref)); new_idx = np.linspace(0, 1, n_new)
                x_ref = np.interp(new_idx, x_idx, x_ref).astype(np.float32)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"No se pudo leer referencia: {e}")
            return

        try:
            x_cur = record_audio(dur, fs=fs, channels=1, device=self.input_device_index)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"Error grabando micrófono: {e}")
            return

        res = analyze_pair(x_ref, x_cur, fs)
        self._set_eval(res["overall"] == "PASSED")

        self.ref_markers = detect_beeps(x_ref, fs)
        self.cur_markers = detect_beeps(x_cur, fs)
        self.ref_segments = build_segments(x_ref, fs, self.ref_markers)
        self.cur_segments = build_segments(x_cur, fs, self.cur_markers)

        self.last_ref, self.last_cur = x_ref, x_cur
        self._clear_waves(); self._plot_wave(self.ax_ref, x_ref, fs); self._plot_wave(self.ax_cur, x_cur, fs); self.canvas.draw_idle()
        self.test_name.set(datetime.now().strftime("Test_%Y-%m-%d_%H-%M-%S"))

        payload = build_json_payload(fs, res, [], self.ref_markers, self.cur_markers,
                                     self.ref_segments, self.cur_segments, str(ref_path), None)
        out = EXPORT_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(out, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
            saved = True
        except Exception as e:
            self._set_messages([f"No se pudo guardar JSON: {e}"]); saved = False

        sent = False
        if saved:
            tb_cfg = self.cfg["thingsboard"]
            if tb_cfg.get("token"):
                sent = send_json_to_thingsboard(payload, tb_cfg["host"], int(tb_cfg["port"]), tb_cfg["token"], bool(tb_cfg["use_tls"]))

        lines = [
            "La prueba ha " + ("aprobado." if res["overall"] == "PASSED" else "fallado."),
            f"JSON: {out}",
            "Resultados enviados a ThingsBoard." if sent else "No se enviaron resultados a ThingsBoard."
        ]
        self._set_messages(lines)
        messagebox.showinfo(APP_NAME, f"Análisis terminado.\nJSON: {out}")

# -------- ConfigDialog --------
class ConfigDialog(tk.Toplevel):
    def __init__(self, master, cfg: dict):
        super().__init__(master)
        self.title("Configuración"); self.resizable(False, False); self.cfg = cfg
        nb = ttk.Notebook(self); nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # General
        f_gen = ttk.Frame(nb); nb.add(f_gen, text="General")
        self.var_oncal = tk.StringVar(value=self.cfg["general"].get("oncalendar","*-*-* 02:00:00"))
        ttk.Label(f_gen, text="OnCalendar (systemd):").grid(row=0, column=0, sticky="w", pady=(8,2))
        ttk.Entry(f_gen, textvariable=self.var_oncal, width=28).grid(row=0, column=1, sticky="w", padx=6, pady=(8,2))

        # Audio
        f_audio = ttk.Frame(nb); nb.add(f_audio, text="Audio")
        self.var_fs  = tk.IntVar(value=int(self.cfg["audio"].get("fs",48000)))
        self.var_dur = tk.DoubleVar(value=float(self.cfg["audio"].get("duration_s",10.0)))
        self.var_pref= tk.StringVar(value=str(self.cfg["audio"].get("prefer_input_name","")))
        ttk.Label(f_audio, text="Sample rate (Hz):").grid(row=0, column=0, sticky="w", pady=(8,2))
        ttk.Entry(f_audio, textvariable=self.var_fs, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=(8,2))
        ttk.Label(f_audio, text="Duración (s):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f_audio, textvariable=self.var_dur, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(f_audio, text="Preferir dispositivo con nombre:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(f_audio, textvariable=self.var_pref, width=28).grid(row=2, column=1, sticky="w", padx=6, pady=2)

        # ThingsBoard
        f_tb = ttk.Frame(nb); nb.add(f_tb, text="ThingsBoard")
        self.var_host  = tk.StringVar(value=self.cfg["thingsboard"].get("host","thingsboard.cloud"))
        self.var_port  = tk.IntVar(value=int(self.cfg["thingsboard"].get("port",1883)))
        self.var_tls   = tk.BooleanVar(value=bool(self.cfg["thingsboard"].get("use_tls",False)))
        self.var_token = tk.StringVar(value=self.cfg["thingsboard"].get("token",""))
        ttk.Label(f_tb, text="Host:").grid(row=0, column=0, sticky="w", pady=(8,2))
        ttk.Entry(f_tb, textvariable=self.var_host, width=28).grid(row=0, column=1, sticky="w", padx=6, pady=(8,2))
        ttk.Label(f_tb, text="Port:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f_tb, textvariable=self.var_port, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(f_tb, text="Usar TLS (8883)", variable=self.var_tls).grid(row=2, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(f_tb, text="Token:").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(f_tb, textvariable=self.var_token, width=40).grid(row=3, column=1, sticky="w", padx=6, pady=2)

        # Pista de referencia
        f_ref = ttk.Frame(nb); nb.add(f_ref, text="Pista de referencia")
        default_ref = str((ASSETS_DIR / "reference_master.wav").resolve())
        current_ref = self.cfg["reference"].get("file", default_ref)
        self.var_ref_path = tk.StringVar(value=current_ref)
        ttk.Label(f_ref, text="La pista de referencia se guardará en:").grid(row=0, column=0, columnspan=2, sticky="w", pady=(8,2))
        ttk.Entry(f_ref, textvariable=self.var_ref_path, width=50, state="readonly").grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(0,8))
        ttk.Button(f_ref, text="Grabar referencia ahora", command=self._record_reference_here).grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Label(f_ref, text="(Usa fs y duración configurados)").grid(row=2, column=1, sticky="w")

        # Botonera
        bar = ttk.Frame(self); bar.pack(fill=tk.X, padx=12, pady=(0,12))
        ttk.Button(bar, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT, padx=6)
        ttk.Button(bar, text="Guardar", command=self._save).pack(side=tk.RIGHT)

        self.grab_set(); self.transient(master)

    def _record_reference_here(self):
        try:
            fs = int(self.var_fs.get()); dur = float(self.var_dur.get())
            x = record_audio(dur, fs=fs, channels=1)
            ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = (ASSETS_DIR / "reference_master.wav").resolve()
            sf.write(str(out_path), x, fs)
            self.var_ref_path.set(str(out_path))
            messagebox.showinfo("Pista de referencia", f"Referencia guardada en:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Pista de referencia", f"No se pudo grabar:\n{e}")

    def _save(self):
        self.cfg["general"]["oncalendar"] = self.var_oncal.get().strip()
        self.cfg["audio"]["fs"] = int(self.var_fs.get())
        self.cfg["audio"]["duration_s"] = float(self.var_dur.get())
        self.cfg["audio"]["prefer_input_name"] = self.var_pref.get().strip()
        self.cfg["thingsboard"]["host"] = self.var_host.get().strip()
        self.cfg["thingsboard"]["port"] = int(self.var_port.get())
        self.cfg["thingsboard"]["use_tls"] = bool(self.var_tls.get())
        self.cfg["thingsboard"]["token"] = self.var_token.get()
        self.cfg["reference"]["file"] = self.var_ref_path.get().strip()
        save_config(self.cfg)
        messagebox.showinfo("Configuración", "Guardado correctamente.")
        self.destroy()

# -------------------- main --------------------
def main():
    root = tb.Window(themename="flatly")
    app = AudioCinemaGUI(root)
    root.geometry("1020x640"); root.minsize(900,600)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
