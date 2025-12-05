#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, traceback, subprocess, re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Any

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
    detect_beeps, build_segments, build_json_payload
)
from iot_tb import send_json_to_thingsboard

APP_NAME = "AudioCinema"
SAVE_DIR = (APP_DIR / "data" / "captures").absolute()
EXPORT_DIR = (APP_DIR / "data" / "reports").absolute()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ENV_INPUT_INDEX = os.environ.get("AUDIOCINEMA_INPUT_INDEX")

INFO_TEXT = (
    "AudioCinema\n\n"
    "Esta aplicación graba, evalúa y compara una pista de PRUEBA con una "
    "pista de REFERENCIA para verificar el estado del sistema de audio.\n\n"
    "Qué hace:\n"
    "• Graba la pista de prueba con el micrófono.\n"
    "• Compara vs. la referencia (RMS, crest, bandas, espectro relativo, P95).\n"
    "• Muestra la forma de onda de ambas pistas.\n"
    "• Exporta un JSON con resultados y (opcional) lo envía a ThingsBoard.\n\n"
    "Sugerencias:\n"
    "• Usa la misma ubicación del micrófono en cada prueba.\n"
    "• Verifica el archivo de referencia en Configuración."
)

# -------------------------------------------------------------------
# UTILIDAD: Leer próximo tiempo del timer systemd
# -------------------------------------------------------------------

def get_next_timer_run(timer_name: str = "audiocinema.timer") -> str:
    """Devuelve el campo NEXT de systemd list-timers."""
    try:
        out = subprocess.check_output(
            ["systemctl", "list-timers", timer_name, "--all"],
            text=True,
            stderr=subprocess.DEVNULL
        )
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if len(lines) < 2:
            return "No programado"

        data_line = lines[1].strip()
        cols = re.split(r"\s{2,}", data_line)
        return cols[0] if cols else "No disponible"

    except Exception:
        return "No disponible"


# -------------------------------------------------------------------
# SELECCIÓN AUTOMÁTICA DE DISPOSITIVO DE AUDIO
# -------------------------------------------------------------------

def pick_input_device(preferred_name_substr: Optional[str] = None) -> Optional[int]:
    import sounddevice as sd
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    if ENV_INPUT_INDEX:
        try:
            idx = int(ENV_INPUT_INDEX)
            if idx < len(devices) and devices[idx].get("max_input_channels", 0) > 0:
                return idx
        except Exception:
            pass

    if preferred_name_substr:
        s = preferred_name_substr.lower()
        for i, d in enumerate(devices):
            if s in str(d["name"]).lower() and d.get("max_input_channels", 0) > 0:
                return i

    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            return i

    return None


# Decorador para capturar errores UI
def ui_action(fn):
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            tb_str = traceback.format_exc()
            messagebox.showerror(APP_NAME, tb_str)
            return None
    return wrapper


# -------------------------------------------------------------------
# CLASE PRINCIPAL DE LA GUI
# -------------------------------------------------------------------

class AudioCinemaGUI:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title(APP_NAME)

        tb.Style(theme="flatly")
        try:
            self.root.configure(bg="#e6e6e6")
        except:
            pass

        # Icono
        self._icon_img = None
        try:
            icon_path = ASSETS_DIR / "audiocinema.png"
            if icon_path.exists():
                self._icon_img = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, self._icon_img)
        except:
            pass

        ensure_dirs()
        self.cfg = load_config()

        # Vars visibles
        self.fs = tk.IntVar(value=int(self._cfg(["audio","fs"], 48000)))
        self.duration = tk.DoubleVar(value=float(self._cfg(["audio","duration_s"], 10.0)))

        self.test_name = tk.StringVar(value="—")
        self.eval_text = tk.StringVar(value="—")
        self.next_eval = tk.StringVar(value="—")

        self.input_device_index = None
        self.last_ref = None
        self.last_cur = None
        self.last_fs = int(self.fs.get())

        self.ref_markers = []
        self.cur_markers = []
        self.ref_segments = []
        self.cur_segments = []

        self._build_ui()
        self._auto_select_input_device()
        self._update_next_eval_label()

    # -------------------------------------------------------------------
    # CONFIG SAFE GET/SET
    # -------------------------------------------------------------------

    def _cfg(self, path: List[str], default=None):
        d = self.cfg
        for key in path:
            if key not in d:
                return default
            d = d[key]
        return d

    def _set_cfg(self, path: List[str], value):
        d = self.cfg
        for key in path[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[path[-1]] = value

    # -------------------------------------------------------------------
    # UI BUILDER
    # -------------------------------------------------------------------

    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=8)
        root_frame.pack(fill=BOTH, expand=True)

        paned = ttk.Panedwindow(root_frame, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True)

        # ------------------------------------------
        # COLUMNA IZQUIERDA
        # ------------------------------------------
        left = ttk.Frame(paned, padding=6)
        paned.add(left, weight=1)

        card = ttk.Frame(left, padding=6)
        card.pack(fill=Y)

        if self._icon_img:
            ttk.Label(card, image=self._icon_img).pack(pady=(0,4))

        ttk.Label(card, text="AudioCinema", font=("Segoe UI", 18, "bold")).pack()

        desc = (
            "Graba, evalúa y analiza tu sistema de audio "
            "para garantizar la mejor experiencia envolvente."
        )
        ttk.Label(card, text=desc, wraplength=220, justify="center").pack(pady=10)

        btn_style = {"bootstyle": PRIMARY, "width": 20}
        tb.Button(card, text="Información",   command=self._show_info, **btn_style).pack(pady=5)
        tb.Button(card, text="Configuración", command=self._popup_config, **btn_style).pack(pady=5)
        tb.Button(card, text="Confirmación",  command=self._popup_confirm, **btn_style).pack(pady=5)
        tb.Button(card, text="Prueba ahora",  command=self._run_once, **btn_style).pack(pady=5)

        # Separador vertical
        paned.add(ttk.Separator(root_frame, orient=VERTICAL))

        # ------------------------------------------
        # COLUMNA DERECHA
        # ------------------------------------------
        right = ttk.Frame(paned, padding=8)
        paned.add(right, weight=4)

        header = ttk.Frame(right)
        header.pack(fill=X, pady=6)

        ttk.Label(header, text="PRUEBA:", font=("Segoe UI",10,"bold")).grid(row=0, column=0, sticky="w")
        ttk.Entry(header, textvariable=self.test_name, width=28, state="readonly", justify="center")\
            .grid(row=0, column=1, sticky="w")

        ttk.Label(header, text="RESULTADO:", font=("Segoe UI",10,"bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.eval_lbl = ttk.Label(header, textvariable=self.eval_text, font=("Segoe UI",11,"bold"))
        self.eval_lbl.grid(row=1, column=1, sticky="w")

        ttk.Label(header, text="PRÓXIMA EVALUACIÓN:", font=("Segoe UI",10,"bold")).grid(row=2, column=0, sticky="w")
        self.next_eval_lbl = ttk.Label(header, textvariable=self.next_eval, font=("Segoe UI",10))
        self.next_eval_lbl.grid(row=2, column=1, sticky="w", pady=5)

        # Gráficos
        fig_card = ttk.Frame(right, padding=4)
        fig_card.pack(fill=BOTH, expand=True)

        self.fig = Figure(figsize=(5,4), dpi=100)
        self.ax_ref = self.fig.add_subplot(2,1,1)
        self.ax_cur = self.fig.add_subplot(2,1,2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_card)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self._clear_waves()
        self.fig.tight_layout()

        # Mensajes
        msg_card = ttk.Frame(right, padding=4)
        msg_card.pack(fill=X)
        ttk.Label(msg_card, text="Mensajes", font=("Segoe UI",10,"bold")).pack(anchor="w")
        self.msg_text = tk.Text(msg_card, height=6, wrap="word")
        self.msg_text.pack(fill=BOTH)

        self._set_messages(["Listo. Presiona «Prueba ahora» para iniciar."])

    # -------------------------------------------------------------------
    def _clear_waves(self):
        for ax, title in ((self.ax_ref,"Pista de referencia"), (self.ax_cur,"Pista de prueba")):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid(True, linestyle=":", axis="x")
        self.canvas.draw_idle()

    def _plot_wave(self, ax, x, fs):
        t = np.arange(len(x)) / fs
        ax.plot(t, x, linewidth=0.8)
        ax.set_xlim(0, t[-1] if len(t) else 1)

    # -------------------------------------------------------------------
    def _set_eval(self, passed):
        if passed is None:
            self.eval_text.set("—")
            self.eval_lbl.configure(foreground="#333")
        elif passed:
            self.eval_text.set("PASSED")
            self.eval_lbl.configure(foreground="#0d8a00")
        else:
            self.eval_text.set("FAILED")
            self.eval_lbl.configure(foreground="#cc0000")

    def _set_messages(self, lines):
        self.msg_text.delete("1.0", tk.END)
        for ln in lines:
            self.msg_text.insert(tk.END, f"• {ln}\n")
        self.msg_text.see(tk.END)

    # -------------------------------------------------------------------
    def _update_next_eval_label(self):
        """Actualiza el timer desde systemd."""
        self.next_eval.set(get_next_timer_run())

    # -------------------------------------------------------------------
    def _auto_select_input_device(self):
        pref = str(self._cfg(["audio","preferred_input_name"], ""))
        self.input_device_index = pick_input_device(pref)

    # -------------------------------------------------------------------
    @ui_action
    def _show_info(self):
        messagebox.showinfo(APP_NAME, INFO_TEXT)

    # -------------------------------------------------------------------
    @ui_action
    def _popup_confirm(self):
        txt = (
            f"Archivo de referencia:\n  {self._cfg(['reference','wav_path'], str(ASSETS_DIR/'reference_master.wav'))}\n\n"
            f"Audio:\n  fs={self._cfg(['audio','fs'],48000)}  duración={self._cfg(['audio','duration_s'],10.0)} s\n"
            f"  preferir dispositivo='{self._cfg(['audio','preferred_input_name'],'')}'\n\n"
            f"Programación (systemd):\n  {self._cfg(['oncalendar'],'*-*-* 02:00:00')}\n"
        )
        messagebox.showinfo("Confirmación", txt)

    # -------------------------------------------------------------------
    @ui_action
    def _popup_config(self):
        w = tk.Toplevel(self.root)
        w.title("Configuración")
        if self._icon_img:
            w.iconphoto(True, self._icon_img)

        frm = ttk.Frame(w, padding=10)
        frm.pack(fill=BOTH, expand=True)

        nb = ttk.Notebook(frm)
        nb.pack(fill=BOTH, expand=True)

        # ---------------- GENERAL ----------------
        g = ttk.Frame(nb)
        nb.add(g, text="General")

        ref_var = tk.StringVar(value=self._cfg(["reference","wav_path"], str(ASSETS_DIR/"reference_master.wav")))
        oncal_var = tk.StringVar(value=self._cfg(["oncalendar"], "*-*-* 02:00:00"))

        ttk.Label(g, text="Archivo referencia:").grid(row=0, column=0, sticky="w")
        ttk.Entry(g, textvariable=ref_var, width=50).grid(row=0, column=1, sticky="w")

        ttk.Label(g, text="OnCalendar (systemd):").grid(row=1, column=0, sticky="w")
        ttk.Entry(g, textvariable=oncal_var, width=30).grid(row=1, column=1, sticky="w")

        # ---------------- AUDIO ----------------
        a = ttk.Frame(nb)
        nb.add(a, text="Audio")

        fs_var = tk.IntVar(value=self._cfg(["audio","fs"], 48000))
        dur_var = tk.DoubleVar(value=self._cfg(["audio","duration_s"], 10.0))
        pref_in = tk.StringVar(value=self._cfg(["audio","preferred_input_name"], ""))

        ttk.Label(a, text="Sample Rate:").grid(row=0, column=0, sticky="w")
        ttk.Entry(a, textvariable=fs_var, width=10).grid(row=0, column=1)

        ttk.Label(a, text="Duración (s):").grid(row=1, column=0, sticky="w")
        ttk.Entry(a, textvariable=dur_var, width=10).grid(row=1, column=1)

        ttk.Label(a, text="Preferir dispositivo:").grid(row=2, column=0, sticky="w")
        ttk.Entry(a, textvariable=pref_in, width=30).grid(row=2, column=1)

        # ---------------- EVALUACIÓN ----------------
        ev = ttk.Frame(nb)
        nb.add(ev, text="Evaluación")

        eval_levels = ["Bajo", "Medio", "Alto"]
        eval_var = tk.StringVar(value=self._cfg(["evaluation","level"], "Medio"))

        ttk.Label(ev, text="Criterios de evaluación:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(ev, textvariable=eval_var, values=eval_levels, state="readonly", width=10)\
            .grid(row=0, column=1, sticky="w")

        ttk.Label(ev, text="Define cuánta tolerancia tendrá la evaluación.").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=4
        )

        # ---------------- FRECUENCIAS ----------------
        fr = ttk.Frame(nb)
        nb.add(fr, text="Frecuencias de grabación")

        trig_enabled = tk.BooleanVar(value=self._cfg(["trigger","enabled"], False))
        min_freq_var = tk.DoubleVar(value=self._cfg(["trigger","min_freq"], 80.0))
        max_freq_var = tk.DoubleVar(value=self._cfg(["trigger","max_freq"], 200.0))

        ttk.Checkbutton(fr, text="Activar grabación por frecuencia", variable=trig_enabled)\
            .grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(fr, text="Frecuencia mínima (Hz):").grid(row=1, column=0, sticky="w")
        ttk.Entry(fr, textvariable=min_freq_var, width=10).grid(row=1, column=1)

        ttk.Label(fr, text="Frecuencia máxima (Hz):").grid(row=2, column=0, sticky="w")
        ttk.Entry(fr, textvariable=max_freq_var, width=10).grid(row=2, column=1)

        # ---------------- BOTONES ----------------
        btns = ttk.Frame(frm)
        btns.pack(fill=X, pady=10)

        def on_save():
            self._set_cfg(["reference","wav_path"], ref_var.get())
            self._set_cfg(["oncalendar"], oncal_var.get())
            self._set_cfg(["audio","fs"], fs_var.get())
            self._set_cfg(["audio","duration_s"], dur_var.get())
            self._set_cfg(["audio","preferred_input_name"], pref_in.get())

            self._set_cfg(["evaluation","level"], eval_var.get())

            self._set_cfg(["trigger","enabled"], trig_enabled.get())
            self._set_cfg(["trigger","min_freq"], min_freq_var.get())
            self._set_cfg(["trigger","max_freq"], max_freq_var.get())

            save_config(self.cfg)

            # Actualizar cabecera
            self._update_next_eval_label()

            messagebox.showinfo(APP_NAME, "Configuración guardada.")
            w.destroy()

        tb.Button(btns, text="Guardar", bootstyle=PRIMARY, command=on_save)\
            .pack(side=RIGHT, padx=5)
        tb.Button(btns, text="Cancelar", bootstyle=SECONDARY, command=w.destroy)\
            .pack(side=RIGHT)

    # -------------------------------------------------------------------
    @ui_action
    def _run_once(self):
        fs = int(self._cfg(["audio","fs"], 48000))
        dur = float(self._cfg(["audio","duration_s"], 10.0))
        self.last_fs = fs

        ref_path = Path(self._cfg(["reference","wav_path"], str(ASSETS_DIR/"reference_master.wav")))
        if not ref_path.exists():
            raise FileNotFoundError(f"No existe archivo de referencia:\n{ref_path}")

        x_ref, fs_ref = sf.read(ref_path, dtype="float32", always_2d=False)
        if x_ref.ndim == 2:
            x_ref = x_ref.mean(axis=1)
        x_ref = normalize_mono(x_ref)

        if fs_ref != fs:
            n_new = int(round(len(x_ref) * fs / fs_ref))
            x_ref = np.interp(
                np.linspace(0,1,n_new), 
                np.linspace(0,1,len(x_ref)), 
                x_ref
            ).astype(np.float32)

        x_cur = record_audio(dur, fs=fs, channels=1, device=self.input_device_index)

        # Evaluación con niveles
        eval_level = self._cfg(["evaluation","level"], "Medio")
        res = analyze_pair(x_ref, x_cur, fs, eval_level)

        self._set_eval(res["overall"] == "PASSED")

        self.ref_markers = detect_beeps(x_ref, fs)
        self.cur_markers = detect_beeps(x_cur, fs)
        self.ref_segments = build_segments(x_ref, fs, self.ref_markers)
        self.cur_segments = build_segments(x_cur, fs, self.cur_markers)

        self.last_ref = x_ref
        self.last_cur = x_cur

        self._clear_waves()
        self._plot_wave(self.ax_ref, x_ref, fs)
        self._plot_wave(self.ax_cur, x_cur, fs)
        self.canvas.draw_idle()

        self.test_name.set(datetime.now().strftime("Test_%Y-%m-%d_%H-%M-%S"))

        payload = build_json_payload(
            fs, res, [], self.ref_markers, self.cur_markers,
            self.ref_segments, self.cur_segments, None, None
        )

        out = EXPORT_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        host = self._cfg(["thingsboard","host"], "thingsboard.cloud")
        port = int(self._cfg(["thingsboard","port"], 1883))
        token = self._cfg(["thingsboard","token"], "")
        use_tls = bool(self._cfg(["thingsboard","use_tls"], False))

        sent = False
        if token:
            sent = send_json_to_thingsboard(payload, host, port, token, use_tls)

        self._set_messages([
            f"La prueba ha {'aprobado' if res['overall']=='PASSED' else 'fallado'}.",
            f"JSON: {out}",
            ("Enviado a ThingsBoard" if sent else "No se envió a ThingsBoard")
        ])

        messagebox.showinfo(APP_NAME, f"Análisis terminado.\nJSON: {out}")


def main():
    root = tb.Window(themename="flatly")
    app = AudioCinemaGUI(root)
    root.geometry("1020x640")
    root.minsize(900,600)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
