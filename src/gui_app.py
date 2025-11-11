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
    normalize_mono, record_audio, analyze_pair,  # <- normalize_mono existe aquí
    detect_beeps, build_segments, crop_same_length,
    build_json_payload, welch_db, BANDS
)

from iot_tb import send_json_to_thingsboard

APP_NAME = "AudioCinema"
SAVE_DIR = (APP_DIR / "data" / "captures").absolute()
EXPORT_DIR = (APP_DIR / "data" / "reports").absolute()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_INPUT_NAME = os.environ.get("AUDIOCINEMA_INPUT_NAME","")
ENV_INPUT_INDEX = os.environ.get("AUDIOCINEMA_INPUT_INDEX")

# -------- util mic --------
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

class AudioCinemaGUI:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title(APP_NAME)

        tb.Style(theme="flatly")
        try:
            self.root.configure(bg="#e6e6e6")  # gris claro
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

        self.input_device_index: Optional[int] = None

        self.test_name = tk.StringVar(value="—")
        self.eval_text = tk.StringVar(value="—")
        self.eval_color = "#333333"

        # buffers último run (p/mostrar onda)
        self.last_ref: Optional[np.ndarray] = None
        self.last_cur: Optional[np.ndarray] = None
        self.last_fs: int = int(self.fs.get())

        # markers/segments (para JSON)
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

        # --------- IZQUIERDA: logo + 4 botones ---------
        left = ttk.Frame(paned, padding=(6,6))
        paned.add(left, weight=1)

        card = ttk.Frame(left, padding=6)
        card.pack(fill=Y, expand=False)

        if self._icon_img is not None:
            ttk.Label(card, image=self._icon_img).pack(anchor="n", pady=(0,4))
        ttk.Label(card, text="AudioCinema", font=("Segoe UI", 18, "bold")).pack(anchor="n")

        desc = ("Graba, evalúa y analiza tu sistema de audio "
                "para garantizar la mejor experiencia envolvente.")
        lbl = ttk.Label(card, text=desc, wraplength=220, justify="center")
        lbl.pack(anchor="n", pady=(6,10))

        # botones azules grandes
        btn_style = {"bootstyle": PRIMARY, "width": 20}
        tb.Button(card, text="Información",   command=self._show_info, **btn_style).pack(pady=6, fill=X)
        tb.Button(card, text="Configuración", command=self._popup_config, **btn_style).pack(pady=6, fill=X)
        tb.Button(card, text="Confirmación",  command=self._popup_confirm, **btn_style).pack(pady=6, fill=X)
        tb.Button(card, text="Prueba ahora",  command=self._run_once, **btn_style).pack(pady=(6,0), fill=X)

        # separador vertical
        sep = ttk.Separator(root_frame, orient=VERTICAL)
        paned.add(sep)

        # --------- DERECHA: cabecera + formas de onda + mensajes ---------
        right = ttk.Frame(paned, padding=(8,6))
        paned.add(right, weight=4)

        header = ttk.Frame(right)
        header.pack(fill=X, pady=(0,8))

        # PRUEBA (no editable)
        ttk.Label(header, text="PRUEBA:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0,6))
        e = ttk.Entry(header, textvariable=self.test_name, width=32, state="readonly", justify="center")
        e.grid(row=0, column=1, sticky="w")

        # EVALUACIÓN badge
        ttk.Label(header, text="EVALUACIÓN:", font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(0,6), pady=(6,0))
        self.eval_lbl = ttk.Label(header, textvariable=self.eval_text, font=("Segoe UI", 11, "bold"), foreground=self.eval_color)
        self.eval_lbl.grid(row=1, column=1, sticky="w", pady=(6,0))

        # Fig con dos ejes (formas de onda)
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
        msg_card.pack(fill=BOTH, expand=False, pady=(6,0))
        ttk.Label(msg_card, text="Mensajes", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.msg_text = tk.Text(msg_card, height=6, wrap="word")
        self.msg_text.pack(fill=BOTH, expand=True)
        self._set_messages(["Listo. Presiona «Prueba ahora» para iniciar."])

    def _clear_waves(self):
        for ax, title in ((self.ax_ref, "Pista de referencia"), (self.ax_cur, "Pista de prueba")):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid(True, axis='x', ls=':')
        self.canvas.draw_idle()

    def _plot_wave(self, ax, x: np.ndarray, fs: int):
        n = len(x)
        t = np.arange(n, dtype=np.float32) / float(fs if fs else 1)
        ax.plot(t, x, linewidth=0.8)
        ax.set_xlim(0.0, float(t[-1]) if n else 1.0)

    def _set_eval(self, passed: Optional[bool]):
        if passed is None:
            self.eval_text.set("—")
            self.eval_lbl.configure(foreground="#333333")
        elif passed:
            self.eval_text.set("PASSED")
            self.eval_lbl.configure(foreground="#0d8a00")
        else:
            self.eval_text.set("FAILED")
            self.eval_lbl.configure(foreground="#cc0000")

    def _set_messages(self, lines: List[str]):
        self.msg_text.delete("1.0", tk.END)
        for ln in lines:
            self.msg_text.insert(tk.END, "• " + ln + "\n")
        self.msg_text.see(tk.END)

    # ----------------- acciones -----------------
    def _auto_select_input_device(self):
        self.input_device_index = pick_input_device(self.cfg["audio"].get("preferred_input_name",""))

    def _show_info(self):
        messagebox.showinfo(APP_NAME, INFO_TEXT)

    def _popup_confirm(self):
        tb_cfg = self.cfg["thingsboard"]
        txt = (
            f"Archivo de referencia:\n  {self.cfg['reference']['wav_path']}\n\n"
            f"Audio:\n  fs={self.cfg['audio']['fs']}  duración={self.cfg['audio']['duration_s']} s\n"
            f"  preferir dispositivo='{self.cfg['audio'].get('preferred_input_name','')}'\n\n"
            f"ThingsBoard:\n  host={tb_cfg['host']}  port={tb_cfg['port']}  TLS={tb_cfg['use_tls']}\n"
            f"  token={tb_cfg['token']}\n\n"
            f"Programación (systemd):\n  {self.cfg['oncalendar']}\n"
        )
        messagebox.showinfo("Confirmación", txt)

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
        ttk.Entry(g, textvariable=ref_var, width=52).grid(row=0, column=1, sticky="we", pady=(6,2))
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

    # ----------------- ejecución de prueba -----------------
    def _run_once(self):
        fs = int(self.fs.get()); dur = float(self.duration.get())
        self.last_fs = fs

        # 1) referencia
        ref_path = Path(self.cfg["reference"]["wav_path"])
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

        # 2) grabar actual
        try:
            x_cur = record_audio(dur, fs=fs, channels=1, device=self.input_device_index)
        except Exception as e:
            messagebox.showerror(APP_NAME, f"Error grabando micrófono: {e}")
            return

        # 3) analizar
        res = analyze_pair(x_ref, x_cur, fs)
        self._set_eval(res["overall"] == "PASSED")

        # 4) beeps/segments para JSON
        self.ref_markers = detect_beeps(x_ref, fs)
        self.cur_markers = detect_beeps(x_cur, fs)
        self.ref_segments = build_segments(x_ref, fs, self.ref_markers)
        self.cur_segments = build_segments(x_cur, fs, self.cur_markers)

        # 5) dibujar ondas
        self.last_ref, self.last_cur = x_ref, x_cur
        self._clear_waves()
        self._plot_wave(self.ax_ref, x_ref, fs)
        self._plot_wave(self.ax_cur, x_cur, fs)
        self.canvas.draw_idle()

        # 6) nombre de la prueba (no editable)
        self.test_name.set(datetime.now().strftime("Test_%Y-%m-%d_%H-%M-%S"))

        # 7) exportar JSON y enviar a TB
        payload = build_json_payload(
            fs, res, [], self.ref_markers, self.cur_markers,
            self.ref_segments, self.cur_segments, None, None
        )
        out = EXPORT_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        ok_json = False
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            ok_json = True
        except Exception as e:
            self._set_messages([f"No se pudo guardar JSON: {e}"])

        sent = False
        if ok_json:
            tb_cfg = self.cfg["thingsboard"]
            if tb_cfg.get("token") and tb_cfg["token"] != "REEMPLAZA_TU_TOKEN":
                sent = send_json_to_thingsboard(payload, tb_cfg["host"], int(tb_cfg["port"]), tb_cfg["token"], bool(tb_cfg["use_tls"]))

        # 8) mensajes
        lines = []
        lines.append("La prueba ha " + ("aprobado." if res["overall"] == "PASSED" else "fallado."))
        lines.append(f"JSON: {out}")
        lines.append("Resultados enviados a ThingsBoard." if sent else "No se enviaron resultados a ThingsBoard.")
        self._set_messages(lines)

        messagebox.showinfo(APP_NAME, f"Análisis terminado.\nJSON: {out}")


class ConfigDialog(tk.Toplevel):
    def __init__(self, master, cfg: dict):
        super().__init__(master)
        self.title("Configuración")
        self.resizable(False, False)
        self.cfg = cfg

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # --- Pestaña General ---
        f_gen = ttk.Frame(nb)
        nb.add(f_gen, text="General")
        ttk.Label(f_gen, text="OnCalendar (systemd):").grid(row=0, column=0, sticky="w", pady=(8,2))
        self.var_oncal = tk.StringVar(value=self.cfg["general"].get("oncalendar","*-*-* 02:00:00"))
        ttk.Entry(f_gen, textvariable=self.var_oncal, width=28).grid(row=0, column=1, sticky="w", padx=6, pady=(8,2))

        # --- Pestaña Audio ---
        f_audio = ttk.Frame(nb)
        nb.add(f_audio, text="Audio")
        self.var_fs = tk.IntVar(value=int(self.cfg["audio"].get("fs", 48000)))
        self.var_dur = tk.DoubleVar(value=float(self.cfg["audio"].get("duration_s", 10.0)))
        self.var_pref = tk.StringVar(value=str(self.cfg["audio"].get("prefer_input_name","")))
        ttk.Label(f_audio, text="Sample rate (Hz):").grid(row=0, column=0, sticky="w", pady=(8,2))
        ttk.Entry(f_audio, textvariable=self.var_fs, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=(8,2))
        ttk.Label(f_audio, text="Duración (s):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f_audio, textvariable=self.var_dur, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(f_audio, text="Preferir dispositivo con nombre:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(f_audio, textvariable=self.var_pref, width=28).grid(row=2, column=1, sticky="w", padx=6, pady=2)

        # --- Pestaña ThingsBoard ---
        f_tb = ttk.Frame(nb)
        nb.add(f_tb, text="ThingsBoard")
        self.var_host = tk.StringVar(value=self.cfg["thingsboard"].get("host","thingsboard.cloud"))
        self.var_port = tk.IntVar(value=int(self.cfg["thingsboard"].get("port",1883)))
        self.var_tls  = tk.BooleanVar(value=bool(self.cfg["thingsboard"].get("use_tls", False)))
        self.var_token = tk.StringVar(value=self.cfg["thingsboard"].get("token",""))
        ttk.Label(f_tb, text="Host:").grid(row=0, column=0, sticky="w", pady=(8,2))
        ttk.Entry(f_tb, textvariable=self.var_host, width=28).grid(row=0, column=1, sticky="w", padx=6, pady=(8,2))
        ttk.Label(f_tb, text="Port:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(f_tb, textvariable=self.var_port, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(f_tb, text="Usar TLS (8883)", variable=self.var_tls).grid(row=2, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(f_tb, text="Token:").grid(row=3, column=0, sticky="w", pady=2)
        # Mostramos el token sin asteriscos (sin show="*")
        ttk.Entry(f_tb, textvariable=self.var_token, width=40).grid(row=3, column=1, sticky="w", padx=6, pady=2)

        # --- NUEVA pestaña: Pista de referencia ---
        f_ref = ttk.Frame(nb)
        nb.add(f_ref, text="Pista de referencia")

        # ruta actual (si no existe, se muestra el default)
        default_ref = str((ASSETS_DIR / "reference_master.wav").resolve())
        current_ref = self.cfg["reference"].get("file", default_ref)
        self.var_ref_path = tk.StringVar(value=current_ref)

        ttk.Label(f_ref, text="La pista de referencia se guardará en:").grid(row=0, column=0, columnspan=2, sticky="w", pady=(8,2))
        self.entry_ref = ttk.Entry(f_ref, textvariable=self.var_ref_path, width=50, state="readonly")
        self.entry_ref.grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(0,8))

        ttk.Button(f_ref, text="Grabar referencia ahora", command=self._record_reference_here)\
            .grid(row=2, column=0, sticky="w", padx=6, pady=6)

        ttk.Label(f_ref, text="(Se grabará con los parámetros de Audio: fs y duración)").grid(row=2, column=1, sticky="w")

        # --- Botonera ---
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=12, pady=(0,12))
        ttk.Button(bar, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT, padx=6)
        ttk.Button(bar, text="Guardar", command=self._save).pack(side=tk.RIGHT)

        self.grab_set()
        self.transient(master)

    def _record_reference_here(self):
        """Graba y guarda en assets/reference_master.wav (o con timestamp)."""
        try:
            fs = int(self.var_fs.get())
            dur = float(self.var_dur.get())

            # grabación
            x = record_audio(dur, fs=fs, channels=1)

            # asegura carpeta
            ASSETS_DIR.mkdir(parents=True, exist_ok=True)

            # nombre fijo (si quieres timestamp, cambia a reference_YYYYmmdd_HHMMSS.wav)
            out_path = (ASSETS_DIR / "reference_master.wav").resolve()
            sf.write(str(out_path), x, fs)

            # refleja ruta en UI y en config local (sin guardar a disco aún)
            self.var_ref_path.set(str(out_path))
            messagebox.showinfo("Pista de referencia", f"Referencia guardada en:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Pista de referencia", f"No se pudo grabar:\n{e}")

    def _save(self):
        # vuelca valores a cfg y guarda
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
