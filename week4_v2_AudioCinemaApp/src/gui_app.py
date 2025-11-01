#!/usr/bin/env python3
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")

from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import json

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app_platform import APP_DIR, REP_DIR, ensure_dirs
from configio import load_config, save_config
from analyzer import (pick_input_device, record, load_ref, analyze_pair,
                      build_json, BANDS)
from iot_tb import send_json_to_thingsboard

PNG_ICON = APP_DIR / "assets" / "audiocinema.png"
ICO_ICON = APP_DIR / "assets" / "audiocinema.ico"

class App:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title("AudioCinema")
        try:
            self.root.wm_class("AudioCinema","AudioCinema")
        except Exception:
            pass
        # icono ventana
        try:
            if PNG_ICON.exists():
                _img = tk.PhotoImage(file=str(PNG_ICON))
                self.root.iconphoto(True, _img)
                self._img_keep = _img
            elif ICO_ICON.exists():
                self.root.iconbitmap(default=str(ICO_ICON))
        except Exception:
            pass

        ensure_dirs()
        self.cfg = load_config()

        # ------------ estado ------------
        self.test_name = tk.StringVar(value=self.cfg["ui"]["last_test_name"])
        self.overall   = tk.StringVar(value="SIN PRUEBA")
        self.msg_lines = 0

        self.fs        = self.cfg["audio"]["fs"]
        self.duration  = self.cfg["audio"]["duration_s"]
        self.pref_in   = self.cfg["audio"]["preferred_input_name"]
        self.ref_wav   = self.cfg["reference"]["wav_path"]

        self.tb_host   = self.cfg["thingsboard"]["host"]
        self.tb_port   = self.cfg["thingsboard"]["port"]
        self.tb_tls    = self.cfg["thingsboard"]["use_tls"]
        self.tb_token  = self.cfg["thingsboard"]["token"]

        self.oncalendar= self.cfg["oncalendar"]

        self.input_index = pick_input_device(self.pref_in)

        # datos del último análisis
        self.last_res = None
        self.last_json_path = None

        self._build_ui()

    # ========================= UI LAYOUT (según boceto) =========================
    def _build_ui(self):
        self.root.geometry("1280x860"); self.root.minsize(1100,720)

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=BOTH, expand=True)

        main.columnconfigure(0, weight=0, minsize=360)   # panel izquierdo
        main.columnconfigure(1, weight=1)                # panel derecho

        # ---------- IZQUIERDA ----------
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,8))

        # LOGO
        try:
            if PNG_ICON.exists():
                logo = tk.PhotoImage(file=str(PNG_ICON))
                lbl = ttk.Label(left, image=logo)
                lbl.image = logo
                lbl.grid(row=0, column=0, sticky="w", pady=(4,8))
        except Exception:
            pass

        # Título
        ttk.Label(left, text="AudioCinema", font=("Helvetica", 18, "bold")).grid(row=1, column=0, sticky="w")

        # OVERALL + nombre de prueba
        over = ttk.Frame(left); over.grid(row=2, column=0, sticky="ew", pady=(10,6))
        over.columnconfigure(1, weight=1)
        ttk.Label(over, text="Prueba:").grid(row=0, column=0, sticky="w")
        ttk.Entry(over, textvariable=self.test_name).grid(row=0, column=1, sticky="ew", padx=(6,0))
        self.badge = tb.Label(over, textvariable=self.overall, bootstyle=SECONDARY, font=("Segoe UI", 12, "bold"))
        self.badge.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8,0))

        # Resultados por canal (tabla)
        lf = tb.Labelframe(left, text="Resultados por canal", bootstyle=INFO)
        lf.grid(row=3, column=0, sticky="nsew", pady=(10,6))
        left.rowconfigure(3, weight=1)

        cols = ("ch","drms","dcrest","lfe","lf","mf","hf","spec","res")
        self.table = ttk.Treeview(lf, columns=cols, show="headings", height=8)
        headers = {"ch":"Canal","drms":"ΔRMS (dB)","dcrest":"ΔCrest (dB)","lfe":"ΔLFE","lf":"ΔLF","mf":"ΔMF","hf":"ΔHF","spec":"Spec P95 (dB)","res":"Resultado"}
        for c in cols:
            self.table.heading(c, text=headers[c])
            self.table.column(c, anchor="center", width=90)
        self.table.column("ch", width=80)
        self.table.pack(fill=BOTH, expand=True, padx=6, pady=6)

        # Mensajes
        logf = tb.Labelframe(left, text="Mensajes", bootstyle=SECONDARY)
        logf.grid(row=4, column=0, sticky="nsew")
        self.log = tk.Text(logf, height=8, wrap="word")
        self.log.pack(fill=BOTH, expand=True, padx=6, pady=6)
        left.rowconfigure(4, weight=1)

        # Botones inferiores
        btns = ttk.Frame(left)
        btns.grid(row=5, column=0, sticky="ew", pady=(10,4))
        tb.Button(btns, text="Ejecutar ahora", bootstyle=SUCCESS, command=self._run_once).pack(fill=X, pady=(0,6))
        tb.Button(btns, text="Configuración", bootstyle=PRIMARY, command=self._open_config).pack(fill=X)

        # ---------- DERECHA (4 gráficas apiladas) ----------
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(0, weight=1)

        gf = tb.Labelframe(right, text="Gráficas", bootstyle=WARNING)
        gf.pack(fill=BOTH, expand=True)

        self.fig = Figure(figsize=(7,8), dpi=100)
        self.ax1 = self.fig.add_subplot(4,1,1)
        self.ax2 = self.fig.add_subplot(4,1,2)
        self.ax3 = self.fig.add_subplot(4,1,3)
        self.ax4 = self.fig.add_subplot(4,1,4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=gf)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self._clear_plots()

    # ========================= Acciones =========================
    def _log(self, s: str):
        self.log.insert(tk.END, s + "\n"); self.log.see(tk.END)

    def _clear_plots(self):
        for ax in (self.ax1,self.ax2,self.ax3,self.ax4):
            ax.clear(); ax.grid(True, ls=":")
        self.ax1.set_title("PSD (Referencia vs Cinema)")
        self.ax2.set_title("Relativo (Cinema - Ref)")
        self.ax3.set_title("Δ Energía por bandas")
        self.ax4.set_title("RMS / Crest")
        self.canvas.draw_idle()

    def _open_config(self):
        cfg = tk.Toplevel(self.root)
        cfg.title("Configuración"); cfg.geometry("520x520"); cfg.resizable(False, False)

        nb = ttk.Notebook(cfg); nb.pack(fill=BOTH, expand=True, padx=8, pady=8)

        # --- Pestaña General ---
        g = ttk.Frame(nb); nb.add(g, text="General")
        ttk.Label(g, text="Archivo de referencia (WAV):").grid(row=0, column=0, sticky="w", pady=(6,2))
        ref_var = tk.StringVar(value=self.ref_wav)
        ent = ttk.Entry(g, textvariable=ref_var, width=48); ent.grid(row=1, column=0, sticky="w")
        def pick_ref():
            p = filedialog.askopenfilename(title="Selecciona referencia.wav",
                                           filetypes=[("WAV", "*.wav"), ("Todos","*.*")],
                                           initialdir=str(Path(self.ref_wav).parent))
            if p: ref_var.set(p)
        ttk.Button(g, text="Buscar…", command=pick_ref).grid(row=1, column=1, padx=6)

        ttk.Label(g, text="Nombre de prueba (UI):").grid(row=2, column=0, sticky="w", pady=(10,2))
        name_var = tk.StringVar(value=self.test_name.get())
        ttk.Entry(g, textvariable=name_var, width=28).grid(row=3, column=0, sticky="w")

        ttk.Label(g, text="Timer (OnCalendar):").grid(row=4, column=0, sticky="w", pady=(10,2))
        oncal_var = tk.StringVar(value=self.oncalendar)
        ttk.Entry(g, textvariable=oncal_var, width=28).grid(row=5, column=0, sticky="w")

        # --- Pestaña Audio ---
        a = ttk.Frame(nb); nb.add(a, text="Audio")
        fs_var = tk.IntVar(value=self.fs); dur_var = tk.DoubleVar(value=self.duration)
        pref_in_var = tk.StringVar(value=self.pref_in)
        ttk.Label(a, text="Sample Rate (Hz):").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=fs_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(a, text="Duración (s):").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=dur_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(a, text="Preferir dispositivo con nombre:").grid(row=2, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=pref_in_var, width=28).grid(row=2, column=1, sticky="w")

        # --- Pestaña ThingsBoard ---
        t = ttk.Frame(nb); nb.add(t, text="ThingsBoard")
        host_var = tk.StringVar(value=self.tb_host)
        port_var = tk.IntVar(value=self.tb_port)
        tls_var  = tk.BooleanVar(value=self.tb_tls)
        token_var= tk.StringVar(value=self.tb_token)

        ttk.Label(t, text="Host:").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=host_var, width=24).grid(row=0, column=1, sticky="w")
        ttk.Label(t, text="Port:").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=port_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(t, text="Usar TLS (8883)", variable=tls_var).grid(row=2, column=1, sticky="w", pady=(6,2))
        ttk.Label(t, text="Token:").grid(row=3, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=token_var, width=40, show="*").grid(row=3, column=1, sticky="w")

        # Botones
        bar = ttk.Frame(cfg); bar.pack(fill=X, padx=8, pady=(0,8))
        def save_and_close():
            # aplicar en memoria
            self.ref_wav = ref_var.get().strip() or self.ref_wav
            self.test_name.set(name_var.get().strip() or self.test_name.get())
            self.oncalendar = oncal_var.get().strip() or self.oncalendar
            self.fs = int(fs_var.get()); self.duration = float(dur_var.get()); self.pref_in = pref_in_var.get()
            self.tb_host = host_var.get().strip(); self.tb_port = int(port_var.get()); self.tb_tls = bool(tls_var.get()); self.tb_token = token_var.get().strip()
            # persistir
            self.cfg["reference"]["wav_path"] = self.ref_wav
            self.cfg["ui"]["last_test_name"] = self.test_name.get()
            self.cfg["oncalendar"] = self.oncalendar
            self.cfg["audio"].update({"fs": self.fs, "duration_s": self.duration, "preferred_input_name": self.pref_in})
            self.cfg["thingsboard"].update({"host": self.tb_host, "port": self.tb_port, "use_tls": self.tb_tls, "token": self.tb_token})
            save_config(self.cfg)
            messagebox.showinfo("Configuración", "Guardado. Si cambiaste el horario, re-instala el timer con install_systemd.sh")
            cfg.destroy()
        ttk.Button(bar, text="Guardar", command=save_and_close).pack(side=RIGHT)
        ttk.Button(bar, text="Cancelar", command=cfg.destroy).pack(side=RIGHT, padx=(0,8))

    def _run_once(self):
        try:
            # 1) cargar referencia
            ref = load_ref(self.ref_wav, self.fs)
            # 2) seleccionar dispositivo y grabar
            self.input_index = pick_input_device(self.pref_in)
            if self.input_index is None:
                messagebox.showerror("Audio", "No se encontró micrófono de entrada.")
                return
            self._log(f"Grabando {self.duration:.1f}s @ {self.fs} Hz (device {self.input_index})…")
            cur = record(self.duration, self.fs, device=self.input_index)
            # 3) alinear longitudes (simple)
            n = min(len(ref), len(cur))
            ref, cur = ref[:n], cur[:n]
            # 4) analizar
            res = analyze_pair(ref, cur, self.fs)
            self.last_res = res
            self._update_overall(res)
            self._update_plots(res)
            self._update_table(res)
            # 5) export JSON (auto en reports/)
            REP_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out = REP_DIR / f"{ts}_report.json"
            payload = build_json(self.fs, res, self.ref_wav, "(live mic)", self.test_name.get())
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            self.last_json_path = out
            self._log(f"✅ Reporte JSON → {out}")
            # 6) enviar a TB
            ok, msg = send_json_to_thingsboard(payload, self.tb_host, self.tb_port, self.tb_token, self.tb_tls)
            self._log(("📤 " if ok else "❌ ") + msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log(f"❌ Error: {e}")

    # ========================= helpers de UI =========================
    def _update_overall(self, res: dict):
        self.overall.set(f"OVERALL: {res['overall']}")
        self.badge.configure(bootstyle=(SUCCESS if res["overall"]=="PASSED" else DANGER))

    def _update_table(self, res: dict):
        self.table.delete(*self.table.get_children())
        diffs = res["diff_bands"]
        self.table.insert("", tk.END, values=(
            "Global",
            f"{res['diff_rms']:.2f}", f"{res['diff_crest']:.2f}",
            f"{diffs['LFE']:.2f}", f"{diffs['LF']:.2f}", f"{diffs['MF']:.2f}", f"{diffs['HF']:.2f}",
            f"{res['spec_dev95']:.2f}", res["overall"]
        ))

    def _update_plots(self, res: dict):
        # PSD
        self.ax1.clear(); self.ax1.grid(True, ls=":")
        self.ax1.semilogx(res["f_ref"], res["psd_ref_db"], label="Ref")
        self.ax1.semilogx(res["f_cur"], res["psd_cur_db"], label="Cinema", alpha=0.9)
        self.ax1.set_ylabel("dB/Hz"); self.ax1.legend()

        # Relativo
        self.ax2.clear(); self.ax2.grid(True, ls=":")
        self.ax2.semilogx(res["f_rel"], res["rel_db"])
        self.ax2.axhline(0, ls='--'); self.ax2.axhline(6, ls=':', lw=1); self.ax2.axhline(-6, ls=':', lw=1)
        self.ax2.set_ylabel("dB")

        # Bandas
        self.ax3.clear(); self.ax3.grid(True, axis='y', ls=":")
        bands = list(BANDS.keys()); vals = [res["diff_bands"][k] for k in bands]
        self.ax3.bar(bands, vals); self.ax3.axhline(0, ls='--', lw=1); self.ax3.set_ylabel("dB")

        # RMS & Crest
        self.ax4.clear(); self.ax4.grid(True, axis='y', ls=":")
        metrics = ["RMS Ref","RMS Cin","ΔRMS","Crest Ref","Crest Cin","ΔCrest"]
        values  = [res["rms_ref"],res["rms_cur"],res["diff_rms"],res["crest_ref"],res["crest_cur"],res["diff_crest"]]
        self.ax4.bar(metrics, values); self.ax4.set_ylabel("dB"); self.ax4.set_xticklabels(metrics, rotation=25, ha="right")

        self.fig.tight_layout(); self.canvas.draw_idle()

# ========================= main =========================
def main():
    theme = "superhero"
    root = tb.Window(themename=theme)
    # WM_CLASS para que el panel use el icono del .desktop
    try: root.wm_class("AudioCinema","AudioCinema")
    except Exception: pass
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
