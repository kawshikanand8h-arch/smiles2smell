#!/usr/bin/env python3
"""Desktop UI for the smell-agent odor predictor.

A modern, single-window interface around predict_smiles.py:

    python odor_ui.py

Features
  - loads the 10-model ensemble in the background (CPU or CUDA)
  - live RDKit structure preview while you type
  - ranked smell profile with bars, top-N selector
  - copy results as text or JSON
"""

import json
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

# ----------------------------------------------------------------------------
# palette
# ----------------------------------------------------------------------------
BG = "#F4F5F7"
CARD = "#FFFFFF"
BORDER = "#E3E6EB"
TEXT = "#111827"
MUTED = "#6B7280"
FAINT = "#9CA3AF"
ACCENT = "#2F6FED"
ACCENT_HOVER = "#1F5BD6"
ACCENT_DISABLED = "#B9CDF6"
ACCENT_SOFT = "#EAF1FE"
TRACK = "#EDF0F4"
ROW_HOVER = "#F6F8FB"
DANGER = "#DC2626"

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
    """Draw a rounded rectangle on a canvas (smooth polygon)."""
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


def blend(c1, c2, t):
    """Linearly blend two #rrggbb colors."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def pick_font(families, *candidates):
    for name in candidates:
        if name in families:
            return name
    return "TkDefaultFont"


# ----------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------


class OdorApp:
    ROW_H = 56
    PAD = 18

    def __init__(self, root):
        self.root = root
        self.models = None
        self.device = None
        self.loading = True
        self.busy = False
        self.rows = []          # full sorted list of (odor, score)
        self.smiles = ""
        self.topn = 10
        self.hover = None
        self._anim = 0
        self._status_job = None
        self._preview_job = None
        self._photo = None

        families = set(tkfont.families(root))
        self.ui = pick_font(families, "Inter", "Segoe UI", "Ubuntu", "Cantarell", "DejaVu Sans", "Arial")
        self.serif = pick_font(families, "Georgia", "DejaVu Serif", "Times New Roman", self.ui)
        self.mono = pick_font(families, "JetBrains Mono", "Cascadia Mono", "DejaVu Sans Mono")

        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            from PIL import Image, ImageTk
            self.Chem, self.Draw = Chem, Draw
            self.ImageTk = ImageTk
            self.rdkit = True
        except ImportError:
            self.rdkit = False

        self._build_window()
        self._build_menu()
        self._build_layout()
        self._set_predict_enabled(False)
        self.root.after(80, self._redraw)
        self._start_loading("cuda")

    # ---------------------------------------------------------------- window
    def _configure_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox", fieldbackground="#F7F8FA", background="#F7F8FA",
                        foreground=TEXT, arrowcolor=MUTED, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, padding=5,
                        relief="flat")
        style.map("TCombobox",
                  fieldbackground=[("readonly", "#F7F8FA")],
                  background=[("readonly", "#F7F8FA")],
                  foreground=[("readonly", TEXT)],
                  selectbackground=[("readonly", "#F7F8FA")],
                  selectforeground=[("readonly", TEXT)])
        style.configure("Vertical.TScrollbar", background="#DDE1E7", troughcolor=CARD,
                        bordercolor=CARD, arrowcolor=MUTED, relief="flat",
                        width=10)
        style.map("Vertical.TScrollbar", background=[("active", "#C9CFD8")])
        self.root.option_add("*TCombobox*Listbox.font", (self.ui, 10))

    def _build_window(self):
        r = self.root
        self._configure_styles()
        r.title("Odor Predictor — smell-agent")
        r.configure(bg=BG)
        w, h = 1120, 748
        r.minsize(1000, 660)
        r.update_idletasks()
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        r.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")

        r.grid_rowconfigure(1, weight=1)
        r.grid_columnconfigure(0, weight=1)

        # header ---------------------------------------------------------
        header = tk.Frame(r, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=26, pady=(18, 6))
        header.grid_columnconfigure(1, weight=1)

        left = tk.Frame(header, bg=BG)
        left.grid(row=0, column=0, sticky="w")
        title = tk.Frame(left, bg=BG)
        title.pack(anchor="w")
        tk.Label(title, text="smell", font=(self.serif, 23, "bold"),
                 bg=BG, fg=TEXT).pack(side="left")
        tk.Label(title, text="-agent", font=(self.serif, 23, "bold"),
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(left, text="Turn a molecule into a smell profile",
                 font=(self.ui, 10), bg=BG, fg=MUTED).pack(anchor="w", pady=(1, 0))

        right = tk.Frame(header, bg=BG)
        right.grid(row=0, column=2, sticky="e")
        tk.Label(right, text="MPNN-POM", font=(self.ui, 9, "bold"),
                 bg=ACCENT_SOFT, fg=ACCENT, padx=10, pady=4).pack(side="right")

        # status bar ------------------------------------------------------
        bar = tk.Frame(r, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        bar.grid(row=2, column=0, sticky="ew")
        inner = tk.Frame(bar, bg=CARD)
        inner.pack(fill="x", padx=26, pady=7)
        self.status = tk.Label(inner, text="Starting…", font=(self.ui, 9),
                               bg=CARD, fg=MUTED, anchor="w")
        self.status.pack(side="left")
        self.status_right = tk.Label(inner, text="10-model ensemble · odor map 256",
                                     font=(self.ui, 9), bg=CARD, fg=FAINT)
        self.status_right.pack(side="right")

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Predict", accelerator="Ctrl+Return",
                              command=self.predict)
        file_menu.add_command(label="Clear", accelerator="Ctrl+L",
                              command=self.clear)
        file_menu.add_separator()
        file_menu.add_command(label="Copy results as text",
                              command=lambda: self.copy_results("txt"))
        file_menu.add_command(label="Copy results as JSON",
                              command=lambda: self.copy_results("json"))
        file_menu.add_separator()
        file_menu.add_command(label="Quit", accelerator="Ctrl+Q",
                              command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-Return>", lambda e: self.predict())
        self.root.bind("<Control-l>", lambda e: self.clear())
        self.root.bind("<Control-q>", lambda e: self.root.destroy())

    # ---------------------------------------------------------------- layout
    def _build_layout(self):
        content = tk.Frame(self.root, bg=BG)
        content.grid(row=1, column=0, sticky="nsew", padx=26, pady=(6, 16))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)

        self._build_left_card(content)
        self._build_results_card(content)

    def _build_left_card(self, parent):
        card = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=0, sticky="nsw", padx=(0, 18))
        body = tk.Frame(card, bg=CARD, width=326)
        body.pack(fill="both", expand=True, padx=18, pady=16)
        body.pack_propagate(False)

        tk.Label(body, text="Molecule", font=(self.ui, 13, "bold"),
                 bg=CARD, fg=TEXT).pack(anchor="w")
        tk.Label(body, text="SMILES string", font=(self.ui, 9),
                 bg=CARD, fg=MUTED).pack(anchor="w", pady=(0, 6))

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(body, textvariable=self.entry_var,
                              font=(self.mono, 11), bg="#F7F8FA", fg=TEXT,
                              relief="flat", highlightthickness=1,
                              highlightbackground=BORDER, highlightcolor=ACCENT,
                              insertbackground=TEXT)
        self.entry.pack(fill="x", ipady=7)
        self.entry.bind("<KeyRelease>", self._on_typing)
        self.entry.bind("<Return>", lambda e: self.predict())
        self.entry.focus_set()

        self.hint = tk.Label(body, text="Try CCO (ethanol)", font=(self.ui, 9),
                             bg=CARD, fg=FAINT, anchor="w")
        self.hint.pack(fill="x", pady=(4, 0))

        # example chips ---------------------------------------------------
        chips = tk.Frame(body, bg=CARD)
        chips.pack(fill="x", pady=(10, 0))
        examples = [
            ("Ethanol", "CCO"),
            ("Vanillin", "COc1cc(C=O)ccc1O"),
        ]
        examples2 = [
            ("Menthol", "CC(C)C1CCC(C)CC1O"),
            ("Limonene", "CC1=CCC(CC1)C(=C)C"),
            ("Acetic acid", "CC(=O)O"),
            ("Benzaldehyde", "O=Cc1ccccc1"),
        ]
        for label, smi in examples:
            self._chip(chips, label, smi).pack(side="left", padx=(0, 6))
        chips2 = tk.Frame(body, bg=CARD)
        chips2.pack(fill="x", pady=(6, 0))
        for label, smi in examples2:
            self._chip(chips2, label, smi).pack(side="left", padx=(0, 6))

        # structure --------------------------------------------------------
        self.structure = tk.Canvas(body, height=196, bg="#FBFCFE",
                                   highlightthickness=1, highlightbackground=BORDER)
        self.structure.pack(fill="x", pady=(12, 0))
        self.structure.bind("<Configure>", lambda e: self._draw_structure())

        # options ----------------------------------------------------------
        opts = tk.Frame(body, bg=CARD)
        opts.pack(fill="x", pady=(12, 0))

        tk.Label(opts, text="Show top", font=(self.ui, 9), bg=CARD, fg=MUTED).grid(row=0, column=0, sticky="w")
        self.topn_var = tk.StringVar(value="10")
        topn = ttk.Combobox(opts, textvariable=self.topn_var, width=5, state="readonly",
                            values=("5", "10", "15", "20", "30", "50", "138"))
        topn.grid(row=1, column=0, sticky="w", pady=(3, 0))
        topn.bind("<<ComboboxSelected>>", self._on_topn)

        tk.Label(opts, text="Device", font=(self.ui, 9), bg=CARD, fg=MUTED).grid(
            row=0, column=1, sticky="w", padx=(16, 0))
        self.device_var = tk.StringVar(value="CUDA" if self._cuda_available() else "CPU")
        self.device_box = ttk.Combobox(opts, textvariable=self.device_var, width=7,
                                       state="readonly", values=("CPU", "CUDA"))
        self.device_box.grid(row=1, column=1, sticky="w", padx=(16, 0), pady=(3, 0))
        self.device_box.bind("<<ComboboxSelected>>", self._on_device)

        # actions ----------------------------------------------------------
        self.predict_btn = tk.Button(body, text="Predict odors", command=self.predict,
                                     font=(self.ui, 11, "bold"), bg=ACCENT, fg="white",
                                     activebackground=ACCENT_HOVER, activeforeground="white",
                                     relief="flat", bd=0, cursor="hand2", pady=9)
        self.predict_btn.pack(fill="x", pady=(16, 0))

        self.clear_btn = tk.Button(body, text="Clear", command=self.clear,
                                   font=(self.ui, 9), bg=CARD, fg=MUTED,
                                   activebackground=CARD, activeforeground=TEXT,
                                   relief="flat", bd=0, cursor="hand2")
        self.clear_btn.pack(pady=(6, 0))

    def _chip(self, parent, label, smiles):
        b = tk.Button(parent, text=label, font=(self.ui, 8),
                      bg=ACCENT_SOFT, fg=ACCENT, activebackground="#DCE8FD",
                      activeforeground=ACCENT, relief="flat", bd=0,
                      cursor="hand2", padx=8, pady=3,
                      command=lambda: self.use_example(smiles))
        return b

    def _build_results_card(self, parent):
        card = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=1, sticky="nsew")
        body = tk.Frame(card, bg=CARD)
        body.pack(fill="both", expand=True, padx=18, pady=16)

        head = tk.Frame(body, bg=CARD)
        head.pack(fill="x")
        tk.Label(head, text="Predictions", font=(self.ui, 13, "bold"),
                 bg=CARD, fg=TEXT).pack(side="left")
        self.count_label = tk.Label(head, text="", font=(self.ui, 9),
                                    bg=CARD, fg=FAINT)
        self.count_label.pack(side="left", padx=(8, 0), pady=(2, 0))

        self.copy_json_btn = self._small_button(head, "Copy JSON", lambda: self.copy_results("json"))
        self.copy_json_btn.pack(side="right")
        self.copy_txt_btn = self._small_button(head, "Copy text", lambda: self.copy_results("txt"))
        self.copy_txt_btn.pack(side="right", padx=(0, 6))

        self.mol_label = tk.Label(body, text="No molecule yet", font=(self.mono, 9),
                                  bg=CARD, fg=FAINT, anchor="w")
        self.mol_label.pack(fill="x", pady=(2, 10))

        tk.Frame(body, bg=BORDER, height=1).pack(fill="x")

        wrap = tk.Frame(body, bg=CARD)
        wrap.pack(fill="both", expand=True)
        self.results = tk.Canvas(wrap, bg=CARD, highlightthickness=0)
        self.results.pack(side="left", fill="both", expand=True)
        self.vbar = ttk.Scrollbar(wrap, orient="vertical", command=self.results.yview,
                                  style="Vertical.TScrollbar")
        self.vbar.pack(side="right", fill="y")
        self.results.configure(yscrollcommand=self.vbar.set)
        self.results.bind("<Configure>", lambda e: self._redraw())
        self.results.bind("<Motion>", self._on_hover)
        self.results.bind("<Leave>", self._clear_hover)
        self.results.bind("<Button-4>", lambda e: self.results.yview_scroll(-3, "units"))
        self.results.bind("<Button-5>", lambda e: self.results.yview_scroll(3, "units"))
        self.results.bind("<MouseWheel>", lambda e: self.results.yview_scroll(int(-e.delta / 120), "units"))

    def _small_button(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd, font=(self.ui, 8),
                         bg=CARD, fg=MUTED, activebackground=BG, activeforeground=TEXT,
                         relief="flat", bd=0, cursor="hand2", padx=8, pady=3,
                         highlightthickness=1, highlightbackground=BORDER)

    # ------------------------------------------------------------- loading
    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _start_loading(self, device):
        self.loading = True
        self.busy = False
        self.device = device
        self.models = None
        self.rows = []
        self._set_predict_enabled(False)
        self._draw_busy_preview()
        self._animate_status(f"Loading 10-model ensemble on {device.upper()}")
        threading.Thread(target=self._load_worker, args=(device,), daemon=True).start()

    def _load_worker(self, device):
        try:
            import predict_smiles as ps
            models = ps.load_models(device)
            self.root.after(0, self._loaded, models, device)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._load_failed, exc)

    def _loaded(self, models, device):
        self.models = models
        self.loading = False
        self.device = device
        self._stop_animation()
        self._set_predict_enabled(True)
        self._set_status(f"Ready · 10 models · {device.upper()}")
        self._redraw()

    def _load_failed(self, exc):
        self.loading = False
        self._stop_animation()
        self._set_status(f"Could not load the ensemble: {exc}", error=True)
        self.results.delete("all")
        w = max(self.results.winfo_width(), 320)
        h = max(self.results.winfo_height(), 300)
        self.results.create_text(w / 2, h / 2, justify="center",
                                 text="Could not load the 10 checkpoints.\n\n"
                                      "Make sure openpom/models/ensemble_models/experiments_1..10 exist.",
                                 font=(self.ui, 10), fill=DANGER)

    def _set_predict_enabled(self, enabled):
        self.predict_btn.configure(
            state="normal" if enabled else "disabled",
            bg=ACCENT if enabled else ACCENT_DISABLED,
            cursor="hand2" if enabled else "arrow")

    def _animate_status(self, base):
        self._anim_base = base
        self._animate_step(0)

    def _animate_step(self, n):
        if not self.loading:
            return
        self._set_status(self._anim_base + "." * (n % 4))
        self.root.after(450, self._animate_step, n + 1)

    def _stop_animation(self):
        self.loading = False  # stops the loop

    def _set_status(self, text, error=False):
        self.status.configure(text=text, fg=DANGER if error else MUTED)

    # -------------------------------------------------------------- inputs
    def _on_typing(self, _event=None):
        smiles = self.entry_var.get().strip()
        if self._preview_job:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(260, self._draw_structure, smiles)
        if smiles:
            self.hint.configure(text="Press Enter or click Predict", fg=FAINT)
        else:
            self.hint.configure(text="Try CCO (ethanol)", fg=FAINT)

    def use_example(self, smiles):
        self.entry_var.set(smiles)
        self.entry.icursor("end")
        self._draw_structure()
        self.predict()

    def _on_topn(self, _event=None):
        try:
            self.topn = int(self.topn_var.get())
        except ValueError:
            self.topn = 10
        self._redraw()

    def _on_device(self, _event=None):
        device = "cuda" if self.device_var.get() == "CUDA" else "cpu"
        if device != self.device and not self.loading:
            self.rows = []
            self.mol_label.configure(text="No molecule yet")
            self._start_loading(device)

    # ------------------------------------------------------------ structure
    def _draw_busy_preview(self):
        c = self.structure
        c.delete("all")
        w = max(c.winfo_width(), 300)
        h = max(c.winfo_height(), 196)
        c.create_text(w / 2, h / 2, text="Loading ensemble…", font=(self.ui, 9), fill=FAINT)

    def _draw_structure(self, smiles=None):
        c = self.structure
        c.delete("all")
        w = max(c.winfo_width(), 300)
        h = max(c.winfo_height(), 196)
        if smiles is None:
            smiles = self.entry_var.get().strip()

        if not smiles:
            c.create_text(w / 2, h / 2, text="Structure preview", font=(self.ui, 9), fill=FAINT)
            return
        if not self.rdkit:
            c.create_text(w / 2, h / 2, text="RDKit not available", font=(self.ui, 9), fill=FAINT)
            return
        if "," in smiles:
            c.create_text(w / 2, h / 2, text="One SMILES at a time", font=(self.ui, 9), fill=DANGER)
            return

        mol = self.Chem.MolFromSmiles(smiles)
        if mol is None:
            c.create_text(w / 2, h / 2, text="Invalid SMILES", font=(self.ui, 10, "bold"), fill=DANGER)
            return

        try:
            img = self.Draw.MolToImage(mol, size=(w - 24, h - 20), kekulize=True)
            self._photo = self.ImageTk.PhotoImage(img)
            c.create_image(w / 2, h / 2, image=self._photo)
        except Exception:  # noqa: BLE001
            c.create_text(w / 2, h / 2, text="Could not render structure",
                          font=(self.ui, 9), fill=FAINT)
            self._photo = None

    # -------------------------------------------------------------- predict
    def predict(self):
        if self.loading:
            self._flash_hint("Ensemble is still loading…", DANGER)
            return
        if self.busy:
            return
        smiles = self.entry_var.get().strip()
        if not smiles:
            self._flash_hint("Enter a SMILES string first", DANGER)
            self.entry.focus_set()
            return
        if "," in smiles:
            self._flash_hint("Only one SMILES at a time", DANGER)
            return
        if self.rdkit and self.Chem.MolFromSmiles(smiles) is None:
            self._flash_hint("That is not a valid SMILES", DANGER)
            self._draw_structure(smiles)
            return
        if self.models is None:
            self._flash_hint("Ensemble not loaded yet", DANGER)
            return

        self.busy = True
        self.smiles = smiles
        self._set_predict_enabled(False)
        self.predict_btn.configure(text="Predicting…", bg=ACCENT)
        self._set_status("Predicting…")
        self.hint.configure(text="Running the 10-model ensemble", fg=FAINT)
        self.mol_label.configure(text=smiles)
        self._draw_structure(smiles)
        self._clear_hover()
        self._show_loading_rows()

        start = time.perf_counter()
        threading.Thread(target=self._predict_worker, args=(smiles, start),
                         daemon=True).start()

    def _predict_worker(self, smiles, start):
        try:
            import predict_smiles as ps
            scores = ps.predict(self.models, smiles)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._predict_failed, exc)
            return
        elapsed = time.perf_counter() - start
        self.root.after(0, self._predict_done, scores, elapsed)

    def _predict_done(self, scores, elapsed):
        import predict_smiles as ps
        flat = scores.reshape(-1)
        order = sorted(range(len(flat)), key=lambda i: flat[i], reverse=True)
        self.rows = [(ps.TASKS[i], float(flat[i])) for i in order]
        self.busy = False
        self._set_predict_enabled(True)
        self.predict_btn.configure(text="Predict odors")
        self.hint.configure(text="Press Enter or click Predict", fg=FAINT)
        self._set_status(f"Ready · 10 models · {self.device.upper()} · "
                         f"{elapsed:.2f}s · {len(self.rows)} odors scored")
        self.count_label.configure(text=f"top {self.topn} of {len(self.rows)}")
        self.results.yview_moveto(0)
        self._redraw()
        self._enable_copy(True)

    def _predict_failed(self, exc):
        self.busy = False
        self._set_predict_enabled(True)
        self.predict_btn.configure(text="Predict odors")
        self._set_status(f"Prediction failed: {exc}", error=True)
        self._flash_hint("Prediction failed", DANGER)
        self._redraw()

    def _show_loading_rows(self):
        c = self.results
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 300)
        c.create_text(w / 2, h / 2, text="Scoring 138 smell words…",
                      font=(self.ui, 10), fill=MUTED)
        c.configure(scrollregion=(0, 0, w, h))
        self.count_label.configure(text="")
        self._enable_copy(False)

    def _enable_copy(self, on):
        state = "normal" if on else "disabled"
        fg = MUTED if on else "#C7CBD1"
        for b in (self.copy_txt_btn, self.copy_json_btn):
            b.configure(state=state, fg=fg,
                        cursor="hand2" if on else "arrow")

    def _flash_hint(self, text, color):
        self.hint.configure(text=text, fg=color)
        self.root.after(2500, lambda: self.hint.configure(
            text="Press Enter or click Predict", fg=FAINT))

    # --------------------------------------------------------------- drawing
    def _visible_rows(self):
        return self.rows[: self.topn]

    def _redraw(self):
        c = self.results
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 300)
        rows = self._visible_rows()

        if not rows:
            c.create_text(w / 2, h / 2 - 12, text="Nothing to show yet",
                          font=(self.ui, 13, "bold"), fill="#4B5563")
            c.create_text(w / 2, h / 2 + 16,
                          text="Type a SMILES string and press Predict",
                          font=(self.ui, 10), fill=FAINT)
            c.configure(scrollregion=(0, 0, w, h))
            return

        n = len(rows)
        total = self.PAD * 2 + n * self.ROW_H
        for i, (name, value) in enumerate(rows):
            y = self.PAD + i * self.ROW_H

            if self.hover == i:
                rounded_rect(c, 6, y - 4, w - 6, y + self.ROW_H - 10, 10,
                             fill=ROW_HOVER, outline="")

            # rank badge
            cx, cy, rad = 27, y + 17, 11
            color = blend(ACCENT, "#8FB4F8", (i / max(n - 1, 1)) * 0.85)
            c.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                          fill=ACCENT_SOFT, outline="")
            c.create_text(cx, cy, text=str(i + 1), font=(self.ui, 8, "bold"), fill=ACCENT)

            # name + score
            c.create_text(48, y + 10, text=name, anchor="w",
                          font=(self.ui, 11, "bold"), fill=TEXT)
            c.create_text(w - 20, y + 10, text=f"{value:.3f}", anchor="e",
                          font=(self.ui, 11, "bold"), fill=TEXT)

            # bar
            x1, x2, by = 48, w - 20, y + 30
            bh = 7
            rounded_rect(c, x1, by, x2, by + bh, bh / 2, fill=TRACK, outline="")
            frac = max(0.0, min(1.0, value))
            fill_w = (x2 - x1) * frac
            if fill_w >= bh:
                rounded_rect(c, x1, by, x1 + fill_w, by + bh, bh / 2,
                             fill=color, outline="")
            else:
                c.create_oval(x1, by, x1 + bh, by + bh, fill=color, outline="")

        c.configure(scrollregion=(0, 0, w, max(total, h)))

    def _on_hover(self, event):
        y = self.results.canvasy(event.y)
        rows = self._visible_rows()
        idx = int((y - self.PAD + 4) // self.ROW_H)
        frac = (y - self.PAD + 4) % self.ROW_H
        if not (0 <= idx < len(rows)) or frac > self.ROW_H - 8:
            idx = None
        if idx != self.hover:
            self.hover = idx
            self._redraw()
        self.results.configure(cursor="hand2" if idx is not None else "")

    def _clear_hover(self, _event=None):
        if self.hover is not None:
            self.hover = None
            self._redraw()
        self.results.configure(cursor="")

    # ---------------------------------------------------------------- clears
    def clear(self):
        if self.loading:
            return
        self.entry_var.set("")
        self.rows = []
        self.smiles = ""
        self.hover = None
        self.mol_label.configure(text="No molecule yet")
        self.count_label.configure(text="")
        self.hint.configure(text="Try CCO (ethanol)", fg=FAINT)
        self._draw_structure("")
        self._enable_copy(False)
        self._redraw()
        self.entry.focus_set()

    # ---------------------------------------------------------------- copies
    def copy_results(self, fmt):
        if not self.rows:
            self._set_status("Nothing to copy yet", error=True)
            self.root.after(1800, lambda: self._set_status(
                f"Ready · 10 models · {self.device.upper()}" if self.models else "Ready"))
            return
        rows = self._visible_rows()
        if fmt == "json":
            text = json.dumps([{"odor": n, "prediction": round(v, 5)} for n, v in rows],
                              indent=2)
        else:
            text = "Odor\tPrediction\n" + "\n".join(
                f"{n}\t{v:.5f}" for n, v in rows)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        self._set_status(f"Copied {len(rows)} results as {fmt.upper()}")
        self.root.after(2000, lambda: self._set_status(
            f"Ready · 10 models · {self.device.upper()}" if self.models else "Ready"))

    def about(self):
        messagebox.showinfo(
            "About",
            "smell-agent odor predictor\n\n"
            "MPNN-POM · 138 smell words · 10-model ensemble\n"
            "Predictions are the mean of ten message-passing networks\n"
            "trained on the OpenPOM curated dataset (4,983 molecules).\n\n"
            "Data: ARY2260/openpom\n"
            "UI: Tkinter + RDKit structure rendering",
            parent=self.root)


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    app = OdorApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
