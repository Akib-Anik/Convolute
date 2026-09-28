from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector

import analysis
import dsp_ops as dsp
from audio_engine import (AudioClip, Clipboard, History, effective_bit_depth,
                          file_dialog_types)
from player import Player
from theme import Theme
from widgets import (Card, ChoiceRow, LevelMeter, NumberRow, SliderRow,
                     StatusToast, action_button, format_time, readout_label)

STUDENT_NAME = "Habibullah Khan"
STUDENT_ID = "2305002"
COURSE_INFO = "CSE Sessional - Signal Processing, Dept. of CSE, BUET"

APP_TITLE = "Convolute"
APP_VERSION = "2.0"

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".mini_audio_editor.json")

SPECTROGRAM_FFT = 1024
SPECTRUM_FFT = 4096
ANALYSIS_WINDOW = "hann"


class _SharedBlitSpanSelector(SpanSelector):
    # A blitting SpanSelector that reuses the app's blit background instead
    # of keeping its own. Its own copy would cost a second full redraw after
    # every draw, and the app's playhead blit would wipe its rectangle. With
    # blitting, dragging a selection repaints only the overlays instead of
    # the whole plot, which would otherwise starve the audio callback.
    def __init__(self, *args, background, **kwargs):
        self._shared_background = background
        super().__init__(*args, useblit=True, **kwargs)

    def update_background(self, event):
        pass

    def _save_blit_background(self, background):
        pass

    def _load_blit_background(self):
        return self._shared_background()

    def update(self):
        # With no background yet a full draw is on its way, and the app's
        # draw handler paints the selection then.
        if self._shared_background() is None:
            return
        super().update()


class MiniAudioEditorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.settings = self._load_settings()
        self.theme = Theme.use("dark")
        ctk.set_appearance_mode("dark")

        self.title(f"{APP_TITLE} {APP_VERSION}")
        self.geometry("1460x900")
        self.minsize(1120, 720)
        self.configure(fg_color=self.theme.palette.bg)

        self.clip: AudioClip | None = None
        self.original_clip: AudioClip | None = None
        self.filepath: str | None = None
        self.dirty = False
        self.history = History(max_depth=50)
        self.clipboard = Clipboard()
        self.selection: tuple[float, float] | None = None
        self.cursor: float | None = None
        self._has_played = False
        self._doc_generation = 0

        self.player = Player()

        self.view_mode = "Waveform"
        self.compare = False
        self.ax_orig = None
        self._overlay_clip: AudioClip | None = None
        self._span_selector = None
        self._playhead_line = None
        self._wave_artist = None
        self._orig_wave_artist = None
        self._time_view: tuple[float, float] | None = None
        self._playhead_pos = None
        self._press_at = None
        self._blit_bg = None
        self._resize_after = None
        self._ui_after = None
        self._last_plot_signature = None
        self._busy = False
        self._busy_label = ""
        self._warned_fallback = False
        self._results: queue.Queue = queue.Queue()

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._refresh_all()
        self._start_ui_timer()

        if not Player.available():
            self.after(400, lambda: self._toast(
                "No audio output found — editing and export still work.",
                "warning", 8000))

    def _load_settings(self) -> dict:
        import json
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_settings(self) -> None:
        import json
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
                json.dump(self.settings, fh, indent=2)
        except Exception:
            pass

    def _build_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_menu()
        self._build_header()
        self._build_body()
        self._build_transport()
        self._build_statusbar()

    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open…", accelerator="Ctrl+O", command=self.open_audio)
        file_menu.add_command(label="Append / Join…", command=self.append_audio)
        file_menu.add_command(label="Insert Audio at Position…",
                              command=self.insert_audio_file)
        file_menu.add_command(label="Overlay / Mix…", command=self.overlay_dialog)
        file_menu.add_command(label="Generate Tone…", command=self.generate_tone_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export…", accelerator="Ctrl+S", command=self.export_audio)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", accelerator="Ctrl+Q", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        edit_menu.add_command(label="Redo", accelerator="Ctrl+Shift+Z", command=self.redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", accelerator="Ctrl+X", command=self.cut)
        edit_menu.add_command(label="Copy", accelerator="Ctrl+C", command=self.copy)
        edit_menu.add_command(label="Paste", accelerator="Ctrl+V", command=self.paste)
        edit_menu.add_command(label="Delete", accelerator="Del", command=self.delete_selection)
        edit_menu.add_command(label="Silence Selection", command=self.silence_selection)
        edit_menu.add_separator()
        edit_menu.add_command(label="Select All", accelerator="Ctrl+A", command=self.select_all)
        edit_menu.add_command(label="Deselect", accelerator="Esc", command=self.clear_selection)
        edit_menu.add_separator()
        edit_menu.add_command(label="Revert to Original", command=self.reset_to_original)
        self.edit_menu = edit_menu
        menubar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Waveform", command=lambda: self.set_view("Waveform"))
        view_menu.add_command(label="Spectrogram", command=lambda: self.set_view("Spectrogram"))
        view_menu.add_command(label="Spectrum", command=lambda: self.set_view("Spectrum"))
        view_menu.add_separator()
        view_menu.add_command(label="Before / After", command=self.toggle_compare)
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", accelerator="Ctrl++", command=lambda: self.zoom(0.5))
        view_menu.add_command(label="Zoom Out", accelerator="Ctrl+-", command=lambda: self.zoom(2.0))
        view_menu.add_command(label="Zoom to Selection", command=self.zoom_to_selection)
        view_menu.add_command(label="Zoom to Fit", accelerator="Ctrl+0", command=self.zoom_to_fit)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_header(self):
        t = self.theme
        bar = ctk.CTkFrame(self, fg_color=t.palette.surface, corner_radius=0,
                           height=52)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w", padx=t.space.lg, pady=t.space.sm)
        ctk.CTkLabel(left, text="◼◣", font=(t.type.family, 18, "bold"),
                     text_color=t.palette.accent).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(left, text=APP_TITLE, font=t.font("lg", "bold"),
                     text_color=t.palette.text).pack(side="left")

        centre = ctk.CTkFrame(bar, fg_color="transparent")
        centre.grid(row=0, column=1)
        self.file_label = ctk.CTkLabel(
            centre, text="No file open", font=t.font("sm"),
            text_color=t.palette.text_dim)
        self.file_label.pack(side="left")
        self.dirty_dot = ctk.CTkLabel(centre, text="", font=t.font("sm", "bold"),
                                      text_color=t.palette.accent, width=16)
        self.dirty_dot.pack(side="left")

        self._build_toolbar()

    def _build_toolbar(self):
        t = self.theme
        bar = ctk.CTkFrame(self, fg_color=t.palette.bg, corner_radius=0, height=44)
        bar.grid(row=1, column=0, sticky="ew", padx=t.space.md, pady=(t.space.sm, 0))

        def group():
            frame = ctk.CTkFrame(bar, fg_color="transparent")
            frame.pack(side="left", padx=(0, t.space.md))
            return frame

        def sep():
            ctk.CTkFrame(bar, fg_color=t.palette.border, width=1, height=22
                         ).pack(side="left", padx=(0, t.space.md), pady=8)

        g = group()
        action_button(g, "Open", self.open_audio, "secondary", width=64).pack(side="left", padx=2)
        action_button(g, "Export", self.export_audio, "secondary", width=68).pack(side="left", padx=2)
        sep()

        g = group()
        self.btn_undo = action_button(g, "Undo", self.undo, "ghost", width=54)
        self.btn_undo.pack(side="left", padx=2)
        self.btn_redo = action_button(g, "Redo", self.redo, "ghost", width=54)
        self.btn_redo.pack(side="left", padx=2)
        sep()

        g = group()
        for label, cmd, width in (("Cut", self.cut, 48), ("Copy", self.copy, 54),
                                  ("Paste", self.paste, 56),
                                  ("Delete", self.delete_selection, 60)):
            btn = action_button(g, label, cmd, "ghost", width=width)
            btn.pack(side="left", padx=2)
            setattr(self, f"btn_{label.lower()}", btn)
        sep()

        g = group()
        action_button(g, "Zoom In", lambda: self.zoom(0.5), "ghost", width=64).pack(side="left", padx=2)
        action_button(g, "Zoom Out", lambda: self.zoom(2.0), "ghost", width=70).pack(side="left", padx=2)
        action_button(g, "Fit", self.zoom_to_fit, "ghost", width=44).pack(side="left", padx=2)
        action_button(g, "Zoom Sel", self.zoom_to_selection, "ghost", width=70).pack(side="left", padx=2)

        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.pack(side="right")
        self.snap_var = ctk.BooleanVar(value=self.settings.get("snap_zero", True))
        ctk.CTkCheckBox(
            right, text="Snap to zero crossings", variable=self.snap_var,
            font=t.font("xs"), text_color=t.palette.text_dim,
            checkbox_width=16, checkbox_height=16, border_width=1,
            fg_color=t.palette.accent, hover_color=t.palette.accent_hover,
            command=self._on_snap_toggle).pack(side="left", padx=t.space.sm)

    def _on_snap_toggle(self):
        self.settings["snap_zero"] = bool(self.snap_var.get())
        self._save_settings()

    def _build_body(self):
        t = self.theme
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew", padx=t.space.md, pady=t.space.sm)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)

        self._build_view_area(body)
        self._build_sidebar(body)

    def _build_view_area(self, parent):
        t = self.theme
        panel = ctk.CTkFrame(parent, fg_color=t.palette.surface,
                             corner_radius=t.space.radius,
                             border_width=1, border_color=t.palette.border)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, t.space.sm))
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(panel, fg_color="transparent", height=44)
        head.grid(row=0, column=0, sticky="ew", padx=t.space.md, pady=(t.space.sm, 0))
        head.grid_columnconfigure(1, weight=1)

        self.view_switch = ctk.CTkSegmentedButton(
            head, values=["Waveform", "Spectrogram", "Spectrum"],
            command=self.set_view, font=t.font("sm"), height=30,
            selected_color=t.palette.accent,
            selected_hover_color=t.palette.accent_hover,
            unselected_color=t.palette.surface_alt,
            unselected_hover_color=t.palette.border,
            text_color=t.palette.text, corner_radius=t.space.radius_sm)
        self.view_switch.set("Waveform")
        self.view_switch.grid(row=0, column=0, sticky="w")

        self.btn_compare = action_button(head, "Before / After",
                                         self.toggle_compare, "secondary",
                                         width=110)
        self.btn_compare.grid(row=0, column=1, sticky="w", padx=(t.space.sm, 0))

        holder = ctk.CTkFrame(panel, fg_color=t.palette.plot_bg,
                              corner_radius=t.space.radius_sm)
        holder.grid(row=1, column=0, sticky="nsew",
                    padx=t.space.md, pady=(t.space.sm, t.space.md))

        self.fig = Figure(figsize=(10, 5), dpi=100, constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.colorbar = None
        t.style_axes(self.fig, self.ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        widget = self.canvas.get_tk_widget()
        widget.configure(highlightthickness=0, bd=0)
        widget.pack(fill="both", expand=True, padx=2, pady=2)
        widget.bind("<Configure>", self._on_canvas_resize, add="+")
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        self.canvas.mpl_connect("draw_event", self._on_canvas_draw)

    def _build_sidebar(self, parent):
        t = self.theme
        side = ctk.CTkFrame(parent, fg_color="transparent", width=330)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.grid_rowconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(
            side, width=330, corner_radius=t.space.radius,
            fg_color=t.palette.surface,
            segmented_button_fg_color=t.palette.surface_alt,
            segmented_button_selected_color=t.palette.accent,
            segmented_button_selected_hover_color=t.palette.accent_hover,
            segmented_button_unselected_color=t.palette.surface_alt,
            segmented_button_unselected_hover_color=t.palette.border,
            text_color=t.palette.text, border_width=1,
            border_color=t.palette.border)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        self.tabs._segmented_button.configure(font=t.font("xs", "bold"))

        for name in ("Edit", "Level", "Effects", "Combine", "Info"):
            self.tabs.add(name)

        self._build_edit_tab(self._scroll_area(self.tabs.tab("Edit")))
        self._build_level_tab(self._scroll_area(self.tabs.tab("Level")))
        self._build_effects_tab(self._scroll_area(self.tabs.tab("Effects")))
        self._build_combine_tab(self._scroll_area(self.tabs.tab("Combine")))
        self._build_info_tab(self._scroll_area(self.tabs.tab("Info")))

    def _scroll_area(self, parent) -> ctk.CTkScrollableFrame:
        t = self.theme
        frame = ctk.CTkScrollableFrame(parent, fg_color="transparent",
                                       scrollbar_button_color=t.palette.border,
                                       scrollbar_button_hover_color=t.palette.text_faint)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        return frame

    def _build_edit_tab(self, root):
        t = self.theme

        card = Card(root, "Selection",
                    "Drag on the waveform to select. Times update as you drag.")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.sel_start = NumberRow(body, "Start", "0.000", "s", minimum=0.0)
        self.sel_start.pack(fill="x")
        self.sel_end = NumberRow(body, "End", "0.000", "s", minimum=0.0)
        self.sel_end.pack(fill="x")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))
        action_button(row, "Apply", self.apply_selection_fields, "secondary",
                      width=60).pack(side="left", expand=True, fill="x",
                                     padx=(0, 3))
        action_button(row, "Select All", self.select_all, "secondary",
                      width=60).pack(side="left", expand=True, fill="x", padx=3)
        action_button(row, "Clear", self.clear_selection, "ghost",
                      width=60).pack(side="left", expand=True, fill="x",
                                     padx=(3, 0))
        action_button(body, "Trim to Selection", self.apply_trim, "primary").pack(
            fill="x", pady=(6, 0))

        card = Card(root, "Reverse")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        action_button(row, "Whole Clip", lambda: self.apply_reverse(False),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(0, 3))
        action_button(row, "Selection", lambda: self.apply_reverse(True),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(3, 0))

        card = Card(root, "Speed",
                    "Resampling changes speed and pitch together, the way "
                    "playing a tape faster or slower does.")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.speed_slider = SliderRow(body, "Resample speed", 0.25, 4.0, 1.0,
                                      "{:.2f}", "×")
        self.speed_slider.pack(fill="x")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        action_button(row, "Whole Clip", lambda: self.apply_time_scale(False),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(0, 3))
        action_button(row, "Selection", lambda: self.apply_time_scale(True),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(3, 0))

    def _build_level_tab(self, root):
        t = self.theme

        card = Card(root, "Gain")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.gain_slider = SliderRow(body, "Gain", -30, 12, 0, "{:+.1f}", " dB")
        self.gain_slider.pack(fill="x")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        action_button(row, "Whole Clip", lambda: self.apply_gain(False),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(0, 3))
        action_button(row, "Selection", lambda: self.apply_gain(True),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(3, 0))

        card = Card(root, "Normalize",
                    "Peak matches the loudest sample; RMS matches average "
                    "power, which tracks perceived loudness more closely.")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.norm_mode = ChoiceRow(body, "Mode", ["peak", "rms"], "peak")
        self.norm_mode.pack(fill="x")
        self.norm_target = SliderRow(body, "Target", -30, 0, -1, "{:.1f}", " dB")
        self.norm_target.pack(fill="x")
        action_button(body, "Normalize", self.apply_normalize, "primary").pack(fill="x")

        card = Card(root, "Fade")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.fade_curve = ChoiceRow(body, "Curve", dsp.FADE_CURVES, "s-curve", width=140)
        self.fade_curve.pack(fill="x")
        self.fade_dur = NumberRow(body, "Duration", "1.0", "s", minimum=0.001,
                                  maximum=600)
        self.fade_dur.pack(fill="x")
        ctk.CTkLabel(body, text="Whole clip", font=t.font("xs"),
                     text_color=t.palette.text_faint, anchor="w").pack(
            fill="x", pady=(6, 2))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        action_button(row, "Fade In", lambda: self.apply_fade("in", False),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(0, 3))
        action_button(row, "Fade Out", lambda: self.apply_fade("out", False),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(3, 0))
        ctk.CTkLabel(body, text="Selection", font=t.font("xs"),
                     text_color=t.palette.text_faint, anchor="w").pack(
            fill="x", pady=(8, 2))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        action_button(row, "Fade In", lambda: self.apply_fade("in", True),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(0, 3))
        action_button(row, "Fade Out", lambda: self.apply_fade("out", True),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(3, 0))

    def _build_effects_tab(self, root):
        t = self.theme

        card = Card(root, "Echo",
                    "Convolution with a decaying impulse train. Runs through "
                    "the FFT, so clip length barely affects the time it takes.")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.echo_delay = NumberRow(body, "Delay", "300", "ms", minimum=1, maximum=5000)
        self.echo_delay.pack(fill="x")
        self.echo_decay = SliderRow(body, "Decay", 0.05, 0.95, 0.5)
        self.echo_decay.pack(fill="x")
        self.echo_repeats = NumberRow(body, "Repeats", "4", "", minimum=1,
                                      maximum=50, integer=True)
        self.echo_repeats.pack(fill="x")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))
        action_button(row, "Whole Clip", lambda: self.apply_echo(False),
                      "primary").pack(side="left", expand=True, fill="x", padx=(0, 3))
        action_button(row, "Selection", lambda: self.apply_echo(True),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(3, 0))

        card = Card(root, "Reverb",
                    "Convolution with a synthesised room response: "
                    "exponentially decaying noise, damped at high frequencies.")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.rev_decay = SliderRow(body, "Decay time", 0.1, 6.0, 1.2, "{:.2f}", " s")
        self.rev_decay.pack(fill="x")
        self.rev_damping = SliderRow(body, "Damping", 0.0, 0.95, 0.35)
        self.rev_damping.pack(fill="x")
        self.rev_predelay = SliderRow(body, "Pre-delay", 0, 200, 20, "{:.0f}", " ms")
        self.rev_predelay.pack(fill="x")
        self.rev_mix = SliderRow(body, "Wet mix", 0.0, 1.0, 0.3)
        self.rev_mix.pack(fill="x")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))
        action_button(row, "Whole Clip", lambda: self.apply_reverb(False),
                      "primary").pack(side="left", expand=True, fill="x", padx=(0, 3))
        action_button(row, "Selection", lambda: self.apply_reverb(True),
                      "secondary").pack(side="left", expand=True, fill="x", padx=(3, 0))

    def _build_combine_tab(self, root):
        t = self.theme

        card = Card(root, "Insert / Join Audio",
                    "Insert another file inside the clip, or append it at "
                    "the end. The crossfade smooths each join.")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.insert_pos = NumberRow(body, "Position", "0.000", "s", minimum=0.0)
        self.insert_pos.pack(fill="x")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))
        action_button(row, "Use Cursor",
                      lambda: self._fill_from_cursor(self.insert_pos),
                      "ghost", width=60).pack(side="left", expand=True,
                                              fill="x", padx=(0, 3))
        action_button(row, "Insert File…", self.insert_audio_file,
                      "primary", width=60).pack(side="left", expand=True,
                                                fill="x", padx=(3, 0))
        action_button(body, "Append File at End…", self.append_audio,
                      "secondary").pack(fill="x", pady=(4, 0))
        self.join_crossfade = NumberRow(body, "Crossfade", "20", "ms",
                                        minimum=0, maximum=5000)
        self.join_crossfade.pack(fill="x", pady=(4, 0))

        card = Card(root, "Insert Silence",
                    "Adds a gap of silence at the chosen position.")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.silence_pos = NumberRow(body, "Position", "0.000", "s",
                                     minimum=0.0)
        self.silence_pos.pack(fill="x")
        self.silence_dur = NumberRow(body, "Duration", "1.0", "s",
                                     minimum=0.001, maximum=600)
        self.silence_dur.pack(fill="x")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))
        action_button(row, "Use Cursor",
                      lambda: self._fill_from_cursor(self.silence_pos),
                      "ghost", width=60).pack(side="left", expand=True,
                                              fill="x", padx=(0, 3))
        action_button(row, "Insert Silence", self.insert_silence,
                      "primary", width=60).pack(side="left", expand=True,
                                                fill="x", padx=(3, 0))

        card = Card(root, "Overlay / Mix",
                    "Add another file on top of this one, e.g. vocals over "
                    "an instrumental. The clip grows if the overlay runs "
                    "past its end, and the mix is scaled down only if it "
                    "would clip.")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        row.columnconfigure(0, weight=1)
        self.overlay_name = ctk.CTkLabel(
            row, text="No file chosen", font=t.font("sm"),
            text_color=t.palette.text_dim, anchor="w", justify="left",
            wraplength=190)
        self.overlay_name.grid(row=0, column=0, sticky="w")
        action_button(row, "Choose…", self.choose_overlay_file, "secondary",
                      width=76).grid(row=0, column=1, sticky="e")
        self.overlay_start = NumberRow(body, "Start at", "0.000", "s",
                                       minimum=0.0)
        self.overlay_start.pack(fill="x", pady=(6, 0))
        action_button(body, "Use Cursor",
                      lambda: self._fill_from_cursor(self.overlay_start),
                      "ghost").pack(fill="x")
        self.overlay_gain = SliderRow(body, "Overlay gain", -30, 12, 0,
                                      "{:+.1f}", " dB")
        self.overlay_gain.pack(fill="x", pady=(6, 0))
        action_button(body, "Mix into Clip", self.mix_overlay,
                      "primary").pack(fill="x", pady=(2, 0))

    def _build_info_tab(self, root):
        t = self.theme

        card = Card(root, "Clip Statistics")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.stats_rows = {}
        self.stats_vars = {}
        for label, _ in analysis.format_statistics(
                analysis.clip_statistics(np.zeros(0), 44100)):
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=label, font=t.font("xs"),
                         text_color=t.palette.text_dim, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            value = readout_label(row, var, font=t.mono_font("xs"),
                                  colour=t.palette.text,
                                  background=t.palette.surface, width=16)
            value.pack(side="right")
            self.stats_rows[label] = value
            self.stats_vars[label] = var
        self.stats_scope = ctk.CTkLabel(
            body, text="Whole clip", font=t.font("xs"),
            text_color=t.palette.text_faint, anchor="w")
        self.stats_scope.pack(fill="x", pady=(6, 0))
        action_button(body, "Refresh", self.refresh_statistics, "secondary").pack(
            fill="x", pady=(6, 0))

        card = Card(root, "Generate")
        card.pack(fill="x", pady=(0, t.space.sm))
        body = card.body()
        self.gen_wave = ChoiceRow(body, "Waveform", dsp.WAVEFORMS, "sine", width=140)
        self.gen_wave.pack(fill="x")
        self.gen_freq = NumberRow(body, "Frequency", "440", "Hz", minimum=1,
                                  maximum=20000)
        self.gen_freq.pack(fill="x")
        self.gen_dur = NumberRow(body, "Duration", "3.0", "s", minimum=0.01,
                                 maximum=600)
        self.gen_dur.pack(fill="x")
        self.gen_sr = ChoiceRow(body, "Sample rate",
                                ["8000", "22050", "44100", "48000"], "44100")
        self.gen_sr.pack(fill="x")
        action_button(body, "Generate New Clip", self.generate_tone,
                      "primary").pack(fill="x", pady=(6, 0))

    def _build_transport(self):
        t = self.theme
        bar = ctk.CTkFrame(self, fg_color=t.palette.surface,
                           corner_radius=t.space.radius, height=64,
                           border_width=1, border_color=t.palette.border)
        bar.grid(row=3, column=0, sticky="ew", padx=t.space.md, pady=(0, t.space.sm))
        bar.grid_propagate(False)
        bar.grid_columnconfigure(2, weight=1)

        controls = ctk.CTkFrame(bar, fg_color="transparent")
        controls.grid(row=0, column=0, padx=t.space.md, pady=t.space.sm)
        self.btn_to_start = action_button(controls, "⏮", self.seek_to_start, "ghost", width=38)
        self.btn_to_start.pack(side="left", padx=2)
        self.btn_play = action_button(controls, "▶", self.toggle_play, "primary", width=52)
        self.btn_play.pack(side="left", padx=2)
        self.btn_stop = action_button(controls, "■", self.stop_playback, "ghost", width=38)
        self.btn_stop.pack(side="left", padx=2)
        self.loop_var = ctk.BooleanVar(value=False)
        self.btn_loop = ctk.CTkSwitch(
            controls, text="Loop", variable=self.loop_var, font=t.font("xs"),
            text_color=t.palette.text_dim, switch_width=34, switch_height=16,
            progress_color=t.palette.accent, command=self._on_loop_toggle)
        self.btn_loop.pack(side="left", padx=(t.space.md, 0))

        times = ctk.CTkFrame(bar, fg_color="transparent")
        times.grid(row=0, column=1, padx=t.space.md)
        self.time_var = tk.StringVar(value="0:00.000")
        self.time_label = tk.Label(times, textvariable=self.time_var,
                                   font=(t.type.mono, 20, "bold"),
                                   fg=t.palette.text, bg=t.palette.surface,
                                   width=10, anchor="w",
                                   bd=0, highlightthickness=0,
                                   padx=0, pady=0)
        self.time_label.pack(anchor="w")
        self.time_total = ctk.CTkLabel(times, text="of 0:00.000",
                                       font=t.mono_font("xs"),
                                       text_color=t.palette.text_faint)
        self.time_total.pack(anchor="w")

        middle = ctk.CTkFrame(bar, fg_color="transparent")
        middle.grid(row=0, column=2, sticky="ew", padx=t.space.md)
        self.sel_var = tk.StringVar(value="No selection")
        self.sel_label = readout_label(middle, self.sel_var, font=t.font("xs"),
                                       colour=t.palette.text_faint,
                                       background=t.palette.surface,
                                       width=44, anchor="w")
        self.sel_label.pack(anchor="w")
        self.scope_var = tk.StringVar(value="")
        self.scope_label = readout_label(
            middle, self.scope_var, font=t.font("xs"),
            colour=t.palette.text_faint, background=t.palette.surface,
            width=44, anchor="w")
        self.scope_label.pack(anchor="w", pady=(4, 0))

        meters = ctk.CTkFrame(bar, fg_color="transparent")
        meters.grid(row=0, column=3, padx=t.space.md)
        self.out_meter = LevelMeter(meters, "L", width=130,
                                    background=t.palette.surface)
        self.out_meter.pack(fill="x", pady=1)
        self.out_meter_r = LevelMeter(meters, "R", width=130,
                                      background=t.palette.surface)
        self.out_meter_r.pack(fill="x", pady=1)

        vol = ctk.CTkFrame(bar, fg_color="transparent")
        vol.grid(row=0, column=4, padx=(0, t.space.lg))
        ctk.CTkLabel(vol, text="Volume", font=t.font("xs"),
                     text_color=t.palette.text_faint).pack(anchor="w")
        self.volume_slider = ctk.CTkSlider(
            vol, from_=0, to=1.5, width=110, height=14,
            command=lambda v: self.player.set_volume(float(v)),
            button_color=t.palette.accent, button_hover_color=t.palette.accent_hover,
            progress_color=t.palette.accent, fg_color=t.palette.surface_alt)
        self.volume_slider.set(1.0)
        self.volume_slider.pack()

    def _on_loop_toggle(self):
        self.player.set_loop(bool(self.loop_var.get()))

    def _build_statusbar(self):
        t = self.theme
        bar = ctk.CTkFrame(self, fg_color="transparent", height=28)
        bar.grid(row=4, column=0, sticky="ew", padx=t.space.md, pady=(0, t.space.sm))
        bar.grid_columnconfigure(0, weight=1)

        self.toast = StatusToast(bar)
        self.toast.grid(row=0, column=0, sticky="ew")

        self.info_var = tk.StringVar(value="")
        self.info_label = readout_label(bar, self.info_var,
                                        font=t.mono_font("xs"),
                                        colour=t.palette.text_faint,
                                        background=t.palette.bg, width=52)
        self.info_label.grid(row=0, column=1, sticky="e", padx=t.space.md)

    def _bind_shortcuts(self):
        mod = "Command" if sys.platform == "darwin" else "Control"
        binds = {
            f"<{mod}-o>": lambda e: self.open_audio(),
            f"<{mod}-s>": lambda e: self.export_audio(),
            f"<{mod}-z>": lambda e: self._if_not_typing(e, self.undo),
            f"<{mod}-Shift-Z>": lambda e: self._if_not_typing(e, self.redo),
            f"<{mod}-y>": lambda e: self._if_not_typing(e, self.redo),
            f"<{mod}-x>": lambda e: self._if_not_typing(e, self.cut),
            f"<{mod}-c>": lambda e: self._if_not_typing(e, self.copy),
            f"<{mod}-v>": lambda e: self._if_not_typing(e, self.paste),
            f"<{mod}-a>": lambda e: self._if_not_typing(e, self.select_all),
            f"<{mod}-q>": lambda e: self._on_close(),
            f"<{mod}-0>": lambda e: self.zoom_to_fit(),
            f"<{mod}-plus>": lambda e: self.zoom(0.5),
            f"<{mod}-equal>": lambda e: self.zoom(0.5),
            f"<{mod}-minus>": lambda e: self.zoom(2.0),
            "<Escape>": lambda e: self.clear_selection(),
            "<Delete>": lambda e: self._if_not_typing(e, self.delete_selection),
            "<BackSpace>": lambda e: self._if_not_typing(e, self.delete_selection),
            "<Home>": lambda e: self._if_not_typing(e, self.seek_to_start),
        }
        for sequence, handler in binds.items():
            self.bind_all(sequence, handler)
        self.bind_all("<space>", self._on_space)

    TEXT_WIDGET_CLASSES = ("Entry", "TEntry", "Text", "TCombobox", "Spinbox",
                           "TSpinbox")

    def _is_typing(self, event) -> bool:
        candidates = [getattr(event, "widget", None)]
        try:
            candidates.append(self.focus_get())
        except Exception:
            pass

        for widget in candidates:
            if widget is None:
                continue
            try:
                if widget.winfo_class() in self.TEXT_WIDGET_CLASSES:
                    return True
                if any(child.winfo_class() in self.TEXT_WIDGET_CLASSES
                       for child in widget.winfo_children()):
                    return True
            except Exception:
                continue
        return False

    def _if_not_typing(self, event, action):
        if self._is_typing(event):
            return None
        action()
        return "break"

    def _on_space(self, event=None):
        if self._is_typing(event):
            return None
        self.toggle_play()
        return "break"

    def _new_document(self):
        self._doc_generation += 1
        self._time_view = None
        self.cursor = None
        self._has_played = False

    def _wait_if_busy(self) -> bool:
        # An effect running in the background works on a snapshot of the
        # clip and replaces the clip when it finishes, so any edit made in
        # the meantime would be silently overwritten. Refuse edits until then.
        if not self._busy:
            return False
        self._toast(f"Wait for {self._busy_label} to finish.", "warning")
        return True

    def _run_async(self, label: str, work, on_done=None):
        if self._wait_if_busy():
            return
        self._busy = True
        self._busy_label = label
        self._toast(f"{label}…", "info", 60000)
        self.configure(cursor="watch")
        generation = self._doc_generation

        def runner():
            try:
                result, error = work(), None
            except Exception as exc:
                result, error = None, exc
                if not isinstance(exc, (ValueError, RuntimeError)):
                    traceback.print_exc()
            self._results.put((label, result, error, on_done, generation))

        threading.Thread(target=runner, daemon=True).start()

    def _drain_results(self):
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                return
            self._finish_async(*item)

    def _finish_async(self, label, result, error, on_done, generation):
        self._busy = False
        self.configure(cursor="")
        if generation != self._doc_generation or self.clip is None:
            self._toast(f"{label} was discarded — the clip changed.",
                        "warning")
            return
        if error is not None:
            self._toast(f"{label} failed: {error}", "error", 8000)
            return
        if on_done is not None:
            on_done(result)

    def _require_clip(self) -> bool:
        if self.clip is None:
            self._toast("Open or generate a clip first.", "warning")
            return False
        return True

    def _selection_samples(self, required: bool = True):
        if self.clip is None:
            return None
        if self.selection is None:
            if required:
                self._toast("Select a region on the waveform first.", "warning")
            return None
        sr = self.clip.sample_rate
        start = int(round(self.selection[0] * sr))
        end = int(round(self.selection[1] * sr))
        start = max(0, min(start, self.clip.n_samples))
        end = max(0, min(end, self.clip.n_samples))

        # The clip's own start and end are already clean edit points, so
        # never snap them inward (Select All must cover every sample).
        if self.snap_var.get():
            if start > 0:
                start = dsp.find_zero_crossing(self.clip.data, start)
            if end < self.clip.n_samples:
                end = dsp.find_zero_crossing(self.clip.data, end)

        if end <= start:
            if required:
                self._toast("The selection is empty.", "warning")
            return None
        return start, end

    def _playhead_seconds(self) -> float:
        if self.clip is None:
            return 0.0
        position = (self.player.position() if self.player.is_playing
                    else (self.cursor or 0.0))
        return float(min(max(position, 0.0), self.clip.duration))

    def _cursor_sample(self) -> int:
        if self.clip is None:
            return 0
        if self.selection is not None:
            return int(round(self.selection[0] * self.clip.sample_rate))
        return int(round(self._playhead_seconds() * self.clip.sample_rate))

    def _halt_playback_for_edit(self) -> bool:
        if not self.player.is_playing:
            return False
        self.cursor = self.player.position()
        self.stop_playback()
        return True

    def _park_playhead(self):
        if self.clip is None:
            self.cursor = None
            return
        self.cursor = float(min(max(self.cursor or 0.0, 0.0),
                                self.clip.duration))
        self._draw_playhead(self.cursor)
        self.time_var.set(format_time(self.cursor))

    def _rewind_after_edit(self, restart: bool = False):
        self.cursor = self._play_range()[0]
        self._park_playhead()
        if restart:
            self.play()

    def _push_undo(self, label: str):
        if self.clip is not None:
            self.history.push(self.clip.data, self.clip.sample_rate, label)

    def _commit(self, data: np.ndarray, message: str, *, view_map=None,
                keep_selection: bool = False):
        assert self.clip is not None
        halted = self._halt_playback_for_edit()
        self.clip.data = np.asarray(data, dtype=np.float64)
        if view_map is not None and self._time_view is not None:
            self._time_view = view_map(*self._time_view)
        self.dirty = True
        if not keep_selection:
            self.selection = None
        else:
            duration = self.clip.duration
            if self.selection is not None:
                lo = min(self.selection[0], duration)
                hi = min(self.selection[1], duration)
                self.selection = (lo, hi) if hi - lo > 1e-4 else None
        self._refresh_all()
        self._rewind_after_edit(restart=halted)
        if halted:
            message = f"{message} Playback restarted from {format_time(self.cursor)}."
        peak = self.clip.peak()
        if peak > 1.0:
            over = dsp.linear_to_db(peak)
            self._toast(f"{message} Peak is {over:+.1f} dB over full scale — "
                        "it will distort on export. Try Normalize.",
                        "warning", 9000)
            return
        self._toast(message, "success")

    def _edit(self, label: str, func, *, view_map=None,
              keep_selection: bool = False, async_: bool = False):
        if self._wait_if_busy():
            return
        if not self._require_clip():
            return
        assert self.clip is not None
        data, sr = self.clip.data, self.clip.sample_rate

        def apply(result):
            self._push_undo(label)
            self._commit(result, f"{label} applied.", view_map=view_map,
                         keep_selection=keep_selection)

        if async_:
            self._run_async(label, lambda: func(data, sr), apply)
        else:
            try:
                apply(func(data, sr))
            except Exception as exc:
                self._error(label, exc)

    def _edit_region(self, label: str, func, *, async_: bool = False):
        if self._wait_if_busy():
            return
        if not self._require_clip():
            return
        bounds = self._selection_samples()
        if bounds is None:
            return
        start, end = bounds
        assert self.clip is not None
        data, sr = self.clip.data, self.clip.sample_rate
        segment = data[start:end].copy()

        def work():
            result = np.asarray(func(segment, sr), dtype=np.float64)
            out = data.copy()
            n = min(len(result), end - start)
            out[start:start + n] = result[:n]
            if n < end - start:
                out[start + n:end] = 0.0
            return out

        def apply(result):
            self._push_undo(label)
            self._commit(result, f"{label} applied to selection.",
                         keep_selection=True)

        if async_:
            self._run_async(label, work, apply)
        else:
            try:
                apply(work())
            except Exception as exc:
                self._error(label, exc)

    def _error(self, label: str, exc: Exception):
        if isinstance(exc, (ValueError, RuntimeError)):
            self._toast(f"{label}: {exc}", "error", 7000)
        else:
            traceback.print_exc()
            self._toast(f"{label} failed: {type(exc).__name__}: {exc}", "error", 8000)

    def _toast(self, message: str, kind: str = "info", duration: int = 4000):
        self.toast.show(message, kind, duration)

    def open_audio(self):
        path = filedialog.askopenfilename(title="Open Audio",
                                          filetypes=file_dialog_types())
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        if not os.path.exists(path):
            self._toast(f"File not found: {os.path.basename(path)}", "error")
            return
        if not self._confirm_discard():
            return
        try:
            clip = AudioClip.from_file(path)
        except Exception as exc:
            messagebox.showerror("Could not open file", str(exc))
            return
        if clip.n_samples == 0:
            self._toast("That file contains no audio.", "error")
            return

        self.stop_playback()
        self._new_document()
        self.clip = clip
        self.original_clip = clip.copy()
        self.filepath = path
        self.dirty = False
        self.history.clear()
        self.selection = None
        self._refresh_all(refit=True)
        self._toast(f"Opened {os.path.basename(path)}", "success")

    def append_audio(self):
        if self._wait_if_busy():
            return
        if not self._require_clip():
            return
        path = filedialog.askopenfilename(title="Append Audio",
                                          filetypes=file_dialog_types())
        if not path:
            return
        try:
            second = AudioClip.from_file(path)
            crossfade = self.join_crossfade.value()
        except Exception as exc:
            self._error("Append", exc)
            return

        assert self.clip is not None
        data = second.data
        note = ""
        if second.sample_rate != self.clip.sample_rate:
            data = dsp.resample_to(data, second.sample_rate, self.clip.sample_rate)
            note = f" (resampled {second.sample_rate}→{self.clip.sample_rate} Hz)"

        base, sr = self.clip.data, self.clip.sample_rate
        self._run_async(
            "Joining",
            lambda: dsp.join(base, data, sr, crossfade_ms=crossfade),
            lambda result: (self._push_undo("Join"),
                            self._commit(result,
                                         f"Joined {os.path.basename(path)}{note}.")))

    def _pick_audio(self, title: str) -> tuple[AudioClip, str] | None:
        path = filedialog.askopenfilename(title=title,
                                          filetypes=file_dialog_types())
        if not path:
            return None
        try:
            clip = AudioClip.from_file(path)
        except Exception as exc:
            self._error(title, exc)
            return None
        if clip.n_samples == 0:
            self._toast("That file contains no audio.", "error")
            return None
        return clip, path

    def _at_clip_rate(self, other: AudioClip) -> tuple[np.ndarray, str]:
        assert self.clip is not None
        if other.sample_rate == self.clip.sample_rate:
            return other.data, ""
        data = dsp.resample_to(other.data, other.sample_rate,
                               self.clip.sample_rate)
        return data, (f" (resampled {other.sample_rate}→"
                      f"{self.clip.sample_rate} Hz)")

    def _fill_from_cursor(self, row: NumberRow):
        if not self._require_clip():
            return
        assert self.clip is not None
        seconds = self._cursor_sample() / self.clip.sample_rate
        row.set(f"{seconds:.3f}")
        self._toast(f"{row.label} set to {format_time(seconds)}.", "info")

    def _insert_position(self, row: NumberRow, label: str) -> int | None:
        # Read a Position field as a sample index inside the clip, snapped
        # to a zero crossing when that option is on. None if invalid.
        assert self.clip is not None
        try:
            seconds = row.value()
        except ValueError as exc:
            self._error(label, exc)
            return None
        if seconds > self.clip.duration + 1e-6:
            self._toast(f"Position is past the end of the clip "
                        f"({format_time(self.clip.duration)}).", "error", 7000)
            return None
        n = self.clip.n_samples
        position = min(int(round(seconds * self.clip.sample_rate)), n)
        if self.snap_var.get() and 0 < position < n:
            position = dsp.find_zero_crossing(self.clip.data, position)
        return position

    def insert_audio_file(self):
        if self._wait_if_busy():
            return
        if not self._require_clip():
            return
        assert self.clip is not None
        try:
            crossfade = self.join_crossfade.value()
        except ValueError as exc:
            self._error("Insert audio", exc)
            return
        position = self._insert_position(self.insert_pos, "Insert audio")
        if position is None:
            return

        picked = self._pick_audio("Insert Audio")
        if picked is None:
            return
        other, path = picked
        data, note = self._at_clip_rate(other)
        base, sr = self.clip.data, self.clip.sample_rate

        def done(result):
            out, start, end = result
            self._push_undo("Insert audio")
            self.selection = (start / sr, end / sr)
            self._commit(out, f"Inserted {os.path.basename(path)} at "
                              f"{format_time(position / sr)}{note}.",
                         keep_selection=True)

        self._run_async(
            "Inserting",
            lambda: dsp.insert_audio(base, data, sr, position, crossfade),
            done)

    def overlay_dialog(self):
        self.tabs.set("Combine")
        if self._overlay_clip is None:
            self.choose_overlay_file()

    def choose_overlay_file(self):
        picked = self._pick_audio("Overlay Audio")
        if picked is None:
            return
        self._overlay_clip, path = picked
        self.overlay_name.configure(
            text=f"{os.path.basename(path)}  "
                 f"({format_time(self._overlay_clip.duration)})",
            text_color=self.theme.palette.text)

    def mix_overlay(self):
        if self._wait_if_busy():
            return
        if not self._require_clip():
            return
        assert self.clip is not None
        if self._overlay_clip is None:
            self._toast("Choose a file to overlay first.", "warning")
            return
        try:
            start_sec = self.overlay_start.value()
        except ValueError as exc:
            self._error("Overlay", exc)
            return
        gain = self.overlay_gain.value()
        data, note = self._at_clip_rate(self._overlay_clip)
        base, sr = self.clip.data, self.clip.sample_rate
        position = int(round(start_sec * sr))
        length = data.shape[0]

        def done(result):
            out, scale = result
            message = f"Mixed overlay at {format_time(start_sec)}{note}."
            if scale < 1.0:
                message += (f" Scaled down {-dsp.linear_to_db(scale):.1f} dB "
                            "to avoid clipping.")
            self._push_undo("Overlay")
            self.selection = (position / sr, (position + length) / sr)
            self._commit(out, message, keep_selection=True)

        self._run_async("Mixing",
                        lambda: dsp.overlay(base, data, position, gain), done)

    def export_audio(self):
        self._export(self.clip.data if self.clip else None, "Export")

    def _export(self, data, label: str):
        if not self._require_clip() or data is None:
            return
        assert self.clip is not None
        path = filedialog.asksaveasfilename(
            title=label, defaultextension=".wav",
            filetypes=file_dialog_types(write=True))
        if not path:
            return

        depth = 16
        if path.lower().endswith((".wav", ".wave", ".flac", ".aiff", ".aif")):
            answer = messagebox.askquestion(
                "Bit depth",
                "Export as 24-bit?\n\n"
                "Yes — 24-bit (larger file, more headroom)\n"
                "No — 16-bit (CD standard, widely compatible)",
                icon="question")
            depth = 24 if answer == "yes" else 16

        written = effective_bit_depth(path, depth)
        try:
            AudioClip(data, self.clip.sample_rate).save(path, bit_depth=depth)
        except Exception as exc:
            messagebox.showerror("Could not save file", str(exc))
            return
        if label == "Export":
            self.dirty = False
            self._refresh_header()
        self._toast(f"Saved {os.path.basename(path)} ({written}-bit)", "success")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        return messagebox.askyesno(
            "Unsaved changes",
            "This clip has unsaved edits. Discard them?")

    def reset_to_original(self):
        if self._wait_if_busy():
            return
        if self.original_clip is None:
            self._toast("Nothing to revert to.", "warning")
            return
        if not messagebox.askyesno("Revert",
                                   "Discard all edits and restore the original?"):
            return
        self._push_undo("Revert")
        self.stop_playback()
        self._new_document()
        self.clip = self.original_clip.copy()
        self.selection = None
        self.dirty = False
        self._refresh_all(refit=True)
        self._toast("Reverted to the original clip.", "success")

    def undo(self):
        if self._wait_if_busy():
            return
        if self.clip is None or not self.history.can_undo():
            return
        label = self.history.undo_label()
        halted = self._halt_playback_for_edit()
        prev = self.history.undo(self.clip.data, self.clip.sample_rate, label)
        if prev is not None:
            self.clip.data, self.clip.sample_rate, _ = prev
            self.selection = None
            self.dirty = True
            self._refresh_all()
            self._rewind_after_edit(restart=halted)
            self._toast(f"Undid {label}.", "info")

    def redo(self):
        if self._wait_if_busy():
            return
        if self.clip is None or not self.history.can_redo():
            return
        label = self.history.redo_label()
        halted = self._halt_playback_for_edit()
        nxt = self.history.redo(self.clip.data, self.clip.sample_rate, label)
        if nxt is not None:
            self.clip.data, self.clip.sample_rate, _ = nxt
            self.selection = None
            self.dirty = True
            self._refresh_all()
            self._rewind_after_edit(restart=halted)
            self._toast(f"Redid {label}.", "info")

    def copy(self):
        bounds = self._selection_samples()
        if bounds is None:
            return
        assert self.clip is not None
        self.clipboard.set(self.clip.data[bounds[0]:bounds[1]],
                           self.clip.sample_rate)
        self._toast(f"Copied {self.clipboard.duration():.3f}s.", "success")
        self._refresh_buttons()

    def cut(self):
        if self._wait_if_busy():
            return
        bounds = self._selection_samples()
        if bounds is None:
            return
        assert self.clip is not None
        start, end = bounds
        try:
            result = dsp.delete_region(self.clip.data, start, end)
        except ValueError as exc:
            self._error("Cut", exc)
            return
        self.clipboard.set(self.clip.data[start:end], self.clip.sample_rate)
        self._push_undo("Cut")
        self._commit(result, f"Cut {self.clipboard.duration():.3f}s.")

    def delete_selection(self):
        if self._wait_if_busy():
            return
        bounds = self._selection_samples()
        if bounds is None:
            return
        assert self.clip is not None
        try:
            result = dsp.delete_region(self.clip.data, *bounds)
        except ValueError as exc:
            self._error("Delete", exc)
            return
        self._push_undo("Delete")
        self._commit(result, "Deleted the selection.")

    def paste(self):
        if self._wait_if_busy():
            return
        if not self._require_clip():
            return
        if self.clipboard.is_empty():
            self._toast("The clipboard is empty.", "warning")
            return
        assert self.clip is not None

        data = self.clipboard.data
        note = ""
        if self.clipboard.sample_rate != self.clip.sample_rate:
            data = dsp.resample_to(data, self.clipboard.sample_rate,
                                   self.clip.sample_rate)
            note = " (resampled)"

        position = self._cursor_sample()
        base = self.clip.data
        bounds = self._selection_samples(required=False)
        try:
            if bounds is not None:
                if bounds[1] - bounds[0] >= base.shape[0]:
                    shape = ((0, base.shape[1]) if base.ndim == 2 else (0,))
                    base = np.zeros(shape, dtype=np.float64)
                else:
                    base = dsp.delete_region(base, *bounds)
                position = bounds[0]
            result = dsp.insert_at(base, data, position)
        except Exception as exc:
            self._error("Paste", exc)
            return
        self._push_undo("Paste")
        self._commit(result, f"Pasted {self.clipboard.duration():.3f}s{note}.")

    def silence_selection(self):
        if self._wait_if_busy():
            return
        bounds = self._selection_samples()
        if bounds is None:
            return
        assert self.clip is not None
        self._push_undo("Silence")
        self._commit(dsp.silence_region(self.clip.data, *bounds),
                     "Silenced the selection.", keep_selection=True)

    def insert_silence(self):
        if self._wait_if_busy():
            return
        if not self._require_clip():
            return
        try:
            duration = self.silence_dur.value()
        except ValueError as exc:
            self._error("Insert silence", exc)
            return
        assert self.clip is not None
        position = self._insert_position(self.silence_pos, "Insert silence")
        if position is None:
            return
        sr = self.clip.sample_rate
        result = dsp.insert_silence(self.clip.data, sr, position, duration)
        added = result.shape[0] - self.clip.n_samples
        self._push_undo("Insert silence")
        self.selection = (position / sr, (position + added) / sr)
        self._commit(result, f"Inserted {duration:g}s of silence at "
                             f"{format_time(position / sr)}.",
                     keep_selection=True)

    def select_all(self):
        if not self._require_clip():
            return
        assert self.clip is not None
        self.selection = (0.0, self.clip.duration)
        self._after_selection_change()

    def clear_selection(self):
        self.selection = None
        self._after_selection_change()

    def _after_selection_change(self):
        self._sync_selection_fields()
        if not self._show_selection_overlay():
            self._replot(force=True)
        self._refresh_footer()
        self._refresh_buttons()
        self.refresh_statistics()

    def apply_selection_fields(self):
        if not self._require_clip():
            return
        assert self.clip is not None
        try:
            start = self.sel_start.value()
            end = self.sel_end.value()
        except ValueError as exc:
            self._error("Selection", exc)
            return
        start = max(0.0, min(start, self.clip.duration))
        end = max(0.0, min(end, self.clip.duration))
        if end - start < 1e-4:
            self._toast("The end time must be after the start time.", "warning")
            return
        self.selection = (start, end)
        self._after_selection_change()

    def _sync_selection_fields(self):
        if self.selection is None:
            self.sel_start.var.set("0.000")
            self.sel_end.var.set("0.000")
        else:
            self.sel_start.var.set(f"{self.selection[0]:.3f}")
            self.sel_end.var.set(f"{self.selection[1]:.3f}")

    def apply_trim(self):
        if self._wait_if_busy():
            return
        bounds = self._selection_samples()
        if bounds is None:
            return
        assert self.clip is not None
        start, end = bounds
        offset = start / self.clip.sample_rate
        self._push_undo("Trim")
        self._commit(self.clip.data[start:end].copy(),
                     f"Trimmed to {(end - start) / self.clip.sample_rate:.3f}s.",
                     view_map=lambda lo, hi: (lo - offset, hi - offset))

    def apply_reverse(self, selection_only: bool):
        if selection_only:
            self._edit_region("Reverse", lambda seg, sr: dsp.reverse(seg))
        else:
            self._edit("Reverse", lambda data, sr: dsp.reverse(data))

    def apply_time_scale(self, selection_only: bool):
        factor = self.speed_slider.value()
        if abs(factor - 1.0) < 1e-3:
            self._toast("Speed is already 1.00× — nothing to do.", "warning")
            return
        if not selection_only:
            self._edit("Resample", lambda data, sr: dsp.time_scale(data, factor),
                       view_map=lambda lo, hi: (lo / factor, hi / factor),
                       async_=True)
            return

        bounds = self._selection_samples()
        if bounds is None:
            return
        assert self.clip is not None
        start, end = bounds
        data = self.clip.data

        self._run_async(
            "Resample",
            lambda: np.concatenate(
                [data[:start], dsp.time_scale(data[start:end], factor),
                 data[end:]], axis=0),
            lambda result: (self._push_undo("Resample selection"),
                            self._commit(result, "Resampled the selection.")))

    def apply_gain(self, selection_only: bool):
        gain = self.gain_slider.value()
        func = lambda data, sr: dsp.apply_gain(data, gain)
        if selection_only:
            self._edit_region(f"Gain {gain:+.1f} dB", func)
        else:
            self._edit(f"Gain {gain:+.1f} dB", func)

    def apply_normalize(self):
        mode = self.norm_mode.value()
        target = self.norm_target.value()
        self._edit(f"Normalize ({mode})",
                   lambda data, sr: dsp.normalize(data, target, mode))

    def apply_fade(self, direction: str, selection_only: bool):
        try:
            duration = self.fade_dur.value()
        except ValueError as exc:
            self._error("Fade", exc)
            return
        curve = self.fade_curve.value()
        op = dsp.fade_in if direction == "in" else dsp.fade_out
        label = f"Fade {direction}"
        func = lambda data, sr: op(data, sr, duration, curve)
        if selection_only:
            self._edit_region(label, func)
        else:
            self._edit(label, func)

    def apply_echo(self, selection_only: bool):
        try:
            delay = self.echo_delay.value()
            repeats = self.echo_repeats.value()
        except ValueError as exc:
            self._error("Echo", exc)
            return
        decay = self.echo_decay.value()

        if selection_only:
            bounds = self._selection_samples()
            if bounds is None:
                return
            assert self.clip is not None
            data, sr = self.clip.data, self.clip.sample_rate
            start, end = bounds
            self._run_async(
                "Echo",
                lambda: dsp.echo_region(data, sr, start, end, delay, decay, repeats),
                lambda result: (self._push_undo("Echo (selection)"),
                                self._commit(result,
                                             "Echo applied; the tail rings over "
                                             "the following audio.")))
            return

        self._edit("Echo",
                   lambda data, sr: dsp.echo(data, sr, delay, decay, repeats),
                   async_=True)

    def apply_reverb(self, selection_only: bool):
        params = dict(decay_sec=self.rev_decay.value(),
                      damping=self.rev_damping.value(),
                      mix=self.rev_mix.value(),
                      pre_delay_ms=self.rev_predelay.value())
        func = lambda data, sr: dsp.reverb(data, sr, **params)
        if selection_only:
            self._edit_region("Reverb", func, async_=True)
        else:
            self._edit("Reverb", func, async_=True)

    def generate_tone(self):
        if not self._confirm_discard():
            return
        try:
            freq = self.gen_freq.value()
            duration = self.gen_dur.value()
        except ValueError as exc:
            self._error("Generate", exc)
            return
        sr = int(self.gen_sr.value())
        waveform = self.gen_wave.value()

        if freq >= sr / 2:
            self._toast(
                f"{freq:.0f} Hz is at or above the Nyquist limit for "
                f"{sr} Hz — it would alias. Use a lower frequency or a "
                "higher sample rate.", "error", 8000)
            return
        if duration * sr < 1:
            self._toast("That duration is too short to produce any samples.",
                        "error")
            return

        data = dsp.generate_tone(sr, duration, waveform, freq)
        self.stop_playback()
        self._new_document()
        self.clip = AudioClip(data, sr)
        self.original_clip = self.clip.copy()
        self.filepath = None
        self.dirty = True
        self.history.clear()
        self.selection = None
        self._refresh_all(refit=True)
        self._toast(f"Generated {waveform}, {duration:g}s at {freq:g} Hz.",
                    "success")

    def generate_tone_dialog(self):
        self.tabs.set("Info")
        self._toast("Tone generator is in the Info tab.", "info")

    def _play_range(self) -> tuple[float, float]:
        if self.clip is None:
            return 0.0, 0.0
        duration = self.clip.duration
        if self.selection is not None:
            lo = max(0.0, min(self.selection[0], duration))
            hi = max(lo, min(self.selection[1], duration))
            if hi - lo > 1e-4:
                return lo, hi
        return 0.0, duration

    def toggle_play(self):
        if self.player.is_playing:
            self.stop_playback()
        else:
            self.play()

    def play(self, from_sec: float | None = None):
        if not self._require_clip():
            return
        assert self.clip is not None
        if not Player.available():
            messagebox.showwarning(
                "Playback unavailable",
                "No audio output was found.\n\nInstall 'sounddevice' "
                "(pip install sounddevice), or export the clip and play it "
                "elsewhere — every editing feature still works.")
            return

        start, end = self._play_range()
        position = from_sec if from_sec is not None else self.cursor
        if position is not None:
            position = float(min(max(position, 0.0), self.clip.duration))
            if start <= position < end - 0.01:
                start = position
            elif from_sec is not None:
                start, end = position, self.clip.duration
        if end - start < 1e-4:
            start, end = 0.0, self.clip.duration

        try:
            self.player.set_volume(float(self.volume_slider.get()))
            self.player.play(self.clip.data, self.clip.sample_rate, start, end,
                             loop=bool(self.loop_var.get()))
        except Exception as exc:
            self._error("Playback", exc)
            return
        self._has_played = True
        self.btn_play.configure(text="❚❚")

        if self.player.device_error is not None and not self._warned_fallback:
            self._warned_fallback = True
            self._toast(
                "The audio device refused a stream, so playback is going "
                "through the system player — the playhead and meters will "
                "not move. Reconnecting the output device usually fixes it.",
                "warning", 9000)

    def stop_playback(self):
        if self.clip is not None and Player.supports_transport():
            position = self.player.position()
            if position > 0.0:
                self.cursor = min(position, self.clip.duration)
        self.player.stop()
        self.btn_play.configure(text="▶")
        self.out_meter.reset()
        self.out_meter_r.reset()
        if self.clip is not None and self.cursor:
            self._park_playhead()
        else:
            self._clear_playhead()

    def seek_to_start(self):
        was_playing = self.player.is_playing
        self.stop_playback()
        self.cursor = self._play_range()[0]
        if was_playing:
            self.play()
        elif self.clip is not None:
            self._draw_playhead(self.cursor)
            self.time_var.set(format_time(self.cursor))

    def _on_plot_press(self, event):
        if (event.inaxes is not self.ax or event.button != 1
                or self.clip is None or self.view_mode == "Spectrum"):
            self._press_at = None
            return
        self._press_at = (event.x, event.y, bool(getattr(event, "dblclick",
                                                         False)))

    def _on_selection_edge(self, position: float) -> bool:
        if self.selection is None:
            return False
        lo, hi = self.ax.get_xlim()
        width = float(self.ax.bbox.width) or 1.0
        tolerance = 6.0 * (hi - lo) / width
        return any(abs(position - edge) <= tolerance
                   for edge in self.selection)

    def _on_plot_release(self, event):
        press, self._press_at = self._press_at, None
        if press is None or self.clip is None:
            return
        if event.inaxes is not self.ax or event.button != 1:
            return
        if self.view_mode == "Spectrum" or event.xdata is None:
            return
        if abs(event.x - press[0]) > 3 or abs(event.y - press[1]) > 3:
            return
        position = float(np.clip(event.xdata, 0.0, self.clip.duration))
        if self._on_selection_edge(position):
            return

        playing = self.player.is_playing
        if self.selection is not None and not (
                self.selection[0] <= position <= self.selection[1]):
            self.selection = None
            self.after(0, self._after_selection_change)
            if playing:
                self.play(self.player.position())
            return

        if not (playing or press[2] or self._has_played):
            return
        self.cursor = position
        if playing:
            self.play(position)
        else:
            self._draw_playhead(position)
            self.time_var.set(format_time(position))

    def set_view(self, mode: str):
        self.view_mode = mode
        self.view_switch.set(mode)
        self._replot(force=True)

    def _on_canvas_resize(self, event=None):
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except Exception:
                pass
        self._resize_after = self.after(180, self._finish_resize)

    def _finish_resize(self):
        self._resize_after = None
        self._replot(force=True)

    def _plot_signature(self):
        if self.clip is None:
            return ("empty", self.view_mode, self.compare)
        return (self.view_mode, self.compare, self.clip.n_samples,
                self.clip.sample_rate,
                self.selection,
                float(np.sum(self.clip.data[::997])) if self.clip.n_samples else 0.0)

    def _replot(self, force: bool = False):
        signature = self._plot_signature()
        if not force and signature == self._last_plot_signature:
            return
        self._last_plot_signature = signature

        self._playhead_line = None
        self._wave_artist = None
        self._orig_wave_artist = None
        self._playhead_pos = None
        self._blit_bg = None
        self._remove_span_selector()
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None
        self._ensure_layout()
        for ax in self._axes():
            ax.clear()
            self.theme.style_axes(self.fig, ax)
        if self.ax_orig is not None:
            self._label_compare_axes()

        if self.clip is None:
            self._draw_empty()
        elif self.view_mode == "Waveform":
            self._draw_waveform()
        elif self.view_mode == "Spectrogram":
            self._draw_spectrogram()
        else:
            self._draw_spectrum()

        self._apply_time_view()
        self._capture_background()
        if self.clip is not None and self.cursor is not None \
                and not self.player.is_playing:
            self._draw_playhead(min(self.cursor, self.clip.duration))

    def toggle_compare(self):
        self.compare = not self.compare
        p = self.theme.palette
        if self.compare:
            self.btn_compare.configure(fg_color=p.accent,
                                       hover_color=p.accent_hover,
                                       text_color=p.accent_text)
            self._toast("Before / After: original on top, edited below.",
                        "info")
        else:
            self.btn_compare.configure(fg_color=p.surface_alt,
                                       hover_color=p.border,
                                       text_color=p.text)
        self._replot(force=True)

    def _stacked(self) -> bool:
        # Waveform and spectrogram compare as two rows; the spectrum draws
        # both curves on one set of axes instead.
        return (self.compare and self.clip is not None
                and self.original_clip is not None
                and self.view_mode in ("Waveform", "Spectrogram"))

    def _axes(self):
        return [self.ax] if self.ax_orig is None else [self.ax_orig, self.ax]

    def _ensure_layout(self):
        stacked = self._stacked()
        if stacked == (self.ax_orig is not None):
            return
        self.fig.clear()
        if stacked:
            self.ax_orig = self.fig.add_subplot(211)
            self.ax = self.fig.add_subplot(212, sharex=self.ax_orig)
        else:
            self.ax_orig = None
            self.ax = self.fig.add_subplot(111)
        self._blit_bg = None

    def _label_compare_axes(self):
        p = self.theme.palette
        self.ax_orig.set_title("Original", loc="left", fontsize=9,
                               color=p.text_dim)
        self.ax.set_title("Edited", loc="left", fontsize=9, color=p.text)
        self.ax_orig.tick_params(labelbottom=False)

    def _view_duration(self) -> float:
        # The time span the axes cover: in Before/After both clips must fit.
        if self.clip is None:
            return 1e-6
        duration = self.clip.duration
        if self._stacked():
            duration = max(duration, self.original_clip.duration)
        return max(duration, 1e-6)

    def _draw_empty(self):
        p = self.theme.palette
        self.ax.text(0.5, 0.5,
                     "Open an audio file or generate a tone to begin",
                     ha="center", va="center", transform=self.ax.transAxes,
                     color=p.text_faint, fontsize=11)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(False)

    def _plot_width_px(self) -> int:
        try:
            width = int(self.ax.bbox.width)
            if width > 50:
                return width
        except Exception:
            pass
        try:
            width = int(self.canvas.get_tk_widget().winfo_width())
            if width > 50:
                return width
        except Exception:
            pass
        return 1000

    def _draw_waveform(self):
        assert self.clip is not None
        p = self.theme.palette
        mono = self.clip.to_mono()
        sr = self.clip.sample_rate

        self._wave_artist = None
        self._orig_wave_artist = None
        self._draw_wave_range(0.0, self._view_duration())

        panels = [(self.ax, mono, sr)]
        if self.ax_orig is not None:
            panels.append((self.ax_orig, self.original_clip.to_mono(),
                           self.original_clip.sample_rate))
        for ax, signal_mono, rate in panels:
            rms_t, rms_v = analysis.rms_envelope(signal_mono, rate)
            if len(rms_t) > 1:
                ax.plot(rms_t, rms_v, color=p.rms, linewidth=0.8, alpha=0.55)
                ax.plot(rms_t, -rms_v, color=p.rms, linewidth=0.8, alpha=0.55)
            ax.axhline(0, color=p.border, linewidth=0.6)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("Amplitude")

        self.ax.set_xlim(0, self._view_duration())
        self.ax.set_xlabel("Time (s)")
        self._install_span_selector()

    def _draw_wave_range(self, lo: float, hi: float):
        # The envelope is built at screen resolution for the visible range
        # only, so zooming in reveals real detail down to single samples.
        assert self.clip is not None
        p = self.theme.palette
        self._wave_artist = self._draw_envelope(
            self.ax, self.clip, lo, hi, self._wave_artist, p.wave)
        if self.ax_orig is not None and self.original_clip is not None:
            self._orig_wave_artist = self._draw_envelope(
                self.ax_orig, self.original_clip, lo, hi,
                self._orig_wave_artist, p.text_dim)

    def _draw_envelope(self, ax, clip: AudioClip, lo: float, hi: float,
                       old_artist, colour):
        sr = clip.sample_rate
        width = max(hi - lo, 0.0)
        start = max(0, int(np.floor((lo - width * 0.05) * sr)))
        end = min(clip.n_samples, int(np.ceil((hi + width * 0.05) * sr)) + 1)
        start = min(start, end)
        segment = clip.data[start:end]
        mono = segment.mean(axis=1) if segment.ndim == 2 else segment

        times, low, high = analysis.peak_envelope(
            mono, sr, max_points=self._plot_width_px())
        times = times + start / sr

        if old_artist is not None:
            try:
                old_artist.remove()
            except Exception:
                pass
        if high is None:
            (artist,) = ax.plot(times, low, color=colour, linewidth=0.7,
                                zorder=1.5)
            return artist
        return ax.fill_between(times, low, high, color=colour, linewidth=0,
                               zorder=1.5)

    def _set_time_view(self, lo: float, hi: float):
        # None means "fit": it keeps showing everything even after an edit
        # changes the clip's length.
        full = self._view_duration()
        fits = lo <= 1e-3 and hi >= full - 1e-3
        self._time_view = None if fits else (lo, hi)
        self._apply_time_view()
        self._capture_background()

    def _apply_time_view(self):
        # Replots rebuild the axes at full width; put the user's zoom back.
        # If an edit left the view hanging off either end of the clip, slide
        # it back inside at the same zoom level.
        if self.clip is None or self.view_mode == "Spectrum":
            return
        duration = self._view_duration()
        if self._time_view is None:
            lo, hi = 0.0, duration
        else:
            lo, hi = self._time_view
            width = hi - lo
            if width < 1e-3 or width >= duration:
                lo, hi = 0.0, duration
            elif lo < 0.0:
                lo, hi = 0.0, width
            elif hi > duration:
                lo, hi = duration - width, duration
            self._time_view = (lo, hi)
        self.ax.set_xlim(lo, hi)
        if self._span_selector is not None:
            self._span_selector.minspan = self._min_span()
        if self.view_mode == "Waveform":
            self._draw_wave_range(lo, hi)

    def _draw_spectrogram(self):
        assert self.clip is not None
        p = self.theme.palette
        panels = [(self.ax, self.clip)]
        if self.ax_orig is not None:
            panels.insert(0, (self.ax_orig, self.original_clip))

        mesh = None
        for ax, clip in panels:
            times, freqs, db = analysis.compute_spectrogram(
                clip.data, clip.sample_rate,
                n_fft=SPECTROGRAM_FFT, window=ANALYSIS_WINDOW)
            ax.grid(False)
            ax.set_ylabel("Frequency (Hz)")
            if db.size == 0:
                ax.text(0.5, 0.5, "Clip is too short for a spectrogram",
                        ha="center", va="center", transform=ax.transAxes,
                        color=p.text_faint)
                continue
            # An image, not pcolormesh: the STFT grid is evenly spaced, so
            # it looks the same but redraws several times faster. A slow
            # redraw holds up the audio callback and makes playback crackle.
            dt = times[1] - times[0] if len(times) > 1 else clip.duration
            df = freqs[1] - freqs[0] if len(freqs) > 1 else clip.sample_rate / 2
            mesh = ax.imshow(db, origin="lower", aspect="auto",
                             interpolation="nearest", cmap=p.spectrogram_cmap,
                             vmin=-90, vmax=0,
                             extent=(times[0] - dt / 2, times[-1] + dt / 2,
                                     freqs[0] - df / 2, freqs[-1] + df / 2))
            ax.set_ylim(0, clip.sample_rate / 2)

        if mesh is not None:
            self.colorbar = self.fig.colorbar(mesh, ax=self._axes(), pad=0.01,
                                              aspect=32)
            self.colorbar.set_label("dB", color=p.text_dim, fontsize=8)
            self.colorbar.ax.tick_params(colors=p.text_dim, labelsize=7)
            self.colorbar.outline.set_edgecolor(p.border)

        self.ax.set_xlim(0, self._view_duration())
        self.ax.set_xlabel("Time (s)")
        self._install_span_selector()

    def _draw_spectrum(self):
        assert self.clip is not None
        p = self.theme.palette
        data = self.clip.data
        scope = "whole clip"
        bounds = self._selection_samples(required=False)
        if bounds is not None:
            data = data[bounds[0]:bounds[1]]
            scope = "selection"

        freqs, db = analysis.compute_spectrum(
            data, self.clip.sample_rate, n_fft=SPECTRUM_FFT,
            window=ANALYSIS_WINDOW)
        if freqs.size == 0:
            self.ax.text(0.5, 0.5, "Not enough audio for a spectrum",
                         ha="center", va="center", transform=self.ax.transAxes,
                         color=p.text_faint)
            return

        self.ax.fill_between(freqs[1:], -100, db[1:], color=p.wave, alpha=0.35)
        self.ax.plot(freqs[1:], db[1:], color=p.wave_edge, linewidth=1.0,
                     label="Edited")
        if self.compare and self.original_clip is not None:
            orig = self.original_clip.data
            if bounds is not None:
                # Same time range of the original, where it still exists.
                rate = self.original_clip.sample_rate
                lo = int(round(self.selection[0] * rate))
                hi = int(round(self.selection[1] * rate))
                orig = orig[lo:hi]
            o_freqs, o_db = analysis.compute_spectrum(
                orig, self.original_clip.sample_rate, n_fft=SPECTRUM_FFT,
                window=ANALYSIS_WINDOW)
            if o_freqs.size:
                self.ax.plot(o_freqs[1:], o_db[1:], color=p.text_dim,
                             linewidth=1.0, linestyle="--", label="Original")
            legend = self.ax.legend(loc="upper right", fontsize=8,
                                    facecolor=p.surface_alt,
                                    edgecolor=p.border)
            for text in legend.get_texts():
                text.set_color(p.text)
        self.ax.set_xscale("log")
        self.ax.set_xlim(max(20, freqs[1]), self.clip.sample_rate / 2)
        self.ax.set_ylim(-100, 3)
        self.ax.set_xlabel(f"Frequency (Hz) — {scope}")
        self.ax.set_ylabel("Magnitude (dB)")

    def _remove_span_selector(self):
        if self._span_selector is not None:
            try:
                self._span_selector.disconnect_events()
            except Exception:
                pass
            self._span_selector = None

    def _install_span_selector(self):
        self._remove_span_selector()

        p = self.theme.palette
        self._span_selector = _SharedBlitSpanSelector(
            self.ax, self._on_span_select, "horizontal",
            background=lambda: self._blit_bg,
            props=dict(alpha=0.22, facecolor=p.selection),
            handle_props=dict(color=p.selection),
            interactive=True, drag_from_anywhere=False,
            minspan=self._min_span())

        if self.selection is not None:
            try:
                self._span_selector.extents = self.selection
            except Exception:
                self.ax.axvspan(self.selection[0], self.selection[1],
                                color=p.selection, alpha=0.22)

    def _show_selection_overlay(self) -> bool:
        # Redraw just the selection overlay (a blit) when the view has one;
        # the Spectrum view depends on the selection, so it must replot.
        selector = self._span_selector
        if (selector is None or self.view_mode == "Spectrum"
                or self._blit_bg is None):
            return False
        try:
            if self.selection is None:
                selector.clear()
            else:
                selector.set_visible(True)
                selector.extents = self.selection
        except Exception:
            return False
        self._last_plot_signature = self._plot_signature()
        return True

    CLICK_SLOP_PX = 4

    def _min_span(self) -> float:
        # The narrowest drag that counts as a selection, in seconds. It is
        # measured in pixels, like the click test in _on_plot_release, so a
        # click whose mouse wobbles a pixel or two seeks instead of leaving
        # behind a sliver of a selection at any zoom level.
        lo, hi = self.ax.get_xlim()
        width = float(self.ax.bbox.width) or 1.0
        return max(1e-4, self.CLICK_SLOP_PX * abs(hi - lo) / width)

    def _restore_selection_span(self):
        selector = self._span_selector
        if selector is None or self.selection is None:
            return
        try:
            selector.set_visible(True)
            selector.extents = self.selection
        except Exception:
            return

    def _on_span_select(self, xmin, xmax):
        if self.clip is None:
            return
        xmin = max(0.0, xmin)
        xmax = min(self.clip.duration, xmax)
        if xmax - xmin < self._min_span():
            if self.selection is not None:
                self.after(0, self._restore_selection_span)
            return
        self.selection = (xmin, xmax)
        self._last_plot_signature = self._plot_signature()
        self._sync_selection_fields()
        self._refresh_footer()
        self._refresh_buttons()
        self.refresh_statistics()
        if self.view_mode == "Spectrum":
            self._replot(force=True)

    def zoom(self, factor: float):
        if self.clip is None or self.view_mode == "Spectrum":
            return
        lo, hi = self.ax.get_xlim()
        centre = (lo + hi) / 2
        half = (hi - lo) * factor / 2
        new_lo = max(0.0, centre - half)
        new_hi = min(self._view_duration(), centre + half)
        if new_hi - new_lo < 1e-3:
            return
        self._set_time_view(new_lo, new_hi)

    def zoom_to_selection(self):
        if self.selection is None or self.view_mode == "Spectrum":
            self._toast("Select a region first.", "warning")
            return
        self._set_time_view(*self.selection)

    def zoom_to_fit(self):
        if self.clip is None or self.view_mode == "Spectrum":
            return
        self._set_time_view(0, self._view_duration())

    def _blit_bbox(self):
        axes, figure = self.ax.bbox, self.fig.bbox
        pad = 3.0
        return Bbox.from_extents(
            max(figure.x0, axes.x0 - pad), max(figure.y0, axes.y0 - pad),
            min(figure.x1, axes.x1 + pad), min(figure.y1, axes.y1 + pad))

    def _capture_background(self):
        # A full draw leaves animated artists out, so it produces a clean
        # background; _on_canvas_draw stores it and paints the playhead back.
        if self._playhead_line is not None:
            self._playhead_line.set_animated(True)
        self.canvas.draw()

    def _on_canvas_draw(self, event):
        # Any full redraw (a replot, a zoom, or the span selector refreshing
        # itself on mouse release) skips the animated playhead, which would
        # otherwise vanish while playback is stopped. Keep the fresh
        # background for blitting and paint the playhead back on top.
        try:
            self._blit_bg = self.canvas.copy_from_bbox(self._blit_bbox())
        except Exception:
            self._blit_bg = None
        self._playhead_pos = None
        self._draw_selection_overlay()
        line = self._playhead_line
        if line is not None and line.get_visible() and line.get_animated():
            self.ax.draw_artist(line)

    def _draw_selection_overlay(self):
        # The selection is animated (blitted), so full draws leave it out and
        # every blit that restores the background has to paint it back.
        selector = self._span_selector
        if selector is None:
            return
        for artist in selector.artists:
            if artist.get_visible():
                self.ax.draw_artist(artist)

    def _draw_playhead(self, position: float):
        if self.clip is None or self.view_mode == "Spectrum":
            return
        p = self.theme.palette
        blitting = self._blit_bg is not None
        fresh = self._playhead_line is None
        if fresh:
            self._playhead_line = self.ax.axvline(
                position, color=p.playhead, linewidth=1.3, animated=blitting)
        else:
            self._playhead_line.set_xdata([position, position])
            self._playhead_line.set_visible(True)

        if blitting:
            try:
                self.canvas.restore_region(self._blit_bg)
                self._draw_selection_overlay()
                self.ax.draw_artist(self._playhead_line)
                self.canvas.blit(self._playhead_region(position, fresh))
                self._playhead_pos = position
                return
            except Exception:
                pass
        self._playhead_line.set_animated(False)
        self._playhead_pos = None
        self.canvas.draw_idle()

    def _playhead_region(self, position: float, fresh: bool):
        bbox = self._blit_bbox()
        previous = getattr(self, "_playhead_pos", None)
        if fresh or previous is None:
            return bbox
        try:
            transform = self.ax.transData.transform
            x_new = float(transform((position, 0.0))[0])
            x_old = float(transform((previous, 0.0))[0])
        except Exception:
            return bbox
        if not (np.isfinite(x_new) and np.isfinite(x_old)):
            return bbox

        pad = 6.0
        x0 = max(min(x_new, x_old) - pad, bbox.x0)
        x1 = min(max(x_new, x_old) + pad, bbox.x1)
        if x1 - x0 < 1.0:
            return bbox
        if x1 - x0 > bbox.width * 0.6:
            return bbox
        return Bbox.from_extents(x0, bbox.y0, x1, bbox.y1)

    def _clear_playhead(self):
        self._playhead_pos = None
        if self._playhead_line is None:
            return
        self._playhead_line.set_visible(False)
        if self._blit_bg is not None:
            try:
                self.canvas.restore_region(self._blit_bg)
                self._draw_selection_overlay()
                self.canvas.blit(self._blit_bbox())
                return
            except Exception:
                pass
        self.canvas.draw_idle()

    def _start_ui_timer(self):
        self._next_frame = time.perf_counter()
        self._tick()

    UI_INTERVAL_MS = 33

    def _tick(self):
        started = time.perf_counter()
        try:
            self._drain_results()
            if self.player.is_playing:
                self._draw_playhead(self.player.position())
                self._update_meters()
                self.time_var.set(format_time(self.player.position()))
            elif self.btn_play.cget("text") != "▶":
                self.stop_playback()
        except Exception:
            traceback.print_exc()

        interval = self.UI_INTERVAL_MS / 1000.0
        self._next_frame = getattr(self, "_next_frame", started) + interval
        now = time.perf_counter()
        if self._next_frame <= now:
            self._next_frame = now + interval
        delay = max(1, int(round((self._next_frame - now) * 1000.0)))
        self._ui_after = self.after(delay, self._tick)

    def _update_meters(self):
        levels = self.player.channel_levels()
        if not levels:
            peak, rms = self.player.level()
            levels = [(peak, rms)]
        left = levels[0]
        right = levels[1] if len(levels) > 1 else left
        self.out_meter.update_level(*left)
        self.out_meter_r.update_level(*right)

    def _refresh_all(self, refit: bool = False):
        self._replot(force=True)
        if refit:
            self.zoom_to_fit()
        self._refresh_header()
        self._refresh_footer()
        self._refresh_buttons()
        self._sync_selection_fields()
        self.refresh_statistics()

    def _refresh_header(self):
        name = (os.path.basename(self.filepath) if self.filepath
                else ("Untitled" if self.clip is not None else "No file open"))
        self.file_label.configure(text=name)
        self.dirty_dot.configure(text="●" if self.dirty else "")

    def _refresh_footer(self):
        if self.clip is None:
            self.time_total.configure(text="of 0:00.000")
            self.sel_var.set("No selection")
            self.scope_var.set("")
            self.info_var.set("")
            return

        self.time_total.configure(text=f"of {format_time(self.clip.duration)}")
        if self.selection is None:
            self.sel_var.set("No selection")
            self.scope_var.set("▶ plays the whole clip")
        else:
            lo, hi = self.selection
            self.sel_var.set(
                f"Selection  {format_time(lo)} → {format_time(hi)}"
                f"   ({hi - lo:.3f}s)")
            self.scope_var.set("▶ plays the selection")

        channels = "Mono" if self.clip.n_channels == 1 else "Stereo"
        self.info_var.set(
            f"{self.clip.sample_rate:,} Hz · {channels} · "
            f"{self.clip.n_samples:,} samples · "
            f"{self.history.depth()} undo")

    def _refresh_buttons(self):
        has_clip = self.clip is not None
        has_selection = self.selection is not None
        has_paste = not self.clipboard.is_empty()

        def enable(widget, condition):
            widget.configure(state="normal" if condition else "disabled")

        enable(self.btn_undo, has_clip and self.history.can_undo())
        enable(self.btn_redo, has_clip and self.history.can_redo())
        enable(self.btn_cut, has_clip and has_selection)
        enable(self.btn_copy, has_clip and has_selection)
        enable(self.btn_delete, has_clip and has_selection)
        enable(self.btn_paste, has_clip and has_paste)
        enable(self.btn_play, has_clip)
        enable(self.btn_stop, has_clip)
        enable(self.btn_to_start, has_clip)

        undo_label = self.history.undo_label()
        redo_label = self.history.redo_label()
        try:
            self.edit_menu.entryconfigure(
                0, label=f"Undo {undo_label}" if undo_label else "Undo",
                state="normal" if self.history.can_undo() else "disabled")
            self.edit_menu.entryconfigure(
                1, label=f"Redo {redo_label}" if redo_label else "Redo",
                state="normal" if self.history.can_redo() else "disabled")
        except Exception:
            pass

    def refresh_statistics(self):
        if self.clip is None:
            for var in self.stats_vars.values():
                var.set("—")
            self.stats_scope.configure(text="No clip loaded")
            return

        data = self.clip.data
        scope = "Whole clip"
        bounds = self._selection_samples(required=False)
        if bounds is not None:
            data = data[bounds[0]:bounds[1]]
            scope = f"Selection ({(bounds[1] - bounds[0]) / self.clip.sample_rate:.3f}s)"

        stats = analysis.clip_statistics(data, self.clip.sample_rate)
        for label, value in analysis.format_statistics(stats):
            var = self.stats_vars.get(label)
            if var is not None and var.get() != value:
                var.set(value)
        self.stats_scope.configure(text=scope)

    def show_shortcuts(self):
        mod = "Cmd" if sys.platform == "darwin" else "Ctrl"
        messagebox.showinfo(
            "Keyboard Shortcuts",
            f"{mod}+O   Open              {mod}+S   Export\n"
            f"{mod}+Z   Undo              {mod}+Shift+Z   Redo\n"
            f"{mod}+X/C/V   Cut / Copy / Paste\n"
            f"{mod}+A   Select all        Esc   Deselect\n"
            f"Del   Delete selection\n"
            f"{mod}+0   Zoom to fit       {mod}+ / -   Zoom in / out\n"
            f"Home   Back to start\n"
            f"Space   Play / Stop\n\n"
            "Drag on the waveform to select a region.\n"
            "Play covers the selection, or the whole clip when there is\n"
            "none. Click while playing to seek there and carry on; click\n"
            "after a play to park the playhead; double-click to place it\n"
            "at any time. A click away from a selection only clears it.")

    def show_about(self):
        messagebox.showinfo(
            "About",
            f"{APP_TITLE} {APP_VERSION}\n\n"
            f"{COURSE_INFO}\n"
            f"Student: {STUDENT_NAME} ({STUDENT_ID})\n\n"
            "An audio editor built with Python, NumPy, SciPy, Matplotlib and "
            "CustomTkinter.\n\n"
            "Time-domain editing (trim, join, insert, overlay/mix, reverse, "
            "cut/copy/paste), resampling, fades and "
            "normalisation, convolution effects (echo and reverb, through "
            "the FFT), and waveform / spectrogram / spectrum visualisation "
            "with a before/after comparison.")

    def _on_close(self):
        if self.dirty and not messagebox.askyesno(
                "Quit", "This clip has unsaved edits. Quit anyway?"):
            return
        for attr in ("_ui_after", "_resize_after"):
            after_id = getattr(self, attr, None)
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except Exception:
                    pass
                setattr(self, attr, None)
        try:
            self.player.close()
        except Exception:
            pass
        self._save_settings()
        self.destroy()
