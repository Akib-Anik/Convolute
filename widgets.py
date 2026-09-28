from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from theme import Theme


def readout_label(master, variable, *, font, colour, background,
                  width: int = 10, anchor: str = "e"):
    return tk.Label(master, textvariable=variable, font=font, fg=colour,
                    bg=background, width=width, anchor=anchor,
                    bd=0, highlightthickness=0, padx=0, pady=0)


class Card(ctk.CTkFrame):
    def __init__(self, master, title: str = "", subtitle: str = "", **kwargs):
        t = Theme.current()
        kwargs.setdefault("fg_color", t.palette.surface)
        kwargs.setdefault("corner_radius", t.space.radius)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", t.palette.border)
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self._row = 0

        if title:
            ctk.CTkLabel(self, text=title.upper(), font=t.font("xs", "bold"),
                         text_color=t.palette.text_dim, anchor="w").grid(
                row=self._row, column=0, sticky="ew",
                padx=t.space.md, pady=(t.space.md, 0))
            self._row += 1
        if subtitle:
            ctk.CTkLabel(self, text=subtitle, font=t.font("xs"),
                         text_color=t.palette.text_faint, anchor="w",
                         justify="left", wraplength=250).grid(
                row=self._row, column=0, sticky="ew",
                padx=t.space.md, pady=(2, 0))
            self._row += 1

    def body(self) -> ctk.CTkFrame:
        t = Theme.current()
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=self._row, column=0, sticky="ew",
                   padx=t.space.md, pady=(t.space.sm, t.space.md))
        frame.columnconfigure(0, weight=1)
        self._row += 1
        return frame


class NumberRow(ctk.CTkFrame):
    def __init__(self, master, label: str, default, unit: str = "",
                 minimum: float | None = None, maximum: float | None = None,
                 integer: bool = False, width: int = 78, hint: str = ""):
        super().__init__(master, fg_color="transparent")
        t = Theme.current()
        self.label = label
        self.minimum = minimum
        self.maximum = maximum
        self.integer = integer

        self.columnconfigure(1, weight=1)
        self.label_widget = ctk.CTkLabel(self, text=label, font=t.font("sm"),
                                         text_color=t.palette.text_dim,
                                         anchor="w")
        self.label_widget.grid(row=0, column=0, sticky="w", pady=3)

        holder = ctk.CTkFrame(self, fg_color="transparent")
        holder.grid(row=0, column=1, sticky="e", pady=3)

        self.var = ctk.StringVar(value=str(default))
        self.entry = ctk.CTkEntry(
            holder, textvariable=self.var, width=width, height=28,
            font=t.mono_font("sm"), justify="right",
            fg_color=t.palette.surface_alt, border_color=t.palette.border,
            corner_radius=t.space.radius_sm)
        self.entry.pack(side="left")
        if unit:
            ctk.CTkLabel(holder, text=unit, font=t.font("xs"),
                         text_color=t.palette.text_faint, width=30,
                         anchor="w").pack(side="left", padx=(5, 0))

        if hint:
            ctk.CTkLabel(self, text=hint, font=t.font("xs"),
                         text_color=t.palette.text_faint, anchor="w",
                         justify="left", wraplength=260).grid(
                row=1, column=0, columnspan=2, sticky="w", pady=(0, 3))

    def value(self):
        raw = self.var.get().strip()
        try:
            num = float(raw)
        except ValueError:
            raise ValueError(f"{self.label} must be a number (got '{raw}').")
        if num != num or num in (float("inf"), float("-inf")):
            raise ValueError(f"{self.label} must be a finite number "
                             f"(got '{raw}').")
        if self.integer:
            num = int(num)
        if self.minimum is not None and num < self.minimum:
            raise ValueError(f"{self.label} must be at least {self.minimum:g}.")
        if self.maximum is not None and num > self.maximum:
            raise ValueError(f"{self.label} must be at most {self.maximum:g}.")
        return num

    def set(self, value) -> None:
        self.var.set(f"{value:g}" if isinstance(value, float) else str(value))

    def set_label(self, label: str) -> None:
        self.label = label
        self.label_widget.configure(text=label)


class SliderRow(ctk.CTkFrame):
    def __init__(self, master, label: str, minimum: float, maximum: float,
                 default: float, fmt: str = "{:.2f}", suffix: str = "",
                 steps: int | None = None, command=None,
                 background: str | None = None):
        super().__init__(master, fg_color="transparent")
        t = Theme.current()
        self.fmt = fmt
        self.suffix = suffix
        self._command = command
        self.columnconfigure(0, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text=label, font=t.font("sm"),
                     text_color=t.palette.text_dim, anchor="w").grid(
            row=0, column=0, sticky="w")
        self._text = tk.StringVar(value=fmt.format(default) + suffix)
        self.readout = readout_label(
            head, self._text, font=t.mono_font("sm"),
            colour=t.palette.accent,
            background=background or t.palette.surface, width=10)
        self.readout.grid(row=0, column=1, sticky="e")

        self.slider = ctk.CTkSlider(
            self, from_=minimum, to=maximum, number_of_steps=steps,
            command=self._on_change, height=16,
            button_color=t.palette.accent, button_hover_color=t.palette.accent_hover,
            progress_color=t.palette.accent, fg_color=t.palette.surface_alt)
        self.slider.set(default)
        self.slider.grid(row=1, column=0, sticky="ew", pady=(2, 6))

    def _on_change(self, value):
        text = self.fmt.format(float(value)) + self.suffix
        if text != self._text.get():
            self._text.set(text)
        if self._command is not None:
            self._command(float(value))

    def value(self) -> float:
        return float(self.slider.get())

    def set(self, value: float) -> None:
        self.slider.set(value)
        self._on_change(value)


class ChoiceRow(ctk.CTkFrame):
    def __init__(self, master, label: str, values, default=None, command=None,
                 width: int = 130):
        super().__init__(master, fg_color="transparent")
        t = Theme.current()
        self.columnconfigure(1, weight=1)
        ctk.CTkLabel(self, text=label, font=t.font("sm"),
                     text_color=t.palette.text_dim, anchor="w").grid(
            row=0, column=0, sticky="w", pady=3)

        self.var = ctk.StringVar(value=default if default is not None else values[0])
        self.menu = ctk.CTkOptionMenu(
            self, variable=self.var, values=list(values), width=width, height=28,
            font=t.font("sm"), dropdown_font=t.font("sm"), command=command,
            fg_color=t.palette.surface_alt, button_color=t.palette.surface_alt,
            button_hover_color=t.palette.border, text_color=t.palette.text,
            dropdown_fg_color=t.palette.surface_alt,
            dropdown_text_color=t.palette.text,
            corner_radius=t.space.radius_sm)
        self.menu.grid(row=0, column=1, sticky="e", pady=3)

    def value(self) -> str:
        return self.var.get()

    def set(self, value: str) -> None:
        self.var.set(value)


def action_button(master, text: str, command, kind: str = "primary",
                  width: int = 0, **kwargs) -> ctk.CTkButton:
    t = Theme.current()
    p = t.palette
    styles = {
        "primary": dict(fg_color=p.accent, hover_color=p.accent_hover,
                        text_color=p.accent_text),
        "secondary": dict(fg_color=p.surface_alt, hover_color=p.border,
                          text_color=p.text),
        "ghost": dict(fg_color="transparent", hover_color=p.surface_alt,
                      text_color=p.text_dim),
        "danger": dict(fg_color="transparent", hover_color=p.danger,
                       text_color=p.danger, border_width=1, border_color=p.danger),
    }
    opts = dict(styles.get(kind, styles["secondary"]))
    opts.update(kwargs)
    button = ctk.CTkButton(
        master, text=text, command=command, height=30,
        font=t.font("sm", "bold" if kind == "primary" else "normal"),
        corner_radius=t.space.radius_sm, **opts)
    if width:
        button.configure(width=width)
    return button


class LevelMeter(ctk.CTkFrame):
    FLOOR_DB = -60.0

    def __init__(self, master, label: str = "", width: int = 150,
                 background: str | None = None):
        super().__init__(master, fg_color="transparent")
        t = Theme.current()
        self._peak_hold = 0.0
        self._hold_frames = 0
        self._shown_fraction = -1.0
        self._shown_colour = ""
        self._shown_text = ""

        if label:
            ctk.CTkLabel(self, text=label, font=t.font("xs"),
                         text_color=t.palette.text_faint, width=16,
                         anchor="w").pack(side="left", padx=(0, 4))

        self.bar = ctk.CTkProgressBar(
            self, width=width, height=10, corner_radius=3,
            fg_color=t.palette.surface_alt, progress_color=t.palette.success)
        self.bar.set(0.0)
        self.bar.pack(side="left", fill="x", expand=True)

        self._text = tk.StringVar(value="-inf")
        self.readout = readout_label(
            self, self._text, font=t.mono_font("xs"),
            colour=t.palette.text_faint,
            background=background or t.palette.surface, width=6)
        self.readout.pack(side="left", padx=(6, 0))

    def _to_fraction(self, amplitude: float) -> float:
        if amplitude <= 1e-6:
            return 0.0
        import numpy as np
        db = 20.0 * np.log10(amplitude)
        return float(max(0.0, min(1.0, (db - self.FLOOR_DB) / -self.FLOOR_DB)))

    def update_level(self, peak: float, rms: float | None = None) -> None:
        t = Theme.current()
        fraction = self._to_fraction(rms if rms is not None else peak)

        if fraction >= self._peak_hold:
            self._peak_hold = fraction
            self._hold_frames = 12
        elif self._hold_frames > 0:
            self._hold_frames -= 1
        else:
            self._peak_hold = max(fraction, self._peak_hold - 0.03)

        if abs(self._peak_hold - self._shown_fraction) >= 0.005:
            self.bar.set(self._peak_hold)
            self._shown_fraction = self._peak_hold

        if peak >= 0.99:
            colour = t.palette.danger
        elif peak >= 0.89:
            colour = t.palette.warning
        else:
            colour = t.palette.success
        if colour != self._shown_colour:
            self.bar.configure(progress_color=colour)
            self._shown_colour = colour

        if peak <= 1e-6:
            text = "-inf"
        else:
            import numpy as np
            text = f"{20.0 * np.log10(peak):+.1f}"
        if text != self._shown_text:
            self._text.set(text)
            self._shown_text = text

    def reset(self) -> None:
        self._peak_hold = 0.0
        self._hold_frames = 0
        self._shown_fraction = 0.0
        self._shown_text = "-inf"
        self.bar.set(0.0)
        self._text.set("-inf")


class StatusToast(ctk.CTkFrame):
    def __init__(self, master):
        t = Theme.current()
        super().__init__(master, fg_color="transparent",
                         corner_radius=t.space.radius_sm, height=26)
        self.label = ctk.CTkLabel(self, text="", font=t.font("sm"),
                                  text_color=t.palette.text, anchor="w")
        self.label.pack(side="left", padx=t.space.md, pady=5)
        self._after_id = None

    def show(self, message: str, kind: str = "info", duration_ms: int = 4000):
        t = Theme.current()
        colours = {"info": t.palette.text, "success": t.palette.success,
                   "warning": t.palette.warning, "error": t.palette.danger}
        self.label.configure(text=message,
                             text_color=colours.get(kind, t.palette.text))
        self.configure(fg_color=t.palette.surface_alt)
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = self.after(duration_ms, self._clear)

    def _clear(self):
        self._after_id = None
        self.label.configure(text="")
        self.configure(fg_color="transparent")


def format_time(seconds: float, with_ms: bool = True) -> str:
    seconds = max(0.0, float(seconds))
    if seconds != seconds:
        seconds = 0.0
    if with_ms:
        total_ms = int(round(seconds * 1000.0))
        minutes, rest_ms = divmod(total_ms, 60_000)
        return f"{minutes}:{rest_ms / 1000.0:06.3f}"
    total_s = int(round(seconds))
    minutes, rest = divmod(total_s, 60)
    return f"{minutes}:{rest:02d}"
