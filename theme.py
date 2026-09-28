from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True)
class Palette:
    name: str

    bg: str
    surface: str
    surface_alt: str
    border: str

    text: str
    text_dim: str
    text_faint: str

    accent: str
    accent_hover: str
    accent_text: str

    danger: str
    warning: str
    success: str

    plot_bg: str
    grid: str
    wave: str
    wave_edge: str
    rms: str
    playhead: str
    selection: str
    spectrogram_cmap: str

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


DARK = Palette(
    name="dark",
    bg="#0f1216",
    surface="#171b21",
    surface_alt="#1f242c",
    border="#2a3038",
    text="#e6e9ee",
    text_dim="#9aa4b2",
    text_faint="#6b7480",
    accent="#f5a524",
    accent_hover="#ffb648",
    accent_text="#1a1205",
    danger="#e5484d",
    warning="#f5a524",
    success="#30a46c",
    plot_bg="#12161b",
    grid="#252b33",
    wave="#4cc2ff",
    wave_edge="#7fd4ff",
    rms="#f5a524",
    playhead="#ff5c5c",
    selection="#f5a524",
    spectrogram_cmap="magma",
)

LIGHT = Palette(
    name="light",
    bg="#f4f5f7",
    surface="#ffffff",
    surface_alt="#eceef1",
    border="#d5d9e0",
    text="#1a1d21",
    text_dim="#5a6472",
    text_faint="#8b94a1",
    accent="#0b6bcb",
    accent_hover="#0a5cae",
    accent_text="#ffffff",
    danger="#c62828",
    warning="#b26a00",
    success="#1e7d43",
    plot_bg="#ffffff",
    grid="#e2e6ec",
    wave="#2b6cb0",
    wave_edge="#4a8bd0",
    rms="#d97706",
    playhead="#d92d20",
    selection="#f59e0b",
    spectrogram_cmap="viridis",
)


@dataclass(frozen=True)
class Typography:
    family: str = "Segoe UI"
    mono: str = "Consolas"
    size_xs: int = 10
    size_sm: int = 11
    size_md: int = 12
    size_lg: int = 14
    size_xl: int = 18

    @staticmethod
    def for_platform() -> "Typography":
        import sys
        if sys.platform == "darwin":
            return Typography(family="SF Pro Text", mono="SF Mono")
        if sys.platform.startswith("linux"):
            return Typography(family="DejaVu Sans", mono="DejaVu Sans Mono")
        return Typography()


@dataclass(frozen=True)
class Spacing:
    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24
    radius: int = 8
    radius_sm: int = 6


@dataclass
class Theme:
    palette: Palette = field(default_factory=lambda: DARK)
    type: Typography = field(default_factory=Typography.for_platform)
    space: Spacing = field(default_factory=Spacing)

    _active: ClassVar["Theme | None"] = None

    @classmethod
    def current(cls) -> "Theme":
        if cls._active is None:
            cls._active = Theme()
        return cls._active

    @classmethod
    def use(cls, mode: str) -> "Theme":
        cls._active = Theme(palette=LIGHT if mode == "light" else DARK)
        return cls._active

    def font(self, size: str = "md", weight: str = "normal"):
        return (self.type.family, getattr(self.type, f"size_{size}"), weight)

    def mono_font(self, size: str = "sm", weight: str = "normal"):
        return (self.type.mono, getattr(self.type, f"size_{size}"), weight)

    def style_axes(self, fig, ax) -> None:
        p = self.palette
        fig.patch.set_facecolor(p.surface)
        ax.set_facecolor(p.plot_bg)
        for spine in ax.spines.values():
            spine.set_color(p.border)
        ax.tick_params(colors=p.text_dim, labelsize=8, length=3)
        ax.xaxis.label.set_color(p.text_dim)
        ax.yaxis.label.set_color(p.text_dim)
        ax.title.set_color(p.text)
        ax.grid(True, color=p.grid, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
