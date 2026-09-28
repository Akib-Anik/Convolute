from __future__ import annotations

import os

import numpy as np
from scipy.io import wavfile

try:
    import soundfile as sf
    SF_AVAILABLE = True
except Exception:
    sf = None
    SF_AVAILABLE = False


_WAV_EXTS = {".wav", ".wave"}


# The formats the app opens and exports, mapped to libsndfile's format name.
# soundfile guesses the format from the extension and does not recognise
# ".aif" or ".wave", so the name is always passed explicitly.
_SF_FORMATS = {
    ".wav": "WAV",
    ".wave": "WAV",
    ".flac": "FLAC",
    ".ogg": "OGG",
    ".mp3": "MP3",
    ".aiff": "AIFF",
    ".aif": "AIFF",
}


def _sf_extensions() -> set[str]:
    if not SF_AVAILABLE:
        return set()
    try:
        available = set(sf.available_formats())
    except Exception:
        return set()
    return {ext for ext, fmt in _SF_FORMATS.items() if fmt in available}


def _check_supported(ext: str, supported: list[str], verb: str) -> None:
    if ext in supported:
        return
    names = ", ".join(e for e in supported if e != ".wave")
    hint = ("" if SF_AVAILABLE else
            "\nMore formats need the 'soundfile' package:  "
            "pip install soundfile")
    raise ValueError(f"Can't {verb} '{ext or 'files without an extension'}'. "
                     f"Supported formats: {names}.{hint}")


def supported_read_extensions() -> list[str]:
    return sorted(_WAV_EXTS | _sf_extensions())


def supported_write_extensions() -> list[str]:
    return sorted(_WAV_EXTS | _sf_extensions())


def effective_bit_depth(path: str, requested: int) -> int:
    ext = os.path.splitext(path)[1].lower()
    if requested == 24 and ext in _WAV_EXTS and not SF_AVAILABLE:
        return 32
    return requested


def file_dialog_types(write: bool = False):
    exts = supported_write_extensions() if write else supported_read_extensions()
    patterns = " ".join(f"*{e}" for e in exts)
    types = [("Audio files", patterns), ("WAV", "*.wav")]
    for label, ext in (("MP3", ".mp3"), ("FLAC", ".flac"),
                       ("OGG Vorbis", ".ogg")):
        if ext in exts:
            types.append((label, f"*{ext}"))
    if ".aiff" in exts:
        types.append(("AIFF", "*.aiff *.aif"))
    return types


def _pcm_to_float(raw: np.ndarray) -> np.ndarray:
    if raw.dtype == np.int16:
        return raw.astype(np.float64) / 32768.0
    if raw.dtype == np.int32:
        return raw.astype(np.float64) / 2147483648.0
    if raw.dtype == np.uint8:
        return (raw.astype(np.float64) - 128.0) / 128.0
    if raw.dtype in (np.float32, np.float64):
        return np.clip(raw.astype(np.float64), -1.0, 1.0)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        return raw.astype(np.float64) / max(abs(info.min), info.max)
    raise ValueError(f"Unsupported WAV sample format: {raw.dtype}")


def _float_to_pcm(data: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    clipped = np.clip(data, -1.0, 1.0)
    if bit_depth == 32:
        return clipped.astype(np.float32)
    if bit_depth == 24:
        return np.round(clipped * 2147483647.0).astype(np.int32)
    return np.round(clipped * 32767.0).astype(np.int16)


class AudioClip:
    def __init__(self, data: np.ndarray, sample_rate: int):
        self.data = np.asarray(data, dtype=np.float64)
        self.sample_rate = int(sample_rate)

    @property
    def n_channels(self) -> int:
        return 1 if self.data.ndim == 1 else self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate if self.sample_rate else 0.0

    @classmethod
    def from_file(cls, path: str) -> "AudioClip":
        ext = os.path.splitext(path)[1].lower()
        _check_supported(ext, supported_read_extensions(), "open")

        if SF_AVAILABLE:
            try:
                data, sr = sf.read(path, dtype="float64", always_2d=False)
                return cls(np.clip(data, -1.0, 1.0), sr)
            except Exception:
                if ext not in _WAV_EXTS:
                    raise

        try:
            sr, raw = wavfile.read(path)
            return cls(_pcm_to_float(raw), sr)
        except Exception as exc:
            if SF_AVAILABLE:
                try:
                    data, sr = sf.read(path, dtype="float64", always_2d=False)
                    return cls(np.clip(data, -1.0, 1.0), sr)
                except Exception:
                    pass
            raise ValueError(
                f"'{os.path.basename(path)}' is not an audio file this "
                f"build can read ({exc})") from exc

    def save(self, path: str, bit_depth: int = 16) -> None:
        ext = os.path.splitext(path)[1].lower()
        _check_supported(ext, supported_write_extensions(), "save")

        if ext in _WAV_EXTS and SF_AVAILABLE and bit_depth == 24:
            sf.write(path, np.clip(self.data, -1.0, 1.0), self.sample_rate,
                     format="WAV", subtype="PCM_24")
            return

        if ext in _WAV_EXTS:
            wavfile.write(path, self.sample_rate,
                          _float_to_pcm(self.data, bit_depth))
            return

        subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bit_depth)
        if ext in (".ogg", ".mp3"):
            subtype = None
        sf.write(path, np.clip(self.data, -1.0, 1.0), self.sample_rate,
                 format=_SF_FORMATS[ext], subtype=subtype)

    def to_mono(self) -> np.ndarray:
        return self.data.mean(axis=1) if self.data.ndim == 2 else self.data

    def peak(self) -> float:
        return float(np.max(np.abs(self.data))) if self.data.size else 0.0

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.data ** 2))) if self.data.size else 0.0

    def copy(self) -> "AudioClip":
        return AudioClip(self.data.copy(), self.sample_rate)

    def slice_seconds(self, start_sec: float, end_sec: float) -> np.ndarray:
        sr = self.sample_rate
        a = max(0, int(round(start_sec * sr)))
        b = min(self.n_samples, int(round(end_sec * sr)))
        return self.data[a:b].copy()


class History:
    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth
        self._undo_stack: list[tuple[np.ndarray, int, str]] = []
        self._redo_stack: list[tuple[np.ndarray, int, str]] = []

    def push(self, data: np.ndarray, sample_rate: int, label: str = "edit") -> None:
        self._undo_stack.append((data.copy(), int(sample_rate), label))
        if len(self._undo_stack) > self.max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self, data: np.ndarray, sample_rate: int, label: str = "edit"):
        if not self._undo_stack:
            return None
        self._redo_stack.append((data.copy(), int(sample_rate), label))
        return self._undo_stack.pop()

    def redo(self, data: np.ndarray, sample_rate: int, label: str = "edit"):
        if not self._redo_stack:
            return None
        self._undo_stack.append((data.copy(), int(sample_rate), label))
        return self._redo_stack.pop()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo_label(self) -> str:
        return self._undo_stack[-1][2] if self._undo_stack else ""

    def redo_label(self) -> str:
        return self._redo_stack[-1][2] if self._redo_stack else ""

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def depth(self) -> int:
        return len(self._undo_stack)


class Clipboard:
    def __init__(self):
        self.data: np.ndarray | None = None
        self.sample_rate: int = 0

    def set(self, data: np.ndarray, sample_rate: int) -> None:
        if data is None:
            self.clear()
            return
        array = np.asarray(data, dtype=np.float64)
        if array.ndim == 0 or array.size == 0:
            self.clear()
            return
        self.data = array.copy()
        self.sample_rate = int(sample_rate)

    def is_empty(self) -> bool:
        return self.data is None or self.data.shape[0] == 0

    def duration(self) -> float:
        if self.is_empty() or not self.sample_rate:
            return 0.0
        return self.data.shape[0] / self.sample_rate

    def clear(self) -> None:
        self.data = None
        self.sample_rate = 0
