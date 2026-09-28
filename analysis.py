from __future__ import annotations

import numpy as np
from scipy import signal

DB_FLOOR = -100.0


def _mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim == 2 else data


def _to_db(magnitude: np.ndarray, floor_db: float = DB_FLOOR) -> np.ndarray:
    ref = np.max(magnitude) if magnitude.size else 0.0
    if ref <= 0:
        return np.full_like(magnitude, floor_db, dtype=np.float64)
    db = 20.0 * np.log10(np.maximum(magnitude, 1e-12) / ref)
    return np.maximum(db, floor_db)


def compute_spectrogram(data: np.ndarray, sr: int, n_fft: int = 1024,
                        overlap: float = 0.75, window: str = "hann",
                        max_frames: int = 3000):
    mono = np.asarray(_mono(data), dtype=np.float64)
    n = len(mono)
    n_fft = int(np.clip(n_fft, 32, 16384))
    if n < 32:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))
    if n < n_fft:
        n_fft = int(2 ** np.floor(np.log2(n)))

    overlap = float(np.clip(overlap, 0.0, 0.95))
    noverlap = min(int(n_fft * overlap), n_fft - 1)
    hop = max(1, n_fft - noverlap)

    n_frames = 1 + max(0, (n - n_fft) // hop)
    if n_frames > max_frames:
        hop = max(1, int(np.ceil((n - n_fft) / max(1, max_frames - 1))))
        noverlap = min(max(0, n_fft - hop), n_fft - 1)

    freqs, times, spec = signal.spectrogram(
        mono, fs=sr, window=window, nperseg=n_fft, noverlap=noverlap,
        mode="magnitude", scaling="spectrum")

    if spec.shape[1] > max_frames:
        step = int(np.ceil(spec.shape[1] / max_frames))
        spec = spec[:, ::step]
        times = times[::step]

    return times, freqs, _to_db(spec)


def compute_spectrum(data: np.ndarray, sr: int, n_fft: int = 4096,
                     window: str = "hann"):
    mono = np.asarray(_mono(data), dtype=np.float64)
    n = len(mono)
    if n < 32:
        return np.zeros(0), np.zeros(0)
    n_fft = int(min(np.clip(n_fft, 32, 32768), 2 ** int(np.floor(np.log2(n)))))

    freqs, psd = signal.welch(mono, fs=sr, window=window, nperseg=n_fft,
                              noverlap=n_fft // 2, scaling="spectrum")
    return freqs, _to_db(np.sqrt(np.maximum(psd, 0.0)))


def peak_envelope(mono: np.ndarray, sr: int, max_points: int = 20000):
    n = len(mono)
    if n == 0:
        return np.zeros(0), np.zeros(0), None
    if n <= max_points:
        return np.arange(n) / sr, mono, None

    bucket = int(np.ceil(n / max_points))
    n_buckets = int(np.ceil(n / bucket))
    padded = np.full(n_buckets * bucket, mono[-1], dtype=np.float64)
    padded[:n] = mono
    reshaped = padded.reshape(n_buckets, bucket)
    times = (np.arange(n_buckets) * bucket) / sr
    return times, reshaped.min(axis=1), reshaped.max(axis=1)


def rms_envelope(mono: np.ndarray, sr: int, window_ms: float = 50.0,
                 max_points: int = 4000):
    n = len(mono)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    win = max(1, int(sr * window_ms / 1000.0))
    hop = max(1, n // max_points, win // 4)
    n_frames = max(1, (n - win) // hop + 1)
    if n_frames < 2:
        return np.array([0.0]), np.array([float(np.sqrt(np.mean(mono ** 2)))])

    frames = np.lib.stride_tricks.sliding_window_view(mono, win)[::hop][:n_frames]
    return (np.arange(n_frames) * hop + win / 2) / sr, np.sqrt(np.mean(frames ** 2, axis=1))


def clip_statistics(data: np.ndarray, sr: int) -> dict:
    out: dict = {}
    n = data.shape[0]
    out["samples"] = int(n)
    out["duration"] = n / sr if sr else 0.0
    out["sample_rate"] = int(sr)
    out["channels"] = 1 if data.ndim == 1 else data.shape[1]

    if n == 0:
        out.update(peak=0.0, peak_db=DB_FLOOR, rms=0.0, rms_db=DB_FLOOR,
                   crest_db=0.0)
        return out

    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    out["peak"] = peak
    out["rms"] = rms
    out["peak_db"] = 20.0 * np.log10(peak) if peak > 0 else DB_FLOOR
    out["rms_db"] = 20.0 * np.log10(rms) if rms > 0 else DB_FLOOR
    out["crest_db"] = out["peak_db"] - out["rms_db"] if rms > 0 else 0.0
    return out


def format_statistics(stats: dict) -> list[tuple[str, str]]:
    def db(x):
        return "-inf dB" if x <= DB_FLOOR else f"{x:+.2f} dB"

    total_ms = int(round(max(0.0, stats["duration"]) * 1000.0))
    mm, rest_ms = divmod(total_ms, 60_000)
    return [
        ("Duration", f"{mm}:{rest_ms / 1000.0:06.3f}"),
        ("Samples", f"{stats['samples']:,}"),
        ("Sample rate", f"{stats['sample_rate']:,} Hz"),
        ("Channels", "Mono" if stats["channels"] == 1 else
                     f"Stereo ({stats['channels']} ch)"),
        ("Peak level", db(stats["peak_db"])),
        ("RMS level", db(stats["rms_db"])),
        ("Crest factor", f"{stats['crest_db']:.2f} dB"),
    ]


def block_levels(block: np.ndarray) -> tuple[float, float]:
    if block.size == 0:
        return 0.0, 0.0
    return float(np.max(np.abs(block))), float(np.sqrt(np.mean(block ** 2)))
