from __future__ import annotations

import numpy as np
from scipy import signal


def _is_stereo(data: np.ndarray) -> bool:
    return data.ndim == 2


def n_channels(data: np.ndarray) -> int:
    return 1 if data.ndim == 1 else data.shape[1]


def _apply_per_channel(data: np.ndarray, func) -> np.ndarray:
    if not _is_stereo(data):
        return func(data)
    channels = [func(data[:, c]) for c in range(data.shape[1])]
    lengths = {len(c) for c in channels}
    if len(lengths) != 1:
        max_len = max(lengths)
        channels = [np.pad(c, (0, max_len - len(c))) for c in channels]
    return np.stack(channels, axis=1)


def _convolve_all(data: np.ndarray, kernel: np.ndarray, mode: str = "full") -> np.ndarray:
    if kernel.size == 1:
        return data * kernel[0]
    return _apply_per_channel(
        data, lambda x: signal.oaconvolve(x, kernel, mode=mode))


def _lowpass_kernel(cutoff_norm: float, num_taps: int = 63) -> np.ndarray:
    cutoff_norm = float(np.clip(cutoff_norm, 1e-4, 0.499))
    if num_taps % 2 == 0:
        num_taps += 1
    return signal.firwin(num_taps, cutoff_norm * 2.0, window="hamming")


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def linear_to_db(amp: float, floor_db: float = -120.0) -> float:
    amp = abs(float(amp))
    if amp <= 0.0:
        return floor_db
    return max(floor_db, 20.0 * np.log10(amp))


def trim(data: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    n = data.shape[0]
    start = int(round(max(0.0, start_sec) * sr))
    end = int(round(min(n / sr, end_sec) * sr))
    start = max(0, min(start, n))
    end = max(0, min(end, n))
    if end <= start:
        raise ValueError("Trim range is empty: end must be greater than start.")
    return data[start:end].copy()


def _match_channels(a: np.ndarray, b: np.ndarray):
    a_ch = n_channels(a)
    b_ch = n_channels(b)
    target = max(a_ch, b_ch)
    if target == 1:
        return a, b
    if a_ch == 1:
        a = np.repeat(a[:, None], target, axis=1)
    if b_ch == 1:
        b = np.repeat(b[:, None], target, axis=1)
    return a, b


def join(data_a: np.ndarray, data_b: np.ndarray, sr: int, crossfade_ms: float = 0.0) -> np.ndarray:
    a, b = _match_channels(data_a, data_b)
    crossfade_samples = int(round(max(0.0, crossfade_ms) / 1000.0 * sr))
    crossfade_samples = min(crossfade_samples, len(a), len(b))

    if crossfade_samples <= 0:
        return np.concatenate([a, b], axis=0)

    t = np.linspace(0.0, 1.0, crossfade_samples)
    ramp_up = np.sin(0.5 * np.pi * t)
    ramp_down = np.cos(0.5 * np.pi * t)
    if a.ndim == 2:
        ramp_down = ramp_down[:, None]
        ramp_up = ramp_up[:, None]

    a_head = a[:-crossfade_samples]
    overlap = a[-crossfade_samples:] * ramp_down + b[:crossfade_samples] * ramp_up
    b_tail = b[crossfade_samples:]

    return np.concatenate([a_head, overlap, b_tail], axis=0)


def reverse(data: np.ndarray) -> np.ndarray:
    return data[::-1].copy()


def _rational_ratio(factor: float, max_denominator: int = 320):
    from fractions import Fraction
    frac = Fraction(1.0 / float(factor)).limit_denominator(max_denominator)
    up, down = frac.numerator, frac.denominator
    return max(1, up), max(1, down)


def time_scale(data: np.ndarray, factor: float, antialias: bool = True) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Time-scale factor must be positive.")

    n = data.shape[0]
    if n <= 1:
        return data.copy()
    if abs(factor - 1.0) < 1e-9:
        return data.copy()

    if not antialias:
        new_n = max(1, int(round(n / factor)))
        old_index = np.arange(n)
        new_index = np.linspace(0, n - 1, new_n)
        return _apply_per_channel(
            data, lambda x: np.interp(new_index, old_index, x))

    up, down = _rational_ratio(factor)
    if n < 32:
        return time_scale(data, factor, antialias=False)
    out = signal.resample_poly(data, up, down, axis=0)
    return np.asarray(out, dtype=np.float64)


def resample_to(data: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or data.shape[0] == 0:
        return data.copy()
    from fractions import Fraction
    frac = Fraction(int(sr_to), int(sr_from))
    out = signal.resample_poly(data, frac.numerator, frac.denominator, axis=0)
    return np.asarray(out, dtype=np.float64)


def _fade_ramp(n: int, curve: str) -> np.ndarray:
    if n <= 0:
        return np.zeros(0)
    if n == 1:
        return np.array([1.0])
    t = np.linspace(0.0, 1.0, n)
    if curve == "exponential":
        k = 5.0
        return (np.exp(k * t) - 1.0) / (np.exp(k) - 1.0)
    if curve == "logarithmic":
        k = 5.0
        return np.log1p(k * t) / np.log1p(k)
    if curve == "s-curve":
        return 0.5 - 0.5 * np.cos(np.pi * t)
    if curve == "equal-power":
        return np.sin(0.5 * np.pi * t)
    return t


FADE_CURVES = ["linear", "exponential", "logarithmic", "s-curve", "equal-power"]


def fade_in(data: np.ndarray, sr: int, duration_sec: float, curve: str = "linear") -> np.ndarray:
    n = data.shape[0]
    fade_len = min(n, max(0, int(round(duration_sec * sr))))
    out = data.copy()
    if fade_len <= 0:
        return out
    gain = _fade_ramp(fade_len, curve)
    if out.ndim == 2:
        gain = gain[:, None]
    out[:fade_len] = out[:fade_len] * gain
    return out


def fade_out(data: np.ndarray, sr: int, duration_sec: float, curve: str = "linear") -> np.ndarray:
    n = data.shape[0]
    fade_len = min(n, max(0, int(round(duration_sec * sr))))
    out = data.copy()
    if fade_len <= 0:
        return out
    gain = _fade_ramp(fade_len, curve)[::-1]
    if out.ndim == 2:
        gain = gain[:, None]
    out[-fade_len:] = out[-fade_len:] * gain
    return out


def normalize_peak(data: np.ndarray, target: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > target:
        return data * (target / peak)
    return data


def echo_kernel(sr: int, delay_ms: float = 300.0, decay: float = 0.5,
                repeats: int = 4) -> np.ndarray:
    delay_ms = max(1.0, float(delay_ms))
    decay = min(max(float(decay), 0.0), 0.95)
    repeats = min(max(int(repeats), 1), 50)
    delay_samples = max(1, int(round(delay_ms / 1000.0 * sr)))

    kernel = np.zeros(delay_samples * repeats + 1)
    kernel[0] = 1.0
    for k in range(1, repeats + 1):
        kernel[k * delay_samples] += decay ** k
    return kernel


def echo(data: np.ndarray, sr: int, delay_ms: float = 300.0,
         decay: float = 0.5, repeats: int = 4) -> np.ndarray:
    kernel = echo_kernel(sr, delay_ms, decay, repeats)
    return normalize_peak(_convolve_all(data, kernel, mode="full"))


def echo_region(data: np.ndarray, sr: int, start: int, end: int,
                delay_ms: float = 300.0, decay: float = 0.5,
                repeats: int = 4) -> np.ndarray:
    n = data.shape[0]
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    if end <= start:
        return data.copy()

    kernel = echo_kernel(sr, delay_ms, decay, repeats)
    wet = _convolve_all(data[start:end], kernel, mode="full")

    needed = start + wet.shape[0]
    out_len = max(n, needed)
    shape = (out_len, data.shape[1]) if data.ndim == 2 else (out_len,)
    out = np.zeros(shape, dtype=np.float64)

    out[:n] = data
    out[start:end] = 0.0
    out[start:needed] += wet
    return normalize_peak(out)


def reverb_ir(sr: int, decay_sec: float = 1.2, damping: float = 0.35,
              pre_delay_ms: float = 20.0, seed: int = 0) -> np.ndarray:
    decay_sec = float(np.clip(decay_sec, 0.05, 10.0))
    rng = np.random.default_rng(seed)
    n = max(16, int(sr * decay_sec))
    t = np.arange(n) / sr
    envelope = 10.0 ** (-3.0 * t / decay_sec)
    ir = rng.standard_normal(n) * envelope

    damping = float(np.clip(damping, 0.0, 0.95))
    if damping > 0.01:
        cutoff = (1.0 - damping) * 0.45 + 0.03
        ir = np.convolve(ir, _lowpass_kernel(cutoff, 127), mode="same")

    pre = max(0, int(sr * max(0.0, pre_delay_ms) / 1000.0))
    if pre:
        ir = np.concatenate([np.zeros(pre), ir])

    ir[0] += 1.0
    peak = np.max(np.abs(ir))
    return ir / peak if peak > 0 else ir


def reverb(data: np.ndarray, sr: int, decay_sec: float = 1.2,
           damping: float = 0.35, mix: float = 0.3,
           pre_delay_ms: float = 20.0) -> np.ndarray:
    if data.shape[0] == 0:
        return data.copy()
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 0.0:
        return data.copy()

    ir = reverb_ir(sr, decay_sec, damping, pre_delay_ms)
    wet = _convolve_all(data, ir, mode="full")[:data.shape[0]]

    dry_rms = float(np.sqrt(np.mean(data ** 2))) if data.size else 0.0
    wet_rms = float(np.sqrt(np.mean(wet ** 2))) if wet.size else 0.0
    if wet_rms > 1e-9 and dry_rms > 1e-9:
        wet = wet * (dry_rms / wet_rms)

    return normalize_peak((1.0 - mix) * data + mix * wet)


def apply_gain(data: np.ndarray, gain_db: float) -> np.ndarray:
    return data * db_to_linear(gain_db)


def normalize(data: np.ndarray, target_db: float = -1.0,
              mode: str = "peak") -> np.ndarray:
    if data.size == 0:
        return data.copy()
    target = db_to_linear(target_db)

    if mode == "rms":
        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms <= 1e-12:
            return data.copy()
        out = data * (target / rms)
        return normalize_peak(out, 0.99)

    peak = float(np.max(np.abs(data)))
    if peak <= 1e-12:
        return data.copy()
    return data * (target / peak)


def delete_region(data: np.ndarray, start: int, end: int) -> np.ndarray:
    n = data.shape[0]
    start, end = max(0, min(start, n)), max(0, min(end, n))
    if end <= start:
        return data.copy()
    if end - start >= n:
        raise ValueError("Deleting the whole clip would leave nothing behind.")
    return np.concatenate([data[:start], data[end:]], axis=0)


def insert_at(data: np.ndarray, other: np.ndarray, position: int) -> np.ndarray:
    n = data.shape[0]
    position = max(0, min(position, n))
    a, b = _match_channels(data, other)
    return np.concatenate([a[:position], b, a[position:]], axis=0)


def insert_audio(data: np.ndarray, other: np.ndarray, sr: int, position: int,
                 crossfade_ms: float = 0.0):
    n = data.shape[0]
    position = max(0, min(int(position), n))
    a, b = _match_channels(data, other)
    head, tail = a[:position], a[position:]

    first = join(head, b, sr, crossfade_ms)
    overlap = head.shape[0] + b.shape[0] - first.shape[0]
    out = join(first, tail, sr, crossfade_ms)
    start = position - overlap
    return out, start, start + b.shape[0]


def overlay(data: np.ndarray, other: np.ndarray, position: int,
            gain_db: float = 0.0, ceiling: float = 0.99):
    a, b = _match_channels(data, other)
    position = max(0, int(position))
    needed = position + b.shape[0]
    out_len = max(a.shape[0], needed)
    shape = (out_len, a.shape[1]) if a.ndim == 2 else (out_len,)
    out = np.zeros(shape, dtype=np.float64)
    out[:a.shape[0]] = a
    out[position:needed] += b * db_to_linear(gain_db)

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        scale = ceiling / peak
        return out * scale, scale
    return out, 1.0


def insert_silence(data: np.ndarray, sr: int, position: int,
                   duration_sec: float) -> np.ndarray:
    count = max(0, int(round(duration_sec * sr)))
    if count == 0:
        return data.copy()
    shape = (count, data.shape[1]) if data.ndim == 2 else (count,)
    return insert_at(data, np.zeros(shape, dtype=np.float64), position)


def silence_region(data: np.ndarray, start: int, end: int) -> np.ndarray:
    out = data.copy()
    n = out.shape[0]
    start, end = max(0, min(start, n)), max(0, min(end, n))
    if end > start:
        out[start:end] = 0.0
    return out


def find_zero_crossing(data: np.ndarray, index: int, search_radius: int = 2000) -> int:
    n = data.shape[0]
    if n < 2:
        return int(np.clip(index, 0, max(0, n)))
    index = int(np.clip(index, 0, n - 1))

    lo = max(0, index - search_radius)
    hi = min(n - 1, index + search_radius)
    window = data[lo:hi + 1]
    if window.ndim == 2:
        window = window.mean(axis=1)
    if len(window) < 2:
        return index

    crossings = np.flatnonzero(np.diff(np.signbit(window)))
    if crossings.size == 0:
        return index
    return int(lo + crossings[np.argmin(np.abs(lo + crossings - index))])


WAVEFORMS = ["sine", "square", "triangle", "sawtooth", "chirp", "white noise",
             "pink noise"]


def generate_tone(sr: int, duration_sec: float, waveform: str = "sine",
                  frequency: float = 440.0) -> np.ndarray:
    n = int(sr * duration_sec)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)

    t = np.arange(n) / float(sr)
    phase = 2 * np.pi * frequency * t

    if waveform == "square":
        sig = signal.square(phase)
    elif waveform == "triangle":
        sig = signal.sawtooth(phase, width=0.5)
    elif waveform == "sawtooth":
        sig = signal.sawtooth(phase)
    elif waveform == "chirp":
        f1 = min(frequency * 6, sr / 2.0 * 0.95)
        sig = signal.chirp(t, f0=frequency, f1=f1,
                           t1=max(duration_sec, 1e-9), method="linear")
    elif waveform == "white noise":
        sig = np.random.default_rng(0).standard_normal(n)
        sig /= np.max(np.abs(sig)) or 1.0
    elif waveform == "pink noise":
        rng = np.random.default_rng(0)
        spectrum = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        scale = np.ones_like(freqs)
        scale[1:] = 1.0 / np.sqrt(freqs[1:])
        sig = np.fft.irfft(spectrum * scale, n=n)
        sig /= np.max(np.abs(sig)) or 1.0
    else:
        sig = np.sin(phase)

    edge = min(max(1, int(0.01 * sr)), n // 2)
    env = np.ones_like(sig)
    if edge > 0:
        env[:edge] = np.linspace(0, 1, edge)
        env[-edge:] = np.linspace(1, 0, edge)
    return (sig * env * 0.9).astype(np.float64)
