from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    sd = None
    SD_AVAILABLE = False


class _SystemPlayer:
    def __init__(self):
        self._proc = None
        self._tmp = None
        self._thread = None

    def _discard_tmp(self) -> None:
        if self._tmp:
            try:
                os.unlink(self._tmp)
            except Exception:
                pass
            self._tmp = None

    @staticmethod
    def available() -> bool:
        if sys.platform == "win32":
            try:
                import winsound
                return True
            except Exception:
                return False
        return _SystemPlayer._unix_cmd() is not None

    @staticmethod
    def _unix_cmd():
        for cmd in ("afplay", "paplay", "aplay", "play"):
            if shutil.which(cmd):
                return cmd
        return None

    def play(self, data: np.ndarray, sample_rate: int) -> None:
        from audio_engine import AudioClip
        self.stop()
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="mae_play_")
        os.close(fd)
        AudioClip(data, sample_rate).save(path, bit_depth=16)
        self._tmp = path

        if sys.platform == "win32":
            import winsound
            def blocking_play():
                try:
                    winsound.PlaySound(path, winsound.SND_FILENAME)
                finally:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                    if self._tmp == path:
                        self._tmp = None

            self._thread = threading.Thread(target=blocking_play, daemon=True)
            self._thread.start()
            return

        cmd = self._unix_cmd()
        if cmd is None:
            self._discard_tmp()
            raise RuntimeError("No system audio player found.")
        proc = subprocess.Popen(
            [cmd, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._proc = proc
        self._reap(proc, path)

    def _reap(self, proc, path: str) -> None:
        def wait_then_delete():
            try:
                proc.wait()
            except Exception:
                pass
            try:
                os.unlink(path)
            except OSError:
                pass
            if self._tmp == path:
                self._tmp = None

        threading.Thread(target=wait_then_delete, daemon=True).start()

    @property
    def is_playing(self) -> bool:
        if sys.platform == "win32":
            return self._thread is not None and self._thread.is_alive()
        proc = self._proc
        return proc is not None and proc.poll() is None

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        self._discard_tmp()


class Player:
    BLOCK = 1024

    def __init__(self):
        self._stream = None
        self._data: np.ndarray | None = None
        self._sr = 44100
        self._frame = 0
        self._start_frame = 0
        self._end_frame = 0
        self._loop = False
        self._volume = 1.0
        self._peak = 0.0
        self._rms = 0.0
        self._peaks: list[float] = [0.0]
        self._rmss: list[float] = [0.0]
        self._block_frame = 0
        self._block_span = 0
        self._block_time: float | None = None
        self._block_dac: float | None = None
        self._finished = False
        self._lock = threading.Lock()
        self._system = _SystemPlayer()
        self._using_system = False
        self._device_error: Exception | None = None

    @staticmethod
    def available() -> bool:
        return (SD_AVAILABLE and sd is not None) or _SystemPlayer.available()

    @staticmethod
    def supports_transport() -> bool:
        return SD_AVAILABLE and sd is not None

    @property
    def device_error(self) -> Exception | None:
        return self._device_error

    @property
    def is_playing(self) -> bool:
        if self._using_system:
            return self._system.is_playing
        return self._stream is not None and not self._finished

    def position(self) -> float:
        with self._lock:
            sr = self._sr
            frame = self._frame
            base = self._block_frame
            span = self._block_span
            stamp = self._block_time
            dac = self._block_dac
            start, end = self._start_frame, self._end_frame
            looping = self._loop
        if not sr:
            return 0.0
        if stamp is None or span <= 0:
            return frame / sr

        offset = None
        if dac:
            stream = self._stream
            try:
                if stream is not None:
                    offset = float(stream.time) - dac
            except Exception:
                offset = None
        if offset is None:
            offset = min(max(time.perf_counter() - stamp, 0.0), span / sr)

        pos = base + offset * sr
        if looping and end > start:
            if pos >= end:
                pos = start + (pos - start) % (end - start)
        return float(min(max(pos, start), end)) / sr

    def level(self) -> tuple[float, float]:
        with self._lock:
            return self._peak, self._rms

    def channel_levels(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(zip(self._peaks, self._rmss))

    def set_volume(self, volume: float) -> None:
        self._volume = float(np.clip(volume, 0.0, 2.0))

    def set_loop(self, loop: bool) -> None:
        self._loop = bool(loop)

    def play(self, data: np.ndarray, sample_rate: int,
             start_sec: float = 0.0, end_sec: float | None = None,
             loop: bool = False) -> None:
        self.stop()
        if data is None or data.shape[0] == 0:
            return

        n = data.shape[0]
        start = int(np.clip(start_sec * sample_rate, 0, n - 1))
        end = n if end_sec is None else int(np.clip(end_sec * sample_rate, 1, n))
        if end <= start:
            return

        self._data = np.ascontiguousarray(data, dtype=np.float32)
        self._sr = int(sample_rate)
        self._start_frame = start
        self._end_frame = end
        self._frame = start
        self._loop = loop
        self._block_frame = start
        self._block_span = 0
        self._block_time = None
        self._block_dac = None
        self._finished = False
        self._device_error = None
        self._peak = self._rms = 0.0
        n_ch = 1 if self._data.ndim == 1 else self._data.shape[1]
        self._peaks = [0.0] * n_ch
        self._rmss = [0.0] * n_ch

        if not (SD_AVAILABLE and sd is not None):
            if not _SystemPlayer.available():
                raise RuntimeError(
                    "No audio output is available. Install 'sounddevice' "
                    "(pip install sounddevice) or export the clip and play "
                    "it in a media player -- editing works either way.")
            self._using_system = True
            self._system.play(np.asarray(data[start:end], dtype=np.float64),
                              self._sr)
            return

        self._using_system = False
        channels = 1 if self._data.ndim == 1 else self._data.shape[1]

        def callback(outdata, frames, time_info, status):
            data_ref = self._data
            if data_ref is None:
                outdata.fill(0)
                raise sd.CallbackStop

            written = 0
            frame = self._frame
            block_start = frame
            n_ch = outdata.shape[1] if outdata.ndim == 2 else 1
            peaks = np.zeros(n_ch)
            sq_acc = np.zeros(n_ch)

            while written < frames:
                if frame >= self._end_frame:
                    if self._loop:
                        frame = self._start_frame
                    else:
                        break
                take = min(frames - written, self._end_frame - frame)
                chunk = data_ref[frame:frame + take]
                if chunk.ndim == 1:
                    chunk = chunk[:, None]
                block = np.clip(chunk * self._volume, -1.0, 1.0)
                outdata[written:written + take] = block
                if block.size:
                    peaks = np.maximum(peaks, np.max(np.abs(block), axis=0))
                    sq_acc += np.sum(block.astype(np.float64) ** 2, axis=0)
                written += take
                frame += take

            if written < frames:
                outdata[written:] = 0

            now = time.perf_counter()
            dac = getattr(time_info, "outputBufferDacTime", None)

            with self._lock:
                self._frame = frame
                self._block_frame = block_start
                self._block_span = written
                self._block_time = now
                self._block_dac = dac
                if written:
                    per_ch_rms = np.sqrt(sq_acc / written)
                    self._peaks = [float(v) for v in peaks]
                    self._rmss = [float(v) for v in per_ch_rms]
                    self._peak = float(np.max(peaks))
                    self._rms = float(np.sqrt(sq_acc.sum() / (written * n_ch)))
                else:
                    self._peaks = [0.0] * n_ch
                    self._rmss = [0.0] * n_ch
                    self._peak = self._rms = 0.0

            if written == 0:
                self._finished = True
                raise sd.CallbackStop

        def finished():
            self._finished = True

        try:
            self._stream = self._open_stream(channels, callback, finished)
        except Exception as exc:
            if not _SystemPlayer.available():
                raise
            self._device_error = exc
            self._using_system = True
            self._system.play(np.asarray(data[start:end], dtype=np.float64),
                              self._sr)
            return
        self._stream.start()

    def _open_stream(self, channels: int, callback, finished):
        def build():
            return sd.OutputStream(
                samplerate=self._sr, channels=channels, dtype="float32",
                blocksize=self.BLOCK, callback=callback,
                finished_callback=finished)

        try:
            return build()
        except Exception:
            pass

        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        return build()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        if self._using_system:
            try:
                self._system.stop()
            except Exception:
                pass
            self._using_system = False
        self._finished = True
        with self._lock:
            self._peak = self._rms = 0.0
            self._peaks = [0.0] * len(self._peaks)
            self._rmss = [0.0] * len(self._rmss)
            self._block_span = 0
            self._block_time = None
            self._block_dac = None

    def close(self) -> None:
        self.stop()
        self._data = None
