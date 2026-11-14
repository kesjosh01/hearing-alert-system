"""Explicit PCM16 -> log-Mel contract, also implemented in portable C++."""
from __future__ import annotations

import numpy as np

from .common import config


def rfft_power_float32(frames):
    """Match the portable FFT's float32 rounding at each radix-2 butterfly."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 400:
        raise ValueError("Expected [frames,400] windowed samples")
    indices = np.asarray([int(f"{i:09b}"[::-1], 2) for i in range(512)])
    real = np.zeros((len(frames), 512), dtype=np.float32)
    real[:, :400] = frames
    real = real[:, indices].copy()
    imag = np.zeros_like(real)
    cosine = np.cos(-2 * np.pi * np.arange(256) / 512).astype(np.float32)
    sine = np.sin(-2 * np.pi * np.arange(256) / 512).astype(np.float32)
    for length in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        half = length // 2
        r = real.reshape(len(frames), -1, length)
        im = imag.reshape(len(frames), -1, length)
        ar, ai = r[..., :half].copy(), im[..., :half].copy()
        br, bi = r[..., half:].copy(), im[..., half:].copy()
        wr, wi = cosine[::512 // length], sine[::512 // length]
        re, imaginary = br * wr - bi * wi, br * wi + bi * wr
        r[..., half:], im[..., half:] = ar - re, ai - imaginary
        r[..., :half], im[..., :half] = ar + re, ai + imaginary
    return real[:, :257] ** 2 + imag[:, :257] ** 2


class LogMel:
    def __init__(self, options=None):
        self.options = dict(options or config()[0]["frontend"])
        o = self.options
        # v1 is intentionally a single fixed contract shared with firmware LUTs.
        fixed = {"sample_rate": 16000, "window_samples": 16000, "window_hop_samples": 8000,
                 "frame_samples": 400, "frame_hop_samples": 160, "fft_size": 512, "mel_bins": 40,
                 "frame_window": "periodic_hann", "mel_scale": "htk_unnormalized_triangles",
                 "spectrum": "unnormalized_rfft_power", "fft_arithmetic": "float32_radix2",
                 "amplitude": "pcm16_divide_32768_no_peak_normalization",
                 "padding": "short_clip_right_zero_pad_no_stft_end_padding"}
        if o != config()[0]["frontend"] or any(o.get(k) != v for k, v in fixed.items()):
            raise ValueError("A changed frontend needs a new version and regenerated firmware tables")
        if not 0 <= o["lower_hz"] < o["upper_hz"] <= 8000 or not np.isfinite(o["log_floor"]) or o["log_floor"] <= 0:
            raise ValueError("Invalid Mel frequency range or log floor")
        self.frames = 1 + (o["window_samples"] - o["frame_samples"]) // o["frame_hop_samples"]
        self.shape = (self.frames, o["mel_bins"], 1)
        self.hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(o["frame_samples"]) / o["frame_samples"])).astype(np.float32)
        mel = lambda hz: 2595 * np.log10(1 + np.asarray(hz) / 700)
        edges = 700 * (10 ** (np.linspace(mel(o["lower_hz"]), mel(o["upper_hz"]), o["mel_bins"] + 2) / 2595) - 1)
        frequencies = np.arange(o["fft_size"] // 2 + 1) * o["sample_rate"] / o["fft_size"]
        self.mel = np.asarray([
            np.maximum(0, np.minimum((frequencies - a) / (b - a), (c - frequencies) / (c - b)))
            for a, b, c in zip(edges[:-2], edges[1:-1], edges[2:])
        ], dtype=np.float32)

    def __call__(self, pcm):
        pcm = np.asarray(pcm)
        if pcm.dtype != np.int16 or pcm.shape != (self.options["window_samples"],):
            raise ValueError("Expected exactly 16000 mono PCM16 samples; window/pad first")
        o = self.options
        audio = pcm.astype(np.float32) / np.float32(32768)
        frames = np.lib.stride_tricks.sliding_window_view(audio, o["frame_samples"])[::o["frame_hop_samples"]]
        power = rfft_power_float32(frames * self.hann)
        mel_power = power @ self.mel.T
        return np.log(np.maximum(mel_power, np.float32(o["log_floor"]))).astype(np.float32)[..., None]


def window_starts(samples, window=16000, hop=8000):
    if samples <= 0 or window <= 0 or not 0 < hop <= window:
        raise ValueError("Positive audio/window length and 0 < hop <= window required")
    if samples <= window:
        return [0]
    starts = list(range(0, samples - window + 1, hop))
    if starts[-1] != samples - window:
        starts.append(samples - window)
    return starts


def windows(pcm):
    pcm = np.asarray(pcm)
    if pcm.dtype != np.int16 or pcm.ndim != 1:
        raise ValueError("Expected mono PCM16")
    for start in window_starts(len(pcm)):
        chunk = np.zeros(16000, dtype=np.int16)
        length = min(16000, len(pcm) - start)
        chunk[:length] = pcm[start:start + length]
        yield start, chunk


def quantize(values, scale, zero_point):
    if not np.isfinite(scale) or scale <= 0 or not -128 <= zero_point <= 127:
        raise ValueError("Invalid int8 quantization parameters")
    value = np.asarray(values, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Non-finite frontend output")
    value = value / np.float32(scale)
    rounded = np.copysign(np.floor(np.abs(value) + np.float32(0.5)), value)
    return np.clip(rounded + zero_point, -128, 127).astype(np.int8)


def synthetic_pcm(count=24, seed=42):
    """Calibration/check signals, with no claim to represent real acoustics."""
    rng = np.random.default_rng(seed)
    t = np.arange(16000, dtype=np.float64) / 16000
    yield np.zeros(16000, dtype=np.int16)
    for index in range(1, count):
        amplitude = [0.003, 0.03, 0.3, 0.8][index % 4]
        if index % 3 == 0:
            signal = rng.normal(0, amplitude, len(t))
        elif index % 3 == 1:
            signal = amplitude * np.sin(2 * np.pi * [200, 700, 1600, 4000, 7000][index % 5] * t)
        else:
            signal = amplitude * np.sin(2 * np.pi * (300 * t + 3000 * t * t)) * (t < 0.35)
        yield np.clip(np.rint(signal * 32768), -32768, 32767).astype(np.int16)
