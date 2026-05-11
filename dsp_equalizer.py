import io
import math
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# Main audio settings used across the project.
DEFAULT_FS = 44100
EPS = 1e-20
SAFE_PEAK = 0.95
GAIN_MIN_DB = -12.0
GAIN_MAX_DB = 12.0

# Common 10-band equalizer frequencies.
BAND_CENTERS = np.array(
    [31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0],
    dtype=float,
)
BAND_LABELS = ["31.25", "62.5", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]

# Ready-made settings so the user can hear the EQ effect quickly.
PRESETS: Dict[str, List[float]] = {
    "Flat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Smiley / Music": [7, 5, 2.5, -1.5, -4, -3, -1, 2.5, 5, 7],
    "Bass Boost": [8, 7, 5, 2, 0, -1, -2, -2, -1, 0],
    "Vocal Clarity": [-3, -2, -1, 1, 3, 4, 3, 2, 0, -2],
    "Treble Boost": [-2, -2, -1, 0, 0, 1, 3, 5, 6, 5],
    "Warm Sound": [4, 3, 2, 1, 0, -1, -1, 0, 1, 1],
}

PRESET_DESCRIPTIONS: Dict[str, str] = {
    "Flat": "No boost or cut. Good as a neutral baseline.",
    "Smiley / Music": "Boosts bass and treble for a wider music sound.",
    "Bass Boost": "Increases low frequencies for stronger bass.",
    "Vocal Clarity": "Boosts mid frequencies to make speech and vocals clearer.",
    "Treble Boost": "Increases high frequencies for brighter audio.",
    "Warm Sound": "Adds low-mid warmth and slightly softens harshness.",
}

# This text is used in the comparison page.
METHOD_INFO = {
    "RBJ Parametric EQ": {
        "filter_type": "Cascaded IIR biquads",
        "phase": "Non-linear phase",
        "latency": "Almost zero",
        "best_for": "Best default for the Streamlit app because it is fast and works smoothly with sliders.",
    },
    "IIR Butterworth SOS": {
        "filter_type": "Parallel IIR Butterworth band-pass filters",
        "phase": "Non-linear phase",
        "latency": "Almost zero",
        "best_for": "Strong real-time method with stable SOS implementation and good magnitude accuracy.",
    },
    "FIR Kaiser": {
        "filter_type": "Parallel FIR Kaiser-window band-pass filters",
        "phase": "Linear phase",
        "latency": "High: delay = (numtaps - 1) / 2 samples",
        "best_for": "Good for academic comparison and phase-sensitive offline processing, but not the best for interactive apps.",
    },
}


@dataclass
class FilterBank:
    name: str
    filters: list
    latency_samples: int
    structure: str
    ops_per_sample: int


def db_to_linear(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 20.0)


def linear_to_db(x, floor_db: float = -120.0):
    return np.maximum(floor_db, 20.0 * np.log10(np.maximum(np.abs(x), EPS)))


def ensure_mono(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def peak_normalize(x, peak: float = SAFE_PEAK):
    x = np.asarray(x, dtype=float)
    max_abs = float(np.max(np.abs(x)) + EPS)
    if max_abs > peak:
        x = x * (peak / max_abs)
    return x.astype(np.float32)


def normalize_with_report(x, peak: float = SAFE_PEAK):
    x = np.asarray(x, dtype=float)
    max_abs_before = float(np.max(np.abs(x)) + EPS)
    clipping_detected = bool(max_abs_before > 1.0)
    normalization_applied = bool(max_abs_before > peak)
    scale_factor = 1.0

    if normalization_applied:
        scale_factor = float(peak / max_abs_before)
        x = x * scale_factor

    max_abs_after = float(np.max(np.abs(x)) + EPS)
    report = {
        "clipping_detected": clipping_detected,
        "normalization_applied": normalization_applied,
        "max_abs_before_normalization": max_abs_before,
        "max_abs_after_normalization": max_abs_after,
        "normalization_scale_factor": scale_factor,
    }
    return x.astype(np.float32), report


# Simple test signal used when the user does not upload audio.
def generate_demo_audio(fs: int = DEFAULT_FS, duration: float = 6.0):
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)

    bass = 0.35 * np.sin(2 * np.pi * 70 * t)
    mid = 0.20 * np.sin(2 * np.pi * 440 * t) + 0.12 * np.sin(2 * np.pi * 880 * t)
    treble = 0.08 * np.sin(2 * np.pi * 4000 * t)
    noise = 0.02 * rng.standard_normal(n)

    envelope = np.linspace(0.2, 1.0, n)
    x = envelope * (bass + mid + treble + noise)
    return peak_normalize(x, 0.85)


# Convert any WAV sample type into float audio between -1 and 1.
def audio_array_to_float(data):
    data = np.asarray(data)

    if np.issubdtype(data.dtype, np.floating):
        return np.clip(data.astype(float), -1.0, 1.0)

    if np.issubdtype(data.dtype, np.unsignedinteger):
        info = np.iinfo(data.dtype)
        midpoint = (info.max + 1) / 2.0
        return ((data.astype(float) - midpoint) / midpoint).clip(-1.0, 1.0)

    if np.issubdtype(data.dtype, np.signedinteger):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), info.max)
        return (data.astype(float) / scale).clip(-1.0, 1.0)

    return data.astype(float)


def _get_uploaded_extension(uploaded_file):
    name = getattr(uploaded_file, "name", "audio.wav")
    return name.lower().split(".")[-1]


def _read_wav_bytes(file_bytes):
    fs, data = wavfile.read(io.BytesIO(file_bytes))
    return audio_array_to_float(data), int(fs)


def _read_mp3_bytes(file_bytes):
    if AudioSegment is None:
        raise ImportError("MP3 support needs pydub. Install it with: pip install pydub")

    audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
    fs = int(audio.frame_rate)
    channels = int(audio.channels)
    sample_width = int(audio.sample_width)
    samples = np.array(audio.get_array_of_samples()).astype(float)

    if channels > 1:
        samples = samples.reshape((-1, channels))

    scale = float(2 ** (8 * sample_width - 1))
    x = np.clip(samples / scale, -1.0, 1.0)
    return x, fs


# Read WAV or MP3 and prepare it for filtering.
def read_audio_file(uploaded_file, target_fs: int = DEFAULT_FS, max_seconds: int = 90):
    file_bytes = uploaded_file.getvalue()
    ext = _get_uploaded_extension(uploaded_file)

    if ext == "mp3":
        x, fs = _read_mp3_bytes(file_bytes)
    elif ext == "wav":
        x, fs = _read_wav_bytes(file_bytes)
    else:
        raise ValueError("Unsupported audio format. Please upload WAV or MP3.")

    x = ensure_mono(x)

    if fs != target_fs:
        gcd = math.gcd(int(fs), int(target_fs))
        x = signal.resample_poly(x, target_fs // gcd, fs // gcd)
        fs = target_fs

    max_samples = int(max_seconds * fs)
    if len(x) > max_samples:
        x = x[:max_samples]

    return peak_normalize(x, SAFE_PEAK), int(fs)


# Export processed audio as a normal 16-bit WAV file.
def audio_to_wav_bytes(x, fs: int):
    buffer = io.BytesIO()
    y = np.clip(peak_normalize(x, SAFE_PEAK), -1.0, 1.0)
    y_int16 = (y * 32767.0).astype(np.int16)
    wavfile.write(buffer, int(fs), y_int16)
    buffer.seek(0)
    return buffer.getvalue()


# Export processed audio as MP3 for smaller downloadable files.
def audio_to_mp3_bytes(x, fs: int, bitrate: str = "192k"):
    if AudioSegment is None:
        raise ImportError("MP3 export needs pydub. Install it with: pip install pydub")

    y = np.clip(peak_normalize(x, SAFE_PEAK), -1.0, 1.0)
    y_int16 = (y * 32767.0).astype(np.int16)
    audio = AudioSegment(
        y_int16.tobytes(),
        frame_rate=int(fs),
        sample_width=2,
        channels=1,
    )
    buffer = io.BytesIO()
    audio.export(buffer, format="mp3", bitrate=bitrate)
    buffer.seek(0)
    return buffer.getvalue()


# Build lower and upper frequency limits for every EQ band.
def band_edges(centers, fs):
    nyq_guard = fs / 2.0 - 100.0
    edges = []

    for f0 in centers:
        lo = max(20.0, float(f0) / math.sqrt(2.0))
        hi = min(float(f0) * math.sqrt(2.0), nyq_guard)
        if hi <= lo:
            raise ValueError(f"Invalid band for {f0} Hz at fs={fs}")
        edges.append((lo, hi))

    return edges


# Normalize each FIR band at its center frequency.
def normalize_fir_at_freq(b, f0, fs):
    w = np.array([2 * np.pi * f0 / fs])
    _, h = signal.freqz(b, worN=w)
    return b / (np.abs(h[0]) + EPS)


# Normalize each IIR band at its center frequency.
def normalize_sos_at_freq(sos, f0, fs):
    w = np.array([2 * np.pi * f0 / fs])
    _, h = signal.sosfreqz(sos, worN=w)
    out = sos.copy()
    out[0, :3] /= np.abs(h[0]) + EPS
    return out


# FIR has linear phase, so it is useful for comparison and theory.
def design_fir_kaiser_bank(fs, centers=BAND_CENTERS, numtaps: int = 1025, beta: float = 8.6):
    filters = []

    for f0, (lo, hi) in zip(centers, band_edges(centers, fs)):
        b = signal.firwin(
            numtaps,
            [lo, hi],
            pass_zero=False,
            fs=fs,
            window=("kaiser", beta),
            scale=False,
        )
        filters.append(normalize_fir_at_freq(b, f0, fs).astype(float))

    latency = (numtaps - 1) // 2
    ops = len(centers) * (2 * numtaps + 1) + len(centers)
    return FilterBank("FIR Kaiser", filters, latency, f"10 parallel {numtaps}-tap FIR filters", ops)


# IIR SOS is stable and much faster than long FIR filters.
def design_iir_butterworth_bank(fs, centers=BAND_CENTERS, order: int = 4):
    filters = []

    for f0, (lo, hi) in zip(centers, band_edges(centers, fs)):
        sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
        filters.append(normalize_sos_at_freq(sos, f0, fs).astype(float))

    total_sos = sum(len(sos) for sos in filters)
    ops = total_sos * 9 + len(centers) * 3
    return FilterBank("IIR Butterworth SOS", filters, 0, "10 parallel 4th-order Butterworth SOS filters", ops)


# RBJ peaking EQ is the main real-time method in this app.
def rbj_peaking_sos(f0, gain_db, fs, q=math.sqrt(2.0)):
    A = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * math.pi * float(f0) / fs
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=float)


def design_rbj_parametric_chain(fs, gains_db, centers=BAND_CENTERS, q=math.sqrt(2.0)):
    sos = np.vstack([rbj_peaking_sos(f0, g, fs, q) for f0, g in zip(centers, gains_db)])
    return FilterBank("RBJ Parametric EQ", [sos], 0, f"10 cascaded RBJ peaking biquads, Q={q:.3f}", len(sos) * 9)


# Choose the needed filter design from one function.
def design_filter_bank(method: str, fs: int, gains_db=None, fir_taps: int = 1025):
    if method == "FIR Kaiser":
        if fir_taps % 2 == 0:
            fir_taps += 1
        return design_fir_kaiser_bank(fs, numtaps=fir_taps)

    if method == "IIR Butterworth SOS":
        return design_iir_butterworth_bank(fs)

    if method == "RBJ Parametric EQ":
        if gains_db is None:
            gains_db = np.zeros(len(BAND_CENTERS))
        return design_rbj_parametric_chain(fs, gains_db)

    raise ValueError(f"Unknown method: {method}")


def fft_convolve_same(x, h):
    return signal.fftconvolve(x, h, mode="same")


# Graphic EQ style: add the boosted or reduced FIR bands back to the signal.
def apply_fir_graphic_eq(x, bank: FilterBank, gains_db):
    g = db_to_linear(gains_db)
    y = x.astype(float).copy()

    for b, gg in zip(bank.filters, g):
        band = fft_convolve_same(x, b)
        y += (gg - 1.0) * band

    return y


# Same idea as FIR, but using efficient IIR SOS bands.
def apply_iir_graphic_eq(x, bank: FilterBank, gains_db):
    g = db_to_linear(gains_db)
    y = x.astype(float).copy()

    for sos, gg in zip(bank.filters, g):
        band = signal.sosfilt(sos, x)
        y += (gg - 1.0) * band

    return y


# RBJ processes the audio through all peaking filters in one chain.
def apply_rbj_eq(x, bank: FilterBank):
    return signal.sosfilt(bank.filters[0], x).astype(float)


# Main function called by Streamlit.
def process_audio(x, fs: int, gains_db, method: str, fir_taps: int = 1025):
    x = np.asarray(x, dtype=float)
    gains_db = np.asarray(gains_db, dtype=float)
    bank = design_filter_bank(method, fs, gains_db=gains_db, fir_taps=fir_taps)

    start = time.perf_counter()

    if method == "FIR Kaiser":
        y = apply_fir_graphic_eq(x, bank, gains_db)
    elif method == "IIR Butterworth SOS":
        y = apply_iir_graphic_eq(x, bank, gains_db)
    else:
        y = apply_rbj_eq(x, bank)

    elapsed = time.perf_counter() - start
    y, norm_report = normalize_with_report(y, peak=SAFE_PEAK)
    duration = max(len(x) / fs, EPS)

    metrics = {
        "method": method,
        "duration_s": duration,
        "time_s": elapsed,
        "rtf_x": elapsed / duration,
        "latency_samples": bank.latency_samples,
        "latency_ms": 1000.0 * bank.latency_samples / fs,
        "ops_per_sample_est": bank.ops_per_sample,
        "structure": bank.structure,
        "input_peak": float(np.max(np.abs(x)) + EPS),
        **norm_report,
    }
    return y, metrics


# Build the smooth target curve shown beside the real filter response.
def desired_curve(freqs, gains_db):
    return np.interp(
        np.log10(freqs),
        np.log10(BAND_CENTERS),
        np.asarray(gains_db, dtype=float),
        left=float(gains_db[0]),
        right=float(gains_db[-1]),
    )


# Calculate the frequency response for FIR, IIR, or RBJ.
def composite_response(method: str, fs: int, gains_db, fir_taps: int = 1025, n_points: int = 2048):
    freqs = np.logspace(np.log10(20), np.log10(20000), n_points)
    gains_db = np.asarray(gains_db, dtype=float)
    bank = design_filter_bank(method, fs, gains_db=gains_db, fir_taps=fir_taps)
    w = 2 * np.pi * freqs / fs

    if method == "FIR Kaiser":
        h_total = np.ones_like(w, dtype=complex)
        for b, gdb in zip(bank.filters, gains_db):
            _, h = signal.freqz(b, worN=w)
            h_total += (db_to_linear(gdb) - 1.0) * h
    elif method == "IIR Butterworth SOS":
        h_total = np.ones_like(w, dtype=complex)
        for sos, gdb in zip(bank.filters, gains_db):
            _, h = signal.sosfreqz(sos, worN=w)
            h_total += (db_to_linear(gdb) - 1.0) * h
    else:
        _, h_total = signal.sosfreqz(bank.filters[0], worN=w)

    mag_db = linear_to_db(np.abs(h_total))
    target_db = desired_curve(freqs, gains_db)

    center_errors = []
    for f0, gdb in zip(BAND_CENTERS, gains_db):
        idx = np.argmin(np.abs(freqs - f0))
        center_errors.append(float(mag_db[idx] - gdb))
    rms_error = float(np.sqrt(np.mean(np.square(center_errors))))

    return {
        "freqs": freqs,
        "mag_db": mag_db,
        "target_db": target_db,
        "center_rms_error_db": rms_error,
        "latency_ms": 1000.0 * bank.latency_samples / fs,
        "ops_per_sample_est": bank.ops_per_sample,
        "structure": bank.structure,
    }


# FFT shows what changed in the audio spectrum after equalization.
def fft_spectrum(x, fs: int, n_fft: int = 8192):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([]), np.array([])

    n = min(len(x), n_fft)
    segment = x[:n]
    window = np.hanning(n)
    spectrum = np.fft.rfft(segment * window, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    mag_db = linear_to_db(np.abs(spectrum) / (np.sum(window) / 2.0 + EPS))

    return freqs, mag_db


# Create the table used to compare FIR, IIR, and RBJ.
def compare_methods(fs: int, gains_db, fir_taps: int = 1025):
    methods = ["FIR Kaiser", "IIR Butterworth SOS", "RBJ Parametric EQ"]
    rows = []

    for method in methods:
        response = composite_response(method, fs, gains_db, fir_taps=fir_taps)
        info = METHOD_INFO[method]
        rows.append(
            {
                "Method": method,
                "Filter design": info["filter_type"],
                "Phase": info["phase"],
                "Latency (ms)": round(response["latency_ms"], 3),
                "Center RMS Error (dB)": round(response["center_rms_error_db"], 3),
                "Estimated Ops/Sample": int(response["ops_per_sample_est"]),
                "Best Use": info["best_for"],
            }
        )

    return rows
