import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dsp_equalizer import (
    BAND_LABELS,
    DEFAULT_FS,
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    METHOD_INFO,
    PRESET_DESCRIPTIONS,
    PRESETS,
    audio_to_wav_bytes,
    compare_methods,
    composite_response,
    fft_spectrum,
    generate_demo_audio,
    process_audio,
    read_audio_file,
)

st.set_page_config(page_title="DSP Audio Equalizer", page_icon="🎛️", layout="wide")

st.markdown(
    """
    <style>
    .main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {font-size: 1.05rem; color: #5f6368; margin-bottom: 1.2rem;}
    .feature-card {padding: 1rem; border: 1px solid #e5e7eb; border-radius: 1rem; background: #fafafa; min-height: 118px;}
    .small-muted {font-size: 0.9rem; color: #6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "audio" not in st.session_state:
    st.session_state.audio = generate_demo_audio(DEFAULT_FS, 6.0)
    st.session_state.fs = DEFAULT_FS
    st.session_state.audio_name = "Demo signal"
if "processed_audio" not in st.session_state:
    st.session_state.processed_audio = None
if "processed_metrics" not in st.session_state:
    st.session_state.processed_metrics = None
if "last_settings" not in st.session_state:
    st.session_state.last_settings = None

st.sidebar.title("🎛️ Navigation")
page = st.sidebar.radio(
    "Choose page",
    ["Home", "Upload Audio", "Equalizer", "Comparison", "Visualization", "Download"],
)

st.sidebar.divider()
st.sidebar.header("Equalizer Controls")
method = st.sidebar.selectbox(
    "Filter method",
    ["RBJ Parametric EQ", "IIR Butterworth SOS", "FIR Kaiser"],
    index=0,
    help="RBJ is recommended as the main/default method because it is fast and suitable for interactive sliders.",
)

preset = st.sidebar.selectbox("Smart preset", list(PRESETS.keys()), index=1)
st.sidebar.caption(PRESET_DESCRIPTIONS[preset])
fir_taps = st.sidebar.slider("FIR taps", 257, 4097, 1025, step=256)
st.sidebar.caption("The FIR taps value affects only the FIR Kaiser method.")
st.sidebar.caption("Adjust each frequency band in dB.")

gains = []
for label, default in zip(BAND_LABELS, PRESETS[preset]):
    gains.append(
        st.sidebar.slider(
            f"{label} Hz",
            min_value=float(GAIN_MIN_DB),
            max_value=float(GAIN_MAX_DB),
            value=float(default),
            step=0.5,
            key=f"gain_{preset}_{label}",
        )
    )
gains = np.array(gains, dtype=float)
current_settings = (method, preset, fir_taps, tuple(np.round(gains, 3)), st.session_state.audio_name)
if st.session_state.last_settings != current_settings:
    st.session_state.processed_audio = None
    st.session_state.processed_metrics = None


def draw_waveform(x, fs, title):
    duration = len(x) / fs
    max_points = 5000
    step = max(1, len(x) // max_points)
    t = np.arange(len(x))[::step] / fs
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, x[::step], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def draw_spectrum(x, fs, title):
    freqs, mag_db = fft_spectrum(x, fs)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.semilogx(freqs[1:], mag_db[1:], linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(20, min(20000, fs / 2))
    ax.set_ylim(np.max(mag_db) - 100, np.max(mag_db) + 5)
    ax.grid(True, which="both", alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def draw_spectrum_before_after(x, y, fs):
    freqs_in, mag_in = fft_spectrum(x, fs)
    freqs_out, mag_out = fft_spectrum(y, fs)
    top = max(float(np.max(mag_in)), float(np.max(mag_out)))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.semilogx(freqs_in[1:], mag_in[1:], linewidth=1.2, label="Before equalization")
    ax.semilogx(freqs_out[1:], mag_out[1:], linewidth=1.2, label="After equalization")
    ax.set_title("FFT Spectrum Before vs After Equalization")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(20, min(20000, fs / 2))
    ax.set_ylim(top - 100, top + 5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


def draw_filter_response(selected_method):
    methods = ["RBJ Parametric EQ", "IIR Butterworth SOS", "FIR Kaiser"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for m in methods:
        response = composite_response(m, st.session_state.fs, gains, fir_taps=fir_taps)
        linewidth = 2.5 if m == selected_method else 1.1
        alpha = 1.0 if m == selected_method else 0.5
        ax.semilogx(response["freqs"], response["mag_db"], linewidth=linewidth, alpha=alpha, label=m)
    target = composite_response(selected_method, st.session_state.fs, gains, fir_taps=fir_taps)
    ax.semilogx(target["freqs"], target["target_db"], linestyle="--", linewidth=2, label="Target slider curve")
    ax.set_title("Filter Frequency Response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_xlim(20, min(20000, st.session_state.fs / 2))
    ax.set_ylim(-15, 15)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


def show_audio_before_after():
    if st.session_state.processed_audio is None:
        return
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Before: Original Audio")
        st.audio(audio_to_wav_bytes(st.session_state.audio, st.session_state.fs), format="audio/wav")
    with c2:
        st.subheader("After: Processed Audio")
        st.audio(audio_to_wav_bytes(st.session_state.processed_audio, st.session_state.fs), format="audio/wav")


def show_clipping_status(metrics):
    if metrics is None:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input peak", f"{metrics['input_peak']:.3f}")
    c2.metric("Peak before normalize", f"{metrics['max_abs_before_normalization']:.3f}")
    c3.metric("Peak after normalize", f"{metrics['max_abs_after_normalization']:.3f}")
    c4.metric("Scale factor", f"{metrics['normalization_scale_factor']:.3f}")

    if metrics["clipping_detected"]:
        st.warning("Clipping risk detected: the processed signal exceeded ±1. Auto-normalization was applied to protect the output WAV.")
    elif metrics["normalization_applied"]:
        st.info("Auto-normalization was applied because the processed signal was above the safe peak level of 0.95.")
    else:
        st.success("No clipping risk detected. The output signal is already within a safe amplitude range.")


def process_current_audio():
    y, metrics = process_audio(st.session_state.audio, st.session_state.fs, gains, method, fir_taps=fir_taps)
    st.session_state.processed_audio = y
    st.session_state.processed_metrics = metrics
    st.session_state.last_settings = current_settings
    return y, metrics


st.markdown('<div class="main-title">🎛️ 10-Band DSP Audio Equalizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Main method: RBJ Parametric EQ. Includes smart presets, before/after audio, FFT analysis, clipping protection, and filter-response visualization.</div>',
    unsafe_allow_html=True,
)

if page == "Home":
    st.header("Project Idea")
    st.markdown(
        """
        This project builds an interactive digital audio equalizer. The main method is **RBJ Parametric EQ** because it is fast, low-latency, and suitable for real-time slider control.

        The project also compares **FIR Kaiser**, **IIR Butterworth SOS**, and **RBJ Parametric EQ** to show the difference between filter-design approaches.
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Default method", "RBJ EQ")
    c2.metric("Sample rate", f"{st.session_state.fs:,} Hz")
    c3.metric("Bands", "10")

    st.subheader("Smart features added")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<div class="feature-card"><b>Before/After Player</b><br><span class="small-muted">Listen to the original audio and the equalized output side by side.</span></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feature-card"><b>FFT Spectrum</b><br><span class="small-muted">View the frequency spectrum before and after equalization.</span></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feature-card"><b>Smart Presets</b><br><span class="small-muted">Use Flat, Music, Bass Boost, Vocal Clarity, Treble Boost, or Warm Sound.</span></div>', unsafe_allow_html=True)
    f4, f5 = st.columns(2)
    with f4:
        st.markdown('<div class="feature-card"><b>Clipping Protection</b><br><span class="small-muted">Detects clipping risk and applies automatic normalization.</span></div>', unsafe_allow_html=True)
    with f5:
        st.markdown('<div class="feature-card"><b>Filter Response Plot</b><br><span class="small-muted">Shows the designed gain response of FIR, IIR, and RBJ filters.</span></div>', unsafe_allow_html=True)

elif page == "Upload Audio":
    st.header("Upload Audio")
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    max_seconds = st.slider("Maximum duration to process", 10, 180, 90, step=10)

    if uploaded is not None:
        x, fs = read_audio_file(uploaded, DEFAULT_FS, max_seconds=max_seconds)
        st.session_state.audio = x
        st.session_state.fs = fs
        st.session_state.audio_name = uploaded.name
        st.session_state.processed_audio = None
        st.session_state.processed_metrics = None
        st.session_state.last_settings = None
        st.success(f"Loaded {uploaded.name} successfully.")
    if st.button("Use demo signal instead"):
        st.session_state.audio = generate_demo_audio(DEFAULT_FS, 6.0)
        st.session_state.fs = DEFAULT_FS
        st.session_state.audio_name = "Demo signal"
        st.session_state.processed_audio = None
        st.session_state.processed_metrics = None
        st.session_state.last_settings = None
        st.success("Demo signal loaded.")

    st.write(f"Current audio: **{st.session_state.audio_name}**")
    st.audio(audio_to_wav_bytes(st.session_state.audio, st.session_state.fs), format="audio/wav")
    draw_waveform(st.session_state.audio, st.session_state.fs, "Input waveform")
    draw_spectrum(st.session_state.audio, st.session_state.fs, "Input FFT spectrum")

elif page == "Equalizer":
    st.header("Equalizer Page")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Selected filter method")
        info = METHOD_INFO[method]
        st.write(f"**Design:** {info['filter_type']}")
        st.write(f"**Phase:** {info['phase']}")
        st.write(f"**Latency:** {info['latency']}")
        st.write(f"**Best use:** {info['best_for']}")
        st.write(f"**Smart preset:** {preset}")
        st.caption(PRESET_DESCRIPTIONS[preset])
        st.dataframe(pd.DataFrame({"Band (Hz)": BAND_LABELS, "Gain (dB)": gains}), hide_index=True, use_container_width=True)
    with col2:
        draw_filter_response(method)

    if st.button("Process audio", type="primary"):
        y, metrics = process_current_audio()
        st.success("Audio processed successfully.")
        show_clipping_status(metrics)
        show_audio_before_after()

elif page == "Comparison":
    st.header("Comparison Page")
    df = pd.DataFrame(compare_methods(st.session_state.fs, gains, fir_taps=fir_taps))
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown(
        """
        **Final recommendation:** Use **RBJ Parametric EQ** as the main app method.  
        Use **IIR Butterworth SOS** as a strong real-time comparison method.  
        Keep **FIR Kaiser** for academic explanation because it shows linear-phase filter design, but it has higher latency.
        """
    )
    draw_filter_response(method)

elif page == "Visualization":
    st.header("Visualization Page")
    if st.session_state.processed_audio is None:
        st.info("Processing current audio first so the before/after audio and plots are available.")
        process_current_audio()

    x = st.session_state.audio
    y = st.session_state.processed_audio
    fs = st.session_state.fs

    show_clipping_status(st.session_state.processed_metrics)
    show_audio_before_after()

    tab_wave, tab_fft, tab_response = st.tabs(["Waveform", "FFT Before/After", "Filter Response"])
    with tab_wave:
        draw_waveform(x, fs, "Original waveform before equalization")
        draw_waveform(y, fs, "Processed waveform after equalization")
    with tab_fft:
        draw_spectrum_before_after(x, y, fs)
    with tab_response:
        draw_filter_response(method)

elif page == "Download":
    st.header("Download Page")
    if st.session_state.processed_audio is None:
        st.info("Process audio first using the current settings.")
        if st.button("Process now", type="primary"):
            process_current_audio()
            st.success("Audio processed.")
    if st.session_state.processed_audio is not None:
        y = st.session_state.processed_audio
        fs = st.session_state.fs
        metrics = st.session_state.processed_metrics
        show_clipping_status(metrics)
        show_audio_before_after()
        st.download_button(
            "⬇️ Download processed WAV",
            data=audio_to_wav_bytes(y, fs),
            file_name=f"equalized_{method.lower().replace(' ', '_').replace('/', '_')}.wav",
            mime="audio/wav",
        )
        st.write("**Processing metrics:**")
        st.json(metrics)
