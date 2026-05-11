# DSP Audio Equalizer with Streamlit

An interactive audio equalizer web application built with **Python**, **Streamlit**, and **Digital Signal Processing (DSP)** techniques.  
The project allows users to upload a WAV audio file, apply equalization using a **10-band RBJ Parametric EQ**, visualize the audio in both time and frequency domains, and compare different filter design methods.

---

## Project Overview

This project demonstrates how digital filters can be used to modify audio signals in real time.  
The main goal is to build a practical and visual audio equalizer that combines:

- Audio signal processing
- Filter design
- FFT spectrum analysis
- Before/after audio comparison
- Streamlit web deployment

The application uses **RBJ Parametric EQ** as the main equalizer method because it is fast, low-latency, and suitable for interactive slider-based control.

---

## Main Features

### Audio Upload

Users can upload a `.wav` audio file and process it directly inside the web app.

### 10-Band Equalizer

The equalizer provides control over ten common frequency bands:

| Band | Frequency |
|---|---:|
| 1 | 31.25 Hz |
| 2 | 62.5 Hz |
| 3 | 125 Hz |
| 4 | 250 Hz |
| 5 | 500 Hz |
| 6 | 1 kHz |
| 7 | 2 kHz |
| 8 | 4 kHz |
| 9 | 8 kHz |
| 10 | 16 kHz |

These bands allow detailed control over bass, mids, vocals, brightness, and treble.

### Smart Presets

The app includes ready-to-use audio presets such as:

- Flat
- Bass Boost
- Vocal Clarity
- Treble Boost
- Warm Sound
- Smiley / Music

### Before and After Audio Player

Users can listen to both the original and processed audio to clearly hear the effect of the equalizer.

### FFT Spectrum Analysis

The project uses Fast Fourier Transform (FFT) to show the frequency content of the audio before and after processing.

This helps visualize how the equalizer changes the signal in the frequency domain.

### Filter Frequency Response

The app shows the frequency response of the selected equalizer, making it clear how the filter boosts or cuts different frequency bands.

### Clipping Warning and Auto-Normalization

When boosting frequencies, the signal may become too loud and clip.  
The app detects this problem and automatically normalizes the output to a safe peak level of `0.95`.

### Processed Audio Download

After equalization, users can download the processed audio as a WAV file.

---

## Filter Methods Compared

The project compares three DSP filtering approaches:

| Method | Description | Best Use |
|---|---|---|
| RBJ Parametric EQ | Cascaded IIR biquad filters | Best for real-time Streamlit equalizer |
| IIR Butterworth SOS | Stable second-order section filters | Good real-time DSP comparison |
| FIR Kaiser | Linear-phase FIR filter design | Good for academic explanation and offline filtering |

The final application uses **RBJ Parametric EQ** as the default method because it provides fast processing, low latency, and smooth interaction with sliders.

---

## Why RBJ Parametric EQ?

RBJ Parametric EQ was selected as the main method because it is commonly used in practical audio equalizers.  
It works by applying a chain of peaking filters, where each band can boost or reduce a specific frequency range.

Compared with FIR filters, RBJ has much lower latency.  
Compared with general IIR filter banks, RBJ is more suitable for interactive equalizer sliders.

---

## Project Structure

```text
dsp-audio-equalizer-streamlit/
│
├── app.py
├── dsp_equalizer.py
├── requirements.txt
├── README.md
├── COMPARISON.md
├── equalizer.ipynb
├── config.toml
└── .gitignore
