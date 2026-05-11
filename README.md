# 10-Band DSP Audio Equalizer

Interactive Streamlit audio equalizer using **RBJ Parametric EQ** as the main/default method, with FIR and IIR included for comparison.

## Main Features

- Before/after audio player
- FFT spectrum before and after equalization
- Smart presets: Flat, Smiley/Music, Bass Boost, Vocal Clarity, Treble Boost, Warm Sound
- Clipping warning and automatic normalization
- Filter frequency response plot
- FIR vs IIR vs RBJ comparison table
- Download processed WAV file
- Clean Jupyter notebook for filter design and FFT analysis

## Why RBJ is the default

RBJ Parametric EQ is recommended as the main method because it is fast, low-latency, and works smoothly with interactive Streamlit sliders.

IIR Butterworth SOS is kept as a strong real-time comparison method. FIR Kaiser is kept for academic filter-design explanation because it has linear phase, but it has higher latency.

## Project Structure

```text
streamlit_equalizer_final/
├── app.py
├── dsp_equalizer.py
├── equalizer_fft_filter_design_notebook.ipynb
├── requirements.txt
├── COMPARISON.md
├── README.md
├── .gitignore
└── .streamlit/config.toml
```

## Run the Streamlit App

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```bash
.venv\Scripts\activate
streamlit run app.py
```

## Run the Notebook

```bash
jupyter notebook equalizer_fft_filter_design_notebook.ipynb
```

## GitHub Upload Steps

```bash
git init
git add .
git commit -m "Add Streamlit DSP equalizer app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/streamlit-dsp-equalizer.git
git push -u origin main
```

## Recommended Project Title

**Interactive Streamlit Audio Equalizer using RBJ Parametric EQ with FFT-Based Visualization and FIR/IIR/RBJ Filter Design Comparison**
