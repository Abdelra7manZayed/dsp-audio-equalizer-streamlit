# FIR vs IIR vs RBJ Comparison

| Method | Phase | Latency | Speed | Best Use | Recommendation |
|---|---|---:|---|---|---|
| FIR Kaiser | Linear phase | High | Slowest | Academic explanation and offline processing | Keep for comparison |
| IIR Butterworth SOS | Non-linear phase | Almost zero | Fast | Real-time filtering and technical comparison | Strong second option |
| RBJ Parametric EQ | Non-linear phase | Almost zero | Fastest | Interactive equalizer sliders | Main/default method |

## Final Recommendation

Use **RBJ Parametric EQ** as the main method in the Streamlit app because it gives the best practical behavior for interactive audio equalization. It is fast, low-latency, and suitable for live slider changes.

Use **IIR Butterworth SOS** as a strong comparison because it is stable and efficient.

Use **FIR Kaiser** to demonstrate linear-phase filter design knowledge, but not as the main Streamlit method because its latency is higher.

## Important Visualizations

The project includes:

1. Waveform before and after equalization
2. FFT spectrum before and after equalization
3. Filter frequency response plot
4. Method comparison table
5. Clipping detection and auto-normalization metrics
