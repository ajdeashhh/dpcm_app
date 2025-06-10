import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import soundfile as sf

st.set_page_config(layout="wide", page_title="DPCM App", page_icon="🎧")

# Title with custom color
st.markdown(
    "<h1 style='text-align: center; color: #000000;'>Differential Pulse-Code Modulation (DPCM)</h1>",
    unsafe_allow_html=True
)

# Upload audio file
st.markdown("### 🎵 Upload Your Audio File")
uploaded_file = st.file_uploader("Upload the Audio File (.wav)", type=["wav"])

# Sidebar with inputs and explanation
st.sidebar.markdown("<h2 style='font-size: 20px; color: #4B8BBE;'>⚙️ DPCM Parameters</h2>", unsafe_allow_html=True)

st.sidebar.markdown("<div style='font-size:17px; margin: 0; padding: 0;'>🔢 <b>Quantizer Value (bits)</b></div>", unsafe_allow_html=True)
quant_value = st.sidebar.number_input("", min_value=1, max_value=16, value=2)

st.sidebar.markdown("<div style='font-size:17px; margin: 0; padding: 0;'>🔁 <b>Prediction Order</b></div>", unsafe_allow_html=True)
prediction_order = st.sidebar.number_input("", min_value=1, max_value=10, value=1)

# Explanations
with st.sidebar.expander("ℹ️ Parameter Info", expanded=False):
    st.markdown("<div style='font-size:17px; margin: 0; padding: 0;'><b>Quantizer Value (bits)</b></div>", unsafe_allow_html=True)
    st.markdown("""
    Defines how many bits are used to represent the prediction error of each sample.

    - More bits = 🔊 **Better audio quality**, 🐢 **less compression**
    - Fewer bits = 📉 **Lower quality**, ⚡ **more compression**
    - Example:
        - 1 bit = 2 levels (very rough, mostly unintelligible)
        - 2 bits = 4 levels (basic intelligibility, compact size)
        - 8 bits = 256 levels (much higher quality)
    """)
    
    st.markdown("<div style='font-size:17px; margin: 0; padding: 0;'><b>Prediction Order</b></div>", unsafe_allow_html=True)
    st.markdown("""
    The number of past samples used to estimate the current sample.

    - Higher order can improve prediction accuracy but may increase computation.
    - Example:
        - Order 1: Predicts current sample from 1 previous sample.
        - Order 2: Uses 2 prior samples for more refined prediction.
    """)

# DPCM functions
def dpcm_encode(y, order):
    len_y = len(y)
    prediction = np.zeros(len_y)
    error = np.zeros(len_y)
    for i in range(order, len_y):
        prediction[i] = y[i - order]
        error[i] = y[i] - prediction[i]
    return error, prediction

def quantize_error(error, N):
    max_val = np.max(np.abs(error))
    min_val = np.min(error)
    step_size = (max_val - min_val) / (2**N)
    quantized_error = np.round((error - min_val) / step_size)
    quantized_error = quantized_error * step_size + min_val
    return quantized_error, step_size

def dpcm_decode(quantized_error, order, prediction):
    len_y = len(quantized_error)
    reconstructed_signal = np.zeros(len_y)
    reconstructed_signal[:order] = prediction[:order]
    for i in range(order, len_y):
        reconstructed_signal[i] = reconstructed_signal[i - order] + quantized_error[i]
    return reconstructed_signal

# Main logic
if uploaded_file is not None:
    audio_bytes = BytesIO(uploaded_file.read())
    y, fs = sf.read(audio_bytes)
    y = y[:, 0] if y.ndim > 1 else y  # Convert stereo to mono if needed

    len_y = len(y)
    error, prediction = dpcm_encode(y, prediction_order)
    quantized_error, step_size = quantize_error(error, quant_value)
    reconstructed_signal = dpcm_decode(quantized_error, prediction_order, prediction)

    # Metrics
    mse = np.mean((y - reconstructed_signal)**2)
    snr = 10 * np.log10(np.sum(y**2) / np.sum((y - reconstructed_signal)**2))
    compression_ratio = (len_y * 16) / (len_y * quant_value)
    mos = 1 + 0.035 * snr + snr * (snr - 60) * (100 - snr) * 7e-6
    mos = np.clip(mos, 1, 5)

    # Metrics display
    st.markdown("### 📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Square Error", f"{mse:.8f}")
    col2.metric("Compression Ratio", f"{compression_ratio:.2f}")
    col3.metric("SNR (dB)", f"{snr:.2f}")
    col4.metric("MOS (1-5)", f"{mos:.2f}")

    with st.expander("ℹ️ Definitions of Metrics", expanded=False):
        st.markdown("""
        - **Mean Square Error (MSE)**: Measures the average squared difference between the original and reconstructed signals. Lower is better.
        - **Compression Ratio (CR)**: Ratio of original bit size to compressed bit size. Higher = more compression.
        - **Signal-to-Noise Ratio (SNR)**: Ratio of signal power to noise power in dB. Higher = better quality.
        - **MOS (Mean Opinion Score)**: Predicts perceived audio quality from 1 (bad) to 5 (excellent).
        """)

    # Plots
    st.markdown("### 📈 Signal Plots")
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.subplots_adjust(hspace=0.4)

    axs[0, 0].plot(y, color='#1f77b4')
    axs[0, 0].set_title("Original Signal")

    axs[0, 1].plot(quantized_error, color='#ff7f0e')
    axs[0, 1].set_title("Encoder Output (Quantized Error)")

    axs[1, 0].semilogy(np.abs(np.fft.fft(y)), color='#2ca02c')
    axs[1, 0].set_title("Spectrum of the Signal")

    axs[1, 1].plot(reconstructed_signal, color='#d62728')
    axs[1, 1].set_title("Decoder Output (Reconstructed Signal)")

    st.pyplot(fig)

    # Audio playback section
    st.markdown("### 🔊 Audio Playback")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Audio**")
        original_audio_buffer = BytesIO()
        sf.write(original_audio_buffer, y, fs, format='WAV')
        st.audio(original_audio_buffer.getvalue(), format='audio/wav')
    
    with col2:
        st.markdown("**Decoded Audio**")
        decoded_audio_buffer = BytesIO()
        sf.write(decoded_audio_buffer, reconstructed_signal, fs, format='WAV')
        st.audio(decoded_audio_buffer.getvalue(), format='audio/wav')
