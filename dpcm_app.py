import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from io import BytesIO
import soundfile as sf

st.set_page_config(layout="wide", page_title="DPCM App")

st.title("Differential Pulse-Code Modulation (DPCM)")

# Upload audio file
uploaded_file = st.file_uploader("Upload the Audio File (.wav)", type=["wav"])

# Sidebar inputs
st.sidebar.header("DPCM Parameters")
quant_value = st.sidebar.number_input("Enter the Quantizer Value (bits)", min_value=1, max_value=16, value=2)
prediction_order = st.sidebar.number_input("Enter the Prediction Order", min_value=1, max_value=10, value=1)

# Functions
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
    # Read audio
    audio_bytes = BytesIO(uploaded_file.read())
    y, fs = sf.read(audio_bytes)
    y = y[:, 0] if y.ndim > 1 else y  # Convert stereo to mono

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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Square Error", f"{mse:.8f}")
    col2.metric("Compression Ratio", f"{compression_ratio:.2f}")
    col3.metric("SNR (dB)", f"{snr:.2f}")
    col4.metric("MOS (1-5)", f"{mos:.2f}")

    # Plots
    st.subheader("Signal Plots")
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    axs[0, 0].plot(y)
    axs[0, 0].set_title("Original Signal")

    axs[0, 1].plot(quantized_error)
    axs[0, 1].set_title("Encoder Output (Quantized Error)")

    axs[1, 0].semilogy(np.abs(np.fft.fft(y)))
    axs[1, 0].set_title("Spectrum of the Signal")

    axs[1, 1].plot(reconstructed_signal)
    axs[1, 1].set_title("Decoder Output (Reconstructed Signal)")

    st.pyplot(fig)

    # Audio playback section with buttons
    st.subheader("Audio Playback")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Play Original Audio"):
            original_audio_buffer = BytesIO()
            sf.write(original_audio_buffer, y, fs, format='WAV')
            st.audio(original_audio_buffer.getvalue(), format='audio/wav')
    
    with col2:
        if st.button("▶️ Play Decoded Audio"):
            decoded_audio_buffer = BytesIO()
            sf.write(decoded_audio_buffer, reconstructed_signal, fs, format='WAV')
            st.audio(decoded_audio_buffer.getvalue(), format='audio/wav')
