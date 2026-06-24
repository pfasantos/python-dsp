import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.io import wavfile

def fft_pcm(filename, fs,swap=False):
    data = np.fromfile(filename, dtype=">i2")
    if swap == True:
        data = data.byteswap()
    N = len(data)
    spec = np.fft.rfft(data * np.hanning(N))
    freq = np.fft.rfftfreq(N, 1/fs)
    return spec, freq


def fft_dat(filename, fs): 
    data = np.loadtxt(filename)
    #data = data[len(data)//2:]
    pdm_signal = 2.0 * data - 1
    N = len(pdm_signal)
    spec = np.fft.rfft(pdm_signal * np.hanning(N))
    freq = np.fft.rfftfreq(N, 1/fs)
    return spec, freq

def fft(filename, fs, bitrate, pdm=True):
    #leitura do arquivo
    if bitrate == 16:
        data = np.fromfile(filename, dtype=np.int16)
    if bitrate == 32:
        data = np.fromfile(filename, dtype=np.uint32)
    data = data[len(data)//2:]
    if pdm == True:
        data = data.byteswap()
        data2 = data.view(np.uint8)
        bits = np.unpackbits(data2) #bytes para bits
        pdm_signal = 2 * bits.astype(np.float64) - 1 #transformar pra bipolar
    if pdm == False:
        pdm_signal = data
    
    # calcular fft
    N = len(pdm_signal)
    pdm_signal = pdm_signal * np.hanning(N)
    spec = np.fft.rfft(pdm_signal)
    freq = np.fft.rfftfreq(N, 1/fs)
    return spec,freq

def fft_noswap(filename, fs, bitrate, pdm=True):
    #leitura do arquivo
    if bitrate == 16:
        data = np.fromfile(filename, dtype=np.uint16)
    if bitrate == 32:
        data = np.fromfile(filename, dtype=np.uint32)
    data = data[len(data)//2:]
    if pdm == True:
        #data = data.byteswap()
        data2 = data.view(np.uint8)
        bits = np.unpackbits(data2) #bytes para bits
        pdm_signal = 2 * bits - 1 #transformar pra bipolar
    if pdm == False:
        pdm_signal = data
    
    # calcular fft
    N = len(pdm_signal)
    pdm_signal = pdm_signal * np.hanning(N)
    spec = np.fft.rfft(pdm_signal)
    freq = np.fft.rfftfreq(N, 1/fs)
    return spec,freq



def plot(spec, freq):
    # plot
    fig, ax = plt.subplots()
    ax.semilogx(freq, 20*np.log10(np.abs(spec) + 1e-12), linewidth=0.4)
    ax.set_xlabel('Frequência (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Espectro de Frequência (FFT)')
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_download(spec, freq, filename):
    output_dir = "pdmrealres"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots()
    mag_db = 20 * np.log10(np.abs(spec) + 1e-12)
    
    ax.semilogx(freq, mag_db, linewidth=0.5)
    ax.set_xlabel('Frequência (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Espectro de Frequência (FFT)')
    ax.grid(True, which="both", linestyle='--', alpha=0.25)
    #ax.set_ylim(-20,120)
    plt.tight_layout()
    
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}.png"
    filepath = os.path.join(output_dir, output_filename)
    
    plt.savefig(filepath, dpi=300)
    plt.close(fig)

def plot_compare(sp1, sp2, f1, f2):
    # plot
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(4,7))
    ax1.semilogx(f1, 20*np.log10(np.abs(sp1)+1e-12), linewidth=0.5)
    ax1.set_xlabel('Frequência (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('FFT - 1')
    ax1.grid(True, which="both", linestyle='--', alpha=0.3)
    ax2.semilogx(f2, 20*np.log10(np.abs(sp2) + 1e-12), linewidth=0.5)
    ax2.set_xlabel('Frequência (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('FFT - 2')
    ax2.grid(True, which="both", linestyle='--', alpha=0.3)
    ax1.set_xlim(1e1, 2.5e6)
    ax2.set_xlim(1e1, 2.5e6)
    plt.tight_layout()
    plt.show()



def fftplot_filtered_wav(filename, fs, bitrate, fc, order, decim, output_wav="output.wav"):
    # 1. Leitura e Conversão PDM
    dtype = np.uint16 if bitrate == 16 else np.uint32
    data = np.fromfile(filename, dtype=dtype)
    data = data.byteswap()
    data2 = data.view(np.uint8)
    bits = np.unpackbits(data2)
    pdm_signal = 2.0 * bits.astype(np.float64) - 1.0

    # 2. Filtragem
    norm_fc = fc / (fs * 0.5)
    b, a = butter(N=order, Wn=norm_fc, btype='low')
    filtered_data = filtfilt(b, a, pdm_signal)
    
    # 3. Decimação
    fs_decim = int(fs / decim)
    pcm_signal = filtered_data[::decim]
    
    # --- SALVAMENTO DO ARQUIVO DE ÁUDIO ---
    # Normalização para evitar clipping (deixa um headroom de 10%)
    #max_val = np.max(np.abs(pcm_signal))
    #if max_val > 0:
    #    audio_normalized = pcm_signal / max_val * 0.9
    #else:
    #    audio_normalized = pcm_signal

    # Conversão para Int16 (PCM 16-bit)
    #audio_int16 = (audio_normalized * 32767).astype(np.int32)
    
    # Escrita do arquivo WAV
    wavfile.write(output_wav, fs_decim, pcm_signal)
    print(f"Áudio salvo com sucesso: {output_wav}")
    # --------------------------------------
    
    # 4. Cálculo FFT (usando magnitude absoluta)
    N = len(pcm_signal)
    spec = np.abs(np.fft.rfft(pcm_signal))
    freq = np.fft.rfftfreq(N, 1/fs_decim)

    # 5. Plotagem
    fig, ax = plt.subplots()
    ax.semilogx(freq, 20 * np.log10(spec + 1e-12), linewidth=0.5)
    ax.set_xlabel('Frequência (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Espectro de Frequência - Fs Final: {fs_decim} Hz')
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
