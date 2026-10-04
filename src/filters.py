import numpy as np
from scipy import signal

FS = 500.0
L_FREQ = 0.5
H_FREQ = 45.0
LINE_FREQ = 50.0

def design_fir_bandpass(fs=FS, l_freq=L_FREQ, h_freq=H_FREQ,
                        trans_width=0.25, ripple_db=60.0):
    nyq = fs / 2.0
    numtaps, beta = signal.kaiserord(ripple_db, trans_width / nyq)
    if numtaps % 2 == 0:
        numtaps += 1
    taps = signal.firwin(numtaps,
                         [l_freq - trans_width / 2.0, h_freq + trans_width / 2.0],
                         window=("kaiser", beta),
                         pass_zero=False, fs=fs)
    return taps, beta

def design_fir_hamming(fs=FS, l_freq=L_FREQ, h_freq=H_FREQ,
                       trans_width=0.25):
    numtaps = int(np.ceil(3.3 * fs / trans_width))
    if numtaps % 2 == 0:
        numtaps += 1
    taps = signal.firwin(numtaps,
                         [l_freq - trans_width / 2.0, h_freq + trans_width / 2.0],
                         window="hamming", pass_zero=False, fs=fs)
    return taps

def design_iir_bandpass(fs=FS, l_freq=L_FREQ, h_freq=H_FREQ,
                        gpass=0.5, gstop=20.0):
    wp = [l_freq, h_freq]
    ws = [l_freq / 2.0, 60.0]
    N, Wn = signal.buttord(wp, ws, gpass, gstop, fs=fs)
    sos = signal.butter(N, Wn, btype="bandpass", output="sos", fs=fs)
    return sos, N

def design_notch(fs=FS, f0=LINE_FREQ, Q=30.0):
    b, a = signal.iirnotch(f0, Q, fs=fs)
    return signal.tf2sos(b, a)

def sos_response(sos, fs=FS, worN=16384):
    w, h = signal.sosfreqz(sos, worN=worN, fs=fs)
    return w, h

def group_delay_samples(w_hz, h, fs=FS):
    phase = np.unwrap(np.angle(h))
    return -np.gradient(phase, w_hz) * fs / (2.0 * np.pi)

def db(h, floor=-140.0):
    mag = np.abs(h)
    out = 20.0 * np.log10(np.maximum(mag, 1e-12))
    return np.maximum(out, floor)

def apply_fir_zerophase(x, taps):
    x = np.atleast_2d(x)
    d = (len(taps) - 1) // 2
    y = signal.oaconvolve(x, taps[None, :], mode="full", axes=-1)
    return y[:, d:d + x.shape[-1]]

def apply_iir_zerophase(x, sos):
    return signal.sosfiltfilt(sos, np.atleast_2d(x), axis=-1)

def preprocess_array(x, fs=FS, use_notch=True):
    sos_bp, _ = design_iir_bandpass(fs=fs)
    y = np.atleast_2d(x)
    if use_notch:
        y = signal.sosfiltfilt(design_notch(fs=fs), y, axis=-1)
    return signal.sosfiltfilt(sos_bp, y, axis=-1)
