import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl
from functools import reduce
from xcorr import xcorr
from scipy.io.wavfile import read


def to_db(h, N=2):
    return 20*np.log10(np.maximum(np.abs(h)*2/N, 1e-5))


def normalize(sin):
    return sin/np.abs(sin).max()


def split(sin, width, step, fs):
    if width <= 0 or step <= 0:
        raise ValueError()

    frames = []

    step_len = int(step/1000 * fs)
    width_len = int(width/1000 * fs)

    if width_len <= 0 or step_len <= 0 or width_len > len(sin):
        raise ValueError(f'{width_len=}, {step_len=}, {width_len > len(sin)=}')

    for i in range(0, len(sin), step_len):
        f = sin[i:i+width_len]
        if len(f) != width_len:
            break
        frames.append(f)

    return np.array(frames)


def energy(sig):
    return reduce(lambda a, b: a + b, map(lambda x: abs(x)**2, sig))


def autocorrelation(sig, width, step, fs, treshold):
    sig = normalize(sig)
    frames = split(sig, width, step, fs)
    voiced = []
    unvoiced = []
    for frame in frames:
        if energy(frame) < treshold:
            unvoiced.append(frame)
        else:
            voiced.append(frame)
    voiced = np.array(voiced)
    unvoiced = np.array(unvoiced)
    print(voiced.shape, unvoiced.shape)

    lags, voiced_peaks = xcorr(voiced[9], maxlag=50)
    # TODO


def cepstrum(sig, width, step, fs, treshold):
    sig = normalize(sig)
    frames = split(sig, width, step, fs)
    voiced = []
    unvoiced = []
    for frame in frames:
        if energy(frame) < treshold:
            unvoiced.append(frame)
        else:
            voiced.append(frame)
    voiced = np.array(voiced)
    unvoiced = np.array(unvoiced)

    for frame in voiced:
        _, spectrum = sgl.freqz(frame)
        log_spectrum = to_db(spectrum)
