import numpy as np
from functools import reduce
from xcorr import xcorr
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


def normalize(sin):
    return sin/abs(sin).max()


def split(sin, width, step, fs):
    if width <= 0 or step <= 0:
        raise ValueError()

    frames = []

    step_len = int(step/1000 * fs)
    width_len = int(width/1000 * fs)

    if width_len <= 0 or step_len <= 0 or width_len > len(sin):
        raise ValueError()

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
    plt.figure()
    plt.plot(lags, voiced_peaks)
    plt.show()


if __name__ == "__main__":
    Fs, sig = read("cmu_us_bdl_arctic/wav/arctic_a0001.wav")
    print(sig.shape)
    # plt.figure()
    # plt.plot(sig)
    # plt.show()
    autocorrelation(sig, 100, 100, Fs, 10)
