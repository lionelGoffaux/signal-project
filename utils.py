import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl
from xcorr import xcorr
from scipy.io.wavfile import read
import librosa as rosa


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
    return (abs(sig)**2).sum()


def autocorrelation(sig, width, step, fs, threshold):
    sig = normalize(sig)
    frames = split(sig, width, step, fs)
    voiced = []
    unvoiced = []
    result = []

    for frame in frames:
        if energy(frame) < threshold:
            unvoiced.append(frame)
        else:
            voiced.append(frame)

    voiced = np.array(voiced)
    unvoiced = np.array(unvoiced)

    for f in voiced:
        lags, corr = xcorr(f, maxlag=fs//50)
        distance = get_distance(lags, corr)
        result.append(fs/distance if distance != -1 else -1)

    return np.array(result)


def get_distance(lags, corr):
    result = [-1, -1]
    start = 1

    for n in range(2):
        for i in range(start, len(corr)-1):
            if corr[i-1] < corr[i] and corr[i+1] < corr[i]:
                result[n] = i
                start = i+1
                break

    if -1 in result:
        return -1

    return result[1] - result[0]


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


def formants(sig, width, step, fs):
    frames = split(sig, width, step, fs)
    b, a = [1, -0.67], [1]
    roots = []
    for frame in frames:
        filtered_frame = sgl.lfilter(b, a, frame)
        hamming_win = sgl.windows.hamming(filtered_frame.size)
        filtered_frame *= hamming_win  # apply hamming window on the frame
        lpc = rosa.lpc(filtered_frame, int(2 + fs / 1000))
        root = np.roots(lpc)

        for r in root:
            if np.imag(r) >= 0:
                roots.append(r)

    angles = np.angle(roots)
    freq = (angles * (fs / (2 * np.pi)))
    return freq


if __name__ == "__main__":
    x = np.arange(20000)
    sig = np.sin(440 / 10000 * 2 * np.pi * x)
    # plt.figure()
    # plt.plot(x, sig)
    # plt.show()
    formants(sig, 25, 5, 10000)
