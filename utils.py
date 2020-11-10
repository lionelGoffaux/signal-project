import random
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl
from xcorr import xcorr
from scipy.io.wavfile import read


def to_db(h, N=2):
    return 20*np.log10(np.maximum(np.abs(h)*2/N, 1e-5))


def get_timeAxis(fs, sin):
    n = np.arange(len(sin))
    return n/fs


def pick_random_files(n=5, random_state=None):
    man_path = 'cmu_us_bdl_arctic/wav/'
    woman_path = 'cmu_us_slt_arctic/wav/'

    man_files = os.listdir(man_path)
    woman_files = os.listdir(woman_path)

    if random_state is not None:
        random.seed(random_state)

    man_result = map(lambda file: os.path.join(man_path, file),
                     random.sample(man_files, k=n))
    woman_result = map(lambda file: os.path.join(woman_path, file),
                       random.sample(woman_files, k=n))

    return list(man_result), list(woman_result)


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


def frame_energy(frame):
    return (abs(frame)**2).sum()


'''def __autocorrelation(sig, width, step, fs, threshold):
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

    return np.array(result)'''


def get_distance(lags, corr):
    result = [-1, -1]
    start = 1

    for n in range(2):
        for i in range(start, len(corr)-1):
            if corr[i-1] <= corr[i] and corr[i+1] <= corr[i]:
                result[n] = lags[i]
                start = i+1
                break

    if -1 in result:
        return -1

    return result[1] - result[0]


'''def __cepstrum(sig, width, step, fs, treshold):
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
        # TODO'''


def autocorrelation(frame, fs, threshold):
    if frame_energy(frame) < threshold:
        return 0
    lags, corr = xcorr(frame, maxlag=fs//50)
    distance = get_distance(lags, corr)
    return fs/distance


def cepstrum(frame, fs, threshold):
    if frame_energy(frame) < threshold:
        return 0

    start = fs//500
    hamming = sgl.windows.hamming(len(frame))
    frame *= hamming
    logSpectrum = np.log(abs(sgl.freqz(frame)[1]))
    ceps = abs(sgl.freqz(logSpectrum)[1])
    return fs/np.arange(len(ceps))[ceps == ceps[start:].max()][0]


def get_pitch(signal, width, step, fs, threshold, methode=autocorrelation):
    step_len = int(fs*step/1000)
    frames = split(normalize(signal), width, step, fs)
    pitch = []

    for f in frames:
        pitch += [methode(f, fs, threshold)] * step_len

    return pitch


def plot_energy(signal, width, step, fs, threshold=None):
    step_len = int(fs*step/1000)
    t = get_timeAxis(fs, signal)
    frames = split(normalize(signal), width, step, fs)
    energies = []

    for f in frames:
        energies += [frame_energy(f)]*step_len

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    ax[0].plot(t, signal)
    ax[0].set_title('Signal')
    ax[0].set_ylabel('Amplitude (V)')
    ax[0].set_xlabel('Time (s)')
    ax[0].grid()
    ax[0].margins(x=0)

    ax[1].plot(t[:len(energies)], energies, label='energy')
    ax[1].axhline(threshold, c='r', label='threshold')
    ax[1].set_title('Energy')
    ax[1].set_ylabel('Energy')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend()
    ax[1].grid()
    ax[1].margins(x=0)

    plt.show()


def plot_pitch(signal, width, step, fs, threshold, methode=autocorrelation):
    t = get_timeAxis(fs, signal)
    pitch = get_pitch(signal, width, step, fs, threshold, methode)

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    ax[0].plot(t, signal)
    ax[0].set_title('Signal')
    ax[0].set_ylabel('Amplitude (V)')
    ax[0].set_xlabel('Time (s)')
    ax[0].grid()
    ax[0].margins(x=0)

    ax[1].plot(t[:len(pitch)], pitch)
    ax[1].set_title('Pitch')
    ax[1].set_ylabel('Pitch (Hz)')
    ax[1].set_xlabel('Time (s)')
    ax[1].grid()
    ax[1].margins(x=0)
    
    plt.show()
