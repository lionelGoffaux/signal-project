import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft
import scipy.signal as sgl
import seaborn as sns
from scipy.io.wavfile import read
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from filterbanks import filter_banks
from librosa import lpc
from xcorr import xcorr


def get_timeAxis(fs, signal):
    """
    get_timeAxis(fs, signal)
        Compute the time axis which correspond to a signal.

        Parameters
        ----------
        fs : float
            The sampling frequency.
        signal : ndarray


        Returns
        -------
        time : ndarray
            The time axis.
    """
    n = np.arange(len(signal))
    return n/fs


def pick_random_files(n=5, random_state=None):
    """
    pick_random_files(n=5, random_state=None)
        Get n signal(s) randomly for each of the two speakers.

        Parameters
        ----------
        n : integer, optional
            The number of signal for each speaker.
        random_state :  integer, optional


        Returns
        -------
        time : dict
            A dic in which each speaker has a list of signal.
    """
    man_path = 'cmu_us_bdl_arctic/wav/'
    woman_path = 'cmu_us_slt_arctic/wav/'

    man_files = os.listdir(man_path)
    woman_files = os.listdir(woman_path)
    result = {}

    if random_state is not None:
        random.seed(random_state)

    result['bdl'] = []
    for f in map(lambda file: os.path.join(man_path, file),
                 random.sample(man_files, k=n)):
        fs, signal = read(f)
        result['bdl'].append((fs, signal))

    result['slt'] = []

    for f in map(lambda file: os.path.join(woman_path, file),
                 random.sample(woman_files, k=n)):
        fs, signal = read(f)
        result['slt'].append((fs, signal))

    return result


def normalize(signal):
    """
   normalize(signal)
       Normalize a signal.

       Parameters
       ----------
        signal : ndarray

       Returns
       -------
       normalize_signal: ndarray
    """
    return signal / np.abs(signal).max()


def split(signal, width, step, fs):
    """
    split(signal, width, step, fs)
        Split a signal into frames.

        Parameters
        ----------
        signal : ndarray
        width : float
            The width of frame in ms.
        step : float
            The step between two frames in ms.
        fs : float
            The sampling frequency.

        Returns
        -------
        frames: ndarray
    """
    if width <= 0 or step <= 0:
        raise ValueError()

    frames = []

    step_len = int(step/1000 * fs)
    width_len = int(width/1000 * fs)

    if width_len <= 0 or step_len <= 0 or width_len > len(signal):
        raise ValueError(f'{width_len=}, {step_len=}, {width_len > len(signal)=}')

    for i in range(0, len(signal), step_len):
        f = signal[i:i + width_len]
        if len(f) != width_len:
            break
        frames.append(f)

    return np.array(frames)


def frame_energy(frame):
    """
    frame_energy(frame)
        Compute the energy of a frame.

        Parameters
        ----------
        frame : ndarray

        Returns
        -------
        energy: float
    """
    return (abs(frame)**2).sum()


def get_distance(lags, corr):
    """
    get_distance(lags, corr)
        Get the distance between to peaks in the autocorrelation of a
        signal.

        Parameters
        ----------
        lags : ndarray
        corr : ndarray

        Returns
        -------
        distance: int
            In number of samples.
            -1 if not found.
    """
    result = [-1, -1]
    start = 1

    for n in range(2):
        for i in range(start, len(corr)-1):
            if corr[i-1] < corr[i] and corr[i+1] < corr[i]:
                result[n] = lags[i]

                start = i+1
                break

    if -1 in result:
        return -1

    return result[1] - result[0]


def autocorrelation(frame, fs, threshold):
    """
        autocorrelation(frame, fs, threshold)
            Compute the pitch of frame using the autocorrelation method.

            Parameters
            ----------
            frame : ndarray
            fs : float
                The sampling
            threshold : float
                The threshold bellow which the frame is unvoiced.

            Returns
            -------
            pitch : float
    """
    if frame_energy(frame) < threshold:
        return 0
    lags, corr = xcorr(frame, maxlag=fs//50)
    distance = get_distance(lags, corr)
    return fs/distance


def cepstrum(frame, fs, threshold):
    """
        cepstrum(frame, fs, threshold)
            Compute the pitch of frame using the cepstrum method.

            Parameters
            ----------
            frame : ndarray
            fs : float
                The sampling
            threshold : float
                The threshold bellow which the frame is unvoiced.

            Returns
            -------
            pitch : float
    """
    if frame_energy(frame) < threshold:
        return 0

    start = fs//500
    hamming = sgl.windows.hamming(len(frame))
    frame *= hamming
    logSpectrum = np.log(abs(sgl.freqz(frame)[1]))
    ceps = abs(sgl.freqz(logSpectrum)[1])
    return fs/np.arange(len(ceps))[ceps == ceps[start:].max()][0]


def get_pitch(signal, width, step, fs, threshold, method=autocorrelation, extend=True):
    """
        get_pitch(signal, width, step, fs, threshold, method=autocorrelation, extend=True)
            Compute the pitch for all frames of a signal.

            Parameters
            ----------
            signal : ndarray
            width : float
            The width of frame in ms.
            step : float
                The step between two frames in ms.
            fs : float
                The sampling frequency.
            threshold : float
                The threshold bellow which the frame is unvoiced.
            method : function
                The method used to get the pitch of each frame.
            extend : boolean
                If False each frame have only one pitch, and ignore the unvoiced ones.

            Returns
            -------
            pitchs : ndarray
    """
    step_len = int(fs*step/1000)
    frames = split(normalize(signal), width, step, fs)
    pitch = []

    for f in frames:
        p = method(f, fs, threshold)
        pitch += [p] * (step_len if extend else 1) if p != 0 or extend else []

    return np.array(pitch)


def plot_energy(signal, width, step, fs, threshold=None):
    """
        plot_energy(signal, width, step, fs, threshold=None)
            Compute the energy for all frames of a signal.

            Parameters
            ----------
            signal : ndarray
            width : float
            The width of frame in ms.
            step : float
                The step between two frames in ms.
            fs : float
                The sampling frequency.
            threshold : float
                The threshold bellow which the frame is unvoiced.
    """
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


def plot_pitch(signal, width, step, fs, threshold, method=autocorrelation):
    """
        plot_pitch(signal, width, step, fs, threshold, method=autocorrelation, extend=True)
            Plot the pitch for all frames of a signal.

            Parameters
            ----------
            signal : ndarray
            width : float
                The width of frame in ms.
            step : float
                The step between two frames in ms.
            fs : float
                The sampling frequency.
            threshold : float
                The threshold bellow which the frame is unvoiced.
            method : function
                The method used to get the pitch of each frame.
    """
    t = get_timeAxis(fs, signal)
    pitch = get_pitch(signal, width, step, fs, threshold, method)

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


def plot_formant(signal, width, step, fs, nb=4):
    """
    plot_formant(signal, width, step, fs, nb=4)
        Plot the formants for all frames of a signal.

        Parameters
        ----------
        signal : ndarray
        width : float
            The width of frame in ms.
        step : float
            The step between two frames in ms.
        fs : float
            The sampling frequency.
    """
    formant = formants(signal, width, step, fs, nb)
    axis = get_timeAxis(1/(step*1e-3), formant[:, 0])
    plt.figure(figsize=(12, 7))
    plt.title('Formants')
    for i in range(formant.shape[1]):
        plt.plot(axis, formant[:, i], label=f'f{i+1}')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid()
    plt.margins(x=0)
    plt.show()


def formants(signal, width, step, fs, nb=4):
    """
    formants(signal, width, step, fs, nb=4)
        Compute the formants for all frames of a signal.

        Parameters
        ----------
        signal : ndarray
        width : float
            The width of frame in ms.
        step : float
            The step between two frames in ms.
        fs : float
            The sampling frequency.
        nb : integer
            The number of formants considers for each frame.

        Returns
        -------
        formants : ndarray
            An 2 dim array with the formants of each frame.
    """
    frames = split(signal, width, step, fs)
    b, a = [1, -0.67], [1]
    roots = []
    for frame in frames:
        filtered_frame = sgl.lfilter(b, a, frame)
        hamming_win = sgl.windows.hamming(filtered_frame.size)
        filtered_frame *= hamming_win  # apply hamming window on the frame
        lpcs = lpc(filtered_frame, 9)
        root = np.roots(lpcs)
        frame_res = root[root.imag > 0][:nb]
        if len(frame_res < nb):
            frame_res = np.concatenate((frame_res, [0]*(nb-len(frame_res))))
            frame_res = np.sort(frame_res)
        roots.append(frame_res)

    angles = np.angle(roots)
    freq = angles*(fs/(2*np.pi))
    return np.sort(freq, axis=1)


def mfcc(signal, width, step, fs, Ntfd=512):
    """
    mfcc(signal, width, step, fs, Ntfd=512)
        Compute the mfcc for all frames of a signal.

        Parameters
        ----------
        signal : ndarray
        width : float
            The width of frame in ms.
        step : float
            The step between two frames in ms.
        fs : float
            The sampling frequency.
        Ntfd : integer

        Returns
        -------
        formants : ndarray
            An 2 dim array with the NFCC of each frame.
    """
    b, a = [1, -0.97], [1]
    signal = sgl.lfilter(b, a, signal)
    frames = split(signal, width, step, fs)
    P = []

    for frame in frames:
        win_frame = sgl.windows.hamming(frame.size) * frame
        p = (np.abs(sgl.freqz(win_frame, worN=Ntfd)[1]) ** 2) / Ntfd
        P.append(p)

    P = np.array(P)
    filtered_P = filter_banks(P, fs, NFFT=2*Ntfd-1)
    res = scipy.fft.dct(filtered_P, type=2, axis=1, norm='ortho')
    return res[:, :13]


def build_dataset(width=21, step=10, threshold=5, formants_number=4, wav_number=15, random_sate=None):
    """
    build_dataset(width=21, step=10, threshold=5, formants_number=4, wav_number=15, random_sate=None)

        Parameters
        ----------
        width : float
            The width of frame in ms.
        step : float
            The step between two frames in ms.
        formants_number : integer
            The number of formants considers for each frame.
        wav_number : integer
            The number of signal for each speaker.
        random_sate: float

        Returns
        -------
        df : dataframe
    """
    data = pick_random_files(wav_number, random_state=random_sate)

    fs = []
    duration = []
    autocorr_pitch_mean = []
    autocorr_pitch_median = []
    cepstrum_pitch_mean = []
    cepstrum_pitch_median = []
    form = {}
    dmfcc = {}
    speaker = []

    for i in range(13):
        dmfcc[f'mfcc{i}'] = []

    for i in range(formants_number):
        form[f'f{i+1}_mean'] = []

    for spkr, files in data.items():
        for sfs, signal in files:
            sformants = formants(signal, width, step, sfs)
            auto_pitch = get_pitch(signal, width, step, sfs, threshold, method=autocorrelation, extend=False)
            cepstrum_pitch = get_pitch(signal, width, step, sfs, threshold, method=cepstrum, extend=False)
            smfcc = mfcc(signal, width, step, sfs)

            for i in range(smfcc.shape[1]):
                dmfcc[f'mfcc{i}'].append(smfcc[:, i].mean())

            for i in range(formants_number):
                form[f'f{i+1}_mean'].append(sformants[:, i].mean())

            fs.append(sfs)
            duration.append(signal.size / sfs)

            autocorr_pitch_mean.append(auto_pitch.mean())
            autocorr_pitch_median.append(np.median(auto_pitch))
            cepstrum_pitch_mean.append(cepstrum_pitch.mean())
            cepstrum_pitch_median.append(np.median(cepstrum_pitch))
            speaker.append(spkr)

    d = {'fs': fs, 'duration': duration, 'autocorr_pitch_mean': autocorr_pitch_mean,
         'autocorr_pitch_median': autocorr_pitch_median,
         'cepstrum_pitch_mean': cepstrum_pitch_mean, 'cepstrum_pitch_median': cepstrum_pitch_median,
         'speaker': speaker}
    d.update(form)
    d.update(dmfcc)

    return pd.DataFrame(data=d)


def accuracy(y, ypred):
    """
    accuracy(y, ypred)
        Parameters
        ----------
        y : ndarray
            The y targeted.
        ypred : ndarray
            The y predicted.

        Returns
        -------
        accuracy : float
    """
    return (y == ypred).sum()/len(y)


def model1(X):
    return 'slt' if X['mfcc7'] >= -12 else 'bdl'


def model2(X):
    return 'slt' if 165 <= X['cepstrum_pitch_median'] <= 200 else 'bdl'


def model3(X):
    return 'slt' if X['f3_mean'] >= 4470 else 'bdl'


def test_rule_model():
    """
    Test the rule-based model
    """
    df = build_dataset(wav_number=50, random_sate=43)

    df['prediction'] = df.apply(model1, axis=1)
    acc1 = accuracy(df['speaker'], df["prediction"])

    df['prediction'] = df.apply(model2, axis=1)
    acc2 = accuracy(df['speaker'], df["prediction"])

    df['prediction'] = df.apply(model3, axis=1)
    acc3 = accuracy(df['speaker'], df["prediction"])

    print(f'The MFCC7-based model has {acc1*100:.2f}% accuracy.')
    print(f'The cepstrum-pitch-based model has {acc2*100:.2f}% accuracy.')
    print(f'The formant-based model has {acc3*100:.2f}% accuracy.')


def visualize_data():
    """
    Plot the features to visualize.
    """
    warnings.filterwarnings('ignore')
    df = build_dataset(wav_number=15, random_sate=42)
    bdl_df = df[df['speaker'] == 'bdl']
    slt_df = df[df['speaker'] == 'slt']

    pitch_col = ['autocorr_pitch_mean', 'autocorr_pitch_median', 'cepstrum_pitch_mean', 'cepstrum_pitch_median']

    formants_col = []
    for n in range(4):
        formants_col.append(f'f{n + 1}_mean')

    mfcc_loc = []
    for n in [5, 6, 10, 11]:
        mfcc_loc.append(f'mfcc{n}')

    plt.figure(figsize=(12, 8))

    for n, col in enumerate(pitch_col):
        plt.subplot(2, 2, n + 1)
        sns.distplot(bdl_df[col], label='bdl')
        sns.distplot(slt_df[col], label='slt')
        plt.legend()

    plt.show()

    plt.figure(figsize=(12, 8))

    for n, col in enumerate(formants_col):
        plt.subplot(2, 2, n + 1)
        sns.distplot(bdl_df[col], label='bdl')
        sns.distplot(slt_df[col], label='slt')
        plt.legend()

    plt.figure(figsize=(6, 4))
    sns.distplot(bdl_df['mfcc7'], label='bdl')
    sns.distplot(slt_df['mfcc7'], label='slt')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 8))

    for n, col in enumerate(mfcc_loc):
        plt.subplot(2, 2, n + 1)
        sns.distplot(bdl_df[col], label='bdl')
        sns.distplot(slt_df[col], label='slt')
        plt.legend()

    plt.show()


def evaluation(model, X_train, y_train, X_test, y_test):
    """
    evaluation(model, X_train, y_train, X_test, y_test)
        Evaluate the model.

        Parameters
        ----------
        model:
            A sklearn model
        X_train: ndarray
        y_train: ndarray
        X_test: ndarray
        y_test: ndarray
    """
    warnings.filterwarnings('ignore')
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)

    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))

    N, train_score, val_score = learning_curve(model, X_train, y_train, scoring='accuracy',
                                               cv=4, train_sizes=np.linspace(0.1, 1, 10))

    plt.figure(figsize=(6, 4))
    plt.plot(N, train_score.mean(axis=1), label='train_score')
    plt.plot(N, val_score.mean(axis=1), label='val_score')
    plt.legend()
    plt.grid()
    plt.margins(x=0)
    plt.show()


def preprocessing(data):
    """
    preprocessing(data)
        Evaluate the model.

        Parameters
        ----------
        data : datframe

        Returns
        -------
        X : ndarray
        y : ndarray
    """
    data = data.copy()

    X = data.drop(['speaker'], axis=1)
    y = data['speaker']

    return X, y


def visualize_energy():

    data = pick_random_files()

    for speaker in data:
        for fs, signal in data[speaker]:
            plot_energy(signal, 21, 10, fs, 5)


def test_machine_learning():
    """
    Test the machine learning-based model.
    """
    df = build_dataset(wav_number=120, random_sate=42)
    df = df.drop(['fs', 'duration'], axis=1)

    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train = preprocessing(train_set)
    X_test, y_test = preprocessing(test_set)

    print('Decision tree')
    tree = DecisionTreeClassifier(random_state=42)
    evaluation(tree, X_train, y_train, X_test, y_test)

    print('Random forest')
    rforest = RandomForestClassifier(random_state=42)
    evaluation(rforest, X_train, y_train, X_test, y_test)
