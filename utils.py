import numpy as np


def normalize(sin):
    return sin/abs(sin).max()


def split(sin, width, step, fs):
    frames = []

    step_len = int(step/1000 * fs)
    width_len = int(width/1000 * fs)

    for i in range(0, len(sin), step_len):
        f = sin[i:i+width_len]
        if len(f) != width_len:
            break
        frames.append(f)

    return np.array(frames)
