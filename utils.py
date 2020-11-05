import numpy as np
from functools import reduce


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
