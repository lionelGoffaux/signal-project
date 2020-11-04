from functools import reduce


def energy(sig):
    return reduce(lambda a, b: a + b, map(lambda x: abs(x)**2, sig))
