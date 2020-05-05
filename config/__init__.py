

_N0 = 950


def filter(x):
    if x.y == 1:
        return True

    global _N0
    if x.y == 0 and _N0 > 0:
        _N0 = _N0 - 1
        return True

    return False
