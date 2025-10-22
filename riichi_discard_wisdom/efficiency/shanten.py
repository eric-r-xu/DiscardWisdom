from functools import lru_cache
from .constants import TERMINAL_INDICES

def _validate(counts):
    if len(counts) != 34:
        raise ValueError("counts must be length 34")
    if any(c < 0 or c > 4 for c in counts):
        raise ValueError("counts invalid: outside 0..4")
    return list(counts)

def shanten_kokushi(counts):
    c = _validate(counts)
    unique = 0; pair = 0
    for i in range(34):
        if i in TERMINAL_INDICES and c[i] > 0:
            unique += 1
            if c[i] >= 2: pair = 1
    return (13 - unique) + (1 - pair) - 1

def shanten_chiitoi(counts):
    c = _validate(counts)
    pairs = 0; distinct = 0
    for i in range(34):
        if c[i] >= 1: distinct += 1
        if c[i] >= 2: pairs += 1
    return max(7 - pairs, 7 - distinct) - 1

def shanten_standard(counts):
    c = _validate(counts)

    @lru_cache(None)
    def suit_dp(t):
        a = list(t); best = (0,0)
        for i in range(9):
            if a[i] >= 3:
                b = a[:]; b[i]-=3
                m, ta = suit_dp(tuple(b))
                best = max(best, (m+1, ta))
        for i in range(7):
            if a[i] and a[i+1] and a[i+2]:
                b = a[:]; b[i]-=1; b[i+1]-=1; b[i+2]-=1
                m, ta = suit_dp(tuple(b))
                best = max(best, (m+1, ta))
        for i in range(9):
            if a[i] >= 2:
                b = a[:]; b[i]-=2
                m, ta = suit_dp(tuple(b))
                best = max(best, (m, ta+1))
        for i in range(8):
            if a[i] and a[i+1]:
                b = a[:]; b[i]-=1; b[i+1]-=1
                m, ta = suit_dp(tuple(b))
                best = max(best, (m, ta+1))
        for i in range(7):
            if a[i] and a[i+2]:
                b = a[:]; b[i]-=1; b[i+2]-=1
                m, ta = suit_dp(tuple(b))
                best = max(best, (m, ta+1))
        return best

    def honors_dp(h):
        m=t=p=0
        for x in h:
            if x >= 3: m+=1
            elif x == 2: t+=1; p+=1
        return m,t,p

    ms,ts = suit_dp(tuple(c[0:9]))
    mp,tp = suit_dp(tuple(c[9:18]))
    mz,tz = suit_dp(tuple(c[18:27]))
    mh,th,ph = honors_dp(c[27:34])
    M = ms+mp+mz+mh; T = ts+tp+tz+th

    candidates = []
    for use_pair in (0,1):
        m = M; t = T
        if use_pair == 1:
            if t > 0 or ph > 0:
                t = max(0, t-1)
            else:
                continue
        sh = 8 - 2*m - t - use_pair
        candidates.append(sh)
    return min(candidates) if candidates else 8

def shanten_all(counts):
    return min(shanten_standard(counts), shanten_chiitoi(counts), shanten_kokushi(counts))
