from .utils import safe_counts
from .constants import ALL_TILES_REMAINING
from .shanten import shanten_all

def _remaining(base, hand):
    return [max(0, base[i]-hand[i]) for i in range(34)]

def ukeire(counts, tiles_remaining=None):
    c = safe_counts(counts)
    rem = _remaining(ALL_TILES_REMAINING if tiles_remaining is None else tiles_remaining, c)
    cur = shanten_all(c)
    improving = {}
    for i in range(34):
        if rem[i] == 0 or c[i] >= 4: continue
        c2 = c[:]; c2[i] += 1
        if shanten_all(c2) < cur:
            improving[i] = rem[i]
    return {"shanten": cur, "improving_tiles": dict(sorted(improving.items())), "total_improving": sum(improving.values())}
