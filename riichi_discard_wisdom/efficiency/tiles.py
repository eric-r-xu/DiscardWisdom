import re
from .constants import SUIT_OFFSETS, TILE_LABELS_34

def parse_tile(label):
    m = re.fullmatch(r"([0-9])([mpszMPSZ])", label)
    if not m: raise ValueError(f"Bad tile label: {label}")
    d = int(m.group(1)); s = m.group(2).lower()
    if d == 0: d = 5
    if s == "z":
        if not (1 <= d <= 7): raise ValueError("honors 1..7")
        return SUIT_OFFSETS[s] + (d-1)
    if not (1 <= d <= 9): raise ValueError("numbers 1..9")
    return SUIT_OFFSETS[s] + (d-1)

def parse_hand(labels):
    c = [0]*34
    for lab in labels:
        idx = parse_tile(lab)
        c[idx] += 1
        if c[idx] > 4: raise ValueError("Too many copies")
    return c

def parse_tenhou_groups(s):
    c = [0]*34
    for m in re.finditer(r"([mpsz])([0-9]+)", s or ""):
        suit = m.group(1).lower()
        for ch in m.group(2):
            d = int(ch)
            if d == 0: d = 5
            idx = parse_tile(f"{d}{suit}")
            c[idx] += 1
            if c[idx] > 4: raise ValueError("Too many copies")
    return c

def to_string(counts):
    parts = []
    for suit, off, n in (("m",0,9),("p",9,9),("s",18,9),("z",27,7)):
        digits = []
        for i in range(n):
            idx = off + i
            digits.extend(str(i+1) for _ in range(counts[idx]))
        if digits:
            parts.append(suit + "".join(digits))
    return "".join(parts)

TILE_LABELS_34 = TILE_LABELS_34
