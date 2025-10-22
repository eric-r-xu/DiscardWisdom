import re
from .constants import SUIT_OFFSETS

def safe_counts(counts):
    if len(counts) != 34:
        raise ValueError("counts must be length 34")
    if any(c < 0 or c > 4 for c in counts):
        raise ValueError("counts invalid: outside 0..4")
    return list(counts)

def parse_mixed_labels_and_groups_34(s: str):
    s = (s or "").strip()
    if not s: return []
    labs = []
    labs += re.findall(r"[0-9][mpsz]", s)
    for suit, digits in re.findall(r"([mpsz])([0-9]+)", s):
        for ch in digits:
            labs.append(ch + suit)
    return labs

def compute_remaining_from_visible_34(hand_counts_34, visible_labels):
    rem = [0]*34
    for i in range(34): rem[i] = 4
    for i, c in enumerate(hand_counts_34): rem[i] = max(0, rem[i]-c)
    for lab in visible_labels:
        d = int(lab[0]); suit = lab[1]
        if d == 0: d = 5
        if suit == "z": idx = SUIT_OFFSETS[suit] + (d-1)
        else: idx = SUIT_OFFSETS[suit] + (d-1)
        rem[idx] = max(0, rem[idx]-1)
    return rem
