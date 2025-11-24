import re
def empty_counts_38(): return [0]*38
def _idx_from_suit_num(num, suit):
    if suit == "m": return num
    if suit == "p": return 10 + num
    if suit == "s": return 20 + num
    if suit == "z": return 30 + num
    raise ValueError
def label_from_10idx(i: int) -> str:
    if 1 <= i <= 9: return f"{i}m"
    if 11 <= i <= 19: return f"{i-10}p"
    if 21 <= i <= 29: return f"{i-20}s"
    if 31 <= i <= 37: return f"{i-30}z"
    return f"?{i}"
def parse_tenhou_groups_10idx(s: str):
    counts = empty_counts_38()
    for m in re.finditer(r"([mpsz])([0-9]+)", s or ""):
        suit = m.group(1).lower()
        for ch in m.group(2):
            d = int(ch); d = 5 if d == 0 else d
            idx = _idx_from_suit_num(d, suit); counts[idx] += 1
    return counts
def labels_to_10idx(labels):
    counts = empty_counts_38()
    for lab in labels or []:
        d = int(lab[0]); s = lab[1].lower(); d = 5 if d == 0 else d
        idx = _idx_from_suit_num(d, s); counts[idx] += 1
    return counts
def parse_mixed_labels_and_groups(s: str):
    s = (s or '').strip()
    if not s: return []
    labs = re.findall(r"[0-9][mpsz]", s)
    for suit, digits in re.findall(r"([mpsz])([0-9]+)", s):
        for ch in digits: labs.append(ch + suit)
    counts = labels_to_10idx(labs); out = []
    for i,c in enumerate(counts): out.extend([i]*c)
    return out
def compute_remaining_from_visible(hand_counts_38, visible_indices):
    rem = [0]*38
    for i in range(1,10): rem[i]=4
    for i in range(11,20): rem[i]=4
    for i in range(21,30): rem[i]=4
    for i in range(31,38): rem[i]=4
    for i,c in enumerate(hand_counts_38): rem[i] = max(0, rem[i]-c)
    for idx in visible_indices: rem[idx] = max(0, rem[idx]-1)
    return rem
