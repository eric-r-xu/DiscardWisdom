import argparse, json, sys, re
from .efficiency.tiles import parse_tenhou_groups, parse_hand, to_string, TILE_LABELS_34
from .efficiency.ukeire import ukeire
from .efficiency.shanten import shanten_all
from .efficiency.utils import compute_remaining_from_visible_34, parse_mixed_labels_and_groups_34

# ---------------- Emoji + formatting helpers ----------------

# Unicode mahjong emoji mapping (suit-first keys)
EMOJI = {
    # honors
    "z1": "🀀", "z2": "🀁", "z3": "🀂", "z4": "🀃", "z5": "🀆", "z6": "🀅", "z7": "🀄",
    # manzu
    "m1": "🀇", "m2": "🀈", "m3": "🀉", "m4": "🀊", "m5": "🀋", "m6": "🀌", "m7": "🀍", "m8": "🀎", "m9": "🀏", "m0": "🀋",
    # souzu
    "s1": "🀐", "s2": "🀑", "s3": "🀒", "s4": "🀓", "s5": "🀔", "s6": "🀕", "s7": "🀖", "s8": "🀗", "s9": "🀘", "s0": "🀔",
    # pinzu
    "p1": "🀙", "p2": "🀚", "p3": "🀛", "p4": "🀜", "p5": "🀝", "p6": "🀞", "p7": "🀟", "p8": "🀠", "p9": "🀡", "p0": "🀝",
}

def reverse_tenhou_string(s: str) -> str:
    """
    Turn Tenhou groups like '123m456p77z' into 'm123p456z77' (i.e., 7z -> z7).
    """
    def repl(m):
        digits, suit = m.group(1), m.group(2)
        return f"{suit}{digits}"
    return re.sub(r'([0-9]+)([mpsz])', repl, s)

def tenhou_to_emoji_string(s: str) -> str:
    """
    Convert Tenhou groups (digits before suit) into a concatenated emoji string.
    Red fives (0) map to the same emoji as 5.
    """
    out = []
    for m in re.finditer(r'([0-9]+)([mpsz])', s):
        digits, suit = m.group(1), m.group(2)
        for d in digits:
            key = f"{suit}{d}"
            if d == '0':  # red five
                key = f"{suit}0"
            out.append(EMOJI.get(key, ""))
    return "".join(out)

def label_to_reversed(label: str) -> str:
    """
    Convert a single label like '1m' or '7z' (or red '0m' / '5mr') to suit-first 'm1'/'z7'.
    If label has 'r' suffix (e.g., '5mr'), treat as red five -> 'm0'.
    """
    label = label.strip()
    if not label:
        return label
    # Handle red five forms like '5mr', '5pr', '5sr'
    if len(label) >= 3 and label[-1] == 'r' and label[-2] in 'mps':
        suit = label[-2]
        return f"{suit}0"
    # Normal digit + suit like '1m', '7z'
    if len(label) >= 2 and label[0].isdigit() and label[-1] in 'mpsz':
        return f"{label[-1]}{label[0]}"
    # Already suit-first like 'm1'
    if len(label) >= 2 and label[0] in 'mpsz' and label[1].isdigit():
        return label[:2]
    return label

def label_to_emoji(label: str) -> str:
    """
    Map a single label like '1m'/'7z'/'5mr' to the emoji.
    """
    r = label_to_reversed(label)  # now suit-first like 'm1'/'z7' or 'm0'
    key = r[:2]
    return EMOJI.get(key, "")

# ---------------- Core logic ----------------

def best_discard_table(hand, remaining):
    items = []
    best = None
    for i in range(34):
        if hand[i] <= 0: 
            continue
        c2 = hand[:]; c2[i] -= 1
        s = shanten_all(c2)
        u = ukeire(c2, remaining)["total_improving"]
        item = {"index": i, "label": TILE_LABELS_34[i], "shanten": s, "ukeire": u}
        items.append(item)
        if (best is None or
            s < best["shanten"] or
            (s == best["shanten"] and u > best["ukeire"]) or
            (s == best["shanten"] and u == best["ukeire"] and item["label"] < best["label"])):
            best = item
    items.sort(key=lambda x: (x["shanten"], -x["ukeire"], x["label"]))
    return best, items

def main():
    ap = argparse.ArgumentParser(description="Riichi Discard Wisdom: Efficiency (best discard)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--groups", "-g", type=str, help="Tenhou groups: e.g. m123456789p11s11 (14 tiles)")
    g.add_argument("--labels", "-l", nargs="+", help="Tile labels: e.g. 1m 2m 3m ... (14 tiles)")
    ap.add_argument("--visible", type=str, default="", help="Other visible tiles affecting remaining (labels or grouped).")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    ap.add_argument("--verbose", action="store_true", help="Show ranked options with reasoning")
    args = ap.parse_args()

    hand = parse_tenhou_groups(args.groups) if args.groups else parse_hand(args.labels)
    vis_labels = parse_mixed_labels_and_groups_34(args.visible)
    remaining = compute_remaining_from_visible_34(hand, vis_labels)

    best, table = best_discard_table(hand, remaining)

    # Build standard Tenhou string, then reverse & emoji-ize for display.
    hand_std = to_string(hand)                   # e.g., "123m456p77z"
    hand_rev = reverse_tenhou_string(hand_std)   # e.g., "m123p456z77"
    hand_emo = tenhou_to_emoji_string(hand_std)  # emoji sequence for hand

    # Enrich best & table with reversed labels + emoji
    best_out = dict(best)
    best_out["label_reversed"] = label_to_reversed(best["label"])
    best_out["emoji"] = label_to_emoji(best["label"])

    table_out = []
    if args.verbose:
        for it in table:
            row = dict(it)
            row["label_reversed"] = label_to_reversed(it["label"])
            row["emoji"] = label_to_emoji(it["label"])
            table_out.append(row)

    if args.json:
        out = {
            "hand": hand_rev,                       # reversed form (suit first)
            "hand_original": hand_std,              # original Tenhou (for compatibility)
            "hand_emoji": hand_emo,                 # emoji sequence
            "visible": reverse_tenhou_string(args.visible) if args.visible else "",
            "best": best_out,
            "ranked": table_out if args.verbose else None
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # Human-readable text output
    print(f"Hand: {hand_rev}   {hand_emo}")
    print(f"Best discard: {best_out['label_reversed']} {best_out['emoji']}  (shanten={best_out['shanten']}, ukeire={best_out['ukeire']})")
    if args.verbose:
        print("\nRanked options:")
        for it in table_out:
            print(f"  {it['label_reversed']:>3} {it['emoji']:2}  shanten={it['shanten']}  ukeire={it['ukeire']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
