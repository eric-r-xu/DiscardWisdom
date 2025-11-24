import argparse, json, sys
from .efficiency.tiles import parse_tenhou_groups, parse_hand, to_string, TILE_LABELS_34
from .efficiency.ukeire import ukeire
from .efficiency.shanten import shanten_all
from .efficiency.utils import compute_remaining_from_visible_34, parse_mixed_labels_and_groups_34

def best_discard_table(hand, remaining):
    items = []
    best = None
    for i in range(34):
        if hand[i] <= 0: continue
        c2 = hand[:]; c2[i] -= 1
        s = shanten_all(c2)
        u = ukeire(c2, remaining)["total_improving"]
        item = {"index": i, "label": TILE_LABELS_34[i], "shanten": s, "ukeire": u}
        items.append(item)
        if best is None or s < best["shanten"] or (s == best["shanten"] and u > best["ukeire"]) or (s == best["shanten"] and u == best["ukeire"] and item["label"] < best["label"]):
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

    if args.json:
        out = {"hand": to_string(hand), "visible": args.visible, "best": best, "ranked": table if args.verbose else None}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"Hand: {to_string(hand)}")
    print(f"Best discard: {best['label']}  (shanten={best['shanten']}, ukeire={best['ukeire']})")
    if args.verbose:
        print("\nRanked options:")
        for it in table:
            print(f"  {it['label']:>3}  shanten={it['shanten']}  ukeire={it['ukeire']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
