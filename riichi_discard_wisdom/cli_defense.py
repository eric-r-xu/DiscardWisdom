import argparse, json, sys
from .defense.indexing import parse_tenhou_groups_10idx, labels_to_10idx, parse_mixed_labels_and_groups, compute_remaining_from_visible, label_from_10idx
from .defense.safety import evaluate_discard_safety_detailed, explain_rank, _compute_no_chance

def main():
    ap = argparse.ArgumentParser(description="Riichi Discard Wisdom: Defense (safest discard)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--groups", "-g", type=str, help="Tenhou groups (10-index domain)")
    g.add_argument("--labels", "-l", nargs="+", help="Labels list")
    ap.add_argument("--oppo", action="append", default=[], help="Opponent discards (repeatable).")
    ap.add_argument("--after-riichi", action="append", default=[], help="After-riichi discards (repeatable).")
    ap.add_argument("--visible", type=str, default="", help="Other visible tiles affecting remaining.")
    ap.add_argument("--riichi", type=int, default=None, help="(Deprecated) Single riichi tile index (10-based).")
    ap.add_argument("--riichi-map", action="append", default=[], help="Map riichi as 'opponent:tile' (e.g., 1:15). Repeat up to 3.")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    ap.add_argument("--verbose", action="store_true", help="Show ranked options with reasoning")
    ap.add_argument("--explain", action="store_true", help="Include per-tile derivation fields")
    ap.add_argument("--top-only", action="store_true", help="Only output the single safest tile")
    args = ap.parse_args()

    hand = parse_tenhou_groups_10idx(args.groups) if args.groups else labels_to_10idx(args.labels)

    opp_lists = [parse_mixed_labels_and_groups(s) for s in args.oppo]
    ar_lists = [parse_mixed_labels_and_groups(s) for s in args.after_riichi]
    opp_all = [i for lst in opp_lists for i in lst]
    ar_all  = [i for lst in ar_lists for i in lst]
    other_visible = parse_mixed_labels_and_groups(args.visible)
    all_visible = opp_all + ar_all + other_visible

    remaining = compute_remaining_from_visible(hand, all_visible)
    ranks, details = evaluate_discard_safety_detailed(hand, opp_all, remaining, ar_all, args.riichi)
    no_chance = _compute_no_chance(remaining)
    # Build riichi mapping (opponent index -> tile). Supports --riichi-map "k:tile" or legacy --riichi.
    riichi_map = {}
    for entry in args.riichi_map:
        try:
            k_str, t_str = entry.split(":", 1)
            k = int(k_str)
            t = int(t_str)
            riichi_map[k] = t
        except Exception:
            raise SystemExit(f"Bad --riichi-map entry: {entry}. Use k:tile, e.g., 1:15")
    # Legacy: single --riichi provided: apply to all opponents if any; else to opponent 1
    if not riichi_map and args.riichi is not None:
        if opp_lists:
            for k in range(1, len(opp_lists)+1):
                riichi_map[k] = args.riichi
        else:
            riichi_map[1] = args.riichi

    # Compute ranks per-riichi opponent; combine by worst-threat (min rank across riichi players)
    per_riichi_ranks = {}
    combined = [0]*38
    if riichi_map:
        for k, tile in riichi_map.items():
            disc_k = opp_lists[k-1] if 1 <= k <= len(opp_lists) else []
            ar_k = ar_lists[k-1] if 1 <= k <= len(ar_lists) else []
            rk, dk = evaluate_discard_safety_detailed(hand, disc_k, remaining, ar_k, tile)
            per_riichi_ranks[str(k)] = rk
            for i in range(38):
                if hand[i] <= 0: continue
                if combined[i] == 0: combined[i] = rk[i]
                else: combined[i] = min(combined[i], rk[i])
        ranks = combined
        # details: fall back to last computed dk for structure; explanations include per-opponent info below
        details = dk
    else:
        # Fallback to previous single-threat behavior across flattened discards
        per_riichi_ranks = {}
        # ranks, details already computed above

    opp_ctx = {}
    for idx, lst in enumerate(opp_lists, start=1):
        opp_ctx[str(idx)] = {"discards": lst, "after_riichi": ar_lists[idx-1] if idx-1 < len(ar_lists) else []}

    items = []
    for i, c in enumerate(hand):
        if c <= 0: continue
        d = details[i]
        it = {"index": i, "label": label_from_10idx(i), "count_in_hand": c, "rank": ranks[i],
              "reason": explain_rank(i, ranks[i], {"remaining": remaining, "opponents": opp_ctx, "no_chance": no_chance})}
        if args.explain:
            it.update({"remaining": d["remaining"], "suji_passed": d["suji_passed"], "genbutsu": d["genbutsu"], "after_riichi": d["after_riichi"], "kabe_applied": d["kabe_applied"], "base_rank": d["base_rank"], "final_rank": d["final_rank"]})
        items.append(it)

    best_rank = max((it["rank"] for it in items), default=0)
    bests = [it for it in items if it["rank"] == best_rank]; bests.sort(key=lambda x: x["label"])
    best = bests[0] if bests else None
    items_sorted = sorted(items, key=lambda x: (-x["rank"], x["label"]))
    if args.top_only: items_sorted = [best] if best else []

    if args.json:
        out = {"hand": "".join([label_from_10idx(i) for i, c in enumerate(hand) for _ in range(c)]),
               "riichi_map": riichi_map,
               "visible_count": len(all_visible), "opponents": opp_ctx, "other_visible": [label_from_10idx(i) for i in other_visible],
               "best": best, "ranked": items_sorted if args.verbose else None,
               "per_riichi_ranks": per_riichi_ranks}
        print(json.dumps(out, ensure_ascii=False, indent=2)); return 0

    if best:
        print(f"Safest discard: {best['label']}  (rank={best['rank']})")
        print(f"Reason: {best['reason']}")
    if args.verbose and items_sorted:
        print("\nRanked safety:")
        for it in items_sorted:
            print(f"  {it['label']:>3}  rank={it['rank']:>2}  {it['reason']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
