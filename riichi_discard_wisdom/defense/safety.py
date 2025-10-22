from typing import List, Dict, Set
from .indexing import label_from_10idx

EXPLANATIONS: Dict[int, str] = {
    15: "Genbutsu (already discarded by the riichi player) or after-riichi discard.",
    14: "Terminal/Honor confirmed safe (0 remaining) or terminal suji with 0 remaining.",
    13: "Honor with 1 remaining.",
    12: "Reserved.",
    11: "Reserved.",
    10: "Honor with 2 remaining.",
     9: "Suji middle (4/5/6).",
     8: "Suji edge-leaning (2/8).",
     7: "Suji gutshot-leaning (3/7).",
     6: "Honor with 3 remaining.",
     5: "Terminal not confirmed by suji.",
     4: "Reserved.",
     3: "Non-suji (2/8).",
     2: "Non-suji (3/7).",
     1: "Non-suji (4/5/6).",
     0: "Unknown / not in hand.",
}
def _compute_no_chance(remaining: List[int]) -> Set[int]:
    nc: Set[int] = set()
    for base in (0,10,20):
        if remaining[base+1]==0 and remaining[base+4]==0: nc.update({base+2, base+3})
        if remaining[base+6]==0 and remaining[base+9]==0: nc.update({base+7, base+8})
    return nc
def check_is_suji(tile: int, opponent_discards: List[int], remaining_tiles: List[int], riichi_tile: int) -> bool:
    sujiA = tile - 3; sujiB = tile + 3
    def same(a,b): return a//10 == b//10
    if sujiA % 10 == 0 or not same(sujiA, tile): okA = True
    else:
        if sujiA == riichi_tile: return False
        okA = (sujiA in opponent_discards) or (remaining_tiles[sujiA+1]==0) or (remaining_tiles[sujiA+2]==0)
    if sujiB % 10 == 0 or not same(sujiB, tile): okB = True
    else:
        if sujiB == riichi_tile: return False
        okB = (sujiB in opponent_discards) or (remaining_tiles[sujiB-1]==0) or (remaining_tiles[sujiB-2]==0)
    return okA and okB
def evaluate_discard_safety(hand_38: List[int], opponent_discards: List[int], remaining_tiles_38: List[int],
                            tiles_discarded_after_riichi: List[int], riichi_tile: int) -> List[int]:
    safety = [0]*38; no_chance = _compute_no_chance(remaining_tiles_38)
    for i, cnt in enumerate(hand_38):
        if cnt <= 0: continue
        if i in opponent_discards or i in tiles_discarded_after_riichi: safety[i] = 15; continue
        if i < 30 and (i % 10 in (1,9)):
            safety[i] = (14 - remaining_tiles_38[i]) if check_is_suji(i, opponent_discards, remaining_tiles_38, riichi_tile) else 5; continue
        if i > 30:
            rem = remaining_tiles_38[i]; safety[i] = 14 if rem==0 else 13 if rem==1 else 10 if rem==2 else 6; continue
        num = i % 10; sp = check_is_suji(i, opponent_discards, remaining_tiles_38, riichi_tile)
        safety[i] = (9 if num in (4,5,6) else 8 if num in (2,8) else 7) if sp else (1 if num in (4,5,6) else 3 if num in (2,8) else 2)
        if 2 <= num <= 8 and i in no_chance and safety[i] < 14: safety[i] = min(14, safety[i]+1)
    return safety
def _same_suit(a:int,b:int)->bool: return a//10 == b//10
def _tile_analysis(tile_idx:int, hand_38:List[int], opponent_discards:List[int], remaining_tiles_38:List[int],
                   tiles_discarded_after_riichi:List[int], riichi_tile:int, no_chance:set) -> dict:
    detail = {"index": tile_idx,"is_in_hand": hand_38[tile_idx] > 0,"is_terminal": tile_idx < 30 and (tile_idx % 10 in (1,9)),
              "is_honor": tile_idx > 30,"remaining": remaining_tiles_38[tile_idx],"genbutsu": tile_idx in opponent_discards,
              "after_riichi": tile_idx in tiles_discarded_after_riichi,"riichi_tile": riichi_tile,"suji_passed": False,
              "base_rank": 0,"kabe_applied": False,"final_rank": 0}
    if not detail["is_in_hand"]: return detail
    if detail["genbutsu"] or detail["after_riichi"]: detail["base_rank"]=detail["final_rank"]=15; detail["suji_passed"]=True; return detail
    if detail["is_terminal"]:
        sp = check_is_suji(tile_idx, opponent_discards, remaining_tiles_38, riichi_tile); detail["suji_passed"]=sp
        detail["base_rank"] = (14 - remaining_tiles_38[tile_idx]) if sp else 5; detail["final_rank"] = detail["base_rank"]; return detail
    if detail["is_honor"]:
        rem = remaining_tiles_38[tile_idx]; detail["base_rank"] = 14 if rem==0 else 13 if rem==1 else 10 if rem==2 else 6
        detail["final_rank"] = detail["base_rank"]; return detail
    num = tile_idx % 10; sp = check_is_suji(tile_idx, opponent_discards, remaining_tiles_38, riichi_tile); detail["suji_passed"]=sp
    br = (9 if num in (4,5,6) else 8 if num in (2,8) else 7) if sp else (1 if num in (4,5,6) else 3 if num in (2,8) else 2)
    if 2 <= num <= 8 and tile_idx in no_chance and br < 14: br += 1; detail["kabe_applied"]=True
    detail["base_rank"]=br; detail["final_rank"]=br; return detail
def evaluate_discard_safety_detailed(hand_38: List[int], opponent_discards: List[int], remaining_tiles_38: List[int],
                                     tiles_discarded_after_riichi: List[int], riichi_tile: int):
    nc = _compute_no_chance(remaining_tiles_38); ranks = [0]*38; details = {}
    for i in range(38):
        d = _tile_analysis(i, hand_38, opponent_discards, remaining_tiles_38, tiles_discarded_after_riichi, riichi_tile, nc)
        details[i]=d; ranks[i]= d["final_rank"] if d["is_in_hand"] else 0
    return ranks, details
def explain_rank(tile_idx: int, rank: int, context: dict) -> str:
    base = EXPLANATIONS.get(rank, "Uncategorized."); label = label_from_10idx(tile_idx)
    rem = context.get('remaining'); rem_val = '?'
    if isinstance(rem, dict): rem_val = rem.get(tile_idx, '?')
    elif isinstance(rem, list) and 0 <= tile_idx < len(rem): rem_val = rem[tile_idx]
    if tile_idx < 30 and (tile_idx % 10 in (1,9)): base += f" Terminal {label} with remaining={rem_val}."
    if tile_idx > 30: base += f" Honor {label} with remaining={rem_val}."
    if rank == 15:
        opp = context.get("opponents", {}); reasons=[]
        for k, info in opp.items():
            if tile_idx in info.get("after_riichi", []): reasons.append(f"after-riichi of opponent {k}")
            if tile_idx in info.get("discards", []): reasons.append(f"genbutsu vs opponent {k}")
        base += " (" + "; ".join(reasons) + ")" if reasons else " (Genbutsu / after-riichi)"
    nc = context.get("no_chance", set())
    if isinstance(nc, set) and tile_idx in nc and 2 <= (tile_idx % 10) <= 8: base += " No-chance (kabe) boosts safety."
    return base
