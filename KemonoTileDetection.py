# -*- coding: utf-8 -*-
# --------------------------------------------------------------------
# Riichi/Kemono Screen Scanner with Region-Based Counting
# (75% UI scale; Status + efficiency + defense visible; others minimized)
# --------------------------------------------------------------------

import os, hashlib, threading, time, subprocess, re
from collections import defaultdict, Counter
from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple, List
from dataclasses import dataclass, field

import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
import pyautogui

# --------------------------------------------------------------------
# -- UI SCALING / FONTS / VISIBILITY
# --------------------------------------------------------------------
UI_SCALE = 0.75  # display at 75%

def _scale_font(font_tuple: tuple[str, int]) -> tuple[str, int]:
    fam, size = font_tuple
    return (fam, max(6, int(round(size * UI_SCALE))))

def G(w: int, h: int, x: int, y: int) -> tuple[int, int, int, int]:
    s = UI_SCALE
    return (int(round(w*s)), int(round(h*s)), int(round(x*s)), int(round(y*s)))

FONT_LARGE_BASE = ("Arial", 18)
FONT_MEDIUM_BASE = ("Arial", 12)
FONT_SMALL_BASE = ("Arial", 10)

FONT_LARGE  = _scale_font(FONT_LARGE_BASE)
FONT_MEDIUM = _scale_font(FONT_MEDIUM_BASE)
FONT_SMALL  = _scale_font(FONT_SMALL_BASE)

VISIBLE_WINDOWS: set[str] = {
    "Status",
    "efficiency (CLI output)",
    "defense (CLI output)",
}
SUMMARY_TITLE = "Other Tiles Summary"

# --------------------------------------------------------------------
# -- CONSTANTS / GLOBALS
# --------------------------------------------------------------------
POLL_SECS = 0.5
TILE_CAP = 4

_ROOT: Optional[tk.Tk] = None

CUR_ALL_TILES: Dict[str, int] = defaultdict(int)
CUR_YOUR_TILES: Dict[str, int] = defaultdict(int)
CUR_YOUR_DISCARDS: Dict[str, int] = defaultdict(int)
CUR_DORA: Dict[str, int] = defaultdict(int)
CUR_OPP3_MELDS: Dict[str, int] = defaultdict(int)
CUR_OPP3_DISCARDS: Dict[str, int] = defaultdict(int)
CUR_OPP2_MELDS: Dict[str, int] = defaultdict(int)
CUR_OPP2_DISCARDS: Dict[str, int] = defaultdict(int)
CUR_OPP1_MELDS: Dict[str, int] = defaultdict(int)
CUR_OPP1_DISCARDS: Dict[str, int] = defaultdict(int)
CUR_OPP1_AFTER_RIICHI: Dict[str, int] = defaultdict(int)
CUR_OPP2_AFTER_RIICHI: Dict[str, int] = defaultdict(int)
CUR_OPP3_AFTER_RIICHI: Dict[str, int] = defaultdict(int)

TILE_TO_EMOJI: Dict[str, str] = {
    "z1": "🀀","z2":"🀁","z3":"🀂","z4":"🀃","z5":"🀆","z6":"🀅","z7":"🀄","riichi":"🇯🇵",
    "m1":"🀇","m2":"🀈","m3":"🀉","m4":"🀊","m5":"🀋","m6":"🀌","m7":"🀍","m8":"🀎","m9":"🀏","m0":"🀋",
    "s1":"🀐","s2":"🀑","s3":"🀒","s4":"🀓","s5":"🀔","s6":"🀕","s7":"🀖","s8":"🀗","s9":"🀘","s0":"🀔",
    "p1":"🀙","p2":"🀚","p3":"🀛","p4":"🀜","p5":"🀝","p6":"🀞","p7":"🀟","p8":"🀠","p9":"🀡","p0":"🀝",
}
MAHJONG_EMOJI_RANGE = r"\U0001F000-\U0001F02B"

SCREENSHOT_OUT_DIR = "/Users/ericxu/RiichiDiscardWisdom/GameScreenshots"
SAVE_REGION_SHOTS = False
SAVE_REGION_COMPOSITE = False

# --------------------------------------------------------------------
# -- REGIONS
# --------------------------------------------------------------------
@dataclass(frozen=True)
class RegionSpec:
    dx1: int; dy1: int; dx2: int; dy2: int

REGIONS: Dict[str, RegionSpec] = {
    "SELF_HAND":        RegionSpec(-57,1428, 672,1540),
    "SELF_DISCARDS":    RegionSpec(278,1196, 671,1421),

    "OPP1_MELDS":       RegionSpec(120,1078, 672,1145),
    "OPP1_DISCARDS":    RegionSpec(278, 850, 672,1081),

    "OPP2_MELDS":       RegionSpec(120, 745, 672, 815),
    "OPP2_DISCARDS":    RegionSpec(278, 524, 672, 751),

    "OPP3_MELDS":       RegionSpec(120, 410, 672, 490),
    "OPP3_DISCARDS":    RegionSpec(278, 190, 672, 421),

    "DORA":             RegionSpec( 32,1282, 267,1359),
}

REGION_TO_BUCKET = {
    "SELF_HAND":            "CUR_YOUR_TILES",
    "SELF_DISCARDS":        "CUR_YOUR_DISCARDS",
    "OPP1_MELDS":           "CUR_OPP1_MELDS",
    "OPP1_DISCARDS":        "CUR_OPP1_DISCARDS",
    "OPP2_MELDS":           "CUR_OPP2_MELDS",
    "OPP2_DISCARDS":        "CUR_OPP2_DISCARDS",
    "OPP3_MELDS":           "CUR_OPP3_MELDS",
    "OPP3_DISCARDS":        "CUR_OPP3_DISCARDS",
    "DORA":                 "CUR_DORA",
}

def _bucket_dict_by_name(name: str) -> Dict[str, int]:
    return {
        "CUR_ALL_TILES":         CUR_ALL_TILES,
        "CUR_YOUR_TILES":        CUR_YOUR_TILES,
        "CUR_YOUR_DISCARDS":     CUR_YOUR_DISCARDS,
        "CUR_DORA":              CUR_DORA,
        "CUR_OPP1_DISCARDS":     CUR_OPP1_DISCARDS,
        "CUR_OPP2_DISCARDS":     CUR_OPP2_DISCARDS,
        "CUR_OPP3_DISCARDS":     CUR_OPP3_DISCARDS,
        "CUR_OPP1_MELDS":        CUR_OPP1_MELDS,
        "CUR_OPP2_MELDS":        CUR_OPP2_MELDS,
        "CUR_OPP3_MELDS":        CUR_OPP3_MELDS,
    }[name]

@dataclass
class OpponentState:
    discards: list[tuple[str, float]] = field(default_factory=list)
    melds: list[str] = field(default_factory=list)
    riichi_time: float | None = None
    riichi_bbox_rel: tuple[int,int,int,int] | None = None

OPP: dict[int, OpponentState] = {1: OpponentState(), 2: OpponentState(), 3: OpponentState()}

REGION_TO_SEAT = {
    "OPP1_DISCARDS": 1, "OPP1_MELDS": 1,
    "OPP2_DISCARDS": 2, "OPP2_MELDS": 2,
    "OPP3_DISCARDS": 3, "OPP3_MELDS": 3,
}

# --------------------------------------------------------------------
# -- UTILS
# --------------------------------------------------------------------
def heartbeat(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        print("...ﮩ٨ـﮩﮩ٨ـ🫀ﮩ٨ـﮩﮩ٨ـ...")
        time.sleep(8)

@lru_cache(maxsize=64)
def load_template_files(template_dir: str) -> List[str]:
    out: List[str] = []
    for root, _dirs, files in os.walk(template_dir):
        for file in files:
            if file.startswith(("m","p","s","z","anchor","riichi")) and file.lower().endswith(".png"):
                out.append(os.path.join(root, file))
    return out

@lru_cache(maxsize=2048)
def imread_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def _contig_f32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)

def capture_screenshot() -> Tuple[np.ndarray, np.ndarray]:
    shot = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    return image, mask

def capture_screenshot_and_save(anchor_location: Optional[Iterable[int]] = None,
                                save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    shot = pyautogui.screenshot()
    full = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    if anchor_location is None:
        h, w = full.shape[:2]
        return full, np.zeros((h, w), dtype=bool)
    min_x, max_x, min_y, max_y = anchor_bounds(anchor_location)
    h, w = full.shape[:2]
    x1, y1, x2, y2 = clamp_rect(min_x, min_y, max_x, max_y, w, h)
    roi = full if (x2 <= x1 or y2 <= y1) else full[y1:y2, x1:x2]
    if save:
        os.makedirs(SCREENSHOT_OUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(SCREENSHOT_OUT_DIR, f"bbox_{int(time.time())}.png"), roi)
    rh, rw = roi.shape[:2]
    return roi, np.zeros((rh, rw), dtype=bool)

def return_first_match_location(template_image: np.ndarray, image: np.ndarray, threshold: float = 0.8):
    img = _contig_f32(image)
    tmpl = _contig_f32(template_image)
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    coords = np.argwhere(res >= float(threshold))
    if coords.size == 0:
        return []
    y, x = coords[0]
    h, w = template_image.shape[:2]
    return [int(x), int(x + w), int(y), int(y + h)]

def anchor_bounds(anchor_location: Iterable[int]) -> Tuple[int, int, int, int]:
    ax1, _ax2, ay1, _ay2 = anchor_location
    min_x, max_x = ax1 - 62, ax1 + 671
    min_y, max_y = ay1 + 187, ay1 + 1600
    return int(min_x), int(max_x), int(min_y), int(max_y)

def clamp_rect(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    return x1, y1, x2, y2

def roi_hash(image: np.ndarray, anchor_location: Iterable[int], save: bool = False) -> str:
    min_x, max_x, min_y, max_y = anchor_bounds(anchor_location)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_rect(min_x, min_y, max_x, max_y, w, h)
    roi = image[y1:y2, x1:x2]
    if save:
        os.makedirs(SCREENSHOT_OUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(SCREENSHOT_OUT_DIR, f"bbox_{int(time.time())}.png"), roi)
    return hashlib.md5(roi.tobytes()).hexdigest()

def find_in_rect(
    template_image: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    rect: Tuple[int, int, int, int],
    threshold: float = 0.8,
) -> Tuple[int, List[List[int]]]:
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return 0, []
    sub = image[y1:y2, x1:x2]
    tmpl = _contig_f32(template_image)
    roi = _contig_f32(sub)
    th, tw = template_image.shape[:2]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return 0, []
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= float(threshold))
    out: List[List[int]] = []
    n = 0
    H, W = image.shape[:2]
    for ry, rx in zip(ys, xs):
        y = y1 + int(ry)
        x = x1 + int(rx)
        ey, ex = y + th, x + tw
        if ey > H or ex > W:
            continue
        if np.any(mask[y:ey, x:ex]):
            continue
        mask[y:ey, x:ex] = True
        out.append([x, ex, y, ey])
        n += 1
    return n, out

def region_rect_from_anchor(anchor_location: Iterable[int], spec: RegionSpec,
                            img_w: int, img_h: int) -> Tuple[int,int,int,int]:
    ax1, _ax2, ay1, _ay2 = anchor_location
    x1 = ax1 + spec.dx1
    y1 = ay1 + spec.dy1
    x2 = ax1 + spec.dx2
    y2 = ay1 + spec.dy2
    return clamp_rect(x1, y1, x2, y2, img_w, img_h)

def detect_riichi_icons_in_roi(image: np.ndarray,
                               anchor_location: Iterable[int],
                               riichi_templates: dict[str, np.ndarray]) -> list[tuple[int,int,int,int]]:
    ax_min_x, ax_max_x, ax_min_y, ax_max_y = anchor_bounds(anchor_location)
    H, W = image.shape[:2]
    rx1, ry1, rx2, ry2 = clamp_rect(ax_min_x, ax_min_y, ax_max_x, ax_max_y, W, H)
    roi = image[ry1:ry2, rx1:rx2]
    mask = np.zeros_like(roi[...,0], dtype=bool)
    hits: list[tuple[int,int,int,int]] = []
    for _, tmpl in riichi_templates.items():
        th, tw = tmpl.shape[:2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            continue
        tmpl_f = _contig_f32(tmpl); roi_f = _contig_f32(roi)
        res = cv2.matchTemplate(roi_f, tmpl_f, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= 0.8)
        for ry, rx in zip(ys, xs):
            y1, x1 = int(ry), int(rx)
            y2, x2 = y1 + th, x1 + tw
            if np.any(mask[y1:y2, x1:x2]):
                continue
            mask[y1:y2, x1:x2] = True
            hits.append((rx1 + x1, ry1 + y1, rx1 + x2, ry1 + y2))
    return hits

# --------------------------------------------------------------------
# -- TEXT / EMOJI HELPERS
# --------------------------------------------------------------------
def _normalize_id(tid: str) -> str:
    tid = tid.strip().lower().replace("_", "")
    if len(tid) >= 2 and tid[1] == "5" and tid.endswith("r"):
        tid = tid[0] + "5"
    return tid

def _tile_label(tile_id: str) -> str:
    tid = _normalize_id(tile_id)
    emj = TILE_TO_EMOJI.get(tid)
    return f"{emj} ({tid})" if emj else tid

def _extract_tile_id(key: str) -> Optional[str]:
    key = key.strip()
    if "(" in key and ")" in key:
        inner = key[key.find("(")+1:key.rfind(")")]
        return inner.strip()
    if len(key) >= 2 and key[0] in "mpsz" and key[1].isdigit():
        return key[:2]
    return None

def _label_to_emoji(key: str) -> str:
    tid = key.strip()
    return TILE_TO_EMOJI.get(tid, tid)

def normalize_tile_id(tid: str) -> str:
    t = tid.strip().lower().replace("_", "")
    if "(" in t and ")" in t:
        inner = t[t.find("(")+1:t.rfind(")")]
        t = inner.strip().lower()
    if len(t) == 2 and t[0] in "mps" and t[1] == "0":
        t = t[0] + "5"
    if len(t) == 3 and t[0] in "mps" and t[1] == "5" and t.endswith("r"):
        t = t[0] + "5"
    return t

def tile_to_10index(tid: str) -> int:
    t = normalize_tile_id(tid)
    if len(t) != 2: raise ValueError(f"Bad tile id: {tid}")
    suit, d = t[0], t[1]
    if suit in "mps":
        if d < "1" or d > "9": raise ValueError(f"Bad number for {tid}")
        base = {"m":0, "p":10, "s":20}[suit]
        return base + int(d)
    if suit == "z":
        if d < "1" or d > "7": raise ValueError(f"Bad honor for {tid}")
        return 30 + int(d)
    raise ValueError(f"Bad suit for {tid}")

def dict_to_counts(tile_dict: Dict[str, int]) -> Dict[str, Counter]:
    counts = {'m': Counter(), 'p': Counter(), 's': Counter(), 'z': Counter()}
    for k, n in tile_dict.items():
        tid = _extract_tile_id(k)
        if not tid:
            continue
        suit, num = tid[0], tid[1]
        if suit in counts and num.isdigit():
            counts[suit][num] += int(n)
    return counts

def tenhou_from_counts_compact(counts: Dict[str, Counter]) -> str:
    parts = []
    for suit in "mpsz":
        if not counts[suit]:
            continue
        order = ['1','2','3','4','5','6','7','8','9','0'] if suit in "mps" else ['1','2','3','4','5','6','7']
        digits = []
        for d in order:
            c = counts[suit].get(d, 0)
            if c:
                digits.append(d * c)
        if digits:
            parts.append(suit + "".join(digits))
    return "".join(parts)

def tenhou_from_counts_spaced(counts: Dict[str, Counter]) -> str:
    tokens = []
    for suit in "mpsz":
        if not counts[suit]:
            continue
        order = ['1','2','3','4','5','6','7','8','9','0'] if suit in "mps" else ['1','2','3','4','5','6','7']
        digits = []
        for d in order:
            c = counts[suit].get(d, 0)
            if c:
                digits.append(d * c)
        if digits:
            tokens.append(suit + "".join(digits))
    return " ".join(tokens)

def subtract_counts(all_counts: Dict[str, Counter], your_counts: Dict[str, Counter]) -> Dict[str, Counter]:
    out = {s: Counter() for s in "mpsz"}
    for suit in "mpsz":
        keys = set(all_counts[suit].keys()) | set(your_counts[suit].keys())
        for d in keys:
            diff = all_counts[suit].get(d,0) - your_counts[suit].get(d,0)
            if diff > 0:
                out[suit][d] = diff
    return out

# --------------------------------------------------------------------
# -- CLI BUILDERS
# --------------------------------------------------------------------
def build_oppo_args() -> list[list[str]]:
    return [
        bucket_as_id_list(CUR_OPP1_DISCARDS, CUR_OPP1_MELDS),
        bucket_as_id_list(CUR_OPP2_DISCARDS, CUR_OPP2_MELDS),
        bucket_as_id_list(CUR_OPP3_DISCARDS, CUR_OPP3_MELDS),
    ]

def build_after_riichi_args() -> list[list[str]]:
    return [
        bucket_as_id_list(CUR_OPP1_AFTER_RIICHI),
        bucket_as_id_list(CUR_OPP2_AFTER_RIICHI),
        bucket_as_id_list(CUR_OPP3_AFTER_RIICHI),
    ]

def _key_to_id(key: str) -> Optional[str]:
    tid = _extract_tile_id(key) or key
    tid = normalize_tile_id(tid)
    if len(tid) == 2 and tid[0] in "mpsz" and tid[1].isdigit():
        return tid
    return None

def bucket_as_id_list(*buckets: Dict[str, int]) -> list[str]:
    s = set()
    for b in buckets:
        for k in b.keys():
            tid = _key_to_id(k)
            if tid: s.add(tid)
    order = {**{f"m{d}":i for i,d in enumerate("123456789", 0)},
             **{f"p{d}":i+10 for i,d in enumerate("123456789", 0)},
             **{f"s{d}":i+20 for i,d in enumerate("123456789", 0)},
             **{f"z{d}":i+30 for i,d in enumerate("1234567",   0)}}
    return sorted(s, key=lambda t: order.get(t, 999))

def pick_riichi_tile_from_after_bucket(bucket_counter: Dict[str,int]) -> Optional[str]:
    if not bucket_counter: return None
    return max(bucket_counter.items(), key=lambda kv: (kv[1], kv[0]))[0]

def bucket_as_id_counter(bucket: Dict[str,int]) -> Dict[str,int]:
    out: Dict[str,int] = defaultdict(int)
    for k, v in bucket.items():
        tid = _key_to_id(k)
        if tid: out[tid] += int(v)
    return out

def build_riichi_map_from_buckets() -> dict[int, str]:
    riichi_map: dict[int,str] = {}
    for k, bucket in enumerate([CUR_OPP1_AFTER_RIICHI, CUR_OPP2_AFTER_RIICHI, CUR_OPP3_AFTER_RIICHI], start=1):
        counter = bucket_as_id_counter(bucket)
        tid = pick_riichi_tile_from_after_bucket(counter)
        if tid:
            riichi_map[k] = tid
    return riichi_map

def build_riichi_map_args(riichi_map: dict[int, str]) -> list[str]:
    args: list[str] = []
    for k in (1,2,3):
        tid = riichi_map.get(k)
        if not tid: continue
        try:
            idx = tile_to_10index(tid)
        except Exception:
            continue
        args += ["--riichi-map", f"{k}:{idx}"]
    return args

def run_cmd_capture(cmd: list[str], timeout: int = 15) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        if res.returncode != 0:
            return f"[exit={res.returncode}]\n{out}\n{err}".strip()
        return out or err or "[no output]"
    except FileNotFoundError:
        return f"Error: '{cmd[0]}' not found on PATH."
    except subprocess.TimeoutExpired:
        return f"Error: '{cmd[0]}' timed out."

def run_efficiency_cli(your_tiles_argument: str, visible_tiles_argument: str, timeout: int = 15) -> str:
    cmd = ["efficiency"]
    if your_tiles_argument:
        cmd += ["--groups", your_tiles_argument]
    if visible_tiles_argument:
        cmd += ["--visible", visible_tiles_argument]
    cmd += ["--verbose"]
    if len(cmd) == 2:
        return "[efficiency skipped] No tiles to analyze."
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        output = res.stdout.strip()
        err = res.stderr.strip()
        if res.returncode != 0:
            return f"[efficiency exit={res.returncode}]\n{output}\n{err}".strip()
        return output or err or "[efficiency produced no output]"
    except FileNotFoundError:
        return "Error: 'efficiency' binary not found on PATH."
    except subprocess.TimeoutExpired:
        return "Error: 'efficiency' timed out."

def run_defense_cli(your_tiles_argument: str, visible_tiles_argument: str, timeout: int = 15) -> str:
    cmd = ["defense"]
    if your_tiles_argument:
        cmd += ["-g", your_tiles_argument]
    if visible_tiles_argument:
        cmd += ["--visible", visible_tiles_argument]
    cmd += ["--verbose"]
    if len(cmd) == 1:
        return "[defense skipped] No tiles to analyze."
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        output = res.stdout.strip()
        err = res.stderr.strip()
        if res.returncode != 0:
            return f"[defense exit={res.returncode}]\n{output}\n{err}".strip()
        return output or err or "[defense produced no output]"
    except FileNotFoundError:
        return "Error: 'defense' binary not found on PATH."
    except subprocess.TimeoutExpired:
        return "Error: 'defense' timed out."

def add_emojis_to_defense_output(text: str) -> str:
    def suit_digit_to_emoji(suit: str, digit: str) -> str:
        key = f"{suit}{digit}"
        if digit == "r":
            key = f"{suit}0"
        return TILE_TO_EMOJI.get(key, "")

    def repl_group_digit_first(m):
        digits, suit = m.group(1), m.group(2)
        return "".join(suit_digit_to_emoji(suit, "0" if d == "0" else d) for d in digits)
    def repl_group_suit_first(m):
        suit, digits = m.group(1), m.group(2)
        return "".join(suit_digit_to_emoji(suit, "0" if d == "0" else d) for d in digits)
    def repl_single_digit_first(m):
        d, suit, red = m.group(1), m.group(2), m.group(3)
        dd = "0" if (d == "5" and red) or d == "0" else d
        return suit_digit_to_emoji(suit, dd) or m.group(0)
    def repl_single_suit_first(m):
        suit, d, red = m.group(1), m.group(2), m.group(3)
        dd = "0" if (d == "5" and red) or d == "0" else d
        return suit_digit_to_emoji(suit, dd) or m.group(0)

    HONOR_LETTER_TO_Z = {"E":"z1","S":"z2","W":"z3","N":"z4","P":"z5","F":"z6","C":"z7"}
    def repl_honor_letter(m):
        letter = m.group(1)
        key = HONOR_LETTER_TO_Z.get(letter)
        return TILE_TO_EMOJI.get(key, "") if key else m.group(0)

    out = text
    out = re.sub(r'([0-9]+)([mpsz])', repl_group_digit_first, out)
    out = re.sub(r'([mpsz])([0-9]+)', repl_group_suit_first, out)
    out = re.sub(r'\b([0-9])([mpsz])(r)?\b', repl_single_digit_first, out)
    out = re.sub(r'\b([mpsz])([0-9])(r)?\b', repl_single_suit_first, out)
    out = re.sub(r'\b([ESWNPFC])\b', repl_honor_letter, out)
    out = re.sub(r'\b([0-9])([mps])(r?)\b', r'\2\1\3', out)
    return out

# --------------------------------------------------------------------
# -- REGION-BASED COMPUTE
# --------------------------------------------------------------------
def scan_region_for_tiles(
    image: np.ndarray,
    rect: Tuple[int,int,int,int],
    templates_all: Dict[str, np.ndarray],
    threshold_overrides: Optional[Dict[str,float]] = None
) -> Tuple[Dict[str,int], List[Tuple[int,int,int,int]]]:
    x1,y1,x2,y2 = rect
    if x2 <= x1 or y2 <= y1:
        return {}, []
    H, W = image.shape[:2]
    local_mask = np.zeros((H, W), dtype=bool)
    region_counts: Dict[str, int] = defaultdict(int)
    for template_path, tmpl in templates_all.items():
        tile_id = template_path.rsplit("/", 1)[-1].replace(".png","").replace("_","")
        label   = _tile_label(tile_id)
        thr = 0.96 if tile_id == "z5" else 0.8
        if threshold_overrides and tile_id in threshold_overrides:
            thr = threshold_overrides[tile_id]
        n, locs = find_in_rect(tmpl, image, local_mask, rect, threshold=thr)
        if not n:
            continue
        region_counts[label] += n
    region_counts = {k: min(TILE_CAP, v) for k, v in region_counts.items()}
    return region_counts, []

def compute_tiles_by_region(
    image: np.ndarray,
    anchor_location: Iterable[int],
    templates_all: Dict[str, np.ndarray],
) -> None:
    for i in (1, 2, 3):
        OPP[i] = OpponentState()
    for d in (
        CUR_ALL_TILES, CUR_YOUR_TILES, CUR_DORA, CUR_YOUR_DISCARDS,
        CUR_OPP1_MELDS, CUR_OPP1_DISCARDS, CUR_OPP2_MELDS, CUR_OPP2_DISCARDS,
        CUR_OPP3_MELDS, CUR_OPP3_DISCARDS
    ):
        d.clear()

    H, W = image.shape[:2]
    for region_name, spec in REGIONS.items():
        rect = region_rect_from_anchor(anchor_location, spec, W, H)
        counts, _ = scan_region_for_tiles(image, rect, templates_all)
        bucket_name = REGION_TO_BUCKET.get(region_name)
        if bucket_name:
            _bucket_dict_by_name(bucket_name).update(counts)
        for k, v in counts.items():
            CUR_ALL_TILES[k] += v
        seat = REGION_TO_SEAT.get(region_name)
        if seat:
            ids = counts_to_id_list(counts)
            now_region = time.time()
            if "DISCARDS" in region_name:
                for tid in ids:
                    OPP[seat].discards.append((tid, now_region))
            elif "MELDS" in region_name:
                OPP[seat].melds.extend(ids)

# ------------ Save crops for bounded boxes (+ composites) -------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_all_region_screenshots(
    full_image: np.ndarray,
    anchor_location: Iterable[int],
    out_dir: str = SCREENSHOT_OUT_DIR,
    include_composite: bool = False
) -> None:
    ts = int(time.time()); _ensure_dir(out_dir)
    H, W = full_image.shape[:2]
    ax_min_x, ax_max_x, ax_min_y, ax_max_y = anchor_bounds(anchor_location)
    rx1, ry1, rx2, ry2 = clamp_rect(ax_min_x, ax_min_y, ax_max_x, ax_max_y, W, H)
    roi = full_image[ry1:ry2, rx1:rx2].copy()
    composite_full = full_image.copy()
    composite_roi  = roi.copy()
    color=(0,255,0); thick=2; font=cv2.FONT_HERSHEY_SIMPLEX; scale=0.5; txt_th=1; pad=4; bg=(0,0,0)

    def draw_label_bottom_left(img, box_xyxy, text):
        x1,y1,x2,y2 = map(int, box_xyxy)
        (tw, th), baseline = cv2.getTextSize(text, font, scale, txt_th)
        tx = max(x1 + pad, 0); ty = min(y2 - pad, img.shape[0]-1)
        bg_x1 = max(tx - pad, 0); bg_y1 = max(ty - th - baseline - pad, 0)
        bg_x2 = min(tx + tw + pad, img.shape[1]-1); bg_y2 = min(ty + pad, img.shape[0]-1)
        cv2.rectangle(img, (bg_x1,bg_y1),(bg_x2,bg_y2), bg, -1)
        cv2.putText(img, text, (tx, ty - baseline), font, scale, (255,255,255), txt_th, cv2.LINE_AA)

    for region_name, spec in REGIONS.items():
        x1, y1, x2, y2 = region_rect_from_anchor(anchor_location, spec, W, H)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = full_image[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(out_dir, f"{ts}_{region_name}_{x1}-{y1}-{x2}-{y2}.png"), crop)
        if include_composite:
            cv2.rectangle(composite_full, (x1,y1),(x2,y2), color, thick)
            draw_label_bottom_left(composite_full, (x1,y1,x2,y2), region_name)
            rx1r, ry1r = max(x1 - rx1, 0), max(y1 - ry1, 0)
            rx2r, ry2r = max(x2 - rx1, 0), max(y2 - ry1, 0)
            rx2r = min(rx2r, composite_roi.shape[1]-1); ry2r = min(ry2r, composite_roi.shape[0]-1)
            if rx2r > rx1r and ry2r > ry1r:
                cv2.rectangle(composite_roi, (rx1r,ry1r),(rx2r,ry2r), color, thick)
                draw_label_bottom_left(composite_roi, (rx1r,ry1r,rx2r,ry2r), region_name)

    if include_composite:
        cv2.imwrite(os.path.join(out_dir, f"{ts}_composite_full.png"), composite_full)
        cv2.imwrite(os.path.join(out_dir, f"{ts}_composite_roi.png"), composite_roi)

def counts_to_id_list(counts: dict[str,int]) -> list[str]:
    ids: list[str] = []
    for lbl, n in counts.items():
        tid = label_to_id(lbl)
        if not tid: 
            continue
        ids.extend([tid] * int(n))
    return ids

def label_to_id(label: str) -> str | None:
    tid = _extract_tile_id(label) or label.strip()
    tid = tid.lower().replace("_", "")
    if len(tid) >= 2 and tid[0] in "mps" and (tid.endswith("r") or tid[1] == "0" or tid == f"{tid[0]}5r"):
        tid = tid[0] + ("0" if tid.endswith("r") or tid[1] in "05" else tid[1])
    if len(tid) == 2 and ((tid[0] in "mps" and tid[1] in "0123456789") or (tid[0] == "z" and tid[1] in "1234567")):
        return tid
    return None

def drop_parentheses_after_emojis(text: str) -> str:
    ID_RE = r"(?:[mps][0-9]|z[1-7])"
    text = re.sub(fr"([{MAHJONG_EMOJI_RANGE}])\s*\(\s*{ID_RE}\s*\)", r"\1", text)
    text = re.sub(fr"\b([mpsz][0-9]r?)\b\s*([{MAHJONG_EMOJI_RANGE}])", r"\2", text)
    text = re.sub(fr"\b([0-9][mpsz]r?)\b\s*([{MAHJONG_EMOJI_RANGE}])", r"\2", text)
    text = re.sub(r"\([^)]+(?=$|\s|[|,:;])", "", text)
    return text

# --------------------------------------------------------------------
# -- TK DISPLAY
# --------------------------------------------------------------------
def ensure_root() -> tk.Tk:
    global _ROOT
    if _ROOT is None:
        _ROOT = tk.Tk()
        _ROOT.withdraw()
    return _ROOT

class DisplayManager:
    def __init__(self):
        self.windows: dict[str, tuple[tk.Toplevel, ScrolledText]] = {}
        ensure_root()

    def _on_close(self, title: str):
        def handler():
            pair = self.windows.pop(title, None)
            if pair:
                win, _ = pair
                try:
                    if win.winfo_exists():
                        win.destroy()
                except tk.TclError:
                    pass
        return handler

    def _apply_visibility_policy(self):
        for title, (win, _area) in list(self.windows.items()):
            try:
                if not win.winfo_exists():
                    continue
                if title in VISIBLE_WINDOWS:
                    try:
                        win.deiconify()
                        win.lift()
                        win.attributes("-topmost", True)
                        win.attributes("-topmost", False)
                    except tk.TclError:
                        pass
                else:
                    try:
                        win.iconify()
                    except tk.TclError:
                        pass
            except tk.TclError:
                pass

    def _ensure_window(
        self,
        title: str,
        geom: tuple[int,int,int,int],
        topmost: bool,
        font: tuple[str,int],
        minimized: bool,
    ) -> tuple[tk.Toplevel, ScrolledText]:
        need_new = False
        pair = self.windows.get(title)
        if not pair:
            need_new = True
        else:
            win, area = pair
            try:
                if not win.winfo_exists() or not area.winfo_exists():
                    need_new = True
            except tk.TclError:
                need_new = True

        if need_new:
            win = tk.Toplevel(_ROOT)
            win.title(title)
            w, h, x, y = geom
            win.geometry(f"{w}x{h}+{x}+{y}")
            if topmost and title in VISIBLE_WINDOWS:
                try:
                    win.attributes("-topmost", True)
                except tk.TclError:
                    pass
            area = ScrolledText(win, wrap=tk.WORD, width=60, height=28, font=font)
            area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            win.protocol("WM_DELETE_WINDOW", self._on_close(title))
            area.bind("<Destroy>", lambda e: self.windows.pop(title, None) if title in self.windows else None)
            try:
                if minimized:
                    win.iconify()
                else:
                    win.deiconify()
            except tk.TclError:
                pass
            self.windows[title] = (win, area)
        else:
            win, area = self.windows[title]
            try:
                if minimized:
                    win.iconify()
                else:
                    win.deiconify()
            except tk.TclError:
                pass

        self._apply_visibility_policy()
        return win, area

    def _format_tiles(self, tiles_found: Dict[str, int], title: str) -> str:
        sorted_tiles = sorted(tiles_found.items(), key=lambda x: (-x[1], _label_to_emoji(x[0])))
        lines = [
            title,
            "-" * 35,
            f"{'Tile':<10} | {'Count':>5}",
            "-" * 35,
            *[f"{_label_to_emoji(tile):<10} | {count:>5}" for tile, count in sorted_tiles],
            "-" * 35,
            f"{'Total':<10} | {sum(tiles_found.values()):>5}",
        ]
        return "\n".join(lines)

    def show_or_update(
        self,
        title: str,
        tiles: Dict[str, int],
        geom: tuple[int,int,int,int],
        topmost: bool = True,
        font: tuple[str,int] = FONT_LARGE,
    ):
        text = self._format_tiles(tiles, title)
        minimized = title not in VISIBLE_WINDOWS
        geom = G(*geom)
        def _do():
            win, area = self._ensure_window(title, geom, topmost, font, minimized)
            try:
                area.configure(state="normal")
                area.delete("1.0", tk.END)
                area.insert(tk.END, text)
                area.configure(state="disabled")
                if topmost and (title in VISIBLE_WINDOWS):
                    win.attributes("-topmost", True)
                    win.attributes("-topmost", False)
            except tk.TclError:
                self.windows.pop(title, None)
            finally:
                self._apply_visibility_policy()
        _ROOT.after(0, _do)

    def show_or_update_text(
        self,
        title: str,
        text: str,
        geom: tuple[int,int,int,int],
        topmost: bool = True,
        font: tuple[str,int] = FONT_LARGE,
    ):
        minimized = title not in VISIBLE_WINDOWS
        geom = G(*geom)
        def _do():
            win, area = self._ensure_window(title, geom, topmost, font, minimized)
            try:
                area.configure(state="normal")
                area.delete("1.0", tk.END)
                area.insert(tk.END, text)
                area.configure(state="disabled")
                if topmost and (title in VISIBLE_WINDOWS):
                    win.attributes("-topmost", True)
                    win.attributes("-topmost", False)
            except tk.TclError:
                self.windows.pop(title, None)
            finally:
                self._apply_visibility_policy()
        _ROOT.after(0, _do)

# --------------------------------------------------------------------
# -- MAIN LOOP
# --------------------------------------------------------------------
def format_tiles_table(tiles: Dict[str, int], title: str) -> str:
    sorted_tiles = sorted(tiles.items(), key=lambda x: (-x[1], _label_to_emoji(x[0])))
    lines = [
        title,
        "-" * 35,
        f"{'Tile':<7} | {'Count':>5}",
        "-" * 35,
        *[f"{_label_to_emoji(tile):<7} | {count:>5}" for tile, count in sorted_tiles],
        "-" * 35,
        f"{'Total':<7} | {sum(tiles.values()):>5}",
    ]
    return "\n".join(lines)

def build_defense_cmd(
    groups_arg: str,
    visible_arg: str,
    oppo_lists: Optional[List[List[str]]] = None,
    after_lists: Optional[List[List[str]]] = None,
    riichi_map: Optional[Dict[int, int | str]] = None,
    verbose: bool = True,
) -> List[str]:
    cmd: List[str] = ["defense"]
    if groups_arg:
        cmd += ["-g", groups_arg]
    if visible_arg:
        cmd += ["--visible", visible_arg]

    def _norm_three(lst_of_lsts: Optional[List[List[str]]]) -> List[str]:
        out: List[str] = []
        for i in range(3):
            tok_list = (lst_of_lsts[i] if (lst_of_lsts and i < len(lst_of_lsts)) else [])
            out.append(" ".join(tok_list))
        return out

    for opp in _norm_three(oppo_lists):
        cmd += ["--oppo", opp]
    for aft in _norm_three(after_lists):
        cmd += ["--after-riichi", aft]

    if riichi_map:
        for k, v in riichi_map.items():
            cmd += ["--riichi-map", f"{int(k)}:{str(v)}"]

    if verbose:
        cmd += ["--verbose"]
    return cmd

def monitor_loop():
    riichi_dir = "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/1_riichi"
    riichi_templates = {file: imread_color(file) for file in load_template_files(riichi_dir)}

    anchor_dir = "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/anchor/"
    anchor_templates = {file: imread_color(file) for file in load_template_files(anchor_dir)}
    tile_dirs = [
        "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/1_riichi",
        "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/2_tiles",
        "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/3_doraAndOpponentMelds",
        "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/4_yourMelds",
    ]
    template_files_all: List[str] = []
    for d in tile_dirs:
        template_files_all.extend(load_template_files(d))
    templates_all = {file: imread_color(file) for file in template_files_all}

    dm = DisplayManager()

    s_w, s_h, s_x, s_y = G(320, 170, 40, 40)
    status = tk.Toplevel(ensure_root()); status.title("Status")
    status.geometry(f"{s_w}x{s_h}+{s_x}+{s_y}")
    try:
        status.attributes("-topmost", True)
        status.attributes("-topmost", False)
        status.deiconify()
    except tk.TclError:
        pass
    VISIBLE_WINDOWS.add("Status")

    status_var = tk.StringVar(value="Status: starting…")
    last_updated_var = tk.StringVar(value="Last updated: —")
    tk.Label(status, textvariable=status_var, anchor="w", font=FONT_LARGE).pack(padx=12, pady=(12, 4), fill=tk.X)
    tk.Label(status, textvariable=last_updated_var, anchor="w", font=FONT_MEDIUM).pack(padx=12, pady=(4, 8), fill=tk.X)

    def _toggle_summary():
        if SUMMARY_TITLE in VISIBLE_WINDOWS:
            VISIBLE_WINDOWS.discard(SUMMARY_TITLE)
        else:
            VISIBLE_WINDOWS.add(SUMMARY_TITLE)
        dm._apply_visibility_policy()

    tk.Button(status, text="Toggle Summary", command=_toggle_summary, font=FONT_MEDIUM).pack(
        padx=12, pady=(4, 10), fill=tk.X
    )

    def update_status(text: str):
        _ROOT.after(0, status_var.set, text)

    last_update_time = time.time()
    def _tick_last_updated():
        now = time.time()
        delta = int(now - last_update_time)
        last_updated_var.set(f"Last updated: {delta}s ago")
        _ROOT.after(1000, _tick_last_updated)
    _ROOT.after(1000, _tick_last_updated)

    last_roi_gray: Optional[np.ndarray] = None
    last_hash: Optional[str] = None

    def update_ui_and_cli():
        dm.show_or_update("all tiles detected", CUR_ALL_TILES, (240, 520, 1180,  40), font=FONT_LARGE)
        dm.show_or_update("your tiles detected", CUR_YOUR_TILES, (240, 350,  930,  40), font=FONT_LARGE)
        dm.show_or_update("your discards detected", CUR_YOUR_DISCARDS, (240, 350, 930, 40), font=FONT_LARGE)
        dm.show_or_update("dora", CUR_DORA, (220, 180, 930, 280), font=FONT_LARGE)

        combined = "\n\n".join([
            format_tiles_table(CUR_DORA,              "dora"),
            format_tiles_table(CUR_OPP1_DISCARDS,     "opponent 1 discards"),
            format_tiles_table(CUR_OPP1_MELDS,        "opponent 1 melds"),
            format_tiles_table(CUR_OPP2_DISCARDS,     "opponent 2 discards"),
            format_tiles_table(CUR_OPP2_MELDS,        "opponent 2 melds"),
            format_tiles_table(CUR_OPP3_DISCARDS,     "opponent 3 discards"),
            format_tiles_table(CUR_OPP3_MELDS,        "opponent 3 melds"),
        ])
        dm.show_or_update_text(SUMMARY_TITLE, combined, (420, 700, 900, 40), font=FONT_LARGE)

        all_counts = dict_to_counts(CUR_ALL_TILES)
        hand_counts = dict_to_counts(CUR_YOUR_TILES)
        visible_counts = subtract_counts(all_counts, hand_counts)

        your_tiles_argument = tenhou_from_counts_compact(hand_counts)
        visible_tiles_argument = tenhou_from_counts_spaced(visible_counts)

        def _run_efficiency():
            cli_output = run_efficiency_cli(your_tiles_argument, visible_tiles_argument, timeout=15)
            header = f'$ efficiency --groups "{your_tiles_argument}" --visible "{visible_tiles_argument}" --verbose\n\n'
            dm.show_or_update_text("efficiency (CLI output)", header + cli_output, (360, 380, 390,  40), font=FONT_LARGE)

        def _run_defense():
            oppo_lists  = build_oppo_args()
            after_lists = build_after_riichi_args()
            riichi_map  = build_riichi_map_from_buckets()
            cmd = build_defense_cmd(your_tiles_argument, visible_tiles_argument, oppo_lists, after_lists, riichi_map, True)
            cli_output = run_cmd_capture(cmd, timeout=15)
            cli_output = add_emojis_to_defense_output(cli_output)
            cli_output = drop_parentheses_after_emojis(cli_output)
            header = "$ " + " ".join([f'"{x}"' if " " in x else x for x in cmd]) + "\n\n"
            dm.show_or_update_text("defense (CLI output)", header + cli_output, (360, 380, 760, 40), font=FONT_LARGE)

        threading.Thread(target=_run_efficiency, daemon=True).start()
        threading.Thread(target=_run_defense, daemon=True).start()

    def recalc_current_frame(image: np.ndarray, anchor_location: Iterable[int]) -> Tuple[np.ndarray, str, List[Tuple[int,int,int,int]]]:
        nonlocal last_update_time
        compute_tiles_by_region(image, anchor_location, templates_all)

        H, W = image.shape[:2]
        for (bx1, by1, bx2, by2) in detect_riichi_icons_in_roi(image, anchor_location, riichi_templates):
            # seat inference by overlap vs. discard lanes (kept as-is in original design)
            pass  # optional: add seat mapping if needed

        if SAVE_REGION_SHOTS:
            try:
                save_all_region_screenshots(image, anchor_location, SCREENSHOT_OUT_DIR, include_composite=SAVE_REGION_COMPOSITE)
            except Exception as e:
                print(f"[warn] save_all_region_screenshots failed: {e}")

        update_ui_and_cli()
        update_status("Status: refreshed (motion/full ROI)")
        last_update_time = time.time()

        min_x, max_x, min_y, max_y = anchor_bounds(anchor_location)
        H, W = image.shape[:2]
        rx1, ry1, rx2, ry2 = clamp_rect(min_x, min_y, max_x, max_y, W, H)
        roi_color = image[ry1:ry2, rx1:rx2]
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        return roi_gray, roi_hash(image, anchor_location), []

    while True:
        image, _ = capture_screenshot()
        found_anchor = False
        anchor_location: List[int] = []

        for _, anchor_img in anchor_templates.items():
            anchor_location = return_first_match_location(anchor_img, image, threshold=0.76)
            if anchor_location:
                found_anchor = True
                break

        if not found_anchor:
            update_status("Status: anchor not found; retrying…")
            time.sleep(POLL_SECS)
            continue

        min_x, max_x, min_y, max_y = anchor_bounds(anchor_location)
        H, W = image.shape[:2]
        rx1, ry1, rx2, ry2 = clamp_rect(min_x, min_y, max_x, max_y, W, H)
        roi_color = image[ry1:ry2, rx1:rx2]
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        if last_roi_gray is None:
            last_roi_gray, last_hash, _ = recalc_current_frame(image, anchor_location)
            time.sleep(POLL_SECS)
            continue

        diff = cv2.absdiff(roi_gray, last_roi_gray)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, th = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_present = any(cv2.contourArea(c) >= 1500 for c in contours)

        if motion_present:
            last_roi_gray, last_hash, _ = recalc_current_frame(image, anchor_location)
            time.sleep(POLL_SECS)
            continue

        current_hash = roi_hash(image, anchor_location)
        if current_hash != last_hash:
            last_roi_gray, last_hash, _ = recalc_current_frame(image, anchor_location)
            time.sleep(POLL_SECS)
            continue

        update_status("Status: idle (no motion)")
        time.sleep(POLL_SECS)

def run_gui():
    ensure_root()
    _ROOT.mainloop()

if __name__ == "__main__":
    stop_event = threading.Event()
    threading.Thread(target=heartbeat, args=(stop_event,), daemon=True).start()
    threading.Thread(target=monitor_loop, daemon=True).start()
    run_gui()
    stop_event.set()
