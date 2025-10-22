#########################
# KEMONO TILE DETECTION #
#########################
from __future__ import annotations

import os
from collections import defaultdict, Counter
from typing import Dict, Iterable, Optional, Tuple
from functools import lru_cache
import hashlib
import threading
import time
import subprocess  

import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
import pyautogui

VERBOSE = False  

_ROOT: Optional[tk.Tk] = None

def heartbeat(stop_event):
    while not stop_event.is_set():
        print("...")
        time.sleep(8)

@lru_cache(maxsize=64)
def load_template_files(template_dir: str) -> list[str]:
    out: list[str] = []
    for root, _dirs, files in os.walk(template_dir):
        for file in files:
            if file.startswith(("m", "p", "s", "z", "anchor")) and file.lower().endswith(".png"):
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

def find_num_locations_and_all_locations(
    template_image: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    anchor_location: Iterable[int],
    threshold: float = 0.8,
):
    # Bounds from anchor [x1, x2, y1, y2]
    ax1, _ax2, ay1, _ay2 = anchor_location
    max_x = ax1 + 800
    max_y = ay1 + 1600
    min_x = ax1 - 100
    min_y = ay1

    img = _contig_f32(image)
    tmpl = _contig_f32(template_image)
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)

    ys, xs = np.where(res >= float(threshold))
    th, tw = template_image.shape[:2]

    keep = (ys >= min_y) & (ys <= max_y) & (xs >= min_x) & (xs <= max_x)
    ys, xs = ys[keep], xs[keep]

    all_locations: list[list[int]] = []
    locations = 0
    for y, x in zip(ys, xs):
        end_y, end_x = y + th, x + tw
        if np.any(mask[y:end_y, x:end_x]):
            continue
        mask[y:end_y, x:end_x] = True
        all_locations.append([int(x), int(end_x), int(y), int(end_y)])
        locations += 1

    return locations, all_locations

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

def capture_screenshot():
    """Fast in-memory screenshot -> BGR image, new boolean mask."""
    shot = pyautogui.screenshot()  # PIL (RGB)
    image = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    return image, mask

def _extract_tile_id(key: str) -> Optional[str]:
    """
    Accepts keys like '🀛 (p3)' or '(p3)' or 'p3' and returns 'p3'.
    """
    key = key.strip()
    if "(" in key and ")" in key:
        inner = key[key.find("(")+1:key.rfind(")")]
        return inner.strip()
    if len(key) >= 2 and key[0] in "mpsz" and key[1].isdigit():
        return key[:2]
    return None

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

# ----------------- TK WINDOWS (updatable) -----------------
def ensure_root() -> tk.Tk:
    global _ROOT
    if _ROOT is None:
        _ROOT = tk.Tk()
        _ROOT.withdraw()
    return _ROOT

class DisplayManager:
    """Keeps one window per title and updates its text instead of creating new ones."""
    def __init__(self):
        self.windows: dict[str, tuple[tk.Toplevel, ScrolledText]] = {}
        ensure_root()

    def _format_tiles(self, tiles_found: Dict[str, int], title: str) -> str:
        sorted_tiles = sorted(tiles_found.items(), key=lambda x: (-x[1], x[0]))
        lines = [
            title,
            "-" * 35,
            f"{'Tile':<15} | {'Count':>5}",
            "-" * 35,
            *[f"{tile:<15} | {count:>5}" for tile, count in sorted_tiles],
            "-" * 35,
            f"{'Total':<15} | {sum(tiles_found.values()):>5}",
        ]
        return "\n".join(lines)

    def show_or_update(self, title: str, tiles: Dict[str, int], geom: Tuple[int,int,int,int], topmost: bool = True):
        text = self._format_tiles(tiles, title)
        if title not in self.windows:
            win = tk.Toplevel(_ROOT)
            win.title(title)
            if topmost:
                win.attributes("-topmost", True)
            w,h,x,y = geom
            win.geometry(f"{w}x{h}+{x}+{y}")
            area = ScrolledText(win, wrap=tk.WORD, width=60, height=28, font=("Arial", 12))
            area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            area.insert(tk.END, text)
            area.configure(state="disabled")
            win.protocol("WM_DELETE_WINDOW", win.destroy)
            self.windows[title] = (win, area)
        else:
            win, area = self.windows[title]
            area.configure(state="normal")
            area.delete("1.0", tk.END)
            area.insert(tk.END, text)
            area.configure(state="disabled")
            if topmost:
                win.attributes("-topmost", True)

    def show_or_update_text(self, title: str, text: str, geom: Tuple[int,int,int,int], topmost: bool = True):
        ensure_root()
        if title not in self.windows:
            win = tk.Toplevel(_ROOT)
            win.title(title)
            if topmost:
                win.attributes("-topmost", True)
            w, h, x, y = geom
            win.geometry(f"{w}x{h}+{x}+{y}")
            area = ScrolledText(win, wrap=tk.WORD, width=60, height=28, font=("Courier New", 11))
            area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            area.insert(tk.END, text)
            area.configure(state="disabled")
            win.protocol("WM_DELETE_WINDOW", win.destroy)
            self.windows[title] = (win, area)
        else:
            win, area = self.windows[title]
            def _update():
                area.configure(state="normal")
                area.delete("1.0", tk.END)
                area.insert(tk.END, text)
                area.configure(state="disabled")
                if topmost:
                    win.attributes("-topmost", True)
            _ROOT.after(0, _update)

# ----------------- CONSTANTS -----------------
TILE_TO_EMOJI: Dict[str, str] = {
    "z1": "🀀", "z2": "🀁", "z3": "🀂", "z4": "🀃", "z5": "🀆", "z6": "🀅", "z7": "🀄",
    "m1": "🀇", "m2": "🀈", "m3": "🀉", "m4": "🀊", "m5": "🀋", "m6": "🀌", "m7": "🀍", "m8": "🀎", "m9": "🀏", "m0": "🀋",
    "s1": "🀐", "s2": "🀑", "s3": "🀒", "s4": "🀓", "s5": "🀔", "s6": "🀕", "s7": "🀖", "s8": "🀗", "s9": "🀘", "s0": "🀔",
    "p1": "🀙", "p2": "🀚", "p3": "🀛", "p4": "🀜", "p5": "🀝", "p6": "🀞", "p7": "🀟", "p8": "🀠", "p9": "🀡", "p0": "🀝",
}

# ----------------- PIPELINE HELPERS -----------------
def compute_tiles(image, mask, anchor_location, templates_all):
    tiles_found = defaultdict(int)
    # your_melds = defaultdict(int)
    # doraAndOpponentMelds = defaultdict(int)
    your_tiles = defaultdict(int)

    for template_path, template_image in templates_all.items():
        tile_type = template_path.split("/")[-2]
        tile_id = template_path.rsplit("/", 1)[-1].replace(".png", "").replace("_", "")
        threshold = 0.96 if tile_id == "z5" else 0.8
        emoji = TILE_TO_EMOJI.get(tile_id, "")
        tile_id_emoji = f"{emoji} ({tile_id})" if emoji else f"({tile_id})"

        try:
            num_locations, all_locations = find_num_locations_and_all_locations(
                template_image, image, mask, anchor_location, threshold=threshold
            )
        except Exception:
            continue

        if num_locations > 0:
            tiles_found[tile_id_emoji] += num_locations
            '''
            if tile_type == "yourMelds":
                your_melds[tile_id_emoji] += num_locations
            if tile_type == "doraAndOpponentMelds":
                doraAndOpponentMelds[tile_id_emoji] += num_locations'''

            anchor_y2 = anchor_location[3]
            for x1, x2, y1, y2 in all_locations:
                dy = y2 - anchor_y2
                if 1430 <= dy <= 1470:
                    your_tiles[tile_id_emoji] += 1
    return tiles_found, your_tiles
    # return tiles_found, your_tiles, your_melds, doraAndOpponentMelds

def anchor_bounds(anchor_location):
    ax1, _ax2, ay1, _ay2 = anchor_location
    max_x = ax1 + 800
    max_y = ay1 + 1600
    min_x = ax1 - 100
    min_y = ay1
    return int(min_x), int(max_x), int(min_y), int(max_y)

def roi_hash(image, anchor_location):
    min_x, max_x, min_y, max_y = anchor_bounds(anchor_location)
    h, w = image.shape[:2]
    # clamp to image
    min_x = max(0, min_x); min_y = max(0, min_y)
    max_x = min(w, max_x); max_y = min(h, max_y)
    roi = image[min_y:max_y, min_x:max_x]
    return hashlib.md5(roi.tobytes()).hexdigest()


def run_efficiency_cli(your_tiles_argument: str, visible_tiles_argument: str, timeout: int = 15) -> str:
    cmd = ["efficiency", "--groups", your_tiles_argument,
           "--visible", visible_tiles_argument, "--verbose"]
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

# ----------------- MAIN MONITOR (every 2s) -----------------
def monitor_loop():
    anchor_dir = "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/anchor/"
    anchor_templates = {file: imread_color(file) for file in load_template_files(anchor_dir)}

    tile_dirs = [
        "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/allTiles/tiles",
        "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/allTiles/doraAndOpponentMelds",
        "/Users/ericxu/RiichiDiscardWisdom/tileTemplates/kemono/allTiles/yourMelds",
    ]
    template_files_all: list[str] = []
    for d in tile_dirs:
        template_files_all.extend(load_template_files(d))
    templates_all = {file: imread_color(file) for file in template_files_all}

    dm = DisplayManager()
    last_hash = None

    while True:
        image, mask = capture_screenshot()

        found_anchor = False
        anchor_location: list[int] = []
        for _, anchor_img in anchor_templates.items():
            anchor_location = return_first_match_location(anchor_img, image, threshold=0.76)
            if anchor_location:
                found_anchor = True
                break

        if not found_anchor:
            if VERBOSE: print("Anchor not found; retrying…")
            time.sleep(2)
            continue

        current_hash = roi_hash(image, anchor_location)
        if current_hash != last_hash:
            last_hash = current_hash
            if VERBOSE: print("Change detected in bounded ROI — recomputing…")

            tiles_found, your_tiles = compute_tiles(
                image, mask, anchor_location, templates_all
            )

            # Build Tenhou strings for CLI
            all_counts  = dict_to_counts(tiles_found)
            hand_counts = dict_to_counts(your_tiles)
            visible_counts = subtract_counts(all_counts, hand_counts)

            your_tiles_argument     = tenhou_from_counts_compact(hand_counts)      # e.g. "m123m456m789p11s11"
            visible_tiles_argument  = tenhou_from_counts_spaced(visible_counts)    # e.g. "m111 p0 s5 z7"

            dm.show_or_update("all tiles detected", tiles_found,        (220, 700,  910,  40))
            dm.show_or_update("your tiles detected", your_tiles,        (220, 400,  690,  40))
            # dm.show_or_update("your melded tiles detected", your_melds, (220, 220,  1050,  40))
            # dm.show_or_update("dora and opponent tiles detected", dora_oppo, (220, 220, 1270, 40))

            def _run_and_show():
                cli_output = run_efficiency_cli(your_tiles_argument, visible_tiles_argument, timeout=15)
                header = f'$ efficiency --groups "{your_tiles_argument}" --visible "{visible_tiles_argument}" --verbose\n\n'
                dm.show_or_update_text("efficiency (CLI output)", header + cli_output, (300, 400, 390, 40))
            threading.Thread(target=_run_and_show, daemon=True).start()

        time.sleep(2)  # check every 2 seconds

# ----------------- BOOT -----------------
def run_gui():
    ensure_root()
    _ROOT.mainloop()

if __name__ == "__main__":
    stop_event = threading.Event()
    threading.Thread(target=heartbeat, args=(stop_event,), daemon=True).start()

    threading.Thread(target=monitor_loop, daemon=True).start()

    run_gui()

    stop_event.set()
    print("Heartbeat stopped.")
