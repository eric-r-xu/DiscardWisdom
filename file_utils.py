import os
from collections import defaultdict

def load_template_files(template_dir):
    template_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(template_dir)
        for file in files
        if file.endswith((".PNG", ".png"))
    ]
    return template_files

def initialize_all_tiles():
    all_tiles = {}
    for num in range(1, 10):
        for suit in ["b", "d", "c"]:
            all_tiles[f"{num}{suit}"] = 4
    for tile in ["nwh", "swh", "wwh", "ewh", "gdh", "rdh", "wdh"]:
        all_tiles[tile] = 4
    for num in range(1, 5):
        all_tiles[f"{num}f"] = 2
    return all_tiles
