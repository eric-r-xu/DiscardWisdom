import os
from collections import defaultdict

def load_template_files(template_dir):
    template_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(template_dir)
        for file in files
        if file.endswith((".PNG", ".png")) and file.startswith(("m","p","s","z"))
    ]
    return template_files

def init_all_tiles():
    all_tiles = {}
    for num in range(1, 10):
        for suit in ["b","d","c"]:
            all_tiles[f"{num}{suit}"] = 4
    for tile in ["nwh","swh","wwh","ewh","gdh","rdh","wdh"]:
        all_tiles[f"{tile}"] = 4
    for num in range(1, 5):
        all_tiles[f"{num}f"] = 2
    return all_tiles

def init_tiles_map():
    tiles_map = {}
    for num in range(1, 10):
        num_str = str(num)
        for _suit, _suit_name in {"b":" bamboo", "d": " dot", "c": " character"}.items():
            tiles_map[f"{num_str}{_suit}"] = ''.join([num_str, str(_suit_name)])
    for _tile, _tile_name in {"nwh": "north wind", "swh": "south wind", "wwh": "west wind", "ewh": "east wind", "gdh": "green dragon", "rdh": "red dragon", "wdh": "white dragon"}.items():
        tiles_map[f"{_tile}"] = _tile_name
    return tiles_map
