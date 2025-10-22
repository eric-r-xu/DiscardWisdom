SUIT_OFFSETS = {"m": 0, "p": 9, "s": 18, "z": 27}
TILE_LABELS_34 = tuple(
    [f"{i+1}m" for i in range(9)] +
    [f"{i-8}p" for i in range(9, 18)] +
    [f"{i-17}s" for i in range(18, 27)] +
    ["1z","2z","3z","4z","5z","6z","7z"]
)
ALL_TILES_REMAINING = [4]*34
TERMINAL_INDICES = set([0,8,9,17,18,26] + list(range(27,34)))
