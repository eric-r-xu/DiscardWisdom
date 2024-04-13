from collections import defaultdict


class MahjongHandChecker:
    def __init__(self, tiles, all_tiles):
        self.tiles = (
            tiles.copy()
        ) 
        self.all_tiles = all_tiles.copy()

    def is_pung(self, tile):
        """Check if there's a pung of the given tile."""
        return self.tiles[tile] >= 3

    def is_chow(self, tile):
        """Check if there's a chow starting with the given tile."""
        if tile[-1] in ["b", "d", "c"]:  # Ensure the tile is a suited tile
            # print(f'tile = {tile}')
            num = int(tile[:-1])
            suit = tile[-1]
            # Check the sequence exists
            return all(self.tiles.get(f"{num + i}{suit}", 0) >= 1 for i in range(3))
        return False

    def is_pair(self, tile):
        """Check if there's a pair of the given tile."""
        return self.tiles[tile] >= 2

    def remove_set(self, tile):
        """Remove a pung or chow starting with the given tile, favoring pung first."""
        if self.is_pung(tile):
            self.tiles[tile] -= 3
            return True
        elif self.is_chow(tile):
            num = int(tile[:-1])
            suit = tile[-1]
            for i in range(3):
                self.tiles[f"{num + i}{suit}"] -= 1
            return True
        return False

    def remove_pair(self, tile):
        """Remove a pair of the given tile."""
        self.tiles[tile] -= 2

    def is_all_honors(self):
        """Check if the hand consists entirely of honor tiles (winds and dragons)."""
        honor_tiles = {"ewh", "swh", "wwh", "nwh", "rdh", "gdh", "wdh"}
        for tile in self.tiles:
            if self.tiles[tile] > 0 and tile not in honor_tiles:
                return False
        return True

    def is_thirteen_orphans(self):
        """Check if the hand is a valid Thirteen Orphans hand."""
        required_tiles = {
            "1b",
            "9b",
            "1c",
            "9c",
            "1d",
            "9d",
            "ewh",
            "swh",
            "wwh",
            "nwh",
            "rdh",
            "gdh",
            "wdh",
        }
        found_pair = False
        for tile in required_tiles:
            if self.tiles.get(tile, 0) == 0:
                return False
            if self.tiles.get(tile, 0) > 1:
                found_pair = True
        return found_pair

    def is_seven_pairs(self):
        """Check if the hand consists of exactly seven different pairs."""
        if sum(value == 2 for value in self.tiles.values()) == 7:
            return True
        return False

    def is_all_flowers(self):
        """Check if the hand contains all eight flower tiles."""
        flower_season_tiles = {"1f", "2f", "3f", "4f"}
        return all(self.tiles.get(tile, 0) > 1 for tile in flower_season_tiles)

    def check_special_hands(self):
        """Check for any special hands."""
        if self.is_all_honors():
            print("All Honors")
            return True
        elif self.is_thirteen_orphans():
            print("Thirteen Orphans")
            return True
        elif self.is_seven_pairs():
            print("Seven Pairs")
            return True
        elif self.is_all_flowers():
            print("All Flowers")
            return True
        return False

    def check_winning_hand(self, sets_found=0, pair_found=False):
        if sets_found == 4 and pair_found:
            return True  # Found 4 sets and a pair

        # Try to find a pair if not found yet
        if not pair_found:
            for tile in list(self.tiles):
                if self.is_pair(tile):
                    self.remove_pair(tile)
                    if self.check_winning_hand(sets_found, True):
                        return True
                    self.tiles[tile] += 2  # Backtrack

        # Try to find sets
        for tile in list(self.tiles):
            if self.tiles[tile] > 0 and self.remove_set(tile):
                if self.check_winning_hand(sets_found + 1, pair_found):
                    return True
                # Backtrack
                num = int(tile[:-1])
                suit = tile[-1]
                if self.is_pung(tile):
                    self.tiles[tile] += 3
                else:
                    for i in range(3):
                        self.tiles[f"{num + i}{suit}"] += 1

        return False
        
    def suggest_discard(self):
        potential_discards = {}
        for tile in self.tiles:
            if self.tiles[tile] > 0 and tile[-1] != 'f': # flowers can't be discarded
                # Temporarily remove the tile and evaluate the hand
                original_count = self.tiles[tile]
                self.tiles[tile] -= 1
                potential_discards[tile] = self.evaluate_hand_potential()
                self.tiles[tile] = original_count  # Restore the original tile count

        # Find the tile whose removal minimizes the potential for a winning hand
        least_useful_tile = min(potential_discards, key=potential_discards.get, default=None)
        return least_useful_tile, potential_discards[least_useful_tile] if least_useful_tile else None, potential_discards

    def evaluate_hand_potential(self):
        # Calculate the potential of the hand based on remaining tiles and current hand composition
        score = 0
        for tile, count in self.tiles.items():
            if count >= 3:
                score += 1  # Potential pung already in hand
            elif count == 2 and (self.all_tiles[tile] - count > 0):
                score += 0.5  # Potential pung if one more tile can be drawn

            # Check for potential chows for suited tiles
            if tile[-1] in ['b', 'c', 'd'] and count > 0:
                num = int(tile[:-1])
                suit = tile[-1]
                # Check for potential chows by looking forward and backward one and two tiles
                chow_potentials = [
                    (num-2, num-1, num),
                    (num-1, num, num+1),
                    (num, num+1, num+2)
                ]
                for triplet in chow_potentials:
                    if all((self.all_tiles.get(f"{n}{suit}", 0) - self.tiles.get(f"{n}{suit}", 0)) > 0 for n in triplet if n >= 1 and n <= 9):
                        score += 1  # Possible chow with available tiles

        return score


def initialize_all_tiles():
    # Initialize dictionary to hold the count of all tiles in a Mahjong set
    all_tiles = {}

    # Adding 4 copies of each suit tile (bamboos, dots, characters)
    for num in range(1, 10):
        for suit in [
            "b",
            "d",
            "c",
        ]:  # 'b' for bamboos, 'd' for dots, 'c' for characters
            all_tiles[f"{num}{suit}"] = 4

    # Adding 4 copies of each wind and dragon tile
    for tile in ["nwh", "swh", "wwh", "ewh", "gdh", "rdh", "wdh"]:  # winds and dragons
        all_tiles[tile] = 4

    # Adding 2 copies of each flower tile
    for num in range(1, 5):
        all_tiles[f"{num}f"] = 2

    return all_tiles


# Initialize all tiles
all_tiles = initialize_all_tiles()

# Initialize dictionaries for your tiles and opponent's tiles
your_tiles = defaultdict(int)

print("TILE NOMENCLATURE")
print(
    """\n   
# suits
#     bamboos = '1b' to '9b' , dots = '1d' to '9d', characters = '1c' to '9c'
# winds / honors
#     north = 'nwh', south = 'swh', west = 'wwh', east = 'ewh'
# dragons / honors
#     green dragon = 'gdh', red dragon = 'rdh', white dragon = 'wdh'
# flowers
#     '1f', '2f', '3f', '4f'
\n
""")




while True:
    your_tiles_input_str = input(
        "What are your tiles including melds/exposed and flowers? Please separate each tile by space. Type 'q' to quit: \n"
    )
    if your_tiles_input_str.lower() == 'q':
        break

    tile_count_validator = 0  # To validate the correct number of tiles (excluding flowers)
    for tile in your_tiles_input_str.split():
        if tile not in all_tiles:
            raise ValueError(f"{tile} is not valid")
        if all_tiles[tile] == 0:
            raise ValueError(f"There are too many {tile} tiles. Please re-check input")
    
        all_tiles[tile] -= 1  # Decrement tile from inventory
        if tile[-1] != "f":  # If tile is not a flower, count towards the 14-tile total
            tile_count_validator += 1
        your_tiles[tile] += 1  # Add or increment the tile in your collection
    
    # Ensure the user has exactly 14 tiles (excluding flowers)
    if tile_count_validator != 14:
        raise ValueError(
            f"Invalid number of tiles: Expecting 14, got {tile_count_validator}"
        )


    print("\nValid input\n")
    # print(f"All remaining tiles = {all_tiles}\n")
    # print(f"Your tiles = {your_tiles}")
    
    
    discarded_tiles_input_str = input(
        "\nPlease list all discarded tiles and melded/exposed tiles (including flowers) by opponents. Please separate each tile by space. Type 'q' to quit: \n"
    )

    if discarded_tiles_input_str.lower() == 'q':
        break

    for tile in discarded_tiles_input_str.split():
        if tile not in all_tiles:
            raise ValueError(f"{tile} is not valid")
        if all_tiles[tile] == 0:
            raise ValueError(f"There are too many {tile} tiles. Please re-check input")
        all_tiles[tile] -= 1  # Decrement tile from inventory
    
    
    
    print("\nValid input\n")

    checker = MahjongHandChecker(your_tiles, all_tiles)
    is_winning = False
    is_winning = max(checker.check_winning_hand(), checker.check_special_hands())
    print("Winning hand already?", is_winning)

    if is_winning:
        break
    
    if not is_winning:
        least_useful_tile, least_useful_score, all_discards = checker.suggest_discard()
        print(f"Suggest discarding {discard_tile} with a potential hand score of {discard_score}")
    
        print("\nPotential scores if each tile is discarded:\n")
        for tile, score in sorted(all_discards.items()):
            print(f"{tile}: {score}")
        

# Test Cases
#     Case 1: 1f 3f 1c 5c 5c 1b 6b 5d swh swh wwh nwh nwh wdh wdh 8d
#     Case 2: 1f 3f 1c 5c 5c 1b 6b 5d swh swh wwh nwh nwh wdh wdh 8d 2f 2f 1f 3f 4f 4f