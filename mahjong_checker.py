from collections import defaultdict

class MahjongHandChecker:
    def __init__(self, tiles, all_tiles):
        self.tiles = tiles.copy()
        self.all_tiles = all_tiles.copy()

    def is_pung(self, tile):
        return self.tiles[tile] >= 3

    def is_chow(self, tile):
        if tile[-1] in ["b", "d", "c"]:
            num = int(tile[:-1])
            suit = tile[-1]
            return all(self.tiles.get(f"{num + i}{suit}", 0) >= 1 for i in range(3))
        return False

    def is_pair(self, tile):
        return self.tiles[tile] >= 2

    def remove_set(self, tile):
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
        self.tiles[tile] -= 2

    def is_all_honors(self):
        honor_tiles = {"ewh", "swh", "wwh", "nwh", "rdh", "gdh", "wdh"}
        for tile in self.tiles:
            if self.tiles[tile] > 0 and tile not in honor_tiles:
                return False
        return True

    def is_thirteen_orphans(self):
        required_tiles = {
            "1b", "9b", "1c", "9c", "1d", "9d",
            "ewh", "swh", "wwh", "nwh",
            "rdh", "gdh", "wdh",
        }
        found_pair = False
        for tile in required_tiles:
            if self.tiles.get(tile, 0) == 0:
                return False
            if self.tiles.get(tile, 0) > 1:
                found_pair = True
        return found_pair

    def is_seven_pairs(self):
        return sum(value == 2 for value in self.tiles.values()) == 7

    def is_all_flowers(self):
        flower_season_tiles = {"1f", "2f", "3f", "4f"}
        return all(self.tiles.get(tile, 0) > 1 for tile in flower_season_tiles)

    def check_special_hands(self):
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
            return True
        if not pair_found:
            for tile in list(self.tiles):
                if self.is_pair(tile):
                    self.remove_pair(tile)
                    if self.check_winning_hand(sets_found, True):
                        return True
                    self.tiles[tile] += 2
        for tile in list(self.tiles):
            if self.tiles[tile] > 0 and self.remove_set(tile):
                if self.check_winning_hand(sets_found + 1, pair_found):
                    return True
                num = int(tile[:-1])
                suit = tile[-1]
                if self.is_pung(tile):
                    self.tiles[tile] += 3
                else:
                    for i in range(3):
                        self.tiles[f"{num + i}{suit}"] += 1
        return False

    def discard_wisdom(self):
        discard_scores = {}
        for tile in self.tiles:
            if self.tiles[tile] > 0 and tile[-1] != 'f':
                original_count = self.tiles[tile]
                self.tiles[tile] -= 1
                discard_scores[tile] = self.evaluate_hand_potential()
                self.tiles[tile] = original_count
        return discard_scores

    def evaluate_hand_potential(self):
        score = 0
        for tile, count in self.tiles.items():
            if count >= 3:
                score += 1
            elif count == 2 and (self.all_tiles[tile] - count > 0):
                score += 0.5
            if tile[-1] in ['b', 'c', 'd'] and count > 0:
                num = int(tile[:-1])
                suit = tile[-1]
                chow_potentials = [
                    (num-2, num-1, num),
                    (num-1, num, num+1),
                    (num, num+1, num+2)
                ]
                for triplet in chow_potentials:
                    if all((self.all_tiles.get(f"{n}{suit}", 0) - self.tiles.get(f"{n}{suit}", 0)) > 0 for n in triplet if n >= 1 and n <= 9):
                        score += 1
        return score
