class MahjongHandChecker:
    def __init__(self, your_discardable_tiles, your_tiles, all_tiles):
        self.your_discardable_tiles = your_discardable_tiles.copy()
        self.all_tiles = all_tiles.copy()
        self.your_tiles = your_tiles.copy()

    def gen_discard_scores(self):
        # removes each discardable tile to calculate hand potential
        discard_scores = {}
        for tile in self.your_discardable_tiles:
            if self.your_discardable_tiles[tile] > 0 and tile[-1] != 'f':
                self.your_tiles[tile] -= 1   # removes discardable tile to calculate hand potential
                discard_scores[tile] = self.evaluate_hand_potential()
                self.your_tiles[tile] += 1   # reverts discardable tile to original count
        return discard_scores

    def evaluate_hand_potential(self):
        score = 0
        for tile, count in self.your_tiles.items():
            # pungs
            if count >= 3:
                score += 3  # Full pung
            elif count == 2 and (self.all_tiles[tile] - count >= 1):
                    score += 2  # One tile missing for pung
            elif count == 1 and (self.all_tiles[tile] - count >= 2):
                    score += 1  # Two tiles missing for pung

            # chows
            if tile[-1] in ['b', 'c', 'd'] and count >= 1:
                num = int(tile[:-1])
                suit = tile[-1]
                chow_potentials = [
                    (num-2, num-1, num),
                    (num-1, num, num+1),
                    (num, num+1, num+2)
                ]
                for triplet in chow_potentials:
                    if all(1 <= n <= 9 for n in triplet):
                        needed_tiles = [f"{n}{suit}" for n in triplet if self.your_tiles.get(f"{n}{suit}", 0) == 0]
                        missing_count = len(needed_tiles)
                        if missing_count == 0:
                            score += 3  # Full chow
                        elif missing_count == 1:
                            if all(self.all_tiles.get(t, 0) - self.your_tiles.get(t, 0) > 0 for t in needed_tiles):
                                score += 2  # One tile missing for chow
                        elif missing_count == 2:
                            if all(self.all_tiles.get(t, 0) - self.your_tiles.get(t, 0) > 0 for t in needed_tiles):
                                score += 1  # Two tiles missing for chow
        return score

