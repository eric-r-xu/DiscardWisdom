class MahjongHandChecker:
    def __init__(self, your_discardable_tiles, your_tiles, all_tiles, left_player_discards, top_player_discards, right_player_discards):
        self.your_discardable_tiles = your_discardable_tiles.copy()
        self.all_tiles = all_tiles.copy()
        self.your_tiles = your_tiles.copy()
        self.left_player_discards = left_player_discards.copy()
        self.top_player_discards = top_player_discards.copy()
        self.right_player_discards = right_player_discards.copy()

    # If a discard score is high for a specific tile, it means without that tile, the
    # hand is still very valuable, and thus a higher relative discard score theoretically
    # means that the tile can more safely be discarded over another tile with a lower score (due to taking away hand value)
    def gen_discard_scores(self):
        discard_scores = {}
        for tile in self.your_discardable_tiles:
            # check if tile is discardable (not meld), and tile is not a flower
            if self.your_discardable_tiles[tile] > 0 and tile[-1] != 'f':
                self.your_tiles[tile] -= 1
                potential_score = self.evaluate_hand_potential()
                risk_penalty = self.evaluate_defensive_risk(tile)   # TO DO
                discard_scores[tile] = potential_score - risk_penalty
                # Reverts discardable tile to original count
                self.your_tiles[tile] += 1
        return discard_scores

    def evaluate_hand_potential(self):
        score = 0
        for tile, count in self.your_tiles.items():
            ############# PUNGS ################
            if count >= 3:
                score += 3  # Full pung or kong
            elif count == 2 and (self.all_tiles[tile] - count >= 1):
                score += 2  # One tile missing for pung, and it may still be possible to obtain
            elif count == 1 and (self.all_tiles[tile] - count >= 2):
                score += 1  # Two tiles missing for pung, and it may still be possible to obtain

            ############# CHOWS ################
            # Chows (b = bamboo, c = characters, d = dots)
            if tile[-1] in ['b', 'c', 'd'] and count >= 1:
                num = int(tile[:-1])
                suit = tile[-1]
                chow_potentials = [
                    (num-2, num-1, num),
                    (num-1, num, num+1),
                    (num, num+1, num+2)
                ]
                for triplet in chow_potentials:
                    # evaluate if chow potential is possible for all 3 tiles (e.g. 2d has no (num-2, num-1, num) possibility)
                    if all(1 <= n <= 9 for n in triplet):
                        # your discardable tiles because we need to exclude fixed melds for chow possibilities since they can't be re-arranged
                        needed_tiles = [f"{n}{suit}" for n in triplet if self.your_discardable_tiles.get(f"{
                                                                                                         n}{suit}", 0) == 0]
                        missing_count = len(needed_tiles)
                        if missing_count == 0:
                            score += 3  # Full chow
                        elif missing_count == 1:
                            if all(self.all_tiles.get(t, 0) - self.your_discardable_tiles.get(t, 0) > 0 for t in needed_tiles):
                                score += 2  # One tile missing for chow
                        elif missing_count == 2:
                            if all(self.all_tiles.get(t, 0) - self.your_discardable_tiles.get(t, 0) > 0 for t in needed_tiles):
                                score += 1  # Two tiles missing for chow
        return score

    def evaluate_defensive_risk(self, tile):
        risk = 0
        tile_suit = tile[-1]

        # Calculate risk based on the suits less frequently discarded by each player
        risk += self.suit_scarcity_risk(tile_suit, self.left_player_discards)
        risk += self.suit_scarcity_risk(tile_suit, self.top_player_discards)
        risk += self.suit_scarcity_risk(tile_suit, self.right_player_discards)

        return risk

    def suit_scarcity_risk(self, tile_suit, player_discards):
        # Count the number of discards for each suit
        suit_counts = {'b': 0, 'c': 0, 'd': 0}
        for t in player_discards:
            suit = t[-1]
            if suit in suit_counts:
                suit_counts[suit] += 1

        # Find the suit with the fewest discards
        min_discard_suit = min(suit_counts, key=suit_counts.get)

        # If the tile's suit matches the suit with the fewest discards, increase risk
        if tile_suit == min_discard_suit:
            return 3  # Arbitrary high risk score for suits with fewer discards
        else:
            return 1  # Lower risk for more frequently discarded suits


# USAGE
#   > checker = MahjongHandChecker(your_discardable_tiles, your_tiles, all_tiles, left_player_discards, top_player_discards, right_player_discards)
#   > print(checker.gen_discard_scores())
