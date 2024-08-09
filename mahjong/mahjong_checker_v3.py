class MahjongHandChecker:
    def __init__(self, your_discardable_tiles, your_tiles, all_tiles, left_player_discards, top_player_discards, right_player_discards):
        self.your_discardable_tiles = your_discardable_tiles.copy()
        self.all_tiles = all_tiles.copy()
        self.your_tiles = your_tiles.copy()
        self.left_player_discards = left_player_discards.copy()
        self.top_player_discards = top_player_discards.copy()
        self.right_player_discards = right_player_discards.copy()

    def gen_discard_scores(self):
        discard_scores = {}
        for tile in self.your_discardable_tiles:
            if self.your_discardable_tiles[tile] > 0 and tile[-1] != 'f':
                self.your_tiles[tile] -= 1
                potential_score = self.evaluate_hand_potential()
                risk_penalty = self.evaluate_defensive_risk(tile)
                discard_scores[tile] = potential_score - risk_penalty
                self.your_tiles[tile] += 1
        return discard_scores

    def evaluate_hand_potential(self):
        score = 0
        for tile, count in self.your_tiles.items():
            # Pungs and Kongs
            if count >= 3:
                score += 3
            elif count == 2 and (self.all_tiles[tile] - count >= 1):
                score += 2
            elif count == 1 and (self.all_tiles[tile] - count >= 2):
                score += 1

            # Chows
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
                        needed_tiles = [f"{n}{suit}" for n in triplet if self.your_discardable_tiles.get(f"{n}{suit}", 0) == 0]
                        missing_count = len(needed_tiles)
                        if missing_count == 0:
                            score += 3
                        elif missing_count == 1:
                            if all(self.all_tiles.get(t, 0) - self.your_discardable_tiles.get(t, 0) > 0 for t in needed_tiles):
                                score += 2
                        elif missing_count == 2:
                            if all(self.all_tiles.get(t, 0) - self.your_discardable_tiles.get(t, 0) > 0 for t in needed_tiles):
                                score += 1
        return score

    def evaluate_defensive_risk(self, tile):
        risk = 0
        tile_suit = tile[-1]

        # Calculate risk based on the suits less frequently discarded by each player
        risk += self.suit_scarcity_risk(tile_suit, self.left_player_discards)
        risk += self.suit_scarcity_risk(tile_suit, self.top_player_discards)
        risk += self.suit_scarcity_risk(tile_suit, self.right_player_discards)

        # Apply the 147, 258, 369 rule logic based on right player's discards
        risk -= self.adjust_risk_based_on_discards(tile, self.right_player_discards)  # Subtracting to reduce risk

        return risk

    def suit_scarcity_risk(self, tile_suit, player_discards):
        suit_counts = {'b': 0, 'c': 0, 'd': 0}
        for t in player_discards:
            suit = t[-1]
            if suit in suit_counts:
                suit_counts[suit] += 1

        min_discard_suit = min(suit_counts, key=suit_counts.get)

        if tile_suit == min_discard_suit:
            return 3
        else:
            return 1

    def adjust_risk_based_on_discards(self, tile, right_player_discards):
        risk_adjustment = 0
        tile_suit = tile[-1]
        tile_number = int(tile[:-1])

        for discarded_tile in right_player_discards:
            discarded_suit = discarded_tile[-1]
            discarded_number = int(discarded_tile[:-1])

            if tile_suit == discarded_suit:
                # 147 Rule: Discarding 4 makes 1 or 7 safer to discard
                if discarded_number == 4 and tile_number in [1, 7]:
                    risk_adjustment += 5  # Reduce risk
                # 258 Rule: Discarding 5 makes 2 or 8 safer to discard
                elif discarded_number == 5 and tile_number in [2, 8]:
                    risk_adjustment += 5  # Reduce risk
                # 369 Rule: Discarding 6 makes 3 or 9 safer to discard
                elif discarded_number == 6 and tile_number in [3, 9]:
                    risk_adjustment += 5  # Reduce risk

        return risk_adjustment


# USAGE
# Example data for testing
your_discardable_tiles = {'1b': 1, '4b': 1, '7b': 1, '5c': 1}
your_tiles = {'1b': 1, '4b': 1, '7b': 1, '5c': 1}
all_tiles = {'1b': 4, '4b': 4, '7b': 4, '5c': 4}
left_player_discards = ['3c', '7d']
top_player_discards = ['2b', '5d']
right_player_discards = ['4b', '5b', '6b']

checker = MahjongHandChecker(your_discardable_tiles, your_tiles, all_tiles, left_player_discards, top_player_discards, right_player_discards)
print(checker.gen_discard_scores())
