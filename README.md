# Mahjong Hand Checker

This Python script provides functionality for checking Mahjong hands based on predefined rules and suggestions for potential discards to improve the hand.

## Features

- **Hand Checks**: Determines if the hand meets special conditions like Pung, Chow, Pair, and various special Mahjong hands.
- **Discard Suggestions**: Offers advice on which tiles to discard to potentially improve the hand.
- **Tile Initialization**: Includes a comprehensive setup of a Mahjong set for simulation purposes.

## Requirements

- Python 3.x

## Usage

1. Clone or download this repository to your local machine.
2. Run the script via the command line:
   ```python mahjong_hand_checker.py```
Follow the interactive prompts to input your tiles and see suggested actions.

### Functions
is_pung(tile): Checks for a Pung in the hand.
is_chow(tile): Checks for a Chow in the hand.
is_pair(tile): Checks for a Pair in the hand.
remove_set(tile): Attempts to remove a set from the hand.
remove_pair(tile): Removes a pair of tiles from the hand.
check_special_hands(): Checks for any special hand patterns.
check_winning_hand(): Checks if the current hand is a winning hand.
suggest_discard(): Suggests a tile to discard.

### How It Works
The script initializes a 144 tile Mahjong tile set and accepts user input to simulate a player's hand and discards. 

The hand is then evaluated against Hong Kong Mahjong rules, and suggestions are provided based on the potential of the hand after discarding specific tiles.
