# DiscardWisdom - HK Mahjong Discard Suggestor

🀀	Mahjong Tile East Wind	&#x1F000;
🀁	Mahjong Tile South Wind	&#x1F001;
🀂	Mahjong Tile West Wind	&#x1F002;
🀃	Mahjong Tile North Wind	&#x1F003;
🀄	Mahjong Tile Red Dragon	&#x1F004;
🀅	Mahjong Tile Green Dragon	&#x1F005;
🀆	Mahjong Tile White Dragon	&#x1F006;
🀇	Mahjong Tile One Of Characters	&#x1F007;
🀈	Mahjong Tile Two Of Characters	&#x1F008;
🀉	Mahjong Tile Three Of Characters	&#x1F009;
🀊	Mahjong Tile Four Of Characters	&#x1F00A;
🀋	Mahjong Tile Five Of Characters	&#x1F00B;
🀌	Mahjong Tile Six Of Characters	&#x1F00C;
🀍	Mahjong Tile Seven Of Characters	&#x1F00D;
🀎	Mahjong Tile Eight Of Characters	&#x1F00E;
🀏	Mahjong Tile Nine Of Characters	&#x1F00F;
🀐	Mahjong Tile One Of Bamboos	&#x1F010;
🀑	Mahjong Tile Two Of Bamboos	&#x1F011;
🀒	Mahjong Tile Three Of Bamboos	&#x1F012;
🀓	Mahjong Tile Four Of Bamboos	&#x1F013;
🀔	Mahjong Tile Five Of Bamboos	&#x1F014;
🀕	Mahjong Tile Six Of Bamboos	&#x1F015;

Symbol	Mahjong Tile Name	Unicode Number
🀖	Mahjong Tile Seven Of Bamboos	&#x1F016;
🀗	Mahjong Tile Eight Of Bamboos	&#x1F017;
🀘	Mahjong Tile Nine Of Bamboos	&#x1F018;
🀙	Mahjong Tile One Of Circles	&#x1F019;
🀚	Mahjong Tile Two Of Circles	&#x1F01A;
🀛	Mahjong Tile Three Of Circles	&#x1F01B;
🀜	Mahjong Tile Four Of Circles	&#x1F01C;
🀝	Mahjong Tile Five Of Circles	&#x1F01D;
🀞	Mahjong Tile Six Of Circles	&#x1F01E;
🀟	Mahjong Tile Seven Of Circles	&#x1F01F;
🀠	Mahjong Tile Eight Of Circles	&#x1F020;
🀡	Mahjong Tile Nine Of Circles	&#x1F021;
🀢	Mahjong Tile Plum	&#x1F022;
🀣	Mahjong Tile Orchid	&#x1F023;
🀤	Mahjong Tile Bamboo	&#x1F024;
🀥	Mahjong Tile Chrysanthemum	&#x1F025;
🀦	Mahjong Tile Spring	&#x1F026;
🀧	Mahjong Tile Summer	&#x1F027;
🀨	Mahjong Tile Autumn	&#x1F028;
🀩	Mahjong Tile Winter	&#x1F029;
🀪	Mahjong Tile Joker	&#x1F02A;
🀫	Mahjong Tile Back	&#x1F02B

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
