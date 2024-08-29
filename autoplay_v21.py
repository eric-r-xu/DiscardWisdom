import os
import time
from datetime import datetime
import pytz
import cv2
import numpy as np
import pytesseract
import logging
import queue
import re
import logging
import pyautogui
from pync import Notifier
from collections import defaultdict
from typing import DefaultDict, Tuple
from PIL import Image, ImageFilter
from collections import deque
from threading import Thread, Event
import textwrap
from concurrent.futures import ThreadPoolExecutor
from Levenshtein import distance as levenshtein_distance


class Config:
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M")
    MAX_SAMPLING_RATE_FPS = 120.0
    TIME_LIMIT = 30000
    NO_MOTION_THRESHOLD = 0.1
    NO_MOTION_WARNING = 30
    SCREEN_MATCHING_THRESHOLD = 0.97
    TILE_MATCHING_THRESHOLD = 0.97
    SMALL_TILE_MATCHING_THRESHOLD = 0.93
    OFFERED_TILE_MATCHING_THRESHOLD = 0.93
    TILE_OVERLAP_THRESHOLD = 10
    CHANGE_THRESHOLD = 20
    TEMPLATE_PATH = '/Users/ericxu/Documents/Jupyter/mahjong/templates'
    SCREENSHOT_PATH = '/Users/ericxu/Documents/Jupyter/mahjong/auto_screenshots'
    WINDS = ['E', 'N', 'W', 'S']
    MIN_PIXEL_VALUE = 200

    TEMPLATE_BOUNDARY_MAP = {
        'GameScreen': {"x_min": 1140, "y_min": 1260},
        'NextGame': {"x_min": 1720, "y_min": 1220},
        'Pong': {"x_min": 1930, "y_min": 820},
        'Kong': {"x_min": 1930, "y_min": 820},
        'Woo': {"x_min": 2100, "y_min": 810},
        'WooChow': {"x_min": 1750, "y_min": 810},
        'WooPong': {"x_min": 1750, "y_min": 810},
        'Chow': {"x_min": 1930, "y_min": 820},
        "ChowSelection": {"y_min": 800, "y_max": 1050},
        'YourDiscard': {"x_min": 2000, "y_min": 820},
        'KongPong': {"x_min": 1750, "y_min": 820},
        'PongChow': {"x_min": 1750, "y_min": 820},
        'NotFullScreen': {"x_max": 250, "y_max": 100},
        "player_you_discardable": {"y_min": 940, "y_max": 1190},
        "player_you_melded": {"y_min": 940, "y_max": 1190},
        "large_discards": {"y_min": 222, "y_max": 1200, "x_min": 160, "x_max": 2400},
        "small_discards": {"y_min": 222, "y_max": 1200, "x_min": 160, "x_max": 2400},
        "discards_minimal": {"y_min": 222, "y_max": 1200, "x_min": 160, "x_max": 2400},
        "DetermineWinner": {"x_min": 580, "y_min": 500, "x_max": 645, "y_max": 600},
        "DetermineWinnerBackup": {"x_min": 580, "y_min": 710, "x_max": 645, "y_max": 810},
        "DetermineSelfPick": {"x_min": 1650, "y_min": 400, "y_max": 1200},
        "DetermineWinnerFan": {"x_min": 1600, "x_max": 2000, "y_min": 500, "y_max": 1000},
        "DetermineSeatWind": {"y_min": 660, "y_max": 740, "x_min": 1230, "x_max": 1345},
        "DetermineFireGun2": {"y_min": 700, "y_max": 800, "x_min": 1480, "x_max": 1600},
        "DetermineFireGun3": {"y_min": 925, "y_max": 1025, "x_min": 1480, "x_max": 1600},
        "DetermineFireGun4": {"y_min": 1150, "y_max": 1250, "x_min": 1480, "x_max": 1600},
    }

    CLICK_COORDINATES = {
        "cancel": (1220, 430), "wall_tile": (1100, 540),
        "D13": (980, 540), "D12": (915, 540), "D11": (850, 540),
        "D10": (785, 540), "D9": (720, 540), "D8": (655, 540),
        "D7": (590, 540), "D6": (525, 540), "D5": (460, 540),
        "D4": (395, 540), "D3": (330, 540), "D2": (265, 540),
        "D1": (200, 540), "accept": (1100, 430), "accept_left": (980, 430),
        "x2": (1241, 151), "x": (1118, 200), "close_ad": (1198, 147),
        "exit_ad": (1205, 191), "exit_ad2": (1202, 197), "exit_ad3": (1243, 141),
        "exit_ad4": (668, 604), "next_game": (893, 626), "draw_game": (618, 450),
        "x_left": (114, 168), "ad_close": (632, 437), "xx": (1013, 252),
        "ad_skip_video": (1133, 152), "ad_play": (1068, 202), "ad_white_x": (1238, 175),
        "ad_white_x_black_background": (1245, 150), "ad_white_x2": (1107, 206),
        "google_play_x": (1067, 231), "x7": (1106, 203), "x_top_right": (1187, 151),
    }

    ACTION_SCREEN_CLICK_ORDER = {
        "NextGame": ['next_game'],
        "Draw": ['draw_game'],
        "YourDiscard": ["wall_tile"],
        "Chow": ['accept', 'cancel'],
        "ChowSelection": ["wall_tile", "D13", "D12", "D11", "D10", "D9", "D8", "D7", "D6", "D5", "D4", "D3", "D2", "D1"],
        "Pong": ['accept', 'cancel'],
        "Kong": ['accept', 'cancel'],
        "PongChow": ['accept_left', 'accept', 'cancel'],
        "KongPong": ['accept_left', 'accept', 'cancel'],
        "Ad": ["x_top_right", "x2", "x_left", "exit_ad", "x", "close_ad", "exit_ad4", "exit_ad3", "exit_ad2", "ad_close", "ad_skip_video", "ad_play", "xx"],
        "Woo": ['accept'],
        "WooChow": ['accept_left'],
        "WooPong": ['accept_left'],
        "ad_skip_video": ["ad_skip_video"], "ad_play": ["ad_play"], "ad_white_x": ["ad_white_x"], "ad_white_x_black_background": ["ad_white_x_black_background"],
        "google_play_x": ["google_play_x"], "x7": ["x7"], "ad_white_x2": ["ad_white_x2"],
        "x_top_right": ["x_top_right"]
    }

    GAME_ACTION_SCREENS = ['YourDiscard', 'Chow', 'Pong', 'Woo', 'WooChow', 'WooPong'
                           'Kong', 'ChowSelection', 'KongPong', 'PongChow', 'Draw']

    MELD_SCREENS = ['Chow', 'Pong', 'Woo', 'WooChow', 'WooPong'
                    'Kong', 'KongPong', 'PongChow']


class Utils:
    @staticmethod
    def is_screen(frame, screen_type='GameScreen', threshold=Config.SCREEN_MATCHING_THRESHOLD, xy_search_boundaries=None):
        if xy_search_boundaries is None:
            xy_search_boundaries = {}

        xy_search_boundaries.update(
            Config.TEMPLATE_BOUNDARY_MAP.get(screen_type, {}))
        x_min = xy_search_boundaries.get("x_min", 0)
        x_max = xy_search_boundaries.get("x_max", frame.shape[1])
        y_min = xy_search_boundaries.get("y_min", 0)
        y_max = xy_search_boundaries.get("y_max", frame.shape[0])

        f = frame[y_min:y_max, x_min:x_max]
        fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        folder_path = os.path.join(Config.TEMPLATE_PATH, screen_type)
        png_files = (f for f in os.listdir(folder_path) if f.endswith('.png'))

        sorted_png_files = sorted(png_files, key=lambda f: len(f), reverse=True)

        for template_path in sorted_png_files:
            tg = cv2.imread(os.path.join(
                folder_path, template_path), cv2.IMREAD_GRAYSCALE)
            matched_result = cv2.matchTemplate(fg, tg, cv2.TM_CCOEFF_NORMED)
            match_found = np.any(matched_result >= threshold)

            if match_found:
                yloc, xloc = np.where(matched_result >= threshold)
                msg = f'Match found with {screen_type}/{template_path}'
                print(msg)
                logging.debug(msg)
                return True, [int(min(xloc)), int(max(xloc)), int(min(yloc)), int(max(yloc)), template_path]

        return False, [0, 0, 0, 0, '']

    @staticmethod
    def chars_on_screen(frame: np.ndarray, screen_type: str = 'DetermineFireGun2', min_pixel_value: int = Config.MIN_PIXEL_VALUE) -> bool:
        """
        - Determines if any characters or significant elements are present on the screen for a given `DetermineFireGun{N}` screen type.

        - This method analyzes an image frame using predefined boundaries and image processing techniques 
          to detect the presence of characters or significant elements.

        Parameters:
        - frame (np.ndarray): The image frame to be analyzed.
        - screen_type (str): Specifies the type of screen to analyze, determining the bounding box for the search.
          (Default: 'DetermineFireGun2', which refers to the second player from the top on the end-of-round screen).
        - min_pixel_value (int): The minimum pixel intensity required to detect significant elements on the screen.
          (Default: Config.MIN_PIXEL_VALUE)

        Returns:
        - bool: Returns True if the maximum pixel intensity within the cropped frame is >= 200, indicating the presence 
          of characters or significant elements on the screen. Otherwise, returns False.
        """

        # Retrieve the search boundaries (x_min, x_max, y_min, y_max) for the specified screen type.
        xy_search_boundaries = {}
        xy_search_boundaries.update(
            Config.TEMPLATE_BOUNDARY_MAP.get(screen_type, {})
        )

        # Set the boundaries for cropping the frame based on the retrieved or default values.
        x_min = xy_search_boundaries.get("x_min", 0)
        x_max = xy_search_boundaries.get("x_max", frame.shape[1])
        y_min = xy_search_boundaries.get("y_min", 0)
        y_max = xy_search_boundaries.get("y_max", frame.shape[0])

        # Crop the frame to the specified boundaries.
        f = frame[y_min:y_max, x_min:x_max]

        # Convert the cropped image to a NumPy array.
        cropped_image_np = np.array(f)

        # If the cropped image has 3 channels (color), convert it to grayscale.
        if len(cropped_image_np.shape) == 3:
            cropped_image_np = cv2.cvtColor(
                cropped_image_np, cv2.COLOR_RGB2GRAY
            )

        # Find the maximum pixel value in the grayscale cropped image.
        _, max_val, _, _ = cv2.minMaxLoc(cropped_image_np)

        # Return True if the maximum pixel value >= Config.MIN_PIXEL_VALUE, indicating a significant element is present.
        return max_val >= 200


    @staticmethod
    def save_screenshot(frame: np.ndarray, prefix: str = '', suffix: str = '', text_info: str = '', ms_ts_id: int = int(time.time() * 1000)) -> None:
        """
        - Saves a screenshot of the provided image frame to a specified directory, optionally overlaying text on the image.
        - captures the current timestamp in milliseconds to uniquely name the screenshot file. 
        - text onto the image frame with a red color and a white outline for better visibility. 

        Parameters:
        - frame (np.ndarray): The image frame to save as a screenshot.
        - prefix (str): An optional prefix for the screenshot filename. (Default: '')
        - suffix (str): An optional suffix for the screenshot filename. (Default: '')
        - text_info (str): Optional text to overlay on the image. If provided, the text will be drawn with an outline 
          at the top-left corner of the image. (Default: None)
        - ms_ts_id (int): millisecond timestamp ID (Default: current millisecond timestamp)

        Returns:
        - None: This function saves the screenshot to the file system and prints the file path to the console.

        """

        # Create the directory path based on configuration settings
        dir_path = os.path.join(Config.SCREENSHOT_PATH, Config.TIMESTAMP)
        os.makedirs(dir_path, exist_ok=True)

        # Define the file path with the prefix and timestamp
        path = os.path.join(dir_path, f'{prefix}{ms_ts_id}{suffix}.png')

        # Overlay text on the frame if text_info is provided
        if len(text_info) > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 0, 255)  # Red color in BGR for the text
            thickness = 2  # Thickness for the main text (red)
            # White color in BGR for the outline
            outline_color = (255, 255, 255)
            outline_thickness = thickness + 2
            position = (10, 30)  # Position of the text on the image

            # Wrap the text to insert line breaks every 135 characters
            wrapped_text = textwrap.fill(text_info, width=135)

            # Split the wrapped text into lines
            lines = wrapped_text.split('\n')

            # Define the initial y-coordinate for the text position
            initial_y = position[1]

            # Define the line height (adjust this based on your font size)
            # Adjust the multiplier if needed
            line_height = int(font_scale * 30)

            for i, line in enumerate(lines):
                # Calculate the y-coordinate for the current line
                y_position = initial_y + i * line_height
                current_position = (position[0], y_position)

                # Draw the outline (white)
                cv2.putText(frame, line, current_position, font, font_scale,
                            outline_color, outline_thickness, cv2.LINE_AA)

                # Draw the text (red) on top of the outline
                cv2.putText(frame, line, current_position, font,
                            font_scale, color, thickness, cv2.LINE_AA)

        # Save the frame as a PNG file
        cv2.imwrite(path, frame)

        # log confirmation message
        msg = f"saved screenshot @ {path}"
        print(msg)
        logging.debug(msg)

    @staticmethod
    def current_time_to_pst():
        utc_dt = datetime.utcfromtimestamp(time.time())
        pst_tz = pytz.timezone('America/Los_Angeles')
        pst_dt = pytz.utc.localize(utc_dt).astimezone(pst_tz)
        return str(pst_dt)[0:19]

    @staticmethod
    def find_offered_meld(frame, screen_type='large_discards', threshold=Config.OFFERED_TILE_MATCHING_THRESHOLD):
        frame1 = frame.copy()
        frame1[0:210, :, :] = 0
        frame1[950:, :, :] = 0
        frame1[1390:, :, :] = 0
        frame1[0:210, :, :] = 0
        frame1[1190:1400, 0:1170, :] = 0

        offered_meld_tile = 'NA'

        xy_search_boundaries = {}
        xy_search_boundaries.update(
            Config.TEMPLATE_BOUNDARY_MAP.get(screen_type, {}))
        x_min = xy_search_boundaries.get("x_min", 0)
        x_max = xy_search_boundaries.get("x_max", frame1.shape[1])
        y_min = xy_search_boundaries.get("y_min", 0)
        y_max = xy_search_boundaries.get("y_max", frame1.shape[0])

        # Crop the frames according to the boundaries
        f1 = frame1[y_min:y_max, x_min:x_max]

        # Path to the folder containing templates
        folder_path = os.path.join(Config.TEMPLATE_PATH, screen_type)
        png_files = (f for f in os.listdir(folder_path) if f.endswith('.png'))
        
        sorted_png_files = sorted(png_files, key=lambda f: len(f), reverse=True)

        match_found = False
        for template_path in sorted_png_files:

            # Read the template in color
            template = cv2.imread(os.path.join(
                folder_path, template_path), cv2.IMREAD_COLOR)

            matched_result = cv2.matchTemplate(
                f1, template, cv2.TM_CCOEFF_NORMED)
            match_found = np.any(matched_result >= threshold)
            if match_found:
                offered_meld_tile = template_path.replace('.png', '')
                break

        _text_info = f'possible meld tile detected: {offered_meld_tile}'
        if not match_found:
            Utils.save_screenshot(
                f1, prefix='DEBUG_offered_meld_not_found', text_info=_text_info, suffix='')
        elif match_found:
            Utils.save_screenshot(
                f1, prefix='', text_info=_text_info, suffix='offered_meld_found')
        print(_text_info)
        logging.debug(_text_info)

        return offered_meld_tile



    @staticmethod
    def find_template_in_sections(
        frame: np.ndarray,
        screen_type: str = 'discards_minimal',
        white_threshold: int = 200,  # Adjust based on what you consider "enough white",
        collective_discards: dict = {},
    ):

        # Determine the height and width of the cropped region
        height_start, height_end = 410, 830
        width_start, width_end = 785, 1770

        cropped_image = frame[height_start:height_end, width_start:width_end, :]

        # Calculate the size of each section
        section_height = 84
        section_width = 58  # Rounded up from 57.94 to maintain integer indices

        # Path to the folder containing templates
        folder_path: str = os.path.join(Config.TEMPLATE_PATH, screen_type)
        png_files: List[str] = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(folder_path)
            for file in files if file.endswith('.png')
        ]

        sorted_png_files = sorted(png_files, key=lambda f: len(f), reverse=True)
        msg = f'sorted_png_files = {sorted_png_files}'
        print(msg)
        logging.debug(msg)

        if collective_discards is None:
            collective_discards = {}

        def pad_to_match_size(section: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Pad the smaller of the two images (section or template) so they have matching dimensions.
            """
            section_height, section_width = section.shape[:2]
            template_height, template_width = template.shape[:2]

            # Determine the required padding for height and width
            pad_height = abs(section_height - template_height)
            pad_width = abs(section_width - template_width)

            # Apply padding
            if section_height < template_height:
                # Pad section on the bottom
                section = np.pad(section, ((0, pad_height), (0, 0), (0, 0)), mode='constant')
            elif template_height < section_height:
                # Pad template on the bottom
                template = np.pad(template, ((0, pad_height), (0, 0), (0, 0)), mode='constant')

            if section_width < template_width:
                # Pad section on the right
                section = np.pad(section, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            elif template_width < section_width:
                # Pad template on the right
                template = np.pad(template, ((0, 0), (0, pad_width), (0, 0)), mode='constant')

            return section, template

        def process_template(template_path: str, section: np.ndarray) -> bool:
            # Read the template in color
            template: np.ndarray = cv2.imread(template_path, cv2.IMREAD_COLOR)

            
            try:
                # Perform template matching
                matched_result: np.ndarray = cv2.matchTemplate(section, template, cv2.TM_CCOEFF_NORMED)
            except:
                # Pad the smaller image to match the dimensions of the larger one
                section, template = pad_to_match_size(section, template)
                matched_result: np.ndarray = cv2.matchTemplate(section, template, cv2.TM_CCOEFF_NORMED)

            # Check if any location exceeds the threshold (indicating a match)
            match_found: bool = np.any(matched_result >= 0.92)

            return match_found

        # Extract and process each section
        for i in range(5):  # 5 rows
            for j in range(17):  # 17 columns
                if (i+1,j+1) not in collective_discards:    # only process new
                    section = cropped_image[i * section_height:(i + 1) * section_height,
                                            j * section_width:(j + 1) * section_width, :]
                    for template_path in sorted_png_files:
                        if process_template(template_path, section):    # if match found
                            tile = str(template_path.split('/')[-1]).replace('.png','')
                            collective_discards[(i+1, j+1)] = tile
                            break

        return dict(sorted(collective_discards.items()))



    @staticmethod
    def calc_small_tiles(
        frame: np.ndarray,
        screen_type: str = 'discards_minimal',
        threshold: float = Config.SMALL_TILE_MATCHING_THRESHOLD,
    ) -> DefaultDict[str, int]:

        all_discards_and_melds_found: DefaultDict[str, int] = defaultdict(int)
        ms_ts_id: int = int(time.time() * 1000)

        # Apply the mask to the frame
        frame[0:210, :, :] = 0
        frame[950:, :, :] = 0
        frame[1390:, :, :] = 0
        frame[1190:1400, 0:1170, :] = 0

        # Get the search boundaries
        xy_search_boundaries: Dict[str, int] = Config.TEMPLATE_BOUNDARY_MAP.get(
            screen_type, {})
        x_min: int = xy_search_boundaries.get("x_min", 0)
        x_max: int = xy_search_boundaries.get("x_max", frame.shape[1])
        y_min: int = xy_search_boundaries.get("y_min", 0)
        y_max: int = xy_search_boundaries.get("y_max", frame.shape[0])

        # Crop the frame according to boundaries
        cropped_frame: np.ndarray = frame[y_min:y_max, x_min:x_max]

        # Path to the folder containing templates
        folder_path: str = os.path.join(Config.TEMPLATE_PATH, screen_type)
        png_files: List[str] = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(folder_path)
            for file in files if file.endswith('.png')
        ]

        sorted_png_files = sorted(png_files, key=lambda f: len(f), reverse=True)

        def process_template(template_path: str) -> Tuple[str, int]:
            non_overlapping_matches: List[Tuple[int, int]] = []
            counter: int = 0
            tile: str = 'NA'

            # Read the template in color
            template: np.ndarray = cv2.imread(template_path, cv2.IMREAD_COLOR)

            # Perform template matching
            matched_result: np.ndarray = cv2.matchTemplate(
                cropped_frame, template, cv2.TM_CCOEFF_NORMED)
            match_found: bool = np.any(matched_result >= threshold)

            if match_found:
                print(f'template_path = {template_path}')

                # Extract the PNG file name
                png_file: str = os.path.basename(template_path)

                # Extract the relative subfolder path
                try:
                    subfolder_path: str = os.path.relpath(
                        os.path.dirname(template_path), folder_path)
                except:
                    subfolder_path = ''

                tile = subfolder_path + '_' + png_file.replace('.png', '')
                yloc, xloc = np.where(matched_result >= threshold)
                matches = list(zip(yloc, xloc))

                for match in matches:
                    if not any(np.linalg.norm(np.array(match) - np.array(prev_match)) < Config.TILE_OVERLAP_THRESHOLD for prev_match in non_overlapping_matches):
                        counter += 1
                        non_overlapping_matches.append(match)

            return tile, counter

        # Sequentially process templates
        for template_path in sorted_png_files:
            tile, counter = process_template(template_path)
            if counter > 0:
                all_discards_and_melds_found[tile] += counter

        return all_discards_and_melds_found

    @staticmethod
    def determine_your_discards(frame, screen_type='player_you_discardable', threshold=Config.TILE_MATCHING_THRESHOLD):
        your_discards = defaultdict(int)
        ms_ts_id = int(time.time() * 1000)
        frame1 = frame.copy()

        frame1[0:940, :, :] = 0
        frame1[1190:, :] = 0

        xy_search_boundaries = {}
        xy_search_boundaries.update(
            Config.TEMPLATE_BOUNDARY_MAP.get(screen_type, {}))
        x_min = xy_search_boundaries.get("x_min", 0)
        x_max = xy_search_boundaries.get("x_max", frame1.shape[1])
        y_min = xy_search_boundaries.get("y_min", 0)
        y_max = xy_search_boundaries.get("y_max", frame1.shape[0])

        # Crop the frames according to the boundaries
        f1 = frame1[y_min:y_max, x_min:x_max]

        # Path to the folder containing templates
        folder_path = os.path.join(Config.TEMPLATE_PATH, screen_type)
        png_files = (f for f in os.listdir(folder_path) if f.endswith('.png'))

        sorted_png_files = sorted(png_files, key=lambda f: len(f), reverse=True)

        for template_path in sorted_png_files:
            non_overlapping_matches = []
            counter = 0

            # Read the template in color
            template = cv2.imread(os.path.join(
                folder_path, template_path), cv2.IMREAD_COLOR)

            matched_result = cv2.matchTemplate(
                f1, template, cv2.TM_CCOEFF_NORMED)
            match_found = np.any(matched_result >= threshold)

            if match_found:
                tile = template_path.replace('.png', '')
                yloc, xloc = np.where(matched_result >= threshold)
                matches = list(zip(yloc, xloc))

                for match in matches:
                    if not any(np.linalg.norm(np.array(match) - np.array(prev_match)) < Config.TILE_OVERLAP_THRESHOLD for prev_match in non_overlapping_matches):
                        counter += 1
                        non_overlapping_matches.append(match)

                        # Get the bounding box of the match
                        y, x = match
                        h, w = template.shape[:2]
                        cropped_image = f1[y:y+h, x:x+w]
                        if screen_type == 'player_you_discardable':
                            if x >= 2132 and x < 2152:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_wall_tile.png'
                            elif x >= 1946 and x < 1978:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D13.png'
                            elif x >= 1811 and x < 1840:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D12.png'
                            elif x >= 1682 and x < 1702:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D11.png'
                            elif x >= 1549 and x < 1569:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D10.png'
                            elif x >= 1410 and x < 1435:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D9.png'
                            elif x >= 1278 and x < 1298:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D8.png'
                            elif x >= 1140 and x < 1160:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D7.png'
                            elif x >= 1002 and x < 1025:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D6.png'
                            elif x >= 864 and x < 890:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D5.png'
                            elif x >= 726 and x < 756:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D4.png'
                            elif x >= 588 and x < 620:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D3.png'
                            elif x >= 450 and x < 485:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D2.png'
                            elif x >= 312 and x < 350:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D1.png'
                            else:
                                filename = f'{ms_ts_id}_tile_{tile}_x{x}_D_undetermined.png'
                        elif screen_type == 'player_you_melded':
                            if x >= 290 and x < 315:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D1.png'
                            elif x >= 425 and x < 450:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D2.png'
                            elif x >= 560 and x < 585:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D3.png'
                            elif x >= 700 and x < 720:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D4.png'
                            elif x >= 835 and x < 855:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D5.png'
                            elif x >= 965 and x < 990:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D6.png'
                            elif x >= 1105 and x < 1125:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D7.png'
                            elif x >= 1240 and x < 1260:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D8.png'
                            elif x >= 1375 and x < 1395:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D9.png'
                            elif x >= 1510 and x < 1530:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D10.png'
                            elif x >= 1640 and x < 1665:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D11.png'
                            elif x >= 1780 and x < 1800:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D12.png'
                            else:
                                filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D_undetermined.png'

                        output_path = os.path.join(
                            Config.SCREENSHOT_PATH, Config.TIMESTAMP, filename)
                        msg = f'output_path = {output_path}'
                        print(msg)
                        logging.debug(msg)
                        # Save the cropped image only if undetermined
                        if filename.find('undetermined') > 0:
                            cv2.imwrite(output_path, cropped_image)
                            print(f"Saved {output_path}")

            if counter > 0:
                your_discards[tile] += counter

        _text_info = str(dict(sorted(your_discards.items())))
        Utils.save_screenshot(
            f1, prefix=f'{ms_ts_id}_{screen_type}_', text_info=_text_info)

        return your_discards

    @staticmethod
    def save_and_calc_frame_changes(
        frame1: np.ndarray,
        frame2: np.ndarray,
        all_discards_and_melds_found: DefaultDict[str, int],
        change_threshold: float = Config.CHANGE_THRESHOLD,
        tile_matching_threshold: float = Config.SMALL_TILE_MATCHING_THRESHOLD,
        frame_type: str = 'normal'
    ) -> DefaultDict[str, int]:

        f1 = frame1.copy()
        f2 = frame2.copy()
        ms_ts_id: int = int(time.time() * 1000)
        new_discards_and_melds_found: DefaultDict[str, int] = defaultdict(int)

        def preprocess(fx: np.ndarray, frame_type: str) -> np.ndarray:
            if frame_type == 'normal':
                fx[0:210, :, :] = 0
                fx[950:, :, :] = 0
                fx[1390:, :, :] = 0
                fx[1190:1400, 0:1170, :] = 0
            elif frame_type == 'player_left':
                fx[1200:, :, :] = 1
                fx[:222, :, :] = 1
                fx[:, 375:, :] = 1
                fx[910:, :, :] = 1
                fx[:, :150, :] = 1
            elif frame_type == 'player_right':
                fx[:222, :, :] = 1
                fx[:, :2175, :] = 1
                fx[:, 2400:, :] = 1
                fx[980:, :, :] = 1
            elif frame_type == 'player_across':
                fx[:222, :, :] = 1
                fx[410:, :, :] = 1
                fx[:, 0:450, :] = 1
                fx[:, 2100:, :] = 1
                fx[350:420, 350:1200, :] = 1
            elif frame_type == 'player_you_discardable':
                fx[0:940, :, :] = 1
                fx[1190:, :] = 1

            return fx

        f1 = preprocess(f1, frame_type)
        f2 = preprocess(f2, frame_type)

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)

        # Threshold the difference to highlight changes
        _, diff_thresh = cv2.threshold(
            diff, change_threshold, 255, cv2.THRESH_BINARY)

        # Find contours of the changes
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        highlighted_image = f2.copy()

        # Path to the folder containing templates
        folder_path: str = os.path.join(Config.TEMPLATE_PATH, 'discards_minimal')

        png_files: List[str] = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(folder_path)
            for file in files if file.endswith('.png')
        ]

        sorted_png_files = sorted(png_files, key=lambda f: len(f), reverse=True)

        
        def process_template_within_contour(cropped_frame: np.ndarray, template_path: str) -> bool:
            template: np.ndarray = cv2.imread(template_path, cv2.IMREAD_COLOR)

            # Check if the template is larger than the cropped image
            if template.shape[0] > cropped_frame.shape[0] or template.shape[1] > cropped_frame.shape[1]:
                logging.debug(f"Skipping template {template_path} because it is larger than the cropped image.")
                return False

            # List to store results of template matching for each permutation
            match_results = []
            
            # Define possible cuts, ensuring that the resulting image is not larger than the template
            cuts = [
                (0, max(0, cropped_frame.shape[0] - template.shape[0]), 0, 0), # Bottom cut
                (max(0, cropped_frame.shape[0] - template.shape[0]), 0, 0, 0), # Top cut
                (0, 0, 0, max(0, cropped_frame.shape[1] - template.shape[1])), # Right cut
                (0, 0, max(0, cropped_frame.shape[1] - template.shape[1]), 0)  # Left cut
            ]

            # Apply each cut and perform template matching
            for cut in cuts:
                top, bottom, left, right = cut
                cropped_variant = cropped_frame[top:cropped_frame.shape[0]-bottom, left:cropped_frame.shape[1]-right]

                # Ensure the cropped variant is not larger than the template
                if cropped_variant.shape[0] > template.shape[0] or cropped_variant.shape[1] > template.shape[1]:
                    # Resize the cropped variant to fit within the template
                    resized_cropped_variant = cv2.resize(cropped_variant, (template.shape[1], template.shape[0]))
                else:
                    resized_cropped_variant = cropped_variant

                # Perform template matching
                matched_result: np.ndarray = cv2.matchTemplate(resized_cropped_variant, template, cv2.TM_CCOEFF_NORMED)
                match_found: bool = np.any(matched_result >= Config.SMALL_TILE_MATCHING_THRESHOLD)
                match_results.append(match_found)

            # If any match is found, return True
            return any(match_results)


        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 30 and h >= 30:  # Only process significant changes
                msg = f'CONTOUR w / h : {w} / {h}'
                print(msg)
                logging.debug(msg)
                cropped_frame = f2[y:y+h, x:x+w]
                match_found = False
                for template_path in sorted_png_files:
                    if process_template_within_contour(cropped_frame, template_path):
                        match_found = True
                        parent_directory = os.path.basename(
                            os.path.dirname(template_path))
                        tile_name = parent_directory + '_' + \
                            os.path.basename(template_path).replace('.png', '')

                        if all_discards_and_melds_found.get(tile_name, 0) == 4:
                            msg = f"DEBUG: > 4 tiles found for {tile_name}"
                            print(msg)
                            Notifier.notify(msg)
                            logging.error(msg)
                        if frame_type == 'normal':
                            all_discards_and_melds_found[tile_name] += 1
                            Utils.save_screenshot(cropped_frame, prefix=f'{ms_ts_id}_w{w}_h{h}_', suffix=f'_{tile_name}')

                        new_discards_and_melds_found[tile_name] += 1
                    # exit loop if contour is too small to have more than one possible match
                    if (h*w) < 6300 and match_found:
                        break

                # green highlight border if match found, otherwise red highlight border
                if match_found:
                    cv2.rectangle(highlighted_image, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(highlighted_image, (x, y),
                                  (x + w, y + h), (0, 0, 255), 2)
                    Utils.save_screenshot(
                        cropped_frame, prefix=f'DEBUG_not_found_{ms_ts_id}_w{w}_h{h}_x{x}_y{y}', suffix='')
                    

        _text_info = str(dict(sorted(new_discards_and_melds_found.items())))

        Utils.save_screenshot(
            highlighted_image, prefix=f'HIGHLIGHTED_DELTA_{ms_ts_id}_{frame_type}', text_info=_text_info)

        if frame_type == 'normal':
            return all_discards_and_melds_found
        else:
            return new_discards_and_melds_found

    @staticmethod
    def find_closest_wind(cropped_image, offset=0):
        comparison_set = set(["E", "W", "S", "N"])

        cropped_image_np = np.array(cropped_image)
        cropped_image_np = cropped_image_np[:, :, ::-1]
        gray = cv2.cvtColor(cropped_image_np, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(
            gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary_image, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        extracted_text = pytesseract.image_to_string(
            eroded, config=custom_config)
        letters = re.findall(r'[a-zA-Z]', extracted_text)

        if letters:
            detected_letter = letters[0].upper()

            def similarity_score(a, b):
                return 1 - levenshtein_distance(a, b) / max(len(a), len(b))

            similarities = {letter: similarity_score(
                detected_letter, letter) for letter in comparison_set}
            likely_wind = max(similarities, key=similarities.get)

            if offset == 1:
                wind_map = {'W': 'S', 'S': 'E', 'E': 'N', 'N': 'W'}
                likely_wind = wind_map.get(likely_wind, likely_wind)

            return likely_wind
        else:
            msg = "WARNING: could not find wind"
            print(msg)
            logging.debug(msg)
            Notifier.notify(msg)
            return 'NA'


class ScreenshotCapturer:
    @staticmethod
    def capture():
        try:
            screenshot = pyautogui.screenshot()
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Screenshot capture failed: {e}")
            time.sleep(2)
            return None


class GameFrameQueue:
    def __init__(self, maxlen=2):
        self.queue = deque(maxlen=maxlen)

    def enqueue(self, frame, warning=True):
        if not isinstance(frame, np.ndarray):
            raise ValueError("Only numpy arrays are allowed.")

        # Check if the queue has at least one frame and
        #     if the new frame is the same as the last one
        if warning:
            if self.length() > 0 and np.array_equal(self.queue[-1], frame):
                msg = f'WARNING: {self} New frame identical to last'
                print(msg)
                logging.debug(msg)

                return

            self.queue.append(frame)
        else:
            self.queue.append(frame)

    def length(self):
        return len(self.queue)

    def clear_queue(self):
        # Clear all items in the queue
        self.queue.clear()
        msg = f"{self} queue has been cleared"
        print(msg)
        logging.debug(msg)
        Notifier.notify(msg)

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index += len(self.queue)
            if index < 0 or index >= len(self.queue):
                raise IndexError("Index out of range")
            return self.queue[index]
        else:
            raise TypeError("Index must be an integer")


class MotionDetector:
    def __init__(self, threshold=20):
        self.threshold = threshold

    def detect(self, frame1, frame2, screen_type='WallCount'):
        # Create masks for the regions of interest (ROI)
        mask = np.ones(frame1.shape[:2], dtype=np.uint8)

        if screen_type == 'WallCount':
            mask[:1190, :] = 0
            mask[:, 1450:] = 0
            mask[1190:1400, :1150] = 0
            mask[1390:, :] = 0
            mask[0:210, :] = 0
            mask[1190:1400, 0:1170] = 0
        elif screen_type == 'GameScreenAction':
            self.threshold = 50
            mask[0:210, :] = 0
            mask[950:, :] = 0
            mask[1390:, :] = 0
            mask[1190:1400, 0:1170] = 0
        else:
            self.threshold = 50
            mask[1200:] = 0
            mask[:210] = 0

        # Apply the mask to both frames to zero out non-ROI areas
        f1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        f2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        f1_gray = cv2.bitwise_and(f1_gray, f1_gray, mask=mask)
        f2_gray = cv2.bitwise_and(f2_gray, f2_gray, mask=mask)

        # Calculate the difference and threshold it
        diff = cv2.absdiff(f1_gray, f2_gray)
        _, diff_thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Return whether any contours were found, along with the processed frames
        return len(contours) > 0, frame1, frame2


class ClickMotionDetector(MotionDetector):
    def detect_after_click(self, location, screen_type='GameScreenAction'):
        # capture screenshot before click
        frame1 = ScreenshotCapturer.capture()

        # click location
        pyautogui.click(location)
        print(f"Clicked at {location}")

        # capture screenshot after click
        frame2 = ScreenshotCapturer.capture()

        return self.detect(frame1, frame2, screen_type=screen_type)


# MAIN SCRIPT
if __name__ == "__main__":

    ######### start of logging config #################
    logging.basicConfig(
        # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d-%H:%M:%S',  # Custom date format: YYYY-MM-DD-HH:MM:SS
        handlers=[
            # Log to a file named 'app.log'
            logging.FileHandler(f"logs/{Config.TIMESTAMP}.log"),
            logging.StreamHandler()  # Also log to the console
        ]
    )
    logger = logging.getLogger(__name__)

    logging.getLogger('PIL').setLevel(logging.WARNING)

    ######### end of logging config #################

    msg = f'Starting script @ {Utils.current_time_to_pst()}'
    print(msg)
    Notifier.notify(msg)
    logging.debug(msg)

    start_time = time.time()

    motion_detector = MotionDetector()

    your_discard_queue = GameFrameQueue()
    frame_queue = GameFrameQueue()

    click_motion_detector = ClickMotionDetector()
    last_motion_time = time.time()

    no_motion_before = False
    seat_wind = 'NA'
    wind_order = 'NA'

    all_discards_and_melds_found = defaultdict(int)

    collective_discards = {}

    YOUR_TILES = {key: '' for key in range(1, 15)}

    while (time.time() - start_time) < Config.TIME_LIMIT:

        frame = ScreenshotCapturer.capture()
        frame_queue.enqueue(frame, warning=False)

        ########### start: Detecting motion ##################
        if frame_queue.length() > 1:
            motion_detected, _, _ = motion_detector.detect(
                frame_queue[0], frame_queue[1])
            if motion_detected:
                last_motion_time = time.time()

        no_motion_elapsed_seconds = time.time() - last_motion_time
        if no_motion_elapsed_seconds > Config.NO_MOTION_WARNING:
            msg = f'WARNING: {no_motion_elapsed_seconds} sec of no motion'
            print(msg)
            logging.debug(msg)
            Notifier.notify(msg)
            if not no_motion_before:
                no_motion_before = True
        else:
            no_motion_before = False
        ########### end: Detecting motion ##################

        ########### start: GameScreen, NotFullScreen, Ad, or NextGame template detection ######
        not_full_screen, _nfs = Utils.is_screen(
            frame, screen_type='NotFullScreen')
        game_screen, _gs = Utils.is_screen(frame, screen_type='GameScreen')
        next_game, _ng = Utils.is_screen(frame, screen_type='NextGame')
        ad, _a = Utils.is_screen(frame, screen_type='Ad')
        ########### end: GameScreen, NotFullScreen, Ad, or NextGame template detection ######

        ########### start: Not Full Screen ###########
        if not_full_screen:
            msg = 'full screen not detected'
            print(msg)
            logging.debug(msg)
            Notifier.notify(msg)
            time.sleep(2)
            ########### end: Not Full Screen ###########

        ########### start: Game Screen ###########
        elif game_screen:
            game_frame = frame.copy()
            if seat_wind == 'NA':

                ########### start: Determine Seat Wind ###########
                st = 'DetermineSeatWind'
                xxyy = Config.TEMPLATE_BOUNDARY_MAP.get(st, {})
                cropped_image = game_frame[xxyy.get("y_min", 0):xxyy.get(
                    "y_max", 0), xxyy.get("x_min", 0):xxyy.get("x_max", 0)]
                gray = Image.fromarray(cropped_image).convert('L')
                _, binary_image = cv2.threshold(
                    np.array(gray), 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((1, 1), np.uint8)
                eroded = cv2.erode(cv2.dilate(
                    binary_image, kernel, iterations=1), kernel, iterations=1)
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                extracted_text = pytesseract.image_to_string(
                    eroded, config=custom_config)
                letters = re.findall(r'[a-zA-Z]', extracted_text)
                letter = ''.join(letters)
                if letter in Config.WINDS:
                    msg = f"Seat Wind - {letter}"
                    print(msg)
                    logging.debug(msg)
                    seat_wind = letter
                elif letter not in Config.WINDS:
                    msg = f"WARNING: Invalid Seat wind found: {letter}"
                    print(msg)
                    logging.debug(msg)
                    Notifier.notify(msg)
                else:
                    msg = f"WARNING: Seat wind found"
                    print(msg)
                    logging.debug(msg)
                    Notifier.notify(msg)
                ########### end: Determine Seat Wind ###########

            ########### start: Determine Game Action Screen ###########
            for screen_type in Config.GAME_ACTION_SCREENS:
                gas, _gas = Utils.is_screen(
                    game_frame, screen_type=screen_type)
                if gas:
                    msg = f'{screen_type} detected'
                    Notifier.notify(msg)
                    print(msg)
                    logging.debug(msg)

                    if screen_type == 'YourDiscard':
                        your_discard_queue.enqueue(game_frame)
                        ########### start: find your discardeable tiles and positions ###########
                        # ydt = your discardeablee tiles
                        ydt = Utils.determine_your_discards(
                            game_frame)
                        sorted_ydt = dict(sorted(ydt.items()))
                        sorted_filtered_ydt = {
                            key: value for key, value in sorted_ydt.items() if value != 0}
                        msg = f'your discardeable tiles = {sorted_filtered_ydt}'
                        print(msg)
                        logging.debug(msg)
                        ########### end: find your discardeable tiles and positions ###########

                        ########### start: find your melded tiles and positions ###########
                        # ymt = your melded tiles
                        ymt = Utils.determine_your_discards(
                            game_frame, screen_type='player_you_melded')
                        sorted_ymt = dict(sorted(ymt.items()))
                        sorted_filtered_ymt = {
                            key: value for key, value in sorted_ymt.items() if value != 0}
                        msg = f'your melded tiles = {sorted_filtered_ymt}'
                        print(msg)
                        logging.debug(msg)
                        ########### end: find your melded tiles and positions ###########

                        ########### start: small tile changes ###########
                        if your_discard_queue.length() == 1:
                            all_discards_and_melds_found = Utils.calc_small_tiles(
                                game_frame, screen_type='discards_minimal')
                        elif your_discard_queue.length() > 1:
                            previous_game_frame = your_discard_queue[0]
                            game_frame = your_discard_queue[1]

                            gf1, gf2 = previous_game_frame.copy(), game_frame.copy()
                            all_discards_and_melds_found = Utils.save_and_calc_frame_changes(
                                gf1, gf2, all_discards_and_melds_found, frame_type='normal')

                            ########### end: small tile changes ###########

                        ########## start: collective discards in 85 sections ##############
                        collective_discards = Utils.find_template_in_sections(game_frame, collective_discards=collective_discards)
                        msg = f'collective_discards = {collective_discards}'
                        print(msg)
                        logging.debug(msg)
                        ########## end: collective discards in 85 sections ##############

                    elif screen_type in Config.MELD_SCREENS:
                        offered_meld = None
                        offered_meld = Utils.find_offered_meld(game_frame)
                        if offered_meld:
                            msg = f'Offered meld: {offered_meld}'
                            print(msg)
                            logging.debug(msg)
                        else:
                            msg = f'ERROR: offered meld undetected'
                            logging.error(msg)
                            print(msg)

                    for click_label in Config.ACTION_SCREEN_CLICK_ORDER[screen_type]:
                        c = Config.CLICK_COORDINATES.get(click_label, (0, 0))
                        if c == (0, 0):
                            raise IndexError(
                                f"{click_label} key not found in Config.CLICK_COORDINATES")
                        md, _, _ = click_motion_detector.detect_after_click(
                            c, screen_type='GameScreenAction')
                        if md:
                            msg = f"Motion after clicking {click_label} @ {c}"
                            print(msg)
                            logging.debug(msg)
                            break
        ########### end: Game Screen ###########

        ########### start: Next Screen ###########
        elif next_game:
            screen_type = 'NextGame'
            msg = f"next game detected"
            print(msg)
            Notifier.notify(msg)
            logging.debug(msg)

            st = 'DetermineWinner'
            next_game_frame = frame.copy()
            xxyy = Config.TEMPLATE_BOUNDARY_MAP.get(st, {})
            cropped_image = next_game_frame[xxyy.get("y_min", 0):xxyy.get(
                "y_max", 0), xxyy.get("x_min", 0):xxyy.get("x_max", 0)]
            winner_wind = Utils.find_closest_wind(cropped_image)

            if winner_wind == 'NA':
                msg = 'WARNING: Winner wind not found, attemping `DetermineWinnerBackup`'
                print(msg)
                Notifier.notify(msg)
                logging.debug(msg)

                st = 'DetermineWinnerBackup'
                next_game_frame = frame.copy()
                xxyy = Config.TEMPLATE_BOUNDARY_MAP.get(st, {})
                cropped_image = next_game_frame[xxyy.get("y_min", 0):xxyy.get(
                    "y_max", 0), xxyy.get("x_min", 0):xxyy.get("x_max", 0)]
                winner_wind = Utils.find_closest_wind(cropped_image, offset=1)
                if winner_wind == 'NA':
                    msg = 'WARNING: Winner wind still not found after DetermineWinnerBackup'
                    print(msg)
                    Notifier.notify(msg)
                    logging.debug(msg)
            msg = f'winner wind = {winner_wind}'
            print(msg)
            Notifier.notify(msg)
            logging.debug(msg)
            details = f'_winner_{winner_wind}'

            if seat_wind == 'NA':
                Notifier.notify(
                    'WARNING: no seat wind recorded and winner cannot be determined')
            elif seat_wind == winner_wind:
                details += f'_WINNER'
                msg = 'You won the round!'
            else:
                msg = 'You did not win the round'
            print(msg)
            Notifier.notify(msg, timeout=5)
            logging.debug(msg)

            details += f'_seatwind_{seat_wind}'

            def determine_total_fan():
                st = 'DetermineWinnerFan'
                next_game_frame = frame.copy()
                xxyy = Config.TEMPLATE_BOUNDARY_MAP.get(st, {})
                cropped_image = next_game_frame[xxyy.get("y_min", 0):xxyy.get(
                    "y_max", 0), xxyy.get("x_min", 0):xxyy.get("x_max", 0)]
                cropped_image = Image.fromarray(cropped_image).convert('L').filter(
                    ImageFilter.MedianFilter()).point(lambda p: p > 128 and 255)
                base_width = 1000
                w_percent = (base_width / float(cropped_image.size[0]))
                h_size = int((float(cropped_image.size[1]) * float(w_percent)))
                cropped_image = cropped_image.resize(
                    (base_width, h_size), Image.LANCZOS)
                custom_config = r'--oem 3 --psm 6'
                extracted_text = pytesseract.image_to_string(
                    cropped_image, config=custom_config)
                return re.search(r'Total\s*Fan\s*(\d+)', extracted_text, re.IGNORECASE)

            for _ in range(2):
                match = determine_total_fan()
                if match:
                    total_fan_number = match.group(1)
                    msg = f"Total Fan Number: {total_fan_number}"
                    print(msg)
                    Notifier.notify(msg)
                    logging.debug(msg)
                    details += f'_{total_fan_number}Fan'
                    break
                else:
                    msg = 'WARNING: Total Fan not found, retrying...'
                    print(msg)
                    logging.debug(msg)
                    details += '_NoFanFound'

            st = 'DetermineSelfPick'
            r, _ = Utils.is_screen(frame, screen_type=st)
            details += '_selfpick' if r else '_discardwin'

            if seat_wind in Config.WINDS and winner_wind in Config.WINDS:
                wind_order_map = {'E': 'SWN',
                                  'S': 'WNE', 'N': 'ESW', 'W': 'NES'}
                wind_order = wind_order_map.get(winner_wind, '')
                Notifier.notify(f'wind order = {wind_order}')

                has_chars_on_screen = [Utils.chars_on_screen(frame, screen_type=st) for st in [
                    'DetermineFireGun2', 'DetermineFireGun3', 'DetermineFireGun4']]
                if sum(has_chars_on_screen) == 1:
                    firegunner = wind_order[has_chars_on_screen.index(True)]
                    details += f'_firegun_{firegunner}'

                    if seat_wind == firegunner:
                        msg = f'SHAMEEEE: You are detected as firegun!'
                        Notifier.notify(msg, timeout=5)
                        print(msg)
                        logging.debug(msg)
                        details += '_FIREGUN'

            msg = f'details = {details}'

            Utils.save_screenshot(
                frame, prefix=screen_type + details, text_info=details.replace('_', ' '))
            for click_label in Config.ACTION_SCREEN_CLICK_ORDER[screen_type]:
                c = Config.CLICK_COORDINATES.get(click_label, (0, 0))
                if c == (0, 0):
                    raise IndexError(f"{click_label} key not found")
                md, _, _ = click_motion_detector.detect_after_click(
                    c, screen_type=screen_type)
                if md:
                    msg = f"Motion after clicking {click_label} @ {c}"
                    last_motion_time = time.time()
                    print(msg)
                    logging.debug(msg)
                    break
            seat_wind = 'NA'
            wind_order = 'NA'
            your_discard_queue.clear_queue()
            all_discards_and_melds_found.clear()
            collective_discards = {}
            ########### end: Next Screen ###########

        elif ad:
            screen_type = 'Ad'
            precise_ad_screen = _a[4].replace('.png', '')

            if _a[4].replace('.png', '') in ["ad_skip_video", "ad_play", "ad_white_x", "google_play_x", "x7", "ad_white_x_black_background"]:
                screen_type = precise_ad_screen

            msg = f"{screen_type} detected"
            print(msg)
            Notifier.notify(msg)
            logging.debug(msg)
            Utils.save_screenshot(frame, prefix=screen_type)
            for click_label in Config.ACTION_SCREEN_CLICK_ORDER[screen_type]:
                c = Config.CLICK_COORDINATES.get(click_label, (0, 0))
                if c == (0, 0):
                    raise IndexError(
                        f"{click_label} key not found in Config.CLICK_COORDINATES")
                md, _, _ = click_motion_detector.detect_after_click(
                    c, screen_type=screen_type)
                if md:
                    msg = f"Motion detected after clicking {click_label} @ {c}"
                    last_motion_time = time.time()
                    print(msg)
                    Notifier.notify(msg)
                    logging.debug(msg)
                    break
        else:
            screen_type = 'Unknown'
            msg = f"{screen_type} screen detected"
            print(msg)
            Notifier.notify(msg)
            logging.debug(msg)

        time.sleep(1 / Config.MAX_SAMPLING_RATE_FPS)

    Utils.save_screenshot(frame, prefix='LastScreenshot_')
    msg = f'Ending script @ {Utils.current_time_to_pst()}'
    print(msg)
    Notifier.notify(msg)
    logging.debug(msg)
