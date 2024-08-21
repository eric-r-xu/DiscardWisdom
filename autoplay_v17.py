import os
import time
from datetime import datetime
import pytz
import cv2
import numpy as np
import pytesseract
import queue
import re
import logging
import pyautogui
from pync import Notifier
from collections import defaultdict
from PIL import Image, ImageFilter
from collections import deque
from threading import Thread, Event
from Levenshtein import distance as levenshtein_distance


class Config:
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M")
    MAX_SAMPLING_RATE_FPS = 120.0
    TIME_LIMIT = 30000
    NO_MOTION_THRESHOLD = 0.1
    NO_MOTION_WARNING = 30
    SCREEN_MATCHING_THRESHOLD = 0.97
    TILE_MATCHING_THRESHOLD = 0.97
    OFFERED_TILE_MATCHING_THRESHOLD = 0.9
    TILE_OVERLAP_THRESHOLD = 10
    CHANGE_THRESHOLD_MELDS_FLOWERS = 20
    TEMPLATE_PATH = '/Users/ericxu/Documents/Jupyter/mahjong/templates'
    SCREENSHOT_PATH = '/Users/ericxu/Documents/Jupyter/mahjong/auto_screenshots'
    WINDS = ['E', 'N', 'W', 'S']

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
        "google_play_x": (1067, 231), "x7": (1106, 203),
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
        "Ad": ["xx", "x2", "x_left", "exit_ad", "x", "close_ad", "exit_ad4", "exit_ad3", "exit_ad2", "ad_close", "ad_skip_video", "ad_play"],
        "Woo": ['accept'],
        "WooChow": ['accept_left'],
        "WooPong": ['accept_left'],
        "ad_skip_video": ["ad_skip_video"], "ad_play": ["ad_play"], "ad_white_x": ["ad_white_x"], "ad_white_x_black_background": ["ad_white_x_black_background"],
        "google_play_x": ["google_play_x"], "x7": ["x7"], "ad_white_x2": ["ad_white_x2"],
    }

    GAME_ACTION_SCREENS = ['YourDiscard', 'Chow', 'Pong', 'Woo', 'WooChow', 'WooPong'
                           'Kong', 'ChowSelection', 'KongPong', 'PongChow', 'Draw']

    MELD_SCREENS = ['Chow', 'Pong', 'Woo', 'WooChow', 'WooPong'
                    'Kong', 'KongPong', 'PongChow']


class Utils:
    @staticmethod
    def clear_queue(q):
        try:
            while not q.empty():
                q.get_nowait()
        except queue.Empty:
            pass

    @staticmethod
    def subtract_prior_dict_from_current_dict(current_dict, prior_dict):
        return {key: current_dict[key] - prior_dict.get(key, 0) for key in current_dict}

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

        for template_path in png_files:
            tg = cv2.imread(os.path.join(
                folder_path, template_path), cv2.IMREAD_GRAYSCALE)
            matched_result = cv2.matchTemplate(fg, tg, cv2.TM_CCOEFF_NORMED)
            match_found = np.any(matched_result >= threshold)

            if match_found:
                yloc, xloc = np.where(matched_result >= threshold)
                msg = f'Match found with {screen_type}/{template_path}'
                print(msg)
                return True, [int(min(xloc)), int(max(xloc)), int(min(yloc)), int(max(yloc)), template_path]

        return False, [0, 0, 0, 0, '']

    @staticmethod
    def is_new_on_screen(frame1, frame2=None, screen_type='small_discards', threshold=Config.TILE_MATCHING_THRESHOLD, overlap_threshold=Config.TILE_OVERLAP_THRESHOLD):

        frame1_tiles = defaultdict(int)
        frame2_tiles = defaultdict(int)

        # Set search boundaries
        xy_search_boundaries = {}
        xy_search_boundaries.update(
            Config.TEMPLATE_BOUNDARY_MAP.get(screen_type, {}))
        x_min = xy_search_boundaries.get("x_min", 0)
        x_max = xy_search_boundaries.get("x_max", frame1.shape[1])
        y_min = xy_search_boundaries.get("y_min", 0)
        y_max = xy_search_boundaries.get("y_max", frame1.shape[0])

        # Crop the frames according to the boundaries
        f1 = frame1[y_min:y_max, x_min:x_max]
        f2 = frame2[y_min:y_max, x_min:x_max]

        # Path to the folder containing templates
        folder_path = os.path.join(Config.TEMPLATE_PATH, screen_type)
        png_files = (f for f in os.listdir(folder_path) if f.endswith('.png'))

        for template_path in png_files:
            # Read the template in color
            template = cv2.imread(os.path.join(
                folder_path, template_path), cv2.IMREAD_COLOR)

            # Perform template matching for each channel separately and then combine results
            channels = cv2.split(template)
            match_found1 = False
            match_found2 = False

            for i, channel_template in enumerate(channels):
                matched_result1 = cv2.matchTemplate(
                    f1[:, :, i], channel_template, cv2.TM_CCOEFF_NORMED)
                matched_result2 = cv2.matchTemplate(
                    f2[:, :, i], channel_template, cv2.TM_CCOEFF_NORMED)

                # Combine results by taking the minimum value across channels (conservative match)
                if i == 0:
                    combined_result1 = matched_result1
                    combined_result2 = matched_result2
                else:
                    combined_result1 = np.minimum(
                        combined_result1, matched_result1)
                    combined_result2 = np.minimum(
                        combined_result2, matched_result2)

            # Find non-overlapping matches
            def find_non_overlapping_matches(matched_result):
                yloc, xloc = np.where(matched_result >= threshold)
                matches = list(zip(yloc, xloc))
                non_overlapping_matches = []

                for match in matches:
                    if not any(np.linalg.norm(np.array(match) - np.array(prev_match)) < overlap_threshold for prev_match in non_overlapping_matches):
                        non_overlapping_matches.append(match)

                return non_overlapping_matches

            non_overlapping_matches1 = find_non_overlapping_matches(
                combined_result1)
            non_overlapping_matches2 = find_non_overlapping_matches(
                combined_result2)

            if non_overlapping_matches1:
                msg = f'Match found with {screen_type}/{template_path}'
                print(msg)
                frame1_tiles[f'{template_path}'] += len(
                    non_overlapping_matches1)

            if non_overlapping_matches2:
                msg = f'Match found with {screen_type}/{template_path}'
                print(msg)
                frame2_tiles[f'{template_path}'] += len(
                    non_overlapping_matches2)

        return frame1_tiles, frame2_tiles

    @staticmethod
    def chars_on_screen(frame, screen_type='DetermineFireGun2'):
        xy_search_boundaries = {}
        xy_search_boundaries.update(
            Config.TEMPLATE_BOUNDARY_MAP.get(screen_type, {}))
        x_min = xy_search_boundaries.get("x_min", 0)
        x_max = xy_search_boundaries.get("x_max", frame.shape[1])
        y_min = xy_search_boundaries.get("y_min", 0)
        y_max = xy_search_boundaries.get("y_max", frame.shape[0])

        f = frame[y_min:y_max, x_min:x_max]

        cropped_image_np = np.array(f)
        if len(cropped_image_np.shape) == 3:
            cropped_image_np = cv2.cvtColor(
                cropped_image_np, cv2.COLOR_RGB2GRAY)

        _, max_val, _, _ = cv2.minMaxLoc(cropped_image_np)

        return max_val >= 200

    @staticmethod
    def save_screenshot(frame, prefix='', text_info=None):
        ms_ts_id = int(time.time() * 1000)
        dir_path = os.path.join(Config.SCREENSHOT_PATH, Config.TIMESTAMP)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f'{prefix}_{ms_ts_id}.png')

        # Overlay text on the frame if text_info is provided
        if text_info:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 0, 255)  # Red color in BGR for the text
            thickness = 2  # Thickness for the main text (red)
            # White color in BGR for the outline
            outline_color = (255, 255, 255)
            # Thickness for the outline (white)
            outline_thickness = thickness + 2
            position = (10, 30)  # Position of the text on the image

            # Draw the outline (white)
            cv2.putText(frame, text_info, position, font, font_scale,
                        outline_color, outline_thickness, cv2.LINE_AA)

            # Draw the text (red) on top of the outline
            cv2.putText(frame, text_info, position, font,
                        font_scale, color, thickness, cv2.LINE_AA)

        cv2.imwrite(path, frame)
        msg = f"{prefix} sshot --> {prefix}_{ms_ts_id}.png"
        print(msg)
        if prefix == 'discardSaveTask_':
            Notifier.notify(msg)

    @staticmethod
    def debug_screenshot(frame, details=''):
        ms_ts_id = int(time.time() * 1000)
        dir_path = os.path.join(Config.SCREENSHOT_PATH, Config.TIMESTAMP)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f'DEBUG_{ms_ts_id}{details}.png')
        cv2.imwrite(path, frame)

    @staticmethod
    def current_time_to_pst():
        utc_dt = datetime.utcfromtimestamp(time.time())
        pst_tz = pytz.timezone('America/Los_Angeles')
        pst_dt = pytz.utc.localize(utc_dt).astimezone(pst_tz)
        return str(pst_dt)[0:19]

    @staticmethod
    def find_offered_meld(frame, screen_type='large_discards', threshold=Config.OFFERED_TILE_MATCHING_THRESHOLD, save_screenshot=True):
        frame1 = frame.copy()
        frame1[0:210, :, :] = 0
        frame1[950:, :, :] = 0
        frame1[1390:, :, :] = 0
        frame1[0:210, :, :] = 0
        frame1[1190:1400, 0:1170, :] = 0

        tile = 'NA'

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

        for template_path in png_files:

            # Read the template in color
            template = cv2.imread(os.path.join(
                folder_path, template_path), cv2.IMREAD_COLOR)

            matched_result = cv2.matchTemplate(
                f1, template, cv2.TM_CCOEFF_NORMED)
            match_found = np.any(matched_result >= threshold)
            if match_found:
                tile = template_path.replace('.png', '')

                return tile
        _text_info = f'offered meld: {tile}'
        Utils.save_screenshot(frame, prefix='OfferedMeld',
                              text_info=_text_info)

        return None

    @staticmethod
    def find_all_discards_and_melds(frame, screen_type='small_discards', threshold=Config.TILE_MATCHING_THRESHOLD, save_screenshot=False):
        all_discards_and_melds_found = defaultdict(int)
        ms_ts_id = int(time.time() * 1000)
        frame1 = frame.copy()

        frame1[0:210, :, :] = 0
        frame1[950:, :, :] = 0
        frame1[1390:, :, :] = 0
        frame1[0:210, :, :] = 0
        frame1[1190:1400, 0:1170, :] = 0

        xy_search_boundaries = {}
        xy_search_boundaries.update(
            Config.TEMPLATE_BOUNDARY_MAP.get(screen_type, {}))
        x_min = xy_search_boundaries.get("x_min", 0)
        x_max = xy_search_boundaries.get("x_max", frame1.shape[1])
        y_min = xy_search_boundaries.get("y_min", 0)
        y_max = xy_search_boundaries.get("y_max", frame1.shape[0])

        # Crop the frames according to the boundaries
        f1 = frame1[y_min:y_max, x_min:x_max]

        # save discard meld screenshot for investigation
        """
        filename = f'{ms_ts_id}_{screen_type}_find_all_discards_and_melds.png'
        output_path = os.path.join(
            Config.SCREENSHOT_PATH, Config.TIMESTAMP, filename)
        cv2.imwrite(output_path, f1)
        msg = f"Screenshot saved @ {output_path}"
        print(msg)
        """

        # Path to the folder containing templates
        folder_path = os.path.join(Config.TEMPLATE_PATH, screen_type)
        png_files = (f for f in os.listdir(folder_path) if f.endswith('.png'))

        for template_path in png_files:
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

                        if save_screenshot:
                            # Get the bounding box of the match
                            y, x = match
                            h, w = template.shape[:2]
                            cropped_image = f1[y:y+h, x:x+w]

                            filename = f'{ms_ts_id}_tile_{tile}_xy_{x}{y}.png'
                            output_path = os.path.join(
                                Config.SCREENSHOT_PATH, Config.TIMESTAMP, filename)

                            # Save the cropped image
                            cv2.imwrite(output_path, cropped_image)
                            print(f"Saved {output_path}")

            if counter > 0:
                all_discards_and_melds_found[tile] += counter

        return all_discards_and_melds_found

    @staticmethod
    def determine_your_discards(frame, screen_type='player_you_discardable', threshold=Config.TILE_MATCHING_THRESHOLD, save_screenshot=True):
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

        for template_path in png_files:
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

                        if save_screenshot:
                            # Get the bounding box of the match
                            y, x = match
                            h, w = template.shape[:2]
                            cropped_image = f1[y:y+h, x:x+w]
                            if screen_type == 'player_you_discardable':
                                if x >= 2132 and x < 2152:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_wall_tile.png'
                                elif x >= 1946 and x < 1978:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D13.png'
                                elif x >= 1811 and x < 1835:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D12.png'
                                elif x >= 1682 and x < 1702:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D11.png'
                                elif x >= 1549 and x < 1569:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D10.png'
                                elif x >= 1410 and x < 1430:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D9.png'
                                elif x >= 1278 and x < 1298:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D8.png'
                                elif x >= 1140 and x < 1160:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D7.png'
                                elif x >= 1002 and x < 1022:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D6.png'
                                elif x >= 864 and x < 890:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D5.png'
                                elif x >= 726 and x < 756:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D4.png'
                                elif x >= 588 and x < 620:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D3.png'
                                elif x >= 450 and x < 480:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D2.png'
                                elif x >= 312 and x < 350:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D1.png'
                                else:
                                    filename = f'{ms_ts_id}_tile_{tile}_x{x}_D_undetermined.png'
                            elif screen_type == 'player_you_melded':
                                if x >= 295 and x < 315:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D1.png'
                                elif x >= 430 and x < 450:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D2.png'
                                elif x >= 565 and x < 585:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D3.png'
                                elif x >= 700 and x < 720:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D4.png'
                                elif x >= 835 and x < 855:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D5.png'
                                elif x >= 970 and x < 990:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D6.png'
                                elif x >= 1105 and x < 1125:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D7.png'
                                elif x >= 1240 and x < 1260:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D8.png'
                                elif x >= 1375 and x < 1395:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D9.png'
                                elif x >= 1510 and x < 1530:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D10.png'
                                elif x >= 1645 and x < 1665:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D11.png'
                                elif x >= 1780 and x < 1800:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D12.png'
                                else:
                                    filename = f'{ms_ts_id}_melded_tile_{tile}_x{x}_D_undetermined.png'

                            output_path = os.path.join(
                                Config.SCREENSHOT_PATH, Config.TIMESTAMP, filename)

                            # Save the cropped image
                            cv2.imwrite(output_path, cropped_image)
                            print(f"Saved {output_path}")

            if counter > 0:
                your_discards[tile] += counter

        _text_info = str(dict(sorted(your_discards.items())))
        Utils.save_screenshot(
            f1, prefix=f'{ms_ts_id}_{screen_type}', text_info=_text_info)

        return your_discards

    @staticmethod
    def highlight_frame_changes(frame1, frame2=None, threshold=Config.TILE_OVERLAP_THRESHOLD):

        ms_ts_id = int(time.time() * 1000)

        def preprocess(frame1):
            frame1[0:210, :, :] = 0
            frame1[950:, :, :] = 0
            frame1[1390:, :, :] = 0
            frame1[0:210, :, :] = 0
            frame1[1190:1400, 0:1170, :] = 0
            return frame1

        frame1 = preprocess(frame1)

        if frame2 is not None:
            frame2 = preprocess(frame2)

        _output_path = f'{Config.SCREENSHOT_PATH}/{Config.TIMESTAMP}/DISCARD_MELD_ANALYSIS_{ms_ts_id}_frame_before.png'
        cv2.imwrite(_output_path, frame1)
        print(f'frame before saved to {_output_path}')

        if frame2 is not None:
            _output_path = f'{Config.SCREENSHOT_PATH}/{Config.TIMESTAMP}/DISCARD_MELD_ANALYSIS_{ms_ts_id}_frame_after.png'
            cv2.imwrite(_output_path, frame2)
            print(f'frame after saved to {_output_path}')

        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        if frame2 is not None:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Compute the absolute difference between the frames
            diff = cv2.absdiff(gray1, gray2)

            # Threshold the difference to highlight changes
            _, diff_thresh = cv2.threshold(
                diff, threshold, 255, cv2.THRESH_BINARY)

            # Find contours of the changes
            contours, _ = cv2.findContours(
                diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Save each contour as a separate image
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                cropped_change = frame2[y:y+h, x:x+w]
                crop_output_path = f'{Config.SCREENSHOT_PATH}/{Config.TIMESTAMP}/CONTOUR_{ms_ts_id}_{i}.png'
                cv2.imwrite(crop_output_path, cropped_change)
                print(f'Cropped change saved to {crop_output_path}')

            # Create a mask from the thresholded difference image
            mask = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2BGR)

            # Highlight the changes by adding a color to the areas where changes occurred
            highlighted_area = np.zeros_like(frame1)
            highlighted_area[:, :] = [0, 0, 255]  # Highlight in red

            # Apply the mask to the highlighted area
            highlighted_changes = cv2.bitwise_and(highlighted_area, mask)

            # Combine the original frame with the highlighted changes
            highlighted_image = cv2.addWeighted(
                frame1, 0.5, highlighted_changes, 0.5, 0)

            # Save the highlighted image
            _output_path = f'{Config.SCREENSHOT_PATH}/{Config.TIMESTAMP}/DISCARD_MELD_ANALYSIS_{ms_ts_id}.png'
            cv2.imwrite(_output_path, highlighted_image)
            print(f'finished saving to {_output_path}')

            '''sd1, sd2 = Utils.is_new_on_screen(
                frame1, frame2, screen_type='small_discards')'''

        return

    @staticmethod
    def save_png_frame_changes(frame1, frame2, threshold=Config.CHANGE_THRESHOLD_MELDS_FLOWERS, frame_type='normal', save_contours=False, text_info=None):
        f1 = frame1.copy()
        f2 = frame2.copy()
        ms_ts_id = int(time.time() * 1000)

        def preprocess(f1, frame_type):
            if frame_type == 'normal':
                f1[1200:, :, :] = 1
                f1[:222, :, :] = 1
                f1[:, 0:160, :] = 1
                f1[:, 2400:, :] = 1
            elif frame_type == 'player_left':
                f1[1200:, :, :] = 1
                f1[:222, :, :] = 1
                f1[:, 375:, :] = 1
                f1[910:, :, :] = 1
                f1[:, :150, :] = 1
            elif frame_type == 'player_right':
                f1[:222, :, :] = 1
                f1[:, :2175, :] = 1
                f1[:, 2400:, :] = 1
                f1[980:, :, :] = 1
            elif frame_type == 'player_across':
                f1[:222, :, :] = 1
                f1[410:, :, :] = 1
                f1[:, 0:450, :] = 1
                f1[:, 2100:, :] = 1
                f1[350:420, 350:1200, :] = 1
            elif frame_type == 'player_you_discardable':
                f1[0:940, :, :] = 1
                f1[1190:, :] = 1

            return f1

        f1 = preprocess(f1, frame_type)

        f2 = preprocess(f2, frame_type)

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)

        # Threshold the difference to highlight changes
        _, diff_thresh = cv2.threshold(
            diff, threshold, 255, cv2.THRESH_BINARY)

        # Find contours of the changes
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Save each contour as a separate image
        if save_contours:
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w >= 30 and h >= 30:     # ensure changes are valid and significant
                    cropped_change = frame2[y:y+h, x:x+w]
                    crop_output_path = f'{Config.SCREENSHOT_PATH}/{Config.TIMESTAMP}/contour_{ms_ts_id}_countour_{frame_type}_{i}.png'
                    cv2.imwrite(crop_output_path, cropped_change)
                    print(
                        f'Cropped change saved to {crop_output_path} with x/y/w/h = {x}/{y}/{w}/{h}')

        if frame_type == 'normal':
            # Create a mask from the thresholded difference image
            mask = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2BGR)

            # Highlight the changes by adding a color to the areas where changes occurred
            highlighted_area = np.zeros_like(f1)
            highlighted_area[:, :] = [0, 0, 255]  # Highlight in red

            # Apply the mask to the highlighted area
            highlighted_changes = cv2.bitwise_and(highlighted_area, mask)

            # Combine the original frame with the highlighted changes
            highlighted_image = cv2.addWeighted(
                f1, 0.5, highlighted_changes, 0.5, 0)

            # Save the highlighted image with text info if provided
            if text_info:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (0, 0, 255)  # Red color in BGR for the text
                thickness = 2  # Thickness for the main text (red)
                # White color in BGR for the outline
                outline_color = (255, 255, 255)
                # Thickness for the outline (white)
                outline_thickness = thickness + 2
                position = (10, 30)  # Position of the text on the image

                # Draw the outline (white)
                cv2.putText(highlighted_image, text_info, position, font,
                            font_scale, outline_color, outline_thickness, cv2.LINE_AA)

                # Draw the text (red) on top of the outline
                cv2.putText(highlighted_image, text_info, position,
                            font, font_scale, color, thickness, cv2.LINE_AA)

            _output_path = f'{Config.SCREENSHOT_PATH}/{Config.TIMESTAMP}/{ms_ts_id}_highlighted_delta.png'
            cv2.imwrite(_output_path, highlighted_image)
            print(f'finished saving to {_output_path}')

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
                # Notifier.notify(msg)

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

    def save_screenshot(self, frame):
        ms_ts_id = int(time.time() * 1000)
        path = os.path.join(
            f'{Config.SCREENSHOT_PATH}/{Config.TIMESTAMP}/DAEMON_{ms_ts_id}.png')
        cv2.imwrite(path, frame)
        msg = f"Screenshot saved @ DAEMON_{ms_ts_id}.png"
        print(msg)


class MotionDetector:
    def __init__(self, threshold=20):
        self.threshold = threshold

    def detect(self, frame1, frame2, screen_type='WallCount'):
        f1, f2 = frame1.copy(), frame2.copy()

        if screen_type == 'WallCount':
            f1[:1190, :] = f2[:1190, :] = 0
            f1[:, 1450:] = f2[:, 1450:] = 0
            f1[1190:1400, :1150] = f2[1190:1400, :1150] = 0

            f1[1390:, :, :] = f2[1390:, :, :] = 0
            f1[0:210, :, :] = f2[0:210, :, :] = 0
            f1[1190:1400, 0:1170, :] = f2[1190:1400, 0:1170, :] = 0
        elif screen_type == 'GameScreenAction':
            self.threshold = 50
            f1[0:210, :, :] = f2[0:210, :, :] = 0
            f1[950:, :, :] = f2[950:, :, :] = 0

            f1[1390:, :, :] = f2[1390:, :, :] = 0
            f1[0:210, :, :] = f2[0:210, :, :] = 0
            f1[1190:1400, 0:1170, :] = f2[1190:1400, 0:1170, :] = 0
        else:
            self.threshold = 50
            f1[1200:], f1[:210] = 0, 0
            f2[1200:], f2[:210] = 0, 0

        diff = cv2.absdiff(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))
        _, diff_thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours) > 0, f1, f2


class ClickMotionDetector(MotionDetector):
    def detect_after_click(self, location, screen_type='GameScreenAction'):
        frame1 = ScreenshotCapturer.capture()
        pyautogui.moveTo(location)
        print(f"Moved mouse to {location}")
        pyautogui.click()
        frame2 = ScreenshotCapturer.capture()
        return self.detect(frame1, frame2, screen_type=screen_type)


# Main script
if __name__ == "__main__":
    msg = f'Starting script @ {Utils.current_time_to_pst()}'
    print(msg)
    Notifier.notify(msg)
    start_time = time.time()

    motion_detector = MotionDetector()

    your_discard_queue = GameFrameQueue()
    frame_queue = GameFrameQueue()

    click_motion_detector = ClickMotionDetector()
    last_motion_time = time.time()

    no_motion_before = False
    seat_wind = 'NA'
    wind_order = 'NA'

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
            Notifier.notify(msg)
            if not no_motion_before:

                # Utils.save_screenshot(frame, prefix='NoMotion_')
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
            Notifier.notify(msg, timeout=2)
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
                    seat_wind = letter
                elif letter not in Config.WINDS:
                    msg = f"WARNING: Invalid Seat wind found: {letter}"
                    print(msg)
                    Notifier.notify(msg)
                else:
                    msg = f"WARNING: Seat wind found"
                    print(msg)
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
                    # Utils.save_screenshot(game_frame, prefix=screen_type)

                    if screen_type == 'YourDiscard':
                        your_discard_queue.enqueue(game_frame)
                        all_new_discards_and_melds_found = Utils.find_all_discards_and_melds(
                            game_frame, screen_type='small_discards')
                        sorted_dict = dict(
                            sorted(all_new_discards_and_melds_found.items()))

                        ########### start: find your discardeable tiles and positions ###########
                        # ydt = your discardable tiles
                        ydt = Utils.determine_your_discards(
                            game_frame, save_screenshot=True)
                        sorted_ydt = dict(sorted(ydt.items()))
                        sorted_filtered_ydt = {
                            key: value for key, value in sorted_ydt.items() if value != 0}
                        msg = f'your discardeable tiles = {sorted_filtered_ydt}'
                        print(msg)
                        ########### end: find your discardeable tiles and positions ###########

                        ########### start: find your melded tiles and positions ###########
                        # ymt = your melded tiles
                        ymt = Utils.determine_your_discards(
                            game_frame, screen_type='player_you_melded', save_screenshot=True)
                        sorted_ymt = dict(sorted(ymt.items()))
                        sorted_filtered_ymt = {
                            key: value for key, value in sorted_ymt.items() if value != 0}
                        msg = f'your melded tiles = {sorted_filtered_ymt}'
                        print(msg)
                        ########### end: find your melded tiles and positions ###########

                        if your_discard_queue.length() > 1:
                            previous_game_frame = your_discard_queue[0]
                            previous_all_discards_and_melds_found = Utils.find_all_discards_and_melds(
                                previous_game_frame, screen_type='small_discards')
                            sorted_dict = Utils.subtract_prior_dict_from_current_dict(
                                all_new_discards_and_melds_found, previous_all_discards_and_melds_found)
                            sorted_dict = dict(sorted(sorted_dict.items()))

                            filtered_dict = {
                                key: value for key, value in sorted_dict.items() if value != 0}
                            if filtered_dict == {}:
                                Utils.debug_screenshot(
                                    game_frame, details='_no_changes_found')
                            msg = f'new discards/melds found: {filtered_dict}'
                            print(msg)
                            Utils.save_png_frame_changes(
                                previous_game_frame, game_frame, frame_type='normal', save_contours=False, text_info=str(filtered_dict))
                            Utils.save_png_frame_changes(
                                previous_game_frame, game_frame, frame_type='player_left', save_contours=True)
                            Utils.save_png_frame_changes(
                                previous_game_frame, game_frame, frame_type='player_right', save_contours=True)
                            Utils.save_png_frame_changes(
                                previous_game_frame, game_frame, frame_type='player_across', save_contours=True)

                        # Notifier.notify(msg)
                        # time.sleep(3)
                    elif screen_type in Config.MELD_SCREENS:
                        offered_meld = None
                        offered_meld = Utils.find_offered_meld(game_frame)
                        if offered_meld:
                            msg = f'Offered meld: {offered_meld}'
                        else:
                            msg = f'ERROR: offered meld undetected'
                            Utils.debug_screenshot(
                                game_frame, details='_offered_meld_undetected')
                        print(msg)
                        # Notifier.notify(msg)
                        # time.sleep(3)

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
                            # Notifier.notify(msg)
                            break
        ########### end: Game Screen ###########

        ########### start: Next Screen ###########
        elif next_game:
            screen_type = 'NextGame'
            msg = f"next game detected"
            print(msg)
            Notifier.notify(msg)

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
            msg = f'winner wind = {winner_wind}'
            print(msg)
            Notifier.notify(msg)
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
                    details += f'_{total_fan_number}Fan'
                    break
                else:
                    print('WARNING: Total Fan not found, retrying...')
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
                        details += '_FIREGUN'

            msg = f'details = {details}'
            # Notifier.notify(msg)
            # time.sleep(2)

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
                    break
            seat_wind = 'NA'
            wind_order = 'NA'
            your_discard_queue.clear_queue()
            ########### end: Next Screen ###########
        elif ad:
            screen_type = 'Ad'
            precise_ad_screen = _a[4].replace('.png', '')

            if _a[4].replace('.png', '') in ["ad_skip_video", "ad_play", "ad_white_x", "google_play_x", "x7", "ad_white_x_black_background"]:
                screen_type = precise_ad_screen

            msg = f"{screen_type} detected"
            print(msg)
            Notifier.notify(msg)
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
                    break
        else:
            screen_type = 'Unknown'
            msg = f"{screen_type} screen detected"
            print(msg)
            Notifier.notify(msg)

        time.sleep(1 / Config.MAX_SAMPLING_RATE_FPS)

    Utils.save_screenshot(frame, prefix='LastScreenshot_')
    msg = f'Ending script @ {Utils.current_time_to_pst()}'
    print(msg)
    Notifier.notify(msg)
