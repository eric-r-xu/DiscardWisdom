import os
import sys
import time
from datetime import datetime
import random
import cv2
import numpy as np
import pyautogui
from pync import Notifier
from collections import deque, OrderedDict, defaultdict


# Flow Schematic #################################################################
#     https://github.com/eric-r-xu/DiscardWisdom/blob/main/HKMJ%20Decision%20Tree.png


class Config:
    SAMPLING_RATE_FPS = 100.            # (frames/sec)
    TIME_LIMIT = 30000                  # seconds
    NO_MOTION_THRESHOLD = 0.5           # seconds
    NO_MOTION_CLICK_THRESHOLD = 0.1     # seconds
    CLICK_COORDINATES = {
        "reject": (1220, 430),
        "wall_tile": (1100, 540),
        "D13": (980, 540),
        "D12": (915, 540),
        "D11": (850, 540),
        "D10": (785, 540),
        "D9": (720, 540),
        "D8": (655, 540),
        "D7": (590, 540),
        "D6": (525, 540),
        "D5": (460, 540),
        "D4": (395, 540),
        "D3": (330, 540),
        "D2": (265, 540),
        "D1": (200, 540),
        "accept": (1100, 430),
        "x": (1118, 200),
        "close_ad": (1198, 147),
        "exit_ad": (1205, 191),
        "exit_ad2": (1202, 197),
        "exit_ad3": (1243, 141),
        "exit_ad4": (668, 604),
        "next_game": (893, 626),
        "draw_game": (618, 450),
    }
    YOUR_TURN_CLICK_COORDINATES, OTHER_CLICK_COORDINATES = OrderedDict(), OrderedDict()
    YOUR_TURN_CLICK_ORDER = ['reject', 'wall_tile', 'D13', 'D12', 'D11',
                             'D10', 'D9', 'D8', 'D7', 'D6', 'D5', 'D4', 'D3', 'D2', 'D1', 'accept']
    for k in YOUR_TURN_CLICK_ORDER:
        YOUR_TURN_CLICK_COORDINATES[k] = CLICK_COORDINATES[k]
    OTHER_CLICK_ORDER = ['x', 'close_ad', 'exit_ad', 'exit_ad2',
                         'exit_ad3', 'exit_ad4', 'next_game', 'draw_game']
    OTHER_CLICK_ORDER = OTHER_CLICK_ORDER + YOUR_TURN_CLICK_ORDER
    for k in OTHER_CLICK_ORDER:
        OTHER_CLICK_COORDINATES[k] = CLICK_COORDINATES[k]
    TIMESTAMP = str(datetime.now().strftime("%Y%m%d%H%M"))
    YOUR_TURN_TILES = set(['YourDiscard', 'Kong', 'Pong',
                          'Chow', 'KongPong', 'PongChow', 'ChowSelection'])


class Utils:
    @staticmethod
    def is_screen(
            frame,
            type='GameScreen',
            threshold=0.97,
            xy_search_boundaries={}):

        # SCREEN
        if type == 'GameScreen':
            xy_search_boundaries = {"x_min": 1140, "y_min": 1260}
        elif type == 'NextGame':
            xy_search_boundaries = {"x_min": 1720, "y_min": 1220}
        elif type in ['Pong', 'Kong', 'Chow']:
            xy_search_boundaries = {"x_min": 1930, "y_min": 820}
        elif type == 'YourDiscard':
            xy_search_boundaries = {"x_min": 2000, "y_min": 820}
        elif type in ['KongPong', 'PongChow']:
            xy_search_boundaries = {"x_min": 1750, "y_min": 820}
        elif type == 'NotFullScreen':
            xy_search_boundaries = {"x_max": 250, "y_max": 100}
        else:
            pass

        x_min_boundary = xy_search_boundaries.get("x_min", 0)
        x_max_boundary = xy_search_boundaries.get("x_max", frame.shape[1])
        y_min_boundary = xy_search_boundaries.get("y_min", 0)
        y_max_boundary = xy_search_boundaries.get("y_max", frame.shape[0])

        f = frame[y_min_boundary:y_max_boundary, x_min_boundary:x_max_boundary]
        fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        # TEMPLATE
        folder_path = f'/Users/ericxu/Documents/Jupyter/mahjong/templates/{type}/'
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

        x_min_match, x_max_match, y_min_match, y_max_match = 0, 0, 0, 0
        match_found = 0

        for template_path in png_files:

            tg = cv2.imread(folder_path+template_path, cv2.IMREAD_GRAYSCALE)

            matched_result = cv2.matchTemplate(fg, tg, cv2.TM_CCOEFF_NORMED)

            matched_result_locations = np.where(matched_result >= threshold)

            match_found = int(len(matched_result_locations[0]) > 0)

            # if match found
            if match_found == 1:
                print(f'match found w/ {template_path}')
                x_min_match, x_max_match, y_min_match, y_max_match = min(matched_result_locations[1]), max(
                    matched_result_locations[1]), min(matched_result_locations[0]), max(matched_result_locations[0])

                return bool(match_found), [int(x_min_match), int(x_max_match), int(y_min_match), int(y_max_match)]

        return bool(match_found), [int(x_min_match), int(x_max_match), int(y_min_match), int(y_max_match)]

    @staticmethod
    def save_screenshot(frame, prefix=''):
        ms_ts_id = int(time.time() * 1000)   # millisecond timestamp id
        path = f'/Users/ericxu/Documents/Jupyter/mahjong/auto_screenshots/{Config.TIMESTAMP}/{prefix}_{ms_ts_id}.png'
        cv2.imwrite(path, frame)
        msg = f"{prefix} Screenshot saved @ {ms_ts_id}.png"
        print(msg)


class ScreenshotCapturer:
    @staticmethod
    def capture():
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)


class GameFrameQueue:
    def __init__(self):
        self.queue = deque(maxlen=2)

    def enqueue(self, frame):
        if isinstance(frame, np.ndarray):
            self.queue.append(frame)
        else:
            raise ValueError("Only numpy matrices are allowed.")

    def length(self):
        return len(self.queue)

    def __getitem__(self, index):
        if isinstance(index, int):
            # Allow negative indexing
            if index < 0:
                index += len(self.queue)
            if index < 0 or index >= len(self.queue):
                raise IndexError("Index out of range")
            return self.queue[index]
        else:
            raise TypeError("Index must be an integer")

    def save_screenshots(self, game_frame_queue):
        ms_ts_id = int(time.time() * 1000)   # millisecond timestamp id
        for i, suffix in enumerate(['T1', 'T2']):
            path = f'/Users/ericxu/Documents/Jupyter/mahjong/GameScreenScreenshots/{ms_ts_id}_{suffix}.png'
            cv2.imwrite(path, game_frame_queue[i])
            msg = f"Screenshot saved @ {ms_ts_id}_{suffix}.png"
            print(msg)
            if i == 1:
                path = f'/Users/ericxu/Documents/Jupyter/mahjong/auto_screenshots/{Config.TIMESTAMP}/{ms_ts_id}.png'
                cv2.imwrite(path, game_frame_queue[i])
                msg = f"Screenshot saved @ {ms_ts_id}.png"
                print(msg)


class MotionDetector:
    def __init__(self, threshold=40):
        self.threshold = threshold

    def detect(self, frame1, frame2, type='GameScreen'):

        f1 = frame1.copy()
        f2 = frame2.copy()

        if type == 'GameScreen':
            f2[0:1190, :, :] = 0
            f2[:, 1450:, :] = 0
            f2[1190:1400, 0:1150, :] = 0

            f1[0:1190, :, :] = 0
            f1[:, 1450:, :] = 0
            f1[1190:1400, 0:1150, :] = 0
        elif type == 'other':
            f1[1200:, :, :] = 0
            f1[0:210, :, :] = 0
            f2[1200:, :, :] = 0
            f2[0:210, :, :] = 0

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, diff_thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return len(contours) > 0


class ClickMotionDetector(MotionDetector):
    def detect_after_click(self, location, type='GameScreen'):
        frame1 = ScreenshotCapturer.capture()
        pyautogui.moveTo(location)
        msg = f"Moved mouse to {location}"
        pyautogui.click()
        time.sleep(Config.NO_MOTION_CLICK_THRESHOLD)
        frame2 = ScreenshotCapturer.capture()
        return self.detect(frame1, frame2, type=type)


# Record the start time
start_time = time.time()
msg = f'Start time = {start_time}\n'
print(msg)

# make directory for screenshots
dir_path = os.path.join("auto_screenshots", Config.TIMESTAMP)
os.makedirs(dir_path, exist_ok=True)


game_frame_queue = GameFrameQueue()
non_game_frame_queue = GameFrameQueue()
motion_detector = MotionDetector()

last_motion_time = time.time()
motion_detected = False

click_motion_detector = ClickMotionDetector()


# keeps running Config.TIME_LIMIT
while (time.time() - start_time) < Config.TIME_LIMIT:
    frame = ScreenshotCapturer.capture()
    is_game_frame, match_locations = Utils.is_screen(frame, type='GameScreen')
    msg = f'is_game_frame / match_locations = {is_game_frame} / {match_locations} \n'
    is_not_full_screen_frame, is_not_full_screen_frame_locations = Utils.is_screen(
        frame, type='NotFullScreen')
    if is_game_frame:
        Notifier.notify('game screen detected')
        game_frame_queue.enqueue(frame)
        print(msg)
        if game_frame_queue.length() > 1:
            # detect motion
            if motion_detector.detect(game_frame_queue[0], game_frame_queue[1]):
                game_frame_queue.save_screenshots(game_frame_queue)
                last_motion_time = time.time()
                motion_detected = True
            else:
                motion_detected = False

        your_turn_map = defaultdict(str)

        for tile in Config.YOUR_TURN_TILES:
            found, loc = Utils.is_screen(frame, type=tile)
            your_turn_map[tile] = [found, loc]
            if found:
                Utils.save_screenshot(frame, prefix=f'{tile}_')
                break

        results = [v[0] for k, v in your_turn_map.items()]
        if max(results) == True:
            Notifier.notify('your turn detected')

            print(results)
            your_turn_map.clear()
            for t, c in Config.YOUR_TURN_CLICK_COORDINATES.items():
                if click_motion_detector.detect_after_click(c, type='other'):
                    msg = f"Motion detected after clicking {t} at {c}"
                    Notifier.notify(msg)
                    break

        # Check if no motion
        if not motion_detected and (time.time() - last_motion_time) > Config.NO_MOTION_THRESHOLD:
            print(
                f"No motion detected for {Config.NO_MOTION_THRESHOLD} seconds.")
    elif is_not_full_screen_frame:
        print(
            f'is_not_full_screen_frame_locations = {is_not_full_screen_frame_locations}')
        Notifier.notify('full screen not detected')
        time.sleep(5)
    else:
        non_game_frame_queue.enqueue(frame)
        is_next_game_frame, match_locations = Utils.is_screen(
            frame, type='NextGame')
        if is_next_game_frame:
            Utils.save_screenshot(frame, prefix='NextGame_')
            Notifier.notify('next game screen detected')
        # assume ad
        else:
            Utils.save_screenshot(frame, prefix='MaybeAd_')
            Notifier.notify('maybe ad screen detected')

        if non_game_frame_queue.length() > 1:
            # detect motion
            if motion_detector.detect(non_game_frame_queue[0], non_game_frame_queue[1], type='other'):
                last_motion_time = time.time()
                motion_detected = True
            else:
                motion_detected = False
        # Check if no motion
        if not motion_detected and (time.time() - last_motion_time) > Config.NO_MOTION_THRESHOLD:
            msg = f"No motion detected for {Config.NO_MOTION_THRESHOLD} seconds."
            print(msg)
            for t, c in Config.OTHER_CLICK_COORDINATES.items():
                if click_motion_detector.detect_after_click(c, type='other'):
                    msg = f"Motion detected after clicking {t} at {c}"
                    Notifier.notify(msg)
                    break

    # frames per second sampling rate
    time.sleep(1 / Config.SAMPLING_RATE_FPS)


msg = 'Ending script'
print(msg)
Notifier.notify(msg)
