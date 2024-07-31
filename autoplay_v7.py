
import os
import sys
import time
from datetime import datetime
from io import StringIO
import random
import cv2
import numpy as np
import pyautogui
from pync import Notifier
from collections import deque


# Flow Schematic #################################################################
#     https://github.com/eric-r-xu/DiscardWisdom/blob/main/HKMJ%20Decision%20Tree.png


class Config:
    SAMPLING_RATE_FPS = 100.    # (frames/sec)
    TIME_LIMIT = 30000          # seconds


class Utils:
    @staticmethod
    def is_game_screen(
            screen_frame,
            threshold=0.97,
            xy_search_boundaries={"x_min_boundary": 1140, "y_min_boundary": 1260}):
        game_screen_template_path = '/Users/ericxu/Documents/Jupyter/mahjong/templates/GameScreen/1.png'
        game_screen_template = cv2.imread(
            game_screen_template_path, cv2.IMREAD_GRAYSCALE)

        x_min_boundary = xy_search_boundaries.get("x_min_boundary", 0)
        x_max_boundary = xy_search_boundaries.get(
            "x_max_boundary", screen_frame.shape[1])
        y_min_boundary = xy_search_boundaries.get("y_min_boundary", 0)
        y_max_boundary = xy_search_boundaries.get(
            "y_max_boundary", screen_frame.shape[0])

        screen_frame = screen_frame[y_min_boundary:y_max_boundary,
                                    x_min_boundary:x_max_boundary]
        screen_frame_gray = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2GRAY)

        # use cv2's matchTemplate method to find game screen template within screen_frame_gray
        matched_result = cv2.matchTemplate(
            screen_frame_gray, game_screen_template, cv2.TM_CCOEFF_NORMED)

        x_min_match, x_max_match, y_min_match, y_max_match = 0, 0, 0, 0

        matched_result_locations = np.where(matched_result >= threshold)

        # `match_found` is an integer boolean variable
        #     indicating whether match is found (1) or not (0) at `threshold`
        match_found = int(len(matched_result_locations[0]) > 0)

        # if match found
        if match_found == 1:
            x_min_match, x_max_match, y_min_match, y_max_match = min(matched_result_locations[1]), max(
                matched_result_locations[1]), min(matched_result_locations[0]), max(matched_result_locations[0])

        return bool(match_found), [int(x_min_match), int(x_max_match), int(y_min_match), int(y_max_match)]

    @staticmethod
    def is_next_game_screen(
            screen_frame,
            threshold=0.97,
            xy_search_boundaries={"x_min_boundary": 1720, "y_min_boundary": 1220}):
        your_turn_template_dir = '/Users/ericxu/Documents/Jupyter/mahjong/templates/NextGame/1.png'
        game_screen_template = cv2.imread(
            game_screen_template_path, cv2.IMREAD_GRAYSCALE)

        x_min_boundary = xy_search_boundaries.get("x_min_boundary", 0)
        x_max_boundary = xy_search_boundaries.get(
            "x_max_boundary", screen_frame.shape[1])
        y_min_boundary = xy_search_boundaries.get("y_min_boundary", 0)
        y_max_boundary = xy_search_boundaries.get(
            "y_max_boundary", screen_frame.shape[0])

        screen_frame = screen_frame[y_min_boundary:y_max_boundary,
                                    x_min_boundary:x_max_boundary]
        screen_frame_gray = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2GRAY)

        # use cv2's matchTemplate method to find game screen template within screen_frame_gray
        matched_result = cv2.matchTemplate(
            screen_frame_gray, game_screen_template, cv2.TM_CCOEFF_NORMED)

        x_min_match, x_max_match, y_min_match, y_max_match = 0, 0, 0, 0

        matched_result_locations = np.where(matched_result >= threshold)

        # `match_found` is an integer boolean variable
        #     indicating whether match is found (1) or not (0) at `threshold`
        match_found = int(len(matched_result_locations[0]) > 0)

        # if match found
        if match_found == 1:
            x_min_match, x_max_match, y_min_match, y_max_match = min(matched_result_locations[1]), max(
                matched_result_locations[1]), min(matched_result_locations[0]), max(matched_result_locations[0])

        return bool(match_found), [int(x_min_match), int(x_max_match), int(y_min_match), int(y_max_match)]



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

    def save_screenshots(self, game_frame_queue, dir_path):
        ms_ts_id = int(time.time() * 1000)   # millisecond timestamp id
        for i, suffix in enumerate(['T1', 'T2']):
            path = f'/Users/ericxu/Documents/Jupyter/mahjong/GameScreenScreenshots/{ms_ts_id}_{suffix}.png'
            cv2.imwrite(path, game_frame_queue[i])
            msg = f"Screenshot saved @ {ms_ts_id}_{suffix}.png"
            print(msg)
            if i == 1:
                path = f'/Users/ericxu/Documents/Jupyter/mahjong/{dir_path}/{ms_ts_id}.png'
                cv2.imwrite(path, game_frame_queue[i])
                msg = f"Screenshot saved @ {ms_ts_id}.png"
                print(msg)
                # here!
    def save_screenshot(self, frame, dir_path, prefix=''):
        ms_ts_id = int(time.time() * 1000)   # millisecond timestamp id
        path = f'/Users/ericxu/Documents/Jupyter/mahjong/{dir_path}/{prefix}_{ms_ts_id}.png'
        cv2.imwrite(path, frame)
        msg = f"{prefix} Screenshot saved @ {ms_ts_id}.png"
        print(msg)


class MotionDetector:
    def __init__(self, threshold=40):
        self.threshold = threshold

    def detect(self, frame1, frame2, type='game'):

        f1 = frame1.copy()
        f2 = frame2.copy()

        if type == 'game':
            f2[0:1190, :, :] = 0
            f2[:, 1450:, :] = 0
            f2[1190:1400, 0:1150, :] = 0

            f1[0:1190, :, :] = 0
            f1[:, 1450:, :] = 0
            f1[1190:1400, 0:1150, :] = 0

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, diff_thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return len(contours) > 0


# Record the start time
start_time = time.time()
msg = f'Start time = {start_time}\n'
print(msg)

# make directory for screenshots
timestamp = datetime.now().strftime("%Y%m%d%H%M")
dir_path = os.path.join("auto_screenshots", timestamp)
os.makedirs(dir_path, exist_ok=True)


# instantiate game frame queue with max len 2
game_frame_queue = GameFrameQueue()

motion_detector = MotionDetector()

# keeps running Config.TIME_LIMIT
while (time.time() - start_time) < Config.TIME_LIMIT:
    frame = ScreenshotCapturer.capture()
    is_game_frame, match_locations = Utils.is_game_screen(frame)
    msg = f'is_game_frame / match_locations = {is_game_frame} / {match_locations} \n'

    if is_game_frame:
        game_frame_queue.enqueue(frame)

        # detect motion
        if game_frame_queue.length() > 1:
            if motion_detector.detect(game_frame_queue[0], game_frame_queue[1]):
                game_frame_queue.save_screenshots(game_frame_queue, dir_path)

    else:
        is_next_game_frame, match_locations = Utils.is_next_game_screen(frame)
        if is_next_game:

        


    # frames per second sampling rate
    time.sleep(1 / Config.SAMPLING_RATE_FPS)


msg = 'Ending script'
print(msg)
Notifier.notify(msg)
