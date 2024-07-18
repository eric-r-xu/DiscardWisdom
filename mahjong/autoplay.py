import os
import random
import time
from datetime import datetime

import cv2
import numpy as np
import pyautogui


def shuffle_dict_with_fixed_first(d, fixed_key):
    # Ensure the fixed_key is in the dictionary
    if fixed_key not in d:
        raise KeyError(f"The key {fixed_key} is not in the dictionary.")
    fixed_item = {fixed_key: d[fixed_key]}
    remaining_items = list(d.items())
    remaining_items.remove((fixed_key, d[fixed_key]))
    random.shuffle(remaining_items)
    shuffled_dict = {**fixed_item, **dict(remaining_items)}

    return shuffled_dict


# Holds constants and configuration settings.
class Config:
    NO_MOTION_SEC_THRESHOLD = 0.6
    NO_MOTION_CLICK_THRESHOLD = 0.2
    SAMPLING_RATE_FPS = 10
    MINUTES_BEFORE_STOPPING = 100
    COORDINATES = {
        "D1": (259, 683),
        "D2": (335, 683),
        "D3": (411, 683),
        "D4": (487, 683),
        "D5": (563, 683),
        "D6": (639, 683),
        "D7": (715, 683),
        "D8": (791, 683),
        "D9": (867, 683),
        "D10": (943, 683),
        "D11": (1019, 683),
        "D12": (1095, 683),
        "D13": (1171, 683),
        "wall_tile": (1279, 680),
        "next_game": (1072, 790),
        "accept": (1288, 567),
        "reject": (1396, 561),
        "exit_ad": (1447, 241),
        "draw_game": (742, 568),
        "unpause": (653, 748),
        "exit_ad2": (1443, 249),
    }


# Encapsulates screenshot capturing functionality.
class ScreenshotCapturer:
    @staticmethod
    def capture():
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)


# Handles motion detection between two frames.
class MotionDetector:
    def __init__(self, threshold=30):
        self.threshold = threshold

    def detect(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, diff_thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return len(contours) > 0


# Inherits from MotionDetector and adds functionality for detecting motion after clicking a specified location.
class ClickMotionDetector(MotionDetector):
    def detect_after_click(self, frame2, location):
        pyautogui.moveTo(location)
        pyautogui.click()
        # print(f"Clicked at location: {location}")
        time.sleep(Config.NO_MOTION_CLICK_THRESHOLD)
        frame3 = ScreenshotCapturer.capture()
        return self.detect(frame2, frame3)


# Orchestrates the main logic, managing the state, and handles directory creation and screenshot saving.
class MotionDetectionScript:
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.click_motion_detector = ClickMotionDetector()
        self.start_time = time.time()
        self.no_motion_start_time = None
        self.screenshot_dir = self.create_screenshot_directory()

    def create_screenshot_directory(self):
        # create a directory of format `YYYYMMDDHHMM`
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        dir_path = os.path.join("auto_screenshots", timestamp)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def save_screenshot(self, frame, filename):
        full_path = os.path.join(self.screenshot_dir, filename)
        cv2.imwrite(full_path, frame)
        print(f"Screenshot saved as {full_path}")

    def run(self):
        frame1 = ScreenshotCapturer.capture()
        time.sleep(1 / Config.SAMPLING_RATE_FPS)

        while True:
            frame2 = ScreenshotCapturer.capture()
            motion_detected = self.motion_detector.detect(frame1, frame2)

            if motion_detected:
                self.no_motion_start_time = None
            else:
                if self.no_motion_start_time is None:
                    self.no_motion_start_time = time.time()
                elif (
                    time.time() - self.no_motion_start_time
                    >= Config.NO_MOTION_SEC_THRESHOLD
                ):
                    filename = f"no_motion_screenshot_{int(time.time())}.png"
                    self.save_screenshot(frame2, filename)

                    shuffled_dict = shuffle_dict_with_fixed_first(
                        Config.COORDINATES, "exit_ad2"
                    )
                    for tile_name, coordinates in shuffled_dict.items():
                        if self.click_motion_detector.detect_after_click(
                            frame2, coordinates
                        ):
                            print(f"Motion detected after clicking {tile_name}")
                            break
                    time.sleep(1 / Config.SAMPLING_RATE_FPS)
                    self.no_motion_start_time = None

            frame1 = frame2

            if time.time() - self.start_time >= Config.MINUTES_BEFORE_STOPPING * 60:
                print(
                    f"{Config.MINUTES_BEFORE_STOPPING} minutes have passed. Stopping the script."
                )
                break

            time.sleep(1 / Config.SAMPLING_RATE_FPS)


if __name__ == "__main__":
    script = MotionDetectionScript()
    script.run()
