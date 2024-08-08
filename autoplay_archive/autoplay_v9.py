import os
import time
from datetime import datetime
import pytz
import cv2
import numpy as np
import pytesseract
import re
import pyautogui
from pync import Notifier
from collections import deque, OrderedDict, defaultdict
from PIL import UnidentifiedImageError
from threading import Thread, Event


class Config:
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M")
    SCREENSHOT_SAMPLING_RATE_FPS = 200.0
    SAMPLING_RATE_FPS = 100.0
    TIME_LIMIT = 30000
    NO_MOTION_THRESHOLD = 0.4
    NO_MOTION_WARNING = 5
    NO_MOTION_CLICK_THRESHOLD = 0.001
    SCREEN_MATCHING_THRESHOLD = 0.97
    TEMPLATE_PATH = '/Users/ericxu/Documents/Jupyter/mahjong/templates'
    SCREENSHOT_PATH = '/Users/ericxu/Documents/Jupyter/mahjong/auto_screenshots'

    TEMPLATE_BOUNDARY_MAP = {
        'GameScreen': {"x_min": 1140, "y_min": 1260},
        'NextGame': {"x_min": 1720, "y_min": 1220},
        'Pong': {"x_min": 1930, "y_min": 820},
        'Kong': {"x_min": 1930, "y_min": 820},
        'Woo': {"x_min": 2100, "y_min": 810},
        'Chow': {"x_min": 1930, "y_min": 820},
        "ChowSelection": {"y_min": 800, "y_max": 1050},
        'YourDiscard': {"x_min": 2000, "y_min": 820},
        'KongPong': {"x_min": 1750, "y_min": 820},
        'PongChow': {"x_min": 1750, "y_min": 820},
        'NotFullScreen': {"x_max": 250, "y_max": 100},
        "DetermineWinner": {"x_min": 0, "y_min": 400, "x_max": 1250, "y_max": 800},
        "DetermineSelfPick": {"x_min": 1650, "y_min": 400, "y_max": 1200},
        "DetermineWinnerFan": {"x_min": 1650, "y_min": 400, "y_max": 1200},
        "DetermineFiregun": {}
    }

    # Based on mac 13" screen resolution: 1280x800
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
        "x_left": (114, 168), "ad_close": (632, 437),
    }

    ACTION_SCREEN_CLICK_ORDER = {
        "NextGame": ['next_game'],
        "Draw": ['draw_game'],
        "YourDiscard": ["wall_tile", "D13", "D12", "D11", "D10", "D9", "D8", "D7", "D6", "D5", "D4", "D3", "D2", "D1"],
        "Chow": ['accept', 'cancel'],
        "ChowSelection": ["wall_tile", "D13", "D12", "D11", "D10", "D9", "D8", "D7", "D6", "D5", "D4", "D3", "D2", "D1"],
        "Pong": ['accept', 'cancel'],
        "Kong": ['accept', 'cancel'],
        "PongChow": ['accept_left', 'accept', 'cancel'],
        "KongPong": ['accept_left', 'accept', 'cancel'],
        "Ad": ["x2", "x_left", "exit_ad", "x", "close_ad", "exit_ad4", "exit_ad3", "exit_ad2", "ad_close"],
        "Woo": ['accept', 'cancel']
    }

    GAME_ACTION_SCREENS = ['YourDiscard', 'Chow', 'Pong', 'Woo',
                           'Kong', 'ChowSelection', 'KongPong', 'PongChow', 'Draw']

    NEXT_GAME_CHILDREN = ["DetermineSelfPick", "DetermineFiregun"]



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

        for template_path in png_files:
            tg = cv2.imread(os.path.join(
                folder_path, template_path), cv2.IMREAD_GRAYSCALE)
            matched_result = cv2.matchTemplate(fg, tg, cv2.TM_CCOEFF_NORMED)
            match_found = np.any(matched_result >= threshold)

            if match_found:
                yloc, xloc = np.where(matched_result >= threshold)
                print(f'Match found with {screen_type}/{template_path}')
                return True, [int(min(xloc)), int(max(xloc)), int(min(yloc)), int(max(yloc)), template_path]

        return False, [0, 0, 0, 0, '']

    @staticmethod
    def save_screenshot(frame, prefix=''):
        ms_ts_id = int(time.time() * 1000)  # millisecond timestamp id
        dir_path = os.path.join(Config.SCREENSHOT_PATH, Config.TIMESTAMP)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f'{prefix}_{ms_ts_id}.png')
        cv2.imwrite(path, frame)
        print(f"{prefix} Screenshot saved @ {prefix}_{ms_ts_id}.png")

    @staticmethod
    def current_time_to_pst():
        utc_dt = datetime.utcfromtimestamp(time.time())
        pst_tz = pytz.timezone('America/Los_Angeles')
        pst_dt = pytz.utc.localize(utc_dt).astimezone(pst_tz)
        return str(pst_dt)[0:19]


class ScreenshotCapturer:
    @staticmethod
    def capture():
        try:
            screenshot = pyautogui.screenshot()
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Screenshot capture failed: {e}")
            time.sleep(5)
            return None


class GameFrameQueue:
    def __init__(self, maxlen=2):
        self.queue = deque(maxlen=maxlen)

    def enqueue(self, frame):
        if isinstance(frame, np.ndarray):
            self.queue.append(frame)
        else:
            raise ValueError("Only numpy arrays are allowed.")

    def length(self):
        return len(self.queue)

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
        # Notifier.notify(msg)


class MotionDetector:
    def __init__(self, threshold=20):
        self.threshold = threshold

    def detect(self, frame1, frame2, screen_type='WallCount'):
        f1, f2 = frame1.copy(), frame2.copy()       # masked frames for motion capture
        f1_, f2_ = frame1.copy(), frame2.copy()     # masked frames for screenshots

        if screen_type == 'WallCount':
            f1[:1190, :] = f2[:1190, :] = 0
            f1[:, 1450:] = f2[:, 1450:] = 0
            f1[1190:1400, :1150] = f2[1190:1400, :1150] = 0

            f1_[1390:, :, :] = f2_[1390:, :, :] = 0
            f1_[0:210, :, :] = f2_[0:210, :, :] = 0
            f1_[1190:1400, 0:1170, :] = f2_[1190:1400, 0:1170, :] = 0
        elif screen_type == 'GameScreenAction':
            self.threshold = 50   
            f1[0:210, :, :] = f2[0:210, :, :] = 0
            f1[950:, :, :] = f2[950:, :, :] = 0

            f1_[1390:, :, :] = f2_[1390:, :, :] = 0
            f1_[0:210, :, :] = f2_[0:210, :, :] = 0
            f1_[1190:1400, 0:1170, :] = f2_[1190:1400, 0:1170, :] = 0
        else:               # Ad and Unknown game screens
            self.threshold = 50   
            f1[1200:], f1[:210] = 0, 0
            f2[1200:], f2[:210] = 0, 0

        diff = cv2.absdiff(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))
        _, diff_thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours) > 0, f1_, f2_


class ClickMotionDetector(MotionDetector):
    def detect_after_click(self, location, screen_type='GameScreenAction'):
        frame1 = ScreenshotCapturer.capture()
        pyautogui.moveTo(location)
        print(f"Moved mouse to {location}")
        pyautogui.click()
        # time.sleep(Config.NO_MOTION_CLICK_THRESHOLD)
        frame2 = ScreenshotCapturer.capture()
        return self.detect(frame1, frame2, screen_type=screen_type)


class GameScreenshotSaveTask(Thread):
    def __init__(self, game_frame_queue, motion_detector, stop_event):
        super().__init__()
        self.game_frame_queue = game_frame_queue
        self.motion_detector = motion_detector
        self.stop_event = stop_event
        self.last_motion_time = time.time()
        self.motion_detected = False

    def run(self):
        while not self.stop_event.is_set():
            frame = ScreenshotCapturer.capture()
            is_game_frame, match_locations = Utils.is_screen(
                frame, screen_type='GameScreen')
            if is_game_frame:
                game_frame_queue.enqueue(frame)
                if self.game_frame_queue.length() > 1:
                    detected, f1, f2 = self.motion_detector.detect(
                        self.game_frame_queue[0], self.game_frame_queue[1])
                    if detected:    # change detected at threshold
                        self.game_frame_queue.save_screenshot(f2)
                elif self.game_frame_queue.length() == 1:
                    self.game_frame_queue.save_screenshot(frame)
            time.sleep(1 / Config.SCREENSHOT_SAMPLING_RATE_FPS)


# Main script
if __name__ == "__main__":
    msg = f'Starting script @ {Utils.current_time_to_pst()}'
    print(msg)
    Notifier.notify(msg)
    start_time = time.time()

    game_frame_queue = GameFrameQueue()
    motion_detector = MotionDetector()
    stop_event = Event()

    # Start the background motion detection task
    game_screenshot_save_task = GameScreenshotSaveTask(
        game_frame_queue, motion_detector, stop_event)
    game_screenshot_save_task.daemon = True
    game_screenshot_save_task.start()

    frame_queue = GameFrameQueue()

    click_motion_detector = ClickMotionDetector()

    motion_detected = False
    last_motion_time = time.time()

    no_motion_before = False

    # Main loop
    while (time.time() - start_time) < Config.TIME_LIMIT:
        frame = ScreenshotCapturer.capture()
        frame_queue.enqueue(frame)

        if frame_queue.length() > 1:
            motion_detected, f1, f2 = motion_detector.detect(
                frame_queue[0], frame_queue[1])
            if motion_detected:
                last_motion_time = time.time()

        if frame is None:
            continue

        no_motion_elapsed_seconds = time.time() - last_motion_time
        if no_motion_elapsed_seconds > Config.NO_MOTION_WARNING:
            msg = f'WARNING: {no_motion_elapsed_seconds} seconds of no motion'
            print(msg)
            Notifier.notify(msg)
            if not no_motion_before:
                Utils.save_screenshot(frame, prefix='NoMotion_')
                no_motion_before = True
        else:
            no_motion_before = False

        your_discard, _your_discard_loc = Utils.is_screen(
            frame, screen_type='YourDiscard')
        not_full_screen, _not_full_screen_loc = Utils.is_screen(
            frame, screen_type='NotFullScreen')
        game_screen, _game_screen_loc = Utils.is_screen(
            frame, screen_type='GameScreen')
        next_game, _next_game_loc = Utils.is_screen(
            frame, screen_type='NextGame')
        ad, _ad_loc = Utils.is_screen(frame, screen_type='Ad')

        # game screen
        if not_full_screen:
            Notifier.notify('full screen not detected')
            time.sleep(5)
        elif game_screen:
            # game_action_frame_queue.enqueue(frame)
            for screen_type in Config.GAME_ACTION_SCREENS:
                r, l = Utils.is_screen(frame, screen_type=screen_type)
                if r:
                    msg = f'game action screen {screen_type} found @ {l}'
                    Notifier.notify(msg)
                    print(msg)
                    Utils.save_screenshot(frame, prefix=screen_type)
                    for click_label in Config.ACTION_SCREEN_CLICK_ORDER[screen_type]:
                        c = Config.CLICK_COORDINATES.get(click_label, (0, 0))
                        if c == (0, 0):
                            raise IndexError(
                                f"{click_label} key not found in Config.CLICK_COORDINATES")
                        md, fbc, fac = click_motion_detector.detect_after_click(
                            c, screen_type='GameScreenAction')
                        if md:
                            msg = f"Motion detected after clicking {
                                click_label} @ {c}"
                            print(msg)
                            Notifier.notify(msg)
                            Utils.save_screenshot(
                                fbc, prefix=screen_type + '_before_click_')
                            Utils.save_screenshot(
                                fac, prefix=screen_type + '_after_click_')
                            break
        elif next_game:
            next_game_frame = frame.copy()
            screen_type, details = 'NextGame', ''
            msg = f"next game detected @ {_next_game_loc}"
            print(msg)
            Notifier.notify(msg)

            ############# DetermineWinner ##################
            ################################################
            winner_roi = (580, 650, 500, 580)

            cropped_image = next_game_frame.crop(winner_roi)
            cropped_image = cropped_image.convert('L')  # Convert to grayscale

            extracted_text = pytesseract.image_to_string(cropped_image)
            letters = re.findall(r'[a-zA-Z]', extracted_text)

            if letters:
                letter = ''.join(letters)  # Join all letters together or select a specific letter if needed
                print(f"Extracted Letters: {letter}")
                details += ''.join(['_', str(letter), 'Winner'])
            else:
                print("No letters found.")
                details += '_NoWinnerFound'

            ################################################
            ############# DetermineWinnerFan ###############
            next_game_frame = frame.copy()
            total_fan_roi = (1600, 2000, 500, 1000)
            cropped_image = next_game_frame = frame.copy().crop(roi)

            cropped_image = cropped_image.convert('L')  # Convert to grayscale

            cropped_image = cropped_image.filter(ImageFilter.MedianFilter())  # Reduce noise
            cropped_image = cropped_image.point(lambda p: p > 128 and 255)  # Apply thresholding
            base_width = 1000
            w_percent = (base_width / float(cropped_image.size[0]))
            h_size = int((float(cropped_image.size[1]) * float(w_percent)))
            cropped_image = cropped_image.resize((base_width, h_size), Image.LANCZOS)  # Resize image

            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(cropped_image, config=custom_config)

            # Extract the number after "Total Fan"
            match = re.search(r'Total\s*Fan\s*(\d+)', extracted_text, re.IGNORECASE)

            # Check if a match was found and extract the number
            if match:
                total_fan_number = match.group(1)
                print(f"Total Fan Number: {total_fan_number}")
                details += ''.join(['_', str(total_fan_number), 'Fan'])
            else:
                print("No 'Total Fan' pattern found.")
                details += '_NoFanFound'
            ################################################

            for child_screen_type in Config.NEXT_GAME_CHILDREN:
                r, l = Utils.is_screen(frame, screen_type=child_screen_type)
                if r:
                    msg = f'next game child template {
                        l[4]} for {r} found @ {l[0:-1]}'
                    Notifier.notify(msg)
                    print(msg)
                    details += '_'
                    details += l[4].replace('.png', '')

            Utils.save_screenshot(frame, prefix=screen_type+details)
            for click_label in Config.ACTION_SCREEN_CLICK_ORDER[screen_type]:
                c = Config.CLICK_COORDINATES.get(click_label, (0, 0))
                if c == (0, 0):
                    raise IndexError(
                        f"{click_label} key not found in Config.CLICK_COORDINATES")

                md, fbc, fac = click_motion_detector.detect_after_click(
                    c, screen_type=screen_type)
                if md:
                    msg = f"Motion detected after clicking {click_label} @ {c}"
                    last_motion_time = time.time()
                    print(msg)
                    Notifier.notify(msg)
                    Utils.save_screenshot(
                        fbc, prefix=screen_type + '_before_click_')
                    Utils.save_screenshot(
                        fac, prefix=screen_type + '_after_click_')
                    break
        elif ad:
            screen_type = 'Ad'
            msg = f"ad detected @ {_ad_loc}"
            print(msg)
            Notifier.notify(msg)
            Utils.save_screenshot(frame, prefix=screen_type)
            for click_label in Config.ACTION_SCREEN_CLICK_ORDER[screen_type]:
                c = Config.CLICK_COORDINATES.get(click_label, (0, 0))
                if c == (0, 0):
                    raise IndexError(
                        f"{click_label} key not found in Config.CLICK_COORDINATES")

                md, fbc, fac = click_motion_detector.detect_after_click(
                    c, screen_type=screen_type)
                if md:
                    msg = f"Motion detected after clicking {click_label} @ {c}"
                    last_motion_time = time.time()
                    print(msg)
                    Notifier.notify(msg)
                    Utils.save_screenshot(
                        fbc, prefix=screen_type + '_before_click_')
                    Utils.save_screenshot(
                        fac, prefix=screen_type + '_after_click_')
                    break
        else:
            screen_type = 'Unknown'
            msg = f"{screen_type} screen detected"
            print(msg)
            Notifier.notify(msg)

        time.sleep(1 / Config.SAMPLING_RATE_FPS)

    # Stop the background task and wait for it to finish
    stop_event.set()
    game_screenshot_save_task.join()

    msg = f'Ending script @ {Utils.current_time_to_pst()}'
    print(msg)
    Notifier.notify(msg)
