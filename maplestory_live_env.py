from time import sleep
import os
import cv2
import gymnasium
import mss
import numpy as np
import pyautogui  # Need this imported to fix bug
import pygetwindow as gw
import pytesseract
import win32gui
from gymnasium import Space, spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from keyboard_helper import *

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


class MaplestoryLiveEnv(gymnasium.Env):
    GAME_WINDOW_HEIGHT = 240
    GAME_WINDOW_WIDTH = 420
    GAME_WINDOW_TITLE = "Kaizen v92"
    FRAMES_STACK_COUNT = 4

    MOVEMENT_KEYS = ["left", "up", "right", "down", ""]
    ATTACK_KEYS = ["x", "s", ""]
    MISC_KEYS = ["c", "z", ""]

    EXP_GAINED_REWARD_MULTIPLIER = 1
    HEALTH_LOST_REWARD_MULTIPLIER = 1

    def __init__(self):
        super(MaplestoryLiveEnv, self).__init__()

        self.action_space = spaces.MultiDiscrete(
            [5, 3, 3]  # Arrow Keys  # Different Attacks  # Jump or Loot
        )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.GAME_WINDOW_HEIGHT,
                self.GAME_WINDOW_WIDTH,
                3,  # color channels
            ),
            dtype=np.uint8,
        )

        self.window_id = win32gui.FindWindow(None, self.GAME_WINDOW_TITLE)

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        os.system("cls")
        # Wait for game window to be focused
        foreground_id = win32gui.GetForegroundWindow()
        while foreground_id != self.window_id:
            # for key in self.pressed_keys:
            #     key_up(key)
            #     self.pressed_keys.remove(key)
            foreground_id = win32gui.GetForegroundWindow()
            sleep(1)
            print("Waiting for game window to be focused...")

        self.print_action(actions)

        # Get pressed_keys that aren't in actions
        keys_to_release = list(
            set(self.pressed_keys)
            - (
                set([self.MOVEMENT_KEYS[actions[0]]])
                | set([self.MISC_KEYS[actions[2]]])
            )
        )

        print("Release: ", keys_to_release)
        for key in keys_to_release:
            key_up(key)
            self.pressed_keys.remove(key)

        print("Pressed Keys: ", self.pressed_keys)

        # Perform actions
        if actions[0] != 4 and self.MOVEMENT_KEYS[actions[0]] not in self.pressed_keys:
            key_to_press = self.MOVEMENT_KEYS[actions[0]]
            print("KeyDown: ", key_to_press)
            key_down(key_to_press)
            self.pressed_keys.append(key_to_press)

        if actions[1] != 2:
            key_to_press = self.ATTACK_KEYS[actions[1]]
            print("Press: ", key_to_press)
            press(key_to_press, 1)

        if actions[2] != 2 and self.MISC_KEYS[actions[2]] not in self.pressed_keys:
            key_to_press = self.MISC_KEYS[actions[2]]
            print("KeyDown: ", key_to_press)
            key_down(key_to_press)
            self.pressed_keys.append(key_to_press)

        # Process Game Frame
        game_frame, raw_game_frame = self.get_game_frame()

        # cv2.imshow("Screen Capture", game_frame)
        # cv2.waitKey(100)

        # Reward Calculation
        self.reward = self.get_current_reward(raw_game_frame)
        print("Reward: ", self.reward)

        sleep(1)

        return game_frame, self.reward, self.done, {}

    def reset(self, seed=0) -> np.ndarray:
        self.game_window = gw.getWindowsWithTitle(self.GAME_WINDOW_TITLE)[0]
        self.window_location = {
            "top": self.game_window.top,
            "left": self.game_window.left,
            "width": self.game_window.width,
            "height": self.game_window.height,
        }
        self.done = False
        self.pressed_keys = []

        obs, raw_game_frame = self.get_game_frame()

        self.previous_health = 0
        self.previous_exp = 0
        self.get_current_reward(raw_game_frame)

        return obs, {}

    def render(self):
        pass

    def close(self):
        pass

    # Additional Implementation

    def get_game_frame(self):
        game_frame = np.empty([self.GAME_WINDOW_HEIGHT, self.GAME_WINDOW_WIDTH])

        with mss.mss() as sct:
            sc_grab = sct.grab(self.window_location)
            sct_img = self.frame(sc_grab)

            game_frame = cv2.resize(
                sct_img,
                (self.GAME_WINDOW_WIDTH, self.GAME_WINDOW_HEIGHT),
                # interpolation=cv2.INTER_AREA,
            )

        # print(game_frame)
        return np.array(game_frame), sct_img

    def frame(self, grab):
        im = np.array(grab)
        # im = np.flip(im[:, :, :3], 2)  # 1
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2
        return im

    def print_action(self, actions):
        if actions[0] != 4:
            print("Move: ", self.MOVEMENT_KEYS[actions[0]])
        if actions[1] != 2:
            print("Attack: ", self.ATTACK_KEYS[actions[1]])
        if actions[2] != 2:
            print("Misc: ", self.MISC_KEYS[actions[2]])

    def get_current_reward(self, raw_game_frame) -> int:
        raw_game_frame
        # Crop
        exp_gained = self.get_exp_gained(raw_game_frame)
        healthLost = self.get_health_lost(raw_game_frame)

        print("Exp Gained: ", exp_gained)
        print("Health Lost: ", healthLost)

        return (exp_gained * self.EXP_GAINED_REWARD_MULTIPLIER) - (
            healthLost * self.HEALTH_LOST_REWARD_MULTIPLIER
        )

    def get_health_lost(self, raw_game_frame) -> int:
        health_game_frame = raw_game_frame[948:968, 300:380]
        # Posterize
        health_game_frame[health_game_frame >= 128] = 255
        health_game_frame[health_game_frame < 128] = 0
        # Isolate Blue Channel to remove Parans
        health_game_frame[:, :, 1] = 0
        health_game_frame[:, :, 2] = 0

        cv2.imshow("Screen Capture", health_game_frame)
        cv2.waitKey(100)

        health: str = pytesseract.image_to_string(health_game_frame)
        # Example Capture: 300/500
        print("HealthCapture: ", health)
        health = health.strip().split("/")[0]
        try:
            health = int(health)
        except:
            return 0

        print("Current Health:", health)
        print("Previous Health:", self.previous_health)
        health_lost = max((self.previous_health - health), 0)
        self.previous_health = health

        return health_lost

    def get_exp_gained(self, raw_game_frame) -> int:
        exp_game_frame = raw_game_frame[948:968, 582:700]
        # Posterize
        exp_game_frame[exp_game_frame >= 128] = 255
        exp_game_frame[exp_game_frame < 128] = 0
        # Isolate Blue Channel to remove Parans
        exp_game_frame[:, :, 1] = 0
        exp_game_frame[:, :, 2] = 0

        exp: str = pytesseract.image_to_string(exp_game_frame)
        # Example Capture: 1123 98.16%)
        print("ExpCapture: ", exp)
        exp = exp.strip().split(" ")[0]
        # replace $ and S characters with 5
        exp = exp.replace("$", "5")
        exp = exp.replace("S", "5")
        try:
            exp = int(exp)
        except:
            return 0

        print("Current Exp:", exp)
        exp_gained = max((exp - self.previous_exp), 0)
        self.previous_exp = exp

        return exp_gained
