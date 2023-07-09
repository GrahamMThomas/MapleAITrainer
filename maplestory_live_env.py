from time import sleep
from gymnasium import Space, spaces
import gymnasium
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
import pygetwindow as gw
import cv2
import mss
import win32gui
import win32con
import pyautogui  # Need this imported to fix bug
from keyboard_helper import *


class MaplestoryLiveEnv(gymnasium.Env):
    GAME_WINDOW_HEIGHT = 240
    GAME_WINDOW_WIDTH = 420
    GAME_WINDOW_TITLE = "Kaizen v92"
    FRAMES_STACK_COUNT = 4

    MOVEMENT_KEYS = ["left", "up", "right", "down", ""]
    ATTACK_KEYS = ["x", "s", ""]
    MISC_KEYS = ["c", "z", ""]

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
        game_frame = self.get_game_frame()

        cv2.imshow("Screen Capture", game_frame)
        cv2.waitKey(100)

        # Reward Calculation
        self.total_reward += 1
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward
        sleep(0.25)

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
        self.total_reward = 0
        self.prev_reward = 0
        self.reward = 0
        self.pressed_keys = []

        return self.get_game_frame(), {}

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
        return np.array(game_frame)

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
