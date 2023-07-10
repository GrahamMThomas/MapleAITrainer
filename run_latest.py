import os

from gymnasium.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from maplestory_live_env import MaplestoryLiveEnv
from train_and_log_callback import TrainAndLoggingCallback
from matplotlib import pyplot as plt

CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"


def main():
    env = MaplestoryLiveEnv()
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    state = env.reset()
    # state, reward, done, info = env.step([[4, 2, 2]])
    # state, reward, done, info = env.step([[4, 2, 2]])
    # state, reward, done, info = env.step([[4, 2, 2]])
    # plt.figure(figsize=(20, 16))
    # for idx in range(state.shape[3]):
    #     plt.subplot(1, 4, idx + 1)
    #     plt.imshow(state[0][:, :, idx])
    # plt.show()

    model = PPO.load(CHECKPOINT_DIR + "maplestory_trainer_latest.zip")

    while True:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()
