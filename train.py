# Import Frame Stacker Wrapper and GrayScaling Wrapper
import os

from gymnasium.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from maplestory_live_env import MaplestoryLiveEnv
from train_and_log_callback import TrainAndLoggingCallback

CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"


def main():
    env = MaplestoryLiveEnv()
    # env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    callback = TrainAndLoggingCallback(check_freq=64, save_path=CHECKPOINT_DIR)
    file_path_existing = os.path.join(CHECKPOINT_DIR, "maplestory_trainer_latest.zip")
    if os.path.isfile(file_path_existing):
        print("Loading Existing model: ", file_path_existing)
        model = PPO.load(file_path_existing)
    else:
        print("Creating new model!")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=0.0003,
            gamma=0.85,
            batch_size=32,
            n_steps=64,
        )
    model.set_env(env)
    model.learn(total_timesteps=100000, callback=callback, reset_num_timesteps=False)


if __name__ == "__main__":
    main()
