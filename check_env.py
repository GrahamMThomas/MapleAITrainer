from stable_baselines3.common.env_checker import check_env
from maplestory_live_env import MaplestoryLiveEnv

env = MaplestoryLiveEnv()
check_env(env)
