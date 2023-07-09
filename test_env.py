from maplestory_live_env import MaplestoryLiveEnv


env = MaplestoryLiveEnv()
episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()
    while True:  # not done:
        random_action = env.action_space.sample()
        print("\naction", random_action)
        obs, reward, done, info = env.step(random_action)
        print("reward", reward)
