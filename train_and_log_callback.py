import os
from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

        self.cum_exp = 0
        self.cum_health_loss = 0
        self.cum_reward = 0

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_end(self) -> None:
        self.logger.record("cum_exp", self.cum_exp)
        self.logger.record("cum_health_loss", self.cum_health_loss)
        self.logger.record("cum_reward", self.cum_reward)

        self.cum_exp = 0
        self.cum_health_loss = 0
        self.cum_reward = 0

    def _on_step(self):
        print("Step:", self.n_calls)
        self.cum_exp += self.training_env.get_attr("exp_gained")[0]
        self.cum_health_loss += self.training_env.get_attr("health_lost")[0]
        self.cum_reward += self.training_env.get_attr("reward")[0]

        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, "maplestory_trainer_{}".format(self.n_calls)
            )
            latest_model_path = os.path.join(
                self.save_path, "maplestory_trainer_latest"
            )
            self.model.save(model_path)
            self.model.save(latest_model_path)

        return True
