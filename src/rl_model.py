from src.trading_env import TradingEnv, DiscreteActionsTradingEnv
from typing import Union


class RLModel:
    def __init__(
            self,
            env: Union[TradingEnv, DiscreteActionsTradingEnv],
            n_episodes: int,
            max_steps_episode: int
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.max_steps_episode = max_steps_episode
        self.trained_model = None

    def train(self, **kwargs):
        return
