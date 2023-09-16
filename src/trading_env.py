from abc import ABC
from typing import Any, Union, List, Dict
from itertools import product

import gym
import numpy as np
import pandas as pd

import src.observation as observation


class TradingEnv(gym.Env, ABC):
    def __init__(
            self,
            initial_balance: float,
            stocks_series: Dict[str, pd.DataFrame],
            observations: List[str],
            max_steps: int = 100,
            window_size: int = 20,
    ):
        self.stocks_series = stocks_series

        self.window_size = window_size
        if "sma" in observations:
            self.compute_moving_average(method="sma")
        if "ema" in observations:
            self.compute_moving_average(method="ema")
        # todo: fill NaN with average when necessary

        self.num_stocks = len(stocks_series)
        self.max_steps = max_steps

        self.initial_balance = initial_balance
        self.balance = initial_balance   # Balance = Initial Investment + Net Profit (or - Net Loss)
        self.initial_investment = 0
        self.shares_holding = np.zeros(self.num_stocks)
        self.previous_action = None

        self.observations = observations

        self.current_step = 0
        # action_space -> column0: balance ratios invested ; column1: shares ratios sold
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks, 2), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=np.inf, shape=(self.num_stocks, len(observations)), dtype=np.float32
        # )
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf,
            shape=(self.num_stocks * (len(observations) + 1) + 1, 1),
            dtype=np.float32
        )

    @property
    def current_stocks(self) -> pd.DataFrame:
        stocks = self.stocks_series
        step = self.current_step
        df = pd.concat(
            (stocks[name].iloc[step] for name in stocks.keys()), axis=1, keys=stocks.keys()
        ).T
        return df

    @property
    def roi(self) -> float:
        # ROI = (Net Profit / Initial Investment) * 100
        # Net Profit = Total Gains - Total Costs and Losses
        return 0

    def compute_moving_average(self, method: str):
        stocks = self.stocks_series
        ws = self.window_size
        for df in stocks.values():
            if method == "sma":
                df["SMA"] = df["Close"].rolling(window=ws).mean()
            elif method == "ema":
                df["EMA"] = df["Close"].ewm(span=ws, adjust=False).mean()
            df.fillna(df.mean())

    def reset(self, **kwargs) -> Any:
        self.current_step = 0
        self.balance = self.initial_balance
        self.initial_investment = 0
        self.shares_holding = np.zeros(self.num_stocks)
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        curr_stock = self.current_stocks
        observed = [getattr(observation, f)(curr_stock) for f in self.observations]
        observed += [self.shares_holding]
        # observed = np.row_stack(observed)  # uncomment to get as before
        observed = np.row_stack(observed).ravel()  # comment to get as before
        observed = np.append(observed, self.balance)  # comment to get as before
        return observed

    def step(self, action) -> Any:
        self.previous_action = action.T.ravel()

        # Calculate the amounts to invest or sell for each stock
        invest_amounts = action[:, 0] * self.balance
        curr_prices = observation.close(self.current_stocks)
        sell_amounts = action[:, 1] * self.shares_holding * curr_prices
        if self.current_step >= self.max_steps - 1:  # don't invest + sell it all before the end
            invest_amounts = 0.0
            sell_amounts = self.shares_holding * curr_prices
        total_investment = np.sum(invest_amounts)

        total_sold = np.sum(sell_amounts)
        profit_or_loss = total_sold - total_investment
        self.balance += profit_or_loss
        self.shares_holding += (invest_amounts - sell_amounts)/curr_prices

        # Action description
        # todo: add something to keep track of what was done at the end:
        #  what was sold, and what was invested (maybe for a plot)

        # Calculate the reward (e.g., profit or loss)
        # todo: review and add penalty for transaction fees
        # reward = self.balance / self.initial_balance
        reward = 100 * profit_or_loss / self.initial_balance  # return rate

        # Check if the maximum number of steps is reached
        done = self.current_step >= self.max_steps

        # Update the current step
        self.current_step += 1

        return self._get_observation(), reward, done

    def render(self) -> Any:
        print(
            f"Step: {self.current_step}, "
            f"Balance: {self.balance:.2f}, "
            f"Shares: {np.round(self.shares_holding, 2)}, "
            f"Prices: {np.round(observation.close(self.current_stocks), 2)}, "
            f"Action: {self.previous_action}"
        )

    def close(self):
        pass

    # todo: function to plot different stocks with info
    # todo: function to plot a strategy/series of actions...


class DiscreteActionsTradingEnv(TradingEnv):
    def __init__(self, n_discretize: int, **kwargs):
        super().__init__(**kwargs)
        self.n_discretize = n_discretize
        self.actions = self._get_actions()
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def _get_actions(self):
        raw_actions = np.array(list(product(np.linspace(0, 1, self.n_discretize), repeat=self.num_stocks)))
        inf_one_filter = (np.sum(np.maximum(raw_actions, 0), axis=1) <= 1)
        filtered_actions = raw_actions[inf_one_filter]
        actions = np.array(list(product(filtered_actions, repeat=2)))
        return np.array([a.T for a in actions])

    def step(self, action) -> Any:
        return TradingEnv.step(self, self.actions[action])


if __name__ == "__main__":
    from stock_loader import get_data_stocks

    # Create the trading environment
    data = get_data_stocks(["TSLA", "AAPL", "GOOGL"], "2020-01-10", "2022-01-01")

    # ##############" Plot ################
    import matplotlib.pyplot as plt
    which = ["TSLA", "AAPL", "GOOGL"]
    what = "High"
    time = np.arange(len(data[which[0]]))
    plt.plot(time, data[which[0]][what])
    plt.plot(time, data[which[1]][what])
    plt.plot(time, data[which[2]][what])
    plt.legend(which)
    plt.xlabel("Days")
    plt.ylabel(f"Stock {what}")
    plt.show()
    # #####################################

    # env = TradingEnv(initial_balance=1000, stocks_series=data, observations=["low", "high"])
    env = DiscreteActionsTradingEnv(
        initial_balance=1000, stocks_series=data, observations=["low", "high", "sma"], n_discretize=4
    )

    # Reset the environment to initialize
    obs = env.reset()
    env.render()

    # Basic example of interacting with the environment
    for _ in range(50):
        act = env.action_space.sample()  # Take a random action (sell, hold, buy)
        obs, rwd, don = env.step(act)
        env.render()

    env.close()
