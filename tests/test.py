from src.trading_env import TradingEnv, DiscreteActionsTradingEnv
from src.dqn import DeepQLearning
from src.q_networks import qnet1
from src.stock_loader import get_data_stocks

data = get_data_stocks(
    [
        "TSLA",
        "AAPL",
        # "GOOGL"
    ],
    "2020-01-01",
    "2022-01-01"
)

# observations = ["low", "high"]
# observations = ["volume", "dividends", "low", "high", "sma"]
observations = ["volume", "dividends", "low", "high"]
env = DiscreteActionsTradingEnv(
    initial_balance=1000,
    stocks_series=data,
    observations=observations,
    n_discretize=3,
    window_size=20,
)

model = DeepQLearning(q_network=qnet1, env=env, n_episodes=100, max_steps_episode=80)
trained_dqn = model.train()

breakpoint()
