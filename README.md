# Reinforcement Learning for Multiple Stock Trading

## About the project...
This project is in progress. It aims at developing an agent
that will be able to autonomously trade multiple stocks,
using available data on [yahoo-finance](https://www.yahoo.com/author/yahoo-finance/).

__Warning__: for now, the first tested approach (DQN)
is not giving satisfactory results. Further
investigation and other techniques must be pursued.

## What has been developed...

* The _environment_ for the trading agent (`TradingEnv`), 
using [gym](https://gymnasium.farama.org/)'s framework. 
It specifies:
  - the observations space;
  - the action space;
  - what is done at a step from an action.


* An implementation of the _Deep Q-Learning algorithm_, 
that allows to easily test different deep neural
networks (for the Q-value function predictor).

---
## An introduction to the Trading Environment

For $N$ stocks, starting with an initial balance $B_0 = K > 0$ and an initial
shareholding $S_0 \in {\mathbb{R}^+}^N$, we define at time $t>0$ the
action $A_t = (a^t_{ij}) \in \mathbb{M}\_{N, 2}([0, 1])$
such that $\sum_{1 \leq i \leq N} a^t_{i, j} = 1$ for $j=1, 2$
and:

$$S_{t + 1} = S_t \otimes (\begin{bmatrix}
1 \\
1 \\
\end{bmatrix})
- A_t \begin{bmatrix}
0 \\
1 \\
\end{bmatrix}) + 
B_t X_{t + 1}^{-1} \otimes (A_t \begin{bmatrix}
1 \\
0 \\ 
\end{bmatrix})$$

$$\Delta_t = -(S_{t + 1} - S_t) X_{t + 1} $$

$$B_{t + 1} = B_t + \Delta_t$$

With $X_t$ the stocks prices at time $t$ and using the notation
$X_t^{-1}:=(\frac{1}{x^t_1} ... \frac{1}{x^t_N})$ .  
The first column of $A_t$ corresponds to the percentages of the balance
invested in each stock (one per row) at $t$ while the second column corresponds
to the percentages of holding shares of each stock sold at $t$.

## Usage example

The NN model used for DQN can be defined in `q_networks.py` and then 
imported to be used for the RL algo. 
In the following example, a default non-optimal network 
has been defined as `qnet1`.

```python
from src.trading_env import DiscreteActionsTradingEnv
from src.dqn import DeepQLearning
from src.q_networks import qnet1
from src.stock_loader import get_data_stocks

stock_names = ["TSLA", "AAPL", "GOOGL"]
start_date, end_date = "2020-01-01", "2022-01-01"
data = get_data_stocks(stock_names, start_date, end_date)

observations = ["volume", "dividends", "low", "high"]

env = DiscreteActionsTradingEnv(
    initial_balance=1000,
    stocks_series=data,
    observations=observations,
    n_discretize=3,
)

model = DeepQLearning(q_network=qnet1, env=env, n_episodes=100, max_steps_episode=80)
trained_dqn = model.train()
```