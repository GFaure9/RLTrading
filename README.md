# Reinforcement Learning for Multiple Stock Trading

## About the project...
This project is in progress. It aims at developing an agent
that will be able to autonomously trade multiple stocks,
using available data on [yahoo-finance](https://www.yahoo.com/author/yahoo-finance/).

__Warning__: for now, the first tested approach (DQN)
is not giving satisfactory results. Further
investigation and other techniques must be pursued.

## What has been developed...

* The _environment_ for the trading agent (`TradingEnv`), using [gym](https://gymnasium.farama.org/)'s framework. 
It specifies:
  - the observations space;
  - the action space;
  - what is done at a step from an action.


* An implementation of the _Deep Q-Learning algorithm_, that allows to easily test different deep neural
networks (for the Q-value function predictor).

---
## An introduction to the Trading Environment

For $N$ stocks, starting with an initial balance $B_0 = K > 0$ and an initial
shareholding $S^0 \in {\mathbb{R}^+}^N$, we define at time $t>0$ the
action $A^t \in {\mathbb{M}}_{N, 2}([0, 1])$
such that $\sum_{1\leq i \leq N} A^t_{i, j} = 1$ for $j=1, 2$
and:

$$S_{t + 1} = B_k (A^t_{12} A^t_{22} ... A^t_{N2})^T + X_{t + 1}^-1 (A^t_{11} A^t_{21} ... A^t_{N1})^T$$

$$\Delta_t = 0$$

$$B_{t + 1} = B_t + \Delta_t$$

With $X_t$ the stocks prices at time $t$.

## Usage example

???