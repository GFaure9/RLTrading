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
shareholding $S_0 \in {\mathbb{R}^+}^N$, we define at time $t>0$ the
action $A^t = (a^t_{ij}) \in \mathbb{M}\_{N, 2}([0, 1])$
such that $\sum_{1 \leq i \leq N} a^t_{i, j} = 1$ for $j=1, 2$
and:

$$S_{t + 1} = S_t \otimes (1 - A^t (0  1)^T) + B_t X_{t + 1}^{-1} \otimes (A^t (1  0)^T)$$

$$\Delta_t = -(S_{t + 1} - S_t) X_{t + 1} $$

$$B_{t + 1} = B_t + \Delta_t$$

With $X_t$ the stocks prices at time $t$ and using the notation
$X_t^{-1}:=(\frac{1}{x^t_1} ... \frac{1}{x^t_N}) .

## Usage example

???