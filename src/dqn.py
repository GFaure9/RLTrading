import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.rl_model import RLModel
from src.trading_env import DiscreteActionsTradingEnv
from typing import Callable
from collections import namedtuple, deque


HParams = namedtuple(
    "HParams",
    [
        "learning_rate",
        "gamma",
        "epsilon",
        "epsilon_decay",
        "min_epsilon",
        "replay_memory_size",
        "minibatch_size",
    ]
)


class DeepQLearning(RLModel):
    default_hyperparams = {
        "learning_rate": 0.1,
        "gamma": 0.99,  # Discount factor
        "epsilon": 0.9,
        "epsilon_decay": 0.95,
        "min_epsilon": 0.05,
        "replay_memory_size": 100,  # 1000
        "minibatch_size": 32,  # 2 * 64
    }

    def __init__(
            self,
            env: DiscreteActionsTradingEnv,
            q_network: Callable,
            hyperparams: HParams = None,
            **kwargs
    ):
        super().__init__(env=env, **kwargs)
        self.q_network = q_network
        self.hyperparams = hyperparams if None else HParams(**self.default_hyperparams)

    def _train_batch(self, replay_memory: deque, model: tf.keras.Sequential):
        minibatch_size = self.hyperparams.minibatch_size

        if len(replay_memory) < minibatch_size * 2:
            return

        mini_batch = random.sample(replay_memory, minibatch_size)

        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = model.predict(new_current_states)

        x = []
        y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + self.hyperparams.gamma * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            lr = self.hyperparams.learning_rate
            current_qs[action] = (1 - lr) * current_qs[action] + lr * max_future_q

            x.append(observation)
            y.append(current_qs)

        model.fit(np.array(x), np.array(y), batch_size=minibatch_size, verbose=0, shuffle=True)

    def train(self):
        env = self.env
        input_shape = np.prod(env.observation_space.shape)
        output_shape = env.action_space.n

        network = self.q_network(input_shape, output_shape)

        hparams = self.hyperparams
        init_eps, min_eps, eps_dec = hparams.epsilon, hparams.min_epsilon, hparams.epsilon_decay
        eps = init_eps

        replay_memory = deque(maxlen=hparams.replay_memory_size)

        env.max_steps = self.max_steps_episode

        episodes_rewards = []
        for episode in range(self.n_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            steps_to_update = 0

            while not done:
                # env.render()
                # print("Epsilon = ", eps)

                if np.random.uniform(0, 1) < eps:
                    action = env.action_space.sample()  # Explore
                else:
                    q_values = network.predict(np.array([state.ravel()]))[0]
                    action = np.argmax(q_values)  # Exploit

                new_state, reward, done = env.step(action)

                replay_memory.append([state.ravel(), action, reward, new_state.ravel(), done])

                if steps_to_update % 4 == 0 or done:
                    self._train_batch(replay_memory=replay_memory, model=network)

                episode_reward += reward
                state = new_state

                steps_to_update += 1

                if done:
                    break

            # eps = max(min_eps, eps * eps_dec)  # Decay epsilon
            eps = min_eps + (init_eps - min_eps) * np.exp(-eps_dec * episode)

            print(f"Episode {episode + 1}: Reward = {episode_reward}")
            episodes_rewards.append(episode_reward)

        env.close()

        self.trained_model = network

        plt.plot(episodes_rewards)
        plt.show()
        print(np.array(episodes_rewards).mean())
        breakpoint()

        return network
