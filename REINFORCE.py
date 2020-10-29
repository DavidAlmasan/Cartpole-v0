import sys, os
import numpy as np
from collections import deque
from itertools import accumulate
import random

import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import datasets, layers, models


class REINFORCE():
    def __init__(self):
        # Hyperparams
        self.learning_rate = 1e-5
        self.gamma = 0.9999

        # Training params
        self.max_steps_per_episode = 500
        self.max_episodes = 20000
        self.log_episodes = [int(float(self.max_episodes) * i / 10) for i in range(11)]

        # Env and agent
        self.env = self.create_env()
        self.policy, self.optimizer = self.create_agent()

    def preprocess(self, state):
        return np.expand_dims(state, axis=0)

    def create_env(self):
        env = gym.make('CartPole-v0')
        if self.max_steps_per_episode is not None:
            env._max_episode_steps = self.max_steps_per_episode
        env.reset()
        return env

    def create_agent(self):
        agent = models.Sequential()
        agent.add(layers.Input(4))
        agent.add(layers.Dense(32, activation='relu'))
        agent.add(layers.Dropout(0.3))
        agent.add(layers.Dense(32, activation='relu'))
        agent.add(layers.Dropout(0.3))
        agent.add(layers.Dense(2, activation='linear'))

        optimizer = RMSprop(self.learning_rate)

        return agent, optimizer

    def compute_cumulative_rewards(self, episode):
        rewards = [ep[-1] for ep in episode][::-1]
        rewards = list(accumulate(rewards, lambda x, y: x * self.gamma + y))[::-1]
        rewards = list(np.asarray(rewards) / max(rewards))
        states = [t[0] for t in episode]
        actions =[t[1] for t in episode]

        return states, actions, rewards

    def epsilon_greedy(self, t, s):
        eps = 0.92 ** t
        eps = max(0.01, eps)
        s = np.expand_dims(np.asarray(s), axis=0)
        action_space = np.squeeze(self.policy(s).numpy())
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = np.random.choice(list(range(len(action_space))))
        else:
            # Perform greedy action
            action = np.argmax(action_space)
        return action

    def store_episode(self, policy, use_eps=True):
        episode = []
        s = self.env.reset()
        d = False
        t = 0
        while not d:
            if use_eps:
                action = self.epsilon_greedy(t, s)
            else:
                action = tf.argmax(policy(self.preprocess(s))[0]).numpy()

            s_, r, d, _ = self.env.step(action)
            episode.append([s, action, r])
            s = s_
            t += 1
        states, actions, rewards = self.compute_cumulative_rewards(episode)

        return states, actions, rewards

    def train_episode(self):
        states, actions, rewards = self.store_episode(self.policy)
        states = tf.stack(states)
        actions = tf.stack(actions)
        action_indeces = tf.range(len(actions))
        actions = tf.concat([tf.expand_dims(action_indeces, axis=1), tf.expand_dims(actions, axis=1)], axis=1)
        rewards = tf.cast(tf.stack(rewards), dtype=tf.float64)

        with tf.GradientTape() as tape:
            q_space = tf.clip_by_value(tf.cast(self.policy(states, training=True), dtype=tf.float64),
                                       clip_value_min=0.0001,
                                       clip_value_max=self.max_steps_per_episode)
            q_space_probs = tf.nn.softmax(q_space)
            q_values = tf.gather_nd(q_space, actions)
            q_values_probs = tf.gather_nd(q_space_probs, actions)
            gamma_vector = tf.cast(tf.convert_to_tensor([self.gamma ** i for i in range(len(q_space))]),
                                   dtype=tf.float64)
            rewards = tf.math.multiply(rewards, gamma_vector)
            loss = tf.math.multiply(tf.math.log(q_values_probs),
                                    tf.stop_gradient(rewards))
            loss = tf.reduce_mean(loss) * -1
            # print('loss: ', loss)
            grads = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        return len(states), loss

    def train(self):
        for ep in range(self.max_episodes):
            # print('Storing and training episode: {}'.format(ep + 1))
            length, loss = self.train_episode()
            if ep in self.log_episodes:
                print('Finished training episode {}/{} of length: {} with loss {}'.format(ep + 1,
                                                                                        self.max_episodes,
                                                                                        length,
                                                                                        loss))







        pass

if __name__ == '__main__':
    agent = REINFORCE()
    agent.train()

