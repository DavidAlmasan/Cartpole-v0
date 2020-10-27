import sys, os
import numpy as np
from collections import deque
import random

import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import datasets, layers, models


class DQN():
    def __init__(self):
        # Hyperparams
        self.learning_rate = 0.8
        self.gamma = 1
        self.batch_size = 256

        # Training params
        self.max_steps_per_episode = 500
        self.max_episodes = 200
        self.memory = deque(maxlen=2000)

        # Env and agent
        self.env = self.create_env()
        self.agent, self.target_agent, self.optimizer = self.create_agent()

    def update_agent_weights(self, weights_path):
        self.agent.load_weights(weights_path)
        self.target_agent.load_weights(weights_path)

    def create_env(self):
        env = gym.make('CartPole-v0')
        if self.max_steps_per_episode is not None:
            env._max_episode_steps = self.max_steps_per_episode
        env.reset()
        return env

    def get_agent(self):
        return self.agent

    def get_env(self):
        return self.env

    def save_agent(self, path):
        self.agent.save_weights(path)

    def create_agent(self):
        agent = models.Sequential()
        agent.add(layers.Input(4))
        agent.add(layers.Dense(32, activation='relu'))
        agent.add(layers.Dropout(0.3))
        agent.add(layers.Dense(32, activation='relu'))
        agent.add(layers.Dropout(0.3))
        agent.add(layers.Dense(2, activation='softmax'))

        target_agent = models.Sequential()
        target_agent.add(layers.Input(4))
        target_agent.add(layers.Dense(32, activation='relu'))
        target_agent.add(layers.Dropout(0.3))
        target_agent.add(layers.Dense(32, activation='relu'))
        target_agent.add(layers.Dropout(0.3))
        target_agent.add(layers.Dense(2, activation='softmax'))

        optimizer = RMSprop(self.learning_rate)

        # agent.compile(loss='mse', optimizer=optimizer)
        return agent, target_agent, optimizer

    def epsilon_greedy(self, t, s):
        eps = max(0.05, 1. / (t + 1))
        # eps = 0.1
        s = np.expand_dims(np.asarray(s), axis=0)
        action_space = np.squeeze(self.agent(s).numpy())
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = np.random.choice(list(range(len(action_space))))
        else:
            # Perform greedy action
            action = np.argmax(action_space)
        return action, eps

    def preprocess(self, state):
        return np.expand_dims(state, axis=0)

    def remember(self, state, action, reward, next_state, done, eps):
        self.memory.append((state, action, reward, next_state, done, eps))

    def train(self):
        itx = 0
        for episode in range(self.max_episodes):
            print('Training at episode: {}'.format(episode + 1))
            s = self.env.reset()
            for t in range(self.max_steps_per_episode):
                # Epsilon greedy with eps = 1/(itx+1)
                action, eps = self.epsilon_greedy(itx, s)
                # self.env.render()
                s_, r, d, _ = self.env.step(action)
                self.remember(self.preprocess(s),
                              action,
                              r,
                              self.preprocess(s_),
                              d, eps)
                s = s_

                self.train_step()
                itx += 1
                if d:
                    print('Episode length: {}'.format(t + 1))
                    break
        print('Finished training!')

    def train_step(self):
        batch_x, batch_y = [], []
        batch_size = min(len(self.memory), self.batch_size)
        mini_batch = random.sample(self.memory, batch_size)
        with tf.GradientTape() as tape:
            for state, action, reward, next_state, done, eps in mini_batch:
                # q_table = self.target_agent.predict(state)[0]
                q_value = self.agent(state, training=True)[0][action]
                if done:
                    q_target = reward
                else:
                    q_target = tf.stop_gradient(reward + self.gamma * tf.reduce_max(self.target_agent(next_state)[0]))
                # q_table[action] = q_target
                batch_x.append(q_value)
                batch_y.append(q_target)
            x = tf.stack(batch_x)
            y = tf.stack(batch_y)

            # loss = MeanSquaredError()(y, x) / 2
            loss = tf.reduce_mean(tf.square(y - x)) / 2
            grads = tape.gradient(loss, self.agent.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))

        #     # Copy weights to target_agent
        self.target_agent.set_weights(self.agent.get_weights())


class DoubleDQN(DQN):
    def __init__(self):
        super(DoubleDQN, self).__init__()

    def epsilon_greedy(self, t, s):
        eps = max(0.1, 1. / (t + 1))
        # eps = 0.1
        s = np.expand_dims(np.asarray(s), axis=0)
        action_space = np.squeeze(self.agent(s).numpy() + self.target_agent(s).numpy())
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = np.random.choice(list(range(len(action_space))))
        else:
            # Perform greedy action
            action = np.argmax(action_space)
        return action, eps

    def train_step(self):
        batch_x, batch_y = [], []
        batch_size = min(len(self.memory), self.batch_size)
        mini_batch = random.sample(self.memory, batch_size)
        if np.random.uniform(0, 0.999) < 0.5:
            with tf.GradientTape() as tape:
                for state, action, reward, next_state, done, eps in mini_batch:
                    # q_table = self.target_agent.predict(state)[0]
                    q_value = self.agent(state, training=True)[0][action]
                    if done:
                        q_target = reward
                    else:
                        q_target = tf.stop_gradient(reward + self.gamma * tf.reduce_max(self.target_agent(next_state)[0]))
                    # q_table[action] = q_target
                    batch_x.append(q_value)
                    batch_y.append(q_target)
                x = tf.stack(batch_x)
                y = tf.stack(batch_y)

                # loss = MeanSquaredError()(y, x) / 2
                loss = tf.reduce_mean(tf.square(y - x)) / 2
                grads = tape.gradient(loss, self.agent.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.agent.trainable_variables))
        else:
            with tf.GradientTape() as tape:
                for state, action, reward, next_state, done, eps in mini_batch:
                    # q_table = self.target_agent.predict(state)[0]
                    q_value = self.target_agent(state, training=True)[0][action]
                    if done:
                        q_target = reward
                    else:
                        q_target = tf.stop_gradient(
                            reward + self.gamma * tf.reduce_max(self.agent(next_state)[0]))
                    # q_table[action] = q_target
                    batch_x.append(q_value)
                    batch_y.append(q_target)
                x = tf.stack(batch_x)
                y = tf.stack(batch_y)

                # loss = MeanSquaredError()(y, x) / 2
                loss = tf.reduce_mean(tf.square(y - x)) / 2
                grads = tape.gradient(loss, self.target_agent.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.target_agent.trainable_variables))



def play(environment, policy=None, num_steps=1000):
    """
    Method used to play cartpole using a given policy

    :param policy: Class that takes in the state and reward and outputs an action
    :param num_steps: Number of steps to run the env for
    :return: None
    """
    s = environment.reset()

    if policy == None:
        for step in range(num_steps):
            environment.render()
            s, r, d, _ = environment.step(environment.action_space.sample())
        return

    for step in range(num_steps):
        environment.render()
        s = np.expand_dims(np.asarray(s), axis=0)
        # print('Space: ', s.shape)
        actions = np.squeeze(policy(s, training=False).numpy())
        action = np.argmax(actions)
        s, r, d, _ = environment.step(action)
        if d:
            print('Replay Finished at time step: {}'.format(step + 1))
            sys.exit()


if __name__ == '__main__':
    weights_path = './weights'
    num_steps = 200
    solver = DoubleDQN()

    # Load weights in
    # solver.update_agent_weights(weights_path)

    # Train
    solver.train()
    solver.save_agent(weights_path)
    agent = solver.get_agent()
    agent.load_weights(weights_path)
    env = solver.create_env()
    play(env, agent, num_steps)
