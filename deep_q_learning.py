import sys, os
from os.path import join
import numpy as np
from collections import deque
import random
from termcolor import *
import colorama

import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import datasets, layers, models

from agent import Agent
from utils import bottom_n_percent


colorama.init()
CUR = os.path.abspath(os.path.dirname(__file__))


class DQN():
    def __init__(self, save_path, weights_path, use_dueling=True):
        # Hyperparams
        self.learning_rate = 0.00025
        self.gamma = 1
        self.eps_red_factor = 0.999
        self.batch_size = 32

        # Training params
        self.max_steps_per_episode = 500
        self.max_episodes = 200
        self.wait_episodes = 10
        self.log_episodes = [int(float(self.max_episodes) * i / 10) for i in range(11)]
        self.memory = deque(maxlen=20000)

        # Env and agent
        self.env = self.create_env()
        self.use_dueling = use_dueling
        self.is_double = False
        self.ddqn_prob = 0  # Overloaded by ddqn to be 0.5
        self.agent, self.target_agent, self.optimizer = self.create_agent(use_dueling)

        # Misc
        self.save_path = save_path
        self.weights_path = weights_path

    def validate_agent(self):
        steps = []
        games = 100
        perc = 1
        for _ in range(games):
            steps.append(play(self.env, self.agent, 200, False))
        print('Average timesteps of {} games : {}'.format(games, colored(np.mean(steps), 'green')))
        print('STD of  timesteps of {} games : {}'.format(games, colored(float("{:.2f}".format(np.std(steps))),
                                                                         'green')))
        bot_mean = bottom_n_percent(steps, perc)
        print('Average timesteps of bottom {}% games : {}'.format(perc, colored(bot_mean, 'yellow')))
        print('-----------------------------------')
        return np.mean(steps), float("{:.2f}".format(np.std(steps))), bot_mean

    def update_agent_weights(self, epoch=None):
        ckpt_file = tf.train.latest_checkpoint(self.weights_path)
        if epoch is not None:
            ckpt_file = ckpt_file[:-6] + str(epoch) + '.ckpt'
        print('Loading weights from file: {}'.format(ckpt_file))
        self.agent.load_weights(ckpt_file)
        self.target_agent.load_weights(ckpt_file)

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

    def save_agent(self, path, epoch):
        path = join(path, 'dqn-epoch_{}.ckpt'.format(epoch))
        self.agent.save_weights(path)

    def create_agent(self, use_dueling):
        arch = [512, 256, 64]
        agent = Agent(4, arch, 2, use_dueling)
        target_agent = Agent(4, arch, 2, use_dueling)
        optimizer = Adam(self.learning_rate)

        return agent, target_agent, optimizer

    def epsilon_greedy(self, t, s):
        eps = max(0.1, self.eps_red_factor ** t)
        eps = float("{:.2f}".format(eps))
        s = self.preprocess(s)
        action_space = np.squeeze(self.agent(s).numpy())
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = self.env.action_space.sample()
        else:
            # Perform greedy action
            action = np.argmax(action_space)
        return action, eps

    def preprocess(self, state):
        return np.expand_dims(state, axis=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        itx = 0
        with open(self.save_path, 'w') as loss_file:
            loss_file.write('Params: ' + '\n')
            loss_file.write('is_double: {}'.format(self.is_double) + '\n')
            loss_file.write('dueling: {}'.format(self.use_dueling) + '\n')
            loss_file.write('learning_rate: {}'.format(self.learning_rate) + '\n')
            loss_file.write('gamma: {}'.format(self.gamma) + '\n')
            loss_file.write('batch_size: {}'.format(self.batch_size) + '\n')

        for episode in range(self.max_episodes):
            loss = 0.
            print('Training at episode: {}'.format(episode + 1))
            s = self.env.reset()
            for t in range(self.max_steps_per_episode):
                # Epsilon greedy with eps = 1/(itx+1)
                action, eps = self.epsilon_greedy(itx, s)
                s_, r, d, _ = self.env.step(action)
                if d: r = -100
                self.remember(self.preprocess(s),
                              action,
                              r,
                              self.preprocess(s_),
                              d)
                s = s_
                if episode >= self.wait_episodes:
                    if np.random.uniform(0, 1) > self.ddqn_prob:
                        loss += self.train_step(self.agent, self.target_agent)
                    else:
                        loss += self.train_step(self.target_agent, self.agent)
                itx += 1
                if d:
                    loss = float("{:.2f}".format(loss / (t + 1)))
                    episode_str = 'Episode length: {} with final eps: {}, and loss {}'.format(t + 1, eps, loss)
                    print(episode_str)
                    mu, std, bot_mu = self.validate_agent()
                    log_str = 'mean: {}, std: {}, bottom_mean: {}, loss: {}'.format(mu, std, bot_mu, loss)

                    # if only dqn, save weights from agent to target_agent
                    if self.ddqn_prob == 0:
                        self.target_agent.set_weights(self.agent.get_weights())

                    with open(self.save_path, 'a') as loss_file:
                        loss_file.write(log_str + '\n')
                    break
            if episode in self.log_episodes:
                self.save_agent(self.weights_path, episode)

        print('Finished training!')

    def train_step(self, agent, target_agent):
        if len(self.memory) < self.batch_size:
            return 0.
        minibatch = random.sample(self.memory, self.batch_size)
        state = tf.stack([minibatch[i][0][0] for i in range(self.batch_size)])
        action = [minibatch[i][1] for i in range(self.batch_size)]
        reward = [minibatch[i][2] for i in range(self.batch_size)]
        next_state = tf.stack([minibatch[i][3][0] for i in range(self.batch_size)])
        done = [minibatch[i][4] for i in range(self.batch_size)]

        q_values_t = agent.predict(state)
        q_values_t1 = tf.math.argmax(agent.predict(next_state), axis=-1)

        for i in range(self.batch_size):
            if done[i]:
                q_values_t[i][action[i]] = reward[i]
            else:
                q_values_t[i][action[i]] = reward[i] +\
                                           self.gamma * target_agent.predict(next_state)[i][q_values_t1[i]]

        with tf.GradientTape() as tape:
            predictions = agent(state, training=True)
            loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(q_values_t, predictions)
            grads = tape.gradient(loss, agent.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, agent.trainable_variables))

            return loss


class DoubleDQN(DQN):
    def __init__(self, save_path, weights_path, use_dueling):
        super(DoubleDQN, self).__init__(save_path, weights_path, use_dueling)
        self.is_double = True
        self.ddqn_prob = 0.5

    def epsilon_greedy(self, t, s):
        eps = self.eps_red_factor ** t
        eps = max(0.01, eps)
        eps = float("{:.2f}".format(eps))
        s = self.preprocess(s)
        action_space = np.squeeze(self.agent(s).numpy() + self.target_agent(s).numpy())
        if eps <= np.random.uniform(0, 0.9999):
            # Perform explore action
            action = self.env.action_space.sample()
        else:
            # Perform greedy action
            action = np.argmax(action_space)
        return action, eps


def play(environment, policy=None, num_steps=1000, render=True):
    """
    :param environment: env
    :param policy: agent that plays in given env
    :param num_steps: steps before episode termination
    :param render: whether to render the env
    :return: length of the episode
    """
    s = environment.reset()

    if policy == None:
        for step in range(num_steps):
            if render:
                environment.render()
            s, r, d, _ = environment.step(environment.action_space.sample())
        return

    for step in range(num_steps):
        if render:
            environment.render()
        s = np.expand_dims(np.asarray(s), axis=0)
        # print('Space: ', s.shape)
        actions = np.squeeze(policy(s, training=False).numpy())
        action = np.argmax(actions)
        s, r, d, _ = environment.step(action)
        if d:
            if render:
                print('Replay Finished at time step: {}'.format(step + 1))
            return step + 1
    print('Achieved max steps!!')
    return num_steps


if __name__ == '__main__':
    use_dueling_network = True
    weights_path = join(CUR, 'big_dueling_ddqn_weights')
    save_file = join(CUR, 'big_dueling_ddqn_trained.txt')
    os.makedirs(weights_path, exist_ok=True)
    num_steps = 200

    # Instantiate the solver
    solver = DoubleDQN(save_file, weights_path, use_dueling_network)

    # Load weights in
    # solver.update_agent_weights(180)

    # Train
    solver.train()

    # Assess agent performance
    agent = solver.get_agent()
    solver.update_agent_weights()
    env = solver.create_env()
    play(env, agent, num_steps)
