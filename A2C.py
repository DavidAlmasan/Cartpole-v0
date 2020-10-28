import sys, os
import numpy as np
from collections import deque
from itertools import accumulate
import random
import tqdm
import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import datasets, layers, models


def env_step(action, env):
    """Returns state, reward and done flag given an action."""

    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


def tf_env_step(action, env):
    return tf.numpy_function(env_step, [action, env],
                             [tf.float32, tf.int32, tf.int32])


class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super(ActorCritic, self).__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


class A2CSolver():
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
        self.actor_critic, self.optimizer = self.create_agent()

        # loss
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def preprocess(self, state):
        return np.expand_dims(state, axis=0)

    def create_env(self):
        env = gym.make('CartPole-v0')
        if self.max_steps_per_episode is not None:
            env._max_episode_steps = self.max_steps_per_episode
        env.reset()
        return env

    def create_agent(self):
        agent = ActorCritic(2, 128)
        optimizer = RMSprop(self.learning_rate)

        return agent, optimizer

    def create_episode(self):
        state = self.env.reset()
        done = False
        t = 0
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        while not done:
            # Get actions
            action_space, value = self.actor_critic(self.preprocess(state))
            action_space_probs_t = tf.nn.softmax(action_space)

            # Sample action. idk why random and not eps greedy
            action = tf.random.categorical(action_space, 1)[0, 0]

            # Apply action into env
            state, reward, done, _ = tf_env_step(action, self.env)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_space_probs_t[0, action])

            # Store reward
            rewards = rewards.write(t, reward)

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(self, rewards, standardize: bool = True):
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + 1e-5))

        return returns

    def compute_loss(self, action_probs, values, returns):
        """Computes the combined actor-critic loss."""
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    @tf.function
    def train_step(self) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.create_episode()
            # Calculate expected returns
            returns = self.get_expected_return(rewards)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.actor_critic.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward

    def train(self):
        reward_threshold = 195
        running_reward = 0
        with tqdm.trange(self.max_episodes) as t:
            for i in t:
                episode_reward = int(self.train_step())

                running_reward = episode_reward * 0.01 + running_reward * .99

                t.set_description(f'Episode {i}')
                t.set_postfix(
                    episode_reward=episode_reward, running_reward=running_reward)

                # Show average episode reward every 10 episodes
                if i % 10 == 0:
                    pass  # print(f'Episode {i}: average reward: {avg_reward}')

                if running_reward > reward_threshold:
                    break

        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

if __name__ == '__main__':
    solver = A2CSolver()
    solver.train()