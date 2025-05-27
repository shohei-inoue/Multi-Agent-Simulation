import tensorflow as tf
import numpy as np
from utils.utils import flatten_state

class A2CAgent:
    def __init__(self, env, model, optimizer, gamma=0.99, n_steps=5, max_steps_per_episode=10):
        self.env = env # 環境を追加
        self.model = model # モデルを追加
        self.optimizer = optimizer # Optimizerを追加
        self.gamma = gamma # 割引率
        self.n_steps = n_steps # n_stepsを追加
        self.max_steps_per_episode = max_steps_per_episode  # 最大ステップ数を追加 

    def get_action(self, state):
        state_vec = tf.convert_to_tensor([flatten_state(state)], dtype=tf.float32)
        mu, std, _ = self.model(state_vec)
        action, _ = self.model.sample_action(mu, std)
        action_dict = {
            "th": action[0, 0].numpy(),
            "k_e": action[0, 1].numpy(),
            "k_c": action[0, 2].numpy(),
        }
        return action[0], action_dict

    def collect_trajectory(self, initial_state):
        states, actions, rewards, dones, next_states = [], [], [], [], []
        state = initial_state

        for _ in range(self.n_steps):
            action_tensor, action_dict = self.get_action(state)
            next_state, reward, done, _, _ = self.env.step(action_dict)

            states.append(flatten_state(state))
            actions.append(action_tensor.numpy())
            rewards.append(reward)
            dones.append(done)
            next_states.append(flatten_state(next_state))

            if done:
                state = self.env.reset()
                break
            else:
                state = next_state

        return states, actions, rewards, dones, next_states, state

    def compute_returns_and_advantages(self, states, rewards, dones, next_states):
        returns = []
        advantages = []

        last_state_tensor = tf.convert_to_tensor([next_states[-1]], dtype=tf.float32)
        _, _, last_value = self.model(last_state_tensor)
        last_value = last_value.numpy()[0, 0]

        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        _, _, values = self.model(states_tensor)
        values = tf.squeeze(values, axis=-1).numpy()

        gae = 0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_value - values[t]
            gae = delta + self.gamma * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32)

    def train_one_episode(self):
        state = self.env.reset()
        total_reward = 0
        done = False
        step_count = 0  # ★ ステップカウントを初期化

        while not done and step_count < self.max_steps_per_episode:
            states, actions, rewards, dones, next_states, next_state = self.collect_trajectory(state)
            returns, advantages = self.compute_returns_and_advantages(states, rewards, dones, next_states)

            self.model.train_step(
                self.optimizer,
                tf.convert_to_tensor(states, dtype=tf.float32),
                tf.convert_to_tensor(actions, dtype=tf.float32),
                tf.convert_to_tensor(returns, dtype=tf.float32),
                tf.convert_to_tensor(advantages, dtype=tf.float32),
            )

            total_reward += sum(rewards)
            done = dones[-1]
            state = next_state if not done else self.env.reset()
            step_count += 1  # ★ ステップ数更新

        return total_reward

    def train(self, episodes):
        for episode in range(episodes):
            total_reward = self.train_one_episode()
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")