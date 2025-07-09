from agents.base_agent import BaseAgent
import tensorflow as tf
import json
from utils.utils import flatten_state
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # 非インタラクティブバックエンドを使用
import matplotlib.pyplot as plt
from PIL import Image
import io
from datetime import datetime

class A2CAgent(BaseAgent):
  def __init__(
      self, 
      env, 
      algorithm,
      model, 
      optimizer, 
      gamma, 
      n_steps, 
      max_steps_per_episode,
      action_space
    ):
    # ---- gpu logs -----
    print("[TensorFlow Device Check]")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      print(f"✅ GPU available: {gpus}")
    else:
      print("❌ No GPU detected. Training will use CPU.")
    # ----- request settings -----
    self.env                   = env
    self.algorithm             = algorithm
    self.model                 = model
    self.optimizer             = optimizer
    self.gamma                 = gamma
    self.n_steps               = n_steps
    self.max_steps_per_episode = max_steps_per_episode
    self.action_space          = action_space
    
    # Initialize GIF frames list
    self.env_frames = []
    self.frame_size = None
  

  def get_action(self, state, episode, log_dir: str | None = None): # stateは辞書型で送られてくる
    # 状態をテンソルに変換
    state_vec = tf.convert_to_tensor([flatten_state(state)], dtype=tf.float32)

    # モデルから平均・分散を取得
    mu, std, _ = self.model(state_vec)

     # パラメータをサンプリング
    sampled_params, _ = self.model.sample_action(mu, std)
    sampled_params_np = sampled_params[0].numpy()

    # アルゴリズムにより行動を決定
    action_tensor, action_dict = self.algorithm.policy(state, sampled_params_np, episode=episode, log_dir=log_dir)

    # Tensor（学習用）と Dict（環境用）を返す
    return action_tensor, action_dict
  

  def collect_trajectory(self, initial_state, episode=0, csv_path=None):
    states, actions, rewards, dones, next_states = [], [], [], [], []
    state = initial_state
    step_logs = []

    for step_idx in range(self.n_steps):
        action_tensor, action_dict = self.get_action(state, episode, log_dir=csv_path)
        next_state, reward, done, turncated, infos = self.env.step(action_dict)

        # リアルタイム描画は無効化（コメントアウト）
        # if hasattr(self.env, "render") and self.env._render_flag:
        #     self.env.render(ax=self.env._render_ax)
        # if hasattr(self.algorithm, "render") and self.algorithm._render_flag:
        #     self.algorithm.render(ax_params=self.algorithm._ax_params, ax_polar=self.algorithm._ax_polar)
        
        # リアルタイム描画の更新（無効化）
        # plt.pause(0.01)  # 10ms待機して描画を更新
        
        # GIF保存用のフレームキャプチャ（有効）
        self.capture_frame()

        # ログ構築
        log_row = {"step": step_idx}
        for key, val in state.items():
            val_arr = np.array(val).flatten()
            for i, v in enumerate(val_arr):
                log_row[f"s_{key}_{i}" if len(val_arr) > 1 else f"s_{key}"] = v
        for key, val in action_dict.items():
            log_row[f"a_{key}"] = val
        log_row["reward"] = reward
        log_row["done"] = done
        step_logs.append(log_row)

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
    
    # ログ保存（エピソード単位）
    if csv_path:
        self.save_trajectory_to_csv(csv_path, episode=episode, step_logs=step_logs)

    return states, actions, rewards, dones, next_states, state, step_logs
  

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
  

  def train_one_episode(self, episode: int = 0, log_dir: str | None = None):
    # GIF保存用のフレーム初期化（リアルタイム描画は無効）
    self.env_frames = []
    # self.current_fig = self.setup_rendering()  # リアルタイム描画は無効

    state = self.env.reset()
    total_reward = 0
    done = False
    step_count = 0
    episode_step_logs = []  # エピソード全体のログを収集

    while not done and step_count < self.max_steps_per_episode:
      states, actions, rewards, dones, next_states, next_state, step_logs = self.collect_trajectory(state, episode=episode, csv_path=None)  # CSV保存は後で行う
      returns, advantages = self.compute_returns_and_advantages(states, rewards, dones, next_states)

      self.model.train_step(
        self.optimizer,
        tf.convert_to_tensor(states, dtype=tf.float32),
        tf.convert_to_tensor(actions, dtype=tf.float32),
        tf.convert_to_tensor(returns, dtype=tf.float32),
        tf.convert_to_tensor(advantages, dtype=tf.float32),
      )

      # エピソード全体のログに追加
      episode_step_logs.extend(step_logs)

      total_reward += sum(rewards)
      done = dones[-1]
      state = next_state if not done else self.env.reset()
      step_count += 1 # ステップ数更新
    
    # エピソードごとのステップ数を記録
    self.step_count = step_count
    
    # エピソード全体の軌跡CSVを保存
    if log_dir:
        self.save_episode_trajectory_csv(log_dir, episode, episode_step_logs)
    
    # 終了後にログ出力
    self.save_gif(log_dir, episode)  # GIF保存は有効
    self.save_metrics(log_dir, episode, total_reward)
    self.save_model(log_dir, episode)
    self.log_tensorboard(log_dir, episode, total_reward)
    
    # リアルタイム描画を無効化
    # plt.ioff()
    
    return total_reward


  def train(self, episodes, csv_path):
     for episode in range(episodes):
        total_reward = self.train_one_episode(episode=episode, log_dir=csv_path)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
  

  def setup_rendering(self):
    self.env.capture_frame = self.capture_frame
    plt.ion() # リアルタイム描画を有効化

    # 3列 x 4行 のグリッドに調整（2行目はスペーサー）
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[2.5, 1, 0.2, 1.5])  # 2行目に空行

    ax_env = fig.add_subplot(gs[0, :])  # 環境は横幅全部

    ax_params = [
        fig.add_subplot(gs[1, 0]),  # th
        fig.add_subplot(gs[1, 1]),  # k_e
        fig.add_subplot(gs[1, 2])   # k_c
    ]

    ax_polar = [
        fig.add_subplot(gs[3, 0], polar=True),  # drivability
        fig.add_subplot(gs[3, 1], polar=True),  # exploration
        fig.add_subplot(gs[3, 2], polar=True)   # result
    ]

    fig.subplots_adjust(hspace=1.2)  # 余白を広めに
    fig.tight_layout()

    self.env._render_ax = ax_env
    self.env._render_flag = True
    self.algorithm._ax_params = ax_params
    self.algorithm._ax_polar = ax_polar
    self.algorithm._render_flag = True

    fig.show() # リアルタイム描画を有効化
    return fig
  

  def capture_frame(self):
    # GIF保存用のフレームキャプチャ（リアルタイム描画なし）
    # 環境の描画を非表示で実行してフレームをキャプチャ
    if hasattr(self.env, "render"):
        try:
            # rgb_arrayモードで環境を描画
            img_array = self.env.render(mode='rgb_array', fig_size=8)
            
            # numpy配列をPIL Imageに変換
            img = Image.fromarray(img_array)
            
            # サイズ統一処理
            if hasattr(self, "frame_size") and self.frame_size is not None:
                img = img.resize(self.frame_size)
            else:
                self.frame_size = img.size  # 初回にサイズを保存
            
            self.env_frames.append(img)
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            # フォールバック: 基本的な描画
            fig, ax = plt.subplots(figsize=(8, 8))
            self.env.render(ax=ax)
            
            # フレームをキャプチャ
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            
            # サイズ統一処理
            if hasattr(self, "frame_size") and self.frame_size is not None:
                img = img.resize(self.frame_size)
            else:
                self.frame_size = img.size  # 初回にサイズを保存
            
            self.env_frames.append(img)
            buf.close()
            plt.close(fig)  # メモリリーク防止
            plt.clf()  # 追加のクリーンアップ
  

  def save_gif(self, save_dir: str | None, episode: int):
    from pathlib import Path
    if save_dir is None:
        return
        
    gif_dir = Path(save_dir) / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"episode_{episode:04d}.gif"
    
    if self.env_frames:
        self.env_frames[0].save(
            gif_path,
            save_all=True,
            append_images=self.env_frames[1:],
            duration=100,  # 100ms/frame
            loop=0
        )
        print(f"[Saved GIF] {gif_path}")
    
    self.env_frames.clear()
  

  def save_trajectory_to_csv(
    self,
    csv_path: str,
    episode: int,
    step_logs: List[Dict[str, Any]]
):
    """
    軌跡ログをCSVに保存する（1エピソード分）

    Parameters:
    - csv_path: 保存先パス
    - episode: エピソード番号（ログに含める）
    - step_logs: 各ステップの状態・行動・報酬などのリスト
    """
    df = pd.DataFrame(step_logs)
    df.insert(0, "episode", episode)
    csv_dir = os.path.join(csv_path, "csvs")
    os.makedirs(csv_dir, exist_ok=True)
    file_path = os.path.join(csv_dir, f"trajectory_episode_{episode:04d}.csv")
    df.to_csv(file_path, index=False)
    print(f"Trajectory CSV saved to: {file_path}")


  def save_episode_trajectory_csv(
    self,
    log_dir: str,
    episode: int,
    step_logs: List[Dict[str, Any]]
):
    """
    エピソード全体の軌跡ログをCSVに保存する

    Parameters:
    - log_dir: 保存先ディレクトリ
    - episode: エピソード番号
    - step_logs: エピソード全体の各ステップの状態・行動・報酬などのリスト
    """
    if not step_logs:
        print(f"No trajectory data for episode {episode}")
        return
    
    df = pd.DataFrame(step_logs)
    df.insert(0, "episode", episode)
    
    # グローバルステップ番号を追加
    df["global_step"] = list(range(len(df)))
    
    csv_dir = os.path.join(log_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)
    file_path = os.path.join(csv_dir, f"trajectory_episode_{episode:04d}.csv")
    df.to_csv(file_path, index=False)
    print(f"Episode trajectory CSV saved to: {file_path}")
  

  # ----- データ保存用関数 -----
  def save_metrics(self, log_dir, episode, reward):
    metrics_dir = os.path.join(log_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"episode_{episode:04d}.json")
    
    # 環境から詳細なmetricsを取得
    env_metrics = {}
    if hasattr(self.env, 'scorer'):
        scorer = self.env.scorer
        env_metrics = {
            "final_exploration_rate": scorer.exploration_rate[-1] if scorer.exploration_rate else 0.0,
            "max_exploration_rate": max(scorer.exploration_rate) if scorer.exploration_rate else 0.0,
            "total_steps": len(scorer.exploration_rate),
            "goal_reaching_step": scorer.goal_reaching_step,
            "agent_collision_count": scorer.agent_collision_count,
            "follower_collision_count": scorer.follower_collision_count,
            "total_distance_traveled": scorer.total_distance_traveled,
            "revisit_ratio": scorer.revisit_ratio,
            "revisit_count": scorer.revisit_count
        }
    
    # 環境の状態情報を取得
    env_state = {}
    if hasattr(self.env, 'exploration_ratio'):
        env_state["exploration_ratio"] = self.env.exploration_ratio
    if hasattr(self.env, 'explored_area'):
        env_state["explored_area"] = self.env.explored_area
    if hasattr(self.env, 'total_area'):
        env_state["total_area"] = self.env.total_area
    if hasattr(self.env, 'agent_step'):
        env_state["agent_step"] = self.env.agent_step
    
    # 群の情報を取得
    swarm_info = {}
    if hasattr(self.env, 'swarms'):
        swarm_info = {
            "swarm_count": len(self.env.swarms),
            "swarm_details": []
        }
        for swarm in self.env.swarms:
            swarm_detail = {
                "swarm_id": swarm.swarm_id,
                "leader_id": swarm.leader.id,
                "follower_count": len(swarm.followers),
                "total_robots": swarm.get_robot_count(),
                "exploration_rate": swarm.exploration_rate,
                "step_count": swarm.step_count
            }
            swarm_info["swarm_details"].append(swarm_detail)
    
    # 学習関連の情報
    learning_info = {
        "step_count": getattr(self, 'step_count', 0),
        "model_updated": True  # A2Cでは毎エピソード更新
    }
    
    # 総合metrics
    metrics = {
        "episode": episode,
        "total_reward": reward,
        "environment_metrics": env_metrics,
        "environment_state": env_state,
        "swarm_information": swarm_info,
        "learning_info": learning_info,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


  def save_model(self, log_dir, episode):
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"checkpoint_{episode:04d}.keras")
    self.model.save(path)


  def log_tensorboard(self, log_dir, episode, reward):
    tb_dir = os.path.join(log_dir, "tensorboard")
    writer = tf.summary.create_file_writer(tb_dir)
    with writer.as_default():
        # 基本情報
        tf.summary.scalar("Total Reward", reward, step=episode)
        
        # 環境の詳細情報
        if hasattr(self.env, 'scorer'):
            scorer = self.env.scorer
            tf.summary.scalar("Final Exploration Rate", scorer.exploration_rate[-1] if scorer.exploration_rate else 0.0, step=episode)
            tf.summary.scalar("Max Exploration Rate", max(scorer.exploration_rate) if scorer.exploration_rate else 0.0, step=episode)
            tf.summary.scalar("Total Steps", len(scorer.exploration_rate), step=episode)
            tf.summary.scalar("Agent Collision Count", scorer.agent_collision_count, step=episode)
            tf.summary.scalar("Follower Collision Count", scorer.follower_collision_count, step=episode)
            tf.summary.scalar("Total Distance Traveled", scorer.total_distance_traveled, step=episode)
            tf.summary.scalar("Revisit Ratio", scorer.revisit_ratio, step=episode)
        
        # 環境状態
        if hasattr(self.env, 'exploration_ratio'):
            tf.summary.scalar("Current Exploration Ratio", self.env.exploration_ratio, step=episode)
        if hasattr(self.env, 'explored_area'):
            tf.summary.scalar("Explored Area", self.env.explored_area, step=episode)
        if hasattr(self.env, 'total_area'):
            tf.summary.scalar("Total Area", self.env.total_area, step=episode)
        
        # 群の情報
        if hasattr(self.env, 'swarms'):
            tf.summary.scalar("Swarm Count", len(self.env.swarms), step=episode)
            for i, swarm in enumerate(self.env.swarms):
                tf.summary.scalar(f"Swarm_{i}_Follower_Count", len(swarm.followers), step=episode)
                tf.summary.scalar(f"Swarm_{i}_Total_Robots", swarm.get_robot_count(), step=episode)
                tf.summary.scalar(f"Swarm_{i}_Exploration_Rate", swarm.exploration_rate, step=episode)
        
        # 学習情報
        tf.summary.scalar("Episode Steps", getattr(self, 'step_count', 0), step=episode)
        
        writer.flush()
  

  def save(self, path: str):
    self.model.save(path)


  def load(self, path: str):
    self.model.load(path)


