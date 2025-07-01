from params.agent import AgentParam
from enum import Enum
from models.model import Model
from algorithms.algorithm_factory import select_algorithm
from agents.agent_factory import create_agent
from envs.env import Env
import time
import json
import os
import numpy as np
import tensorflow as tf


class LearningType(str, Enum):
  A2C = "a2c"

class AgentConfig:
  def __init__(self, env: Env, param: AgentParam = AgentParam()):
    self.__env                   = env
    self.__algorithm             = select_algorithm(param.algorithm) # 名前が違う場合はerrorを出す
    self.__is_learning           = param.isLearning
    self.__max_steps_per_episode = param.maxStepsPerEpisode
    self.__learning_episode_num  = 1 # 学習なしの場合

    # 学習ありの場合
    if self.__is_learning:
      self.__learning_type         = param.learningParameter.type
      self.__model                 = Model(param.learningParameter.model)
      self.__learning_late         = param.learningParameter.learningLate
      self.__learning_episode_num  = param.learningParameter.episodeNum

      self.__init__learning_agent(
        env                   = self.__env,
        algorithm             = self.__algorithm,
        type                  = self.__learning_type,
        model                 = self.__model,
        learning_late         = self.__learning_late,
        optimizer             = param.learningParameter.optimizer,
        gamma                 = param.learningParameter.gamma,
        n_steps               = param.learningParameter.nStep,
        max_steps_per_episode = self.__max_steps_per_episode,
        action_space          = env.action_space
      )
    else:
      # 非学習でも policy アルゴリズムインスタンスを生成
      self.__agent = self.__algorithm
    
  
  def __init__learning_agent(self, env, algorithm, type, optimizer, model, learning_late,gamma, n_steps, max_steps_per_episode, action_space):
    """
    学習エージェントを設定
    """
    self.__agent = create_agent(
        agent_type=type,
        env=env,
        algorithm=algorithm,
        model=model,
        optimizer=optimizer,
        learning_late=learning_late,
        gamma=gamma,
        n_steps=n_steps,
        max_steps_per_episode=max_steps_per_episode,
        action_space=action_space
    )
  
  
  def get_action(self, state, episode, log_dir: str = None):
    """
    行動を取得する
    """
    if self.__is_learning:
      action_tensor, action_dict = self.__agent.get_action(state, episode, log_dir)
    else:
      action_tensor, action_dict = self.__agent.policy(state, episode, log_dir)
    
    return action_tensor, action_dict
      
    

    
  def train_one_episode(self, episode: int, log_dir: str):
    """
    oneエピソード分の学習
    """
    total_reward = self.__agent.train_one_episode(episode, log_dir=log_dir) # 報酬を返している

    # TODO 非学習の場合の処理を追加

    # === GIF 保存 ===
    self.__env.save_gif(log_dir=log_dir, episode=episode)

    return total_reward
    
  

  def train(self, log_dir: str):
    """
    学習
    """
    all_rewards = []
    self.__summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, "tensorboard"))
    start_time = time.time()

    for episode in range(self.__learning_episode_num):
        total_reward = self.train_one_episode(episode, log_dir=log_dir)
        all_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        with self.__summary_writer.as_default():
          tf.summary.scalar("Episode Reward", total_reward, step=episode)
          tf.summary.scalar("Final Exploration Rate", self.__env.scorer.exploration_rate[-1] if self.__env.scorer.exploration_rate else 0.0, step=episode)
          tf.summary.scalar("Follower Collisions", self.__env.scorer.follower_collision_count, step=episode)
          tf.summary.scalar("Agent Collisions", self.__env.scorer.agent_collision_count, step=episode)

    end_time = time.time()
    training_duration = end_time - start_time

    self.save_summary(
        log_dir=log_dir,
        rewards=all_rewards,
        duration=training_duration
    )

    self.__summary_writer.close()

    # --- スコア保存 ---
    metrics_path = os.path.join(log_dir, "metrics", "score.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
      json.dump(self.__env.scorer.export_metrics(), f, indent=2)
    
    # モデル保存
    self.save_model(log_dir=log_dir)
  

  def save_summary(self, log_dir: str, rewards: list[float], duration: float):
    scorer = self.__env.scorer
    summary = {
        "total_episodes": len(rewards),
        "average_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "total_time_seconds": round(duration, 2),
        "total_time_hms": f"{int(duration // 3600):02}:{int((duration % 3600) // 60):02}:{int(duration % 60):02}",
        # === 詳細スコア ===
        "goal_reaching_step": scorer.goal_reaching_step,
        "agent_collision_count": scorer.agent_collision_count,
        "follower_collision_count": scorer.follower_collision_count,
        "total_distance_traveled": round(scorer.total_distance_traveled, 2),
        "final_exploration_ratio": round(scorer.exploration_rate[-1], 4) if scorer.exploration_rate else 0.0
    }


    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


  def save_model(self, log_dir: str):
    """
    モデルの保存（学習ありの場合のみ）
    """
    if not self.__is_learning:
        return

    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "agent_model.keras")
    self.__model.model.save(model_path)
    print(f"Model saved to {model_path}")