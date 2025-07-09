"""
Agent configuration and management class.
Handles the creation and configuration of agents based on parameters.
"""

from typing import Dict, Any, Optional
from enum import Enum
import time
import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd

from params.agent import AgentParam
from models.model import Model
from core.factories import algorithm_factory, agent_factory
from core.logging import get_component_logger
from core.interfaces import Configurable, Stateful, Loggable


class LearningType(str, Enum):
    """Supported learning types"""
    A2C = "a2c"
    PPO = "ppo"
    DQN = "dqn"


class AgentConfig(Configurable, Stateful, Loggable):
    """Agent configuration and management class"""
    
    def __init__(self, env, param: AgentParam = AgentParam()):
        self.env = env
        self.param = param
        self.logger = get_component_logger("agent_config")
        
        # Configuration
        self._config = {
            "is_learning": param.isLearning,
            "max_steps_per_episode": param.maxStepsPerEpisode,
            "algorithm": param.algorithm
        }
        
        # State
        self._state = {
            "agent_created": False,
            "training_completed": False
        }
        
        # Components
        self.algorithm = None
        self.model = None
        self.agent = None
        
        # Initialize components
        self._setup_components()
    
    def _setup_components(self):
        """Setup algorithm, model, and agent components"""
        try:
            # Create algorithm
            self.algorithm = algorithm_factory.create(
                self.param.algorithm,
                env=self.env
            )
            self.logger.log_component_event("algorithm_created", {"type": self.param.algorithm})
            
            # Create model if learning is enabled
            if self.param.isLearning and self.param.learningParameter:
                self.model = Model(self.param.learningParameter.model)
                self.logger.log_component_event("model_created", {"type": self.param.learningParameter.model})
            
            # Create agent
            self._create_agent()
            
        except Exception as e:
            self.logger.log_component_event("setup_failed", {"error": str(e)})
            raise
    
    def _create_agent(self):
        """Create the appropriate agent based on configuration"""
        if self.param.isLearning and self.param.learningParameter:
            # Learning agent
            agent_kwargs = {
                "env": self.env,
                "algorithm": self.algorithm,
                "model": self.model,
                "optimizer": self.param.learningParameter.optimizer,
                "gamma": self.param.learningParameter.gamma,
                "n_steps": self.param.learningParameter.nStep,
                "max_steps_per_episode": self.param.maxStepsPerEpisode,
                "action_space": self.env.action_space
            }
            
            self.agent = agent_factory.create_agent(
                self.param.learningParameter.type,
                **agent_kwargs
            )
            
            self.logger.log_component_event("learning_agent_created", {
                "type": self.param.learningParameter.type,
                "episodes": self.param.learningParameter.episodeNum
            })
        else:
            # Non-learning agent (uses algorithm directly)
            self.agent = self.algorithm
            self.logger.log_component_event("non_learning_agent_created")
        
        self._state["agent_created"] = True
    
    def train(self, log_dir: str):
        """Train the agent if learning is enabled"""
        if not self.param.isLearning:
            self.logger.log_component_event("training_skipped", {"reason": "learning_disabled"})
            return
        
        if not self.agent or not hasattr(self.agent, 'train'):
            self.logger.log_component_event("training_failed", {"reason": "agent_not_trainable"})
            return
        
        try:
            self.logger.log_component_event("training_started", {"log_dir": log_dir})
            
            # Train the agent
            if hasattr(self.agent, 'train'):
                self.agent.train(log_dir=log_dir)
            else:
                # Fallback to episode-based training
                self._train_episodes(log_dir)
            
            self._state["training_completed"] = True
            self.logger.log_component_event("training_completed")
            
        except Exception as e:
            self.logger.log_component_event("training_failed", {"error": str(e)})
            raise
    
    def _train_episodes(self, log_dir: str):
        """Fallback training method for agents without built-in training"""
        episodes = self.param.learningParameter.episodeNum if self.param.learningParameter else 1
        
        for episode in range(episodes):
            self.logger.log_component_event("episode_started", {"episode": episode})
            
            # Run single episode
            state = self.env.reset()
            total_reward = 0
            step_count = 0
            
            while step_count < self.param.maxStepsPerEpisode:
                action_tensor, action_dict = self.agent.get_action(
                    state, episode, log_dir=log_dir
                )
                next_state, reward, done, truncated, info = self.env.step(action_dict)
                
                total_reward += reward
                state = next_state
                step_count += 1
                
                if done:
                    break
            
            self.logger.log_performance("episode_reward", total_reward, episode)
            self.logger.log_component_event("episode_completed", {
                "episode": episode,
                "total_reward": total_reward,
                "steps": step_count
            })
    
    def get_action(self, state: Dict[str, Any], episode: int = 0, log_dir: Optional[str] = None):
        """Get action from the agent"""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        
        return self.agent.get_action(state, episode, log_dir)
    
    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self._config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        self._config.update(config)
    
    # Stateful interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self._state.copy()
    
    def set_state(self, state: Dict[str, Any]):
        """Set state"""
        self._state.update(state)
    
    def reset_state(self):
        """Reset to initial state"""
        self._state.update({
            "agent_created": False,
            "training_completed": False
        })
    
    # Loggable interface implementation
    def get_log_data(self) -> Dict[str, Any]:
        """Get data for logging"""
        return {
            "config": self.get_config(),
            "state": self.get_state(),
            "agent_type": type(self.agent).__name__ if self.agent else None,
            "algorithm_type": type(self.algorithm).__name__ if self.algorithm else None,
            "model_type": type(self.model).__name__ if self.model else None
        }
    
    def save_log(self, path: str):
        """Save log data to file"""
        with open(path, 'w') as f:
            json.dump(self.get_log_data(), f, indent=2, default=str)
    
  
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
  
  
  def get_action(self, state, episode, log_dir: str | None = None):
    """
    行動を取得する
    """
    if hasattr(self.__agent, 'get_action'):
      action_tensor, action_dict = self.__agent.get_action(state, episode, log_dir)  # type: ignore
    else:
      # 非学習の場合、アルゴリズムのpolicyメソッドを使用
      # sampled_paramsは固定値を使用
      sampled_params = np.array([self.__algorithm.th, self.__algorithm.k_e, self.__algorithm.k_c])
      action_tensor, action_dict = self.__algorithm.policy(state, sampled_params, episode, log_dir)
    
    return action_tensor, action_dict
      
    

    
  def train_one_episode(self, episode: int, log_dir: str):
    """
    oneエピソード分の学習
    """
    if hasattr(self.__agent, 'train_one_episode'):
      total_reward = self.__agent.train_one_episode(episode, log_dir=log_dir)  # type: ignore
    else:
      # 非学習の場合の処理
      total_reward = self.__run_one_episode(episode, log_dir=log_dir)

    # エピソード終了時にCSVを保存
    self.__env.scorer.save_episode_csv(log_dir, episode)
    
    # エピソード全体のスコアサマリーを保存
    self.save_episode_summary(log_dir, episode, total_reward)

    return total_reward
    
  def __run_one_episode(self, episode: int, log_dir: str):
    """
    非学習の場合の1エピソード実行
    """
    state = self.__env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done and step_count < self.__max_steps_per_episode:
      action_tensor, action_dict = self.get_action(state, episode, log_dir)
      next_state, reward, done, truncated, info = self.__env.step(action_dict)
      
      total_reward += reward
      state = next_state
      step_count += 1

    return total_reward
    
  

  def train(self, log_dir: str):
    """
    学習
    """
    all_rewards = []
    all_episode_summaries = []
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

        # エピソードサマリーを収集
        episode_summary = {
            "episode": episode,
            "total_reward": total_reward,
            "final_exploration_rate": self.__env.scorer.exploration_rate[-1] if self.__env.scorer.exploration_rate else 0.0,
            "max_exploration_rate": max(self.__env.scorer.exploration_rate) if self.__env.scorer.exploration_rate else 0.0,
            "total_steps": len(self.__env.scorer.exploration_rate),
            "goal_reaching_step": self.__env.scorer.goal_reaching_step,
            "agent_collision_count": self.__env.scorer.agent_collision_count,
            "follower_collision_count": self.__env.scorer.follower_collision_count,
            "total_distance_traveled": self.__env.scorer.total_distance_traveled
        }
        all_episode_summaries.append(episode_summary)

    end_time = time.time()
    training_duration = end_time - start_time

    self.save_summary(
        log_dir=log_dir,
        rewards=all_rewards,
        duration=training_duration
    )
    
    # 全エピソードのスコアをCSVとして保存
    self.save_all_episodes_csv(log_dir, all_episode_summaries)

    self.__summary_writer.close()

    # --- スコア保存 ---
    metrics_path = os.path.join(log_dir, "metrics", "score.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
      json.dump(self.__env.scorer.export_metrics(), f, indent=2)
    
    # モデル保存
    self.save_model(log_dir=log_dir)
    
  def save_all_episodes_csv(self, log_dir: str, all_episode_summaries: list):
    """
    全エピソードのスコアをCSVとして保存
    """
    csv_dir = os.path.join(log_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)
    
    # DataFrameに変換
    df = pd.DataFrame(all_episode_summaries)
    
    # CSVとして保存
    csv_path = os.path.join(csv_dir, "all_episodes_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"All episodes summary saved to: {csv_path}")

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

  def save_episode_summary(self, log_dir: str, episode: int, total_reward: float):
    """
    エピソード全体のスコアサマリーを保存
    """
    scorer = self.__env.scorer
    
    # エピソードサマリー
    episode_summary = {
        "episode": episode,
        "total_reward": total_reward,
        "final_exploration_rate": scorer.exploration_rate[-1] if scorer.exploration_rate else 0.0,
        "max_exploration_rate": max(scorer.exploration_rate) if scorer.exploration_rate else 0.0,
        "total_steps": len(scorer.exploration_rate),
        "goal_reaching_step": scorer.goal_reaching_step,
        "agent_collision_count": scorer.agent_collision_count,
        "follower_collision_count": scorer.follower_collision_count,
        "total_distance_traveled": scorer.total_distance_traveled,
        "exploration_rate_curve": scorer.exploration_rate
    }
    
    # JSONとして保存
    summary_dir = os.path.join(log_dir, "metrics")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"episode_{episode:04d}_summary.json")
    
    with open(summary_path, "w") as f:
        json.dump(episode_summary, f, indent=2, default=str)
    
    print(f"Episode {episode} summary saved to: {summary_path}")