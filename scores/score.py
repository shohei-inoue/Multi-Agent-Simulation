"""
Score calculation and evaluation utilities.
Provides tools for calculating and tracking performance scores.
"""

import pandas as pd
import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from params.robot_logging import RobotLoggingConfig
from core.interfaces import Configurable, Stateful, Loggable
from core.logging import get_component_logger


class Score(Configurable, Stateful, Loggable):
    def __init__(self, robot_logging_config: RobotLoggingConfig | None = None):
        self.goal_reaching_step: int | None = None # 目標の探査率に到達したステップ
        self.revisit_ratio = 0.0 # 同じセルへの再訪問率
        self.revisit_count = 0 # 再訪問回数
        self.follower_collision_count = 0 # フォロワの合計衝突回数
        self.agent_collision_count = 0 # エージェントの合計衝突回数
        self.total_distance_traveled: float = 0.0 # 合計走行距離

        self.exploration_rate = [] # 探査率格納用リスト
        
        # エピソードごとの詳細データ
        self.episode_data = {
            'step': [],
            'exploration_rate': [],
            'explored_area': [],
            'total_area': [],
            'agent_collision_flag': [],
            'follower_collision_count': [],
            'reward': []
        }
        
        # ロボットデータ保存設定
        self.robot_logging_config = robot_logging_config or RobotLoggingConfig()
        
        # ロボットデータ（サンプリング）
        self.robot_data = []
    
    def calc_exploration_rate(self, explored_area: int, total_area: int) -> float:
        """
        探査率 = 探査済みセル数 / 探査可能セル数
        計算して履歴に追加
        """
        if total_area == 0:
            rate = 0.0
        else:
            rate = explored_area / total_area
        
        self.exploration_rate.append(rate)
        return rate
    
    def add_step_data(self, step: int, exploration_rate: float, explored_area: int, 
                     total_area: int, agent_collision_flag: int, 
                     follower_collision_count: int, reward: float):
        """
        ステップごとのデータを追加
        """
        self.episode_data['step'].append(step)
        self.episode_data['exploration_rate'].append(exploration_rate)
        self.episode_data['explored_area'].append(explored_area)
        self.episode_data['total_area'].append(total_area)
        self.episode_data['agent_collision_flag'].append(agent_collision_flag)
        self.episode_data['follower_collision_count'].append(follower_collision_count)
        self.episode_data['reward'].append(reward)

    def add_robot_data(self, step: int, robot_id: str, x: float, y: float, 
                      collision_flag: bool, boids_flag: int, distance: float):
        """
        ロボットデータを追加（サンプリング）
        """
        if not self.robot_logging_config.save_robot_data:
            return
        
        # メモリ保護
        if len(self.robot_data) >= self.robot_logging_config.max_robot_records:
            return
        
        # サンプリング条件チェック
        should_save = False
        
        # 衝突時は必ず保存
        if self.robot_logging_config.save_collision_only and collision_flag:
            should_save = True
        # サンプリングレートに基づく保存
        elif np.random.random() < self.robot_logging_config.sampling_rate:
            should_save = True
        
        if should_save:
            robot_record = {}
            if self.robot_logging_config.save_position:
                robot_record.update({'x': x, 'y': y})
            if self.robot_logging_config.save_collision:
                robot_record.update({'collision_flag': collision_flag})
            if self.robot_logging_config.save_boids:
                robot_record.update({'boids_flag': boids_flag})
            if self.robot_logging_config.save_distance:
                robot_record.update({'distance': distance})
            
            robot_record.update({
                'step': step,
                'robot_id': robot_id
            })
            
            self.robot_data.append(robot_record)

    def save_episode_csv(self, log_dir: str, episode: int):
        """
        エピソードごとの詳細データをCSVとして保存
        """
        csv_dir = os.path.join(log_dir, "csvs")
        os.makedirs(csv_dir, exist_ok=True)
        
        # エピソードデータをDataFrameに変換
        df = pd.DataFrame(self.episode_data)
        
        # CSVファイル名
        csv_filename = f"episode_{episode:04d}_exploration.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        # CSVとして保存
        df.to_csv(csv_path, index=False)
        print(f"Episode {episode} exploration data saved to: {csv_path}")
        
        # ロボットデータを保存（設定が有効な場合）
        if self.robot_logging_config.save_robot_data and self.robot_data:
            robot_df = pd.DataFrame(self.robot_data)
            robot_csv_filename = f"episode_{episode:04d}_robots.csv"
            robot_csv_path = os.path.join(csv_dir, robot_csv_filename)
            robot_df.to_csv(robot_csv_path, index=False)
            print(f"Episode {episode} robot data saved to: {robot_csv_path}")
        
        # エピソードデータをリセット
        self.episode_data = {
            'step': [],
            'exploration_rate': [],
            'explored_area': [],
            'total_area': [],
            'agent_collision_flag': [],
            'follower_collision_count': [],
            'reward': []
        }
        
        # ロボットデータをリセット
        self.robot_data = []

    def export_metrics(self) -> dict:
        return {
            "goal_reaching_step": self.goal_reaching_step,
            "revisit_ratio": self.revisit_ratio,
            "revisit_count": self.revisit_count,
            "follower_collision_count": self.follower_collision_count,
            "agent_collision_count": self.agent_collision_count,
            "total_distance_traveled": self.total_distance_traveled,
            "final_exploration_rate": self.exploration_rate[-1] if self.exploration_rate else 0.0,
            "exploration_rate_curve": self.exploration_rate,
        }
    
    # Configurable interface implementation
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "goal_reaching_step": self.goal_reaching_step,
            "revisit_ratio": self.revisit_ratio,
            "revisit_count": self.revisit_count,
            "follower_collision_count": self.follower_collision_count,
            "agent_collision_count": self.agent_collision_count,
            "total_distance_traveled": self.total_distance_traveled
        }
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration"""
        # Score class doesn't have mutable config
        pass
    
    # Stateful interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            "exploration_rate": self.exploration_rate,
            "episode_data": self.episode_data,
            "robot_data": self.robot_data
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set state"""
        # Score class doesn't have mutable state
        pass
    
    def reset_state(self):
        """Reset to initial state"""
        self.exploration_rate = []
        self.episode_data = {
            'step': [],
            'exploration_rate': [],
            'explored_area': [],
            'total_area': [],
            'agent_collision_flag': [],
            'follower_collision_count': [],
            'reward': []
        }
        self.robot_data = []
    
    # Loggable interface implementation
    def get_log_data(self) -> Dict[str, Any]:
        """Get data for logging"""
        return {
            "config": self.get_config(),
            "state": self.get_state(),
            "final_exploration_rate": self.exploration_rate[-1] if self.exploration_rate else 0.0,
            "exploration_rate_curve": self.exploration_rate
        }
    
    def save_log(self, path: str):
        """Save log data to file"""
        with open(path, 'w') as f:
            json.dump(self.get_log_data(), f, indent=2, default=str)


