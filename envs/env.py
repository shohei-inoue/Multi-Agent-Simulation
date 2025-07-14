"""
Environment class for the swarm robot exploration simulation.
Implements the core simulation environment with rendering and logging capabilities.
"""

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mColors
from matplotlib.patches import Circle
import os
import imageio
import copy
import uuid
import pandas as pd
from datetime import datetime
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, List, Optional

from envs.env_map import generate_explored_map, generate_rect_obstacle_map
from .action_space import create_action_space
from .observation_space import create_observation_space, create_initial_state
from .reward import create_reward
from robots.red import Red, BoidsType, RobotRole
from dataclasses import dataclass, asdict
from params.simulation import Param
from params.robot_logging import RobotLoggingConfig
from scores.score import Score
from core.logging import get_component_logger
from core.interfaces import Configurable, Stateful, Loggable, Renderable


@dataclass
class Swarm:
    """群を表すクラス"""
    swarm_id: int
    leader: Red
    followers: List[Red]
    exploration_rate: float = 0.0
    step_count: int = 0
    
    def get_all_robots(self) -> List[Red]:
        """群に属する全ロボットを取得"""
        return [self.leader] + self.followers
    
    def add_follower(self, robot: Red):
        """followerを追加"""
        self.followers.append(robot)
        robot.swarm_id = self.swarm_id
    
    def remove_follower(self, robot: Red):
        """followerを削除"""
        if robot in self.followers:
            self.followers.remove(robot)
    
    def get_robot_count(self) -> int:
        """群のロボット数を取得"""
        return 1 + len(self.followers)  # leader + followers


class Env(gym.Env, Configurable, Stateful, Loggable, Renderable):
    """ 
    Environment class for swarm robot exploration simulation
    """
    def __init__(self, param: Param, action_space=None, observation_space=None, reward_dict=None):
        super().__init__()
        
        # Initialize logging
        self.logger = get_component_logger("environment")
        
        # Configuration
        self._config = {
            "map_width": param.environment.map.width,
            "map_height": param.environment.map.height,
            "robot_num": param.explore.robotNum,
            "finish_rate": param.explore.finishRate
        }
        
        # State
        self._state = {
            "explored_area": 0,
            "exploration_ratio": 0.0,
            "agent_step": 0,
            "collision_flag": False
        }
        
        # Rendering
        self._render_flag = False
        self._render_components = {}

        # map
        self.__map_width            = param.environment.map.width
        self.__map_height           = param.environment.map.height
        self.__map_seed             = param.environment.map.seed
        self.__obstacle_probability = param.environment.obstacle.probability
        self.__obstacle_max_size    = param.environment.obstacle.maxSize
        self.__obstacle_value       = param.environment.obstacle.value

        # robot
        self.__movement_min   = param.robot.movement.min
        self.__movement_max   = param.robot.movement.max
        self.__boids_min      = param.robot.boids.min
        self.__boids_max      = param.robot.boids.max
        self.__avoidance_min  = param.robot.avoidance.min
        self.__avoidance_max  = param.robot.avoidance.max
        self.__offset         = param.robot.offset

        # explore
        self.__boundary_inner       = param.explore.boundary.inner
        self.__boundary_outer       = param.explore.boundary.outer
        self.__mv_mean              = param.explore.mv.mean
        self.__mv_variance          = param.explore.mv.variance
        self.__initial_coordinate_x = param.explore.coordinate.x
        self.__initial_coordinate_y = param.explore.coordinate.y
        self.__robot_num            = param.explore.robotNum
        self.__finish_rate          = param.explore.finishRate
        self.__finish_step          = param.explore.finishStep

        self.action_space        = action_space or create_action_space()
        self.__observation_space = observation_space or create_observation_space()
        self.__reward_dict       = reward_dict or create_reward()

        # ----- set initialize map -----
        self.__map = generate_rect_obstacle_map(
          map_width=self.__map_width, 
          map_height=self.__map_height,
          obstacle_prob=self.__obstacle_probability,
          obstacle_max_size=self.__obstacle_max_size,
          obstacle_val=self.__obstacle_value,
          seed=self.__map_seed
        )

        self.explored_map = generate_explored_map(
          map_width=self.__map_width,
          map_height=self.__map_height
        )

        # ----- set scorer -----
        robot_logging_config = RobotLoggingConfig()
        self.scorer = Score(robot_logging_config=robot_logging_config)

        # ----- set explore infos -----
        self.total_area = np.count_nonzero(self.__map != self.__obstacle_value)  # 探査可能エリア
        self.explored_area          = 0    # 探査したセル数
        self.exploration_ratio      = 0.0  # 1ステップ前の探査率
        self.agent_step             = 0    # エージェントのステップ
        self.collision_flag         = False

        # ----- dynamic leader-follower system -----
        self.current_leader_index   = 0    # 現在のleaderのインデックス（固定）
        self.leader_switch_interval = 10   # leader切り替え間隔（ステップ数）- 現在は使用しない
        self.leader_switch_counter  = 0    # leader切り替えカウンター（現在は使用しない）
        
        # ----- multi-swarm system -----
        self.swarms = []                    # 群のリスト
        self.swarm_id_counter = 0          # 群IDカウンター
        self.initial_swarm_id = 0          # 初期群ID
        
        # ----- leader agents -----
        self.leader_agents = {}            # 各リーダーの独立したエージェント
        self.leader_models = {}            # 各リーダーの独立したモデル
        
        # ----- visualization parameters -----
        self.exploration_radius = self.__boundary_outer      # 探査領域の半径（boundary_outerに合わせる）
        
        # 架空のagent座標（初期化時のみ使用）
        self.__agent_initial_coordinate = np.array([self.__initial_coordinate_y, self.__initial_coordinate_x])
        self.agent_coordinate = self.__agent_initial_coordinate
        
        # 1ステップ前のエージェント位置情報
        self.__agent_initial_previous_coordinate = None 
        self.agent_previous_coordinate           = self.__agent_initial_previous_coordinate 

        # ----- set drawing infos -----
        self.agent_trajectory = [self.agent_coordinate.copy()] # 奇跡の初期化
        self.env_frames = [] # 描画用フレームの初期化

        # -----set robots with dynamic roles -----
        self.robots = [Red(
          id                    = f"robot_{index}",
          movement_min          = self.__movement_min,
          movement_max          = self.__movement_max,
          boids_min             = self.__boids_min,
          boids_max             = self.__boids_max,
          avoidance_min         = self.__avoidance_min,
          avoidance_max         = self.__avoidance_max,
          inner_boundary        = self.__boundary_inner,
          outer_boundary        = self.__boundary_outer,
          map                   = self.__map,
          obstacle_value        = self.__obstacle_value,
          mean                  = self.__mv_mean,
          variance              = self.__mv_variance,
          agent_coordinate      = np.array([self.__initial_coordinate_y, self.__initial_coordinate_x]),
          x                     = self.__initial_coordinate_x + self.__offset.position * math.cos((2 * math.pi * index / (self.__robot_num))),
          y                     = self.__initial_coordinate_y + self.__offset.position * math.sin((2 * math.pi * index / (self.__robot_num))),
          step                  = self.__offset.step,
          amount_of_movement    = self.__offset.amount_of_movement,
          direction_angle       = self.__offset.direction_angle,
          collision_flag        = self.__offset.collision_flag,
          boids_flag            = BoidsType(self.__offset.boids_flag.value),
          estimated_probability = self.__offset.estimated_probability,
          role                  = RobotRole.LEADER if index == 0 else RobotRole.FOLLOWER
        ) for index in range(self.__robot_num)]
        
        # 初期leaderを設定
        self.robots[0].set_role(RobotRole.LEADER)
        self.current_leader = self.robots[0]
        
        # 初期群を作成
        initial_swarm = Swarm(
            swarm_id=self.initial_swarm_id,
            leader=self.robots[0],
            followers=self.robots[1:].copy()
        )
        
        # 全ロボットに群IDを設定
        for robot in self.robots:
            robot.swarm_id = self.initial_swarm_id
        
        self.swarms.append(initial_swarm)
        self.swarm_id_counter = 1

        # ----- set initial state -----
        self.MAX_COLLISION_NUM = 100
        initial_fcd = [np.array([0.0, 0.0], dtype=np.float32)] * self.MAX_COLLISION_NUM
        self.__initial_state = create_initial_state(
          self.__agent_initial_coordinate[1],
          self.__agent_initial_coordinate[0],
          azimuth=0,          # TODO どうするか,
          collision_flag=0,   
          agent_step_count=self.agent_step,
          follower_collision_data=initial_fcd
        )

        self.follower_collision_points = [] # 描画用のfollower衝突リスト

        # 非同期実行用のスレッドプール
        self.follower_executor = ThreadPoolExecutor(max_workers=self.__robot_num)
        self.follower_futures = []

        self.state = self.__initial_state
  

    def reset(self):
        """
        環境の初期化関数
        """
        # ----- reset explore infos -----
        self.agent_step = 0 # ステップ初期化
        self.agent_coordinate = self.__agent_initial_coordinate # エージェントの位置情報の初期化
        self.agent_previous_coordinate = self.__agent_initial_previous_coordinate # 1ステップ前の位置情報の初期化
        self.explored_area = 0 # 探査済みエリアの初期化
        self.exploration_ratio = 0.0
        self.collision_flag = False
        self.explored_map = generate_explored_map(
          map_width=self.__map_width,
          map_height=self.__map_height
        ) # 探査済みマップの初期化

        # ----- reset robot info -----
        self.robots = [Red(
          id                    = f"robot_{index}",
          movement_min          = self.__movement_min,
          movement_max          = self.__movement_max,
          boids_min             = self.__boids_min,
          boids_max             = self.__boids_max,
          avoidance_min         = self.__avoidance_min,
          avoidance_max         = self.__avoidance_max,
          inner_boundary        = self.__boundary_inner,
          outer_boundary        = self.__boundary_outer,
          map                   = self.__map,
          obstacle_value       = self.__obstacle_value,
          mean                  = self.__mv_mean,
          variance              = self.__mv_variance,
          agent_coordinate      = np.array([self.__initial_coordinate_y, self.__initial_coordinate_x]),
          x                     = self.__initial_coordinate_x + self.__offset.position * math.cos((2 * math.pi * index / (self.__robot_num))),
          y                     = self.__initial_coordinate_y + self.__offset.position * math.sin((2 * math.pi * index / (self.__robot_num))),
          step                  = self.__offset.step,
          amount_of_movement    = self.__offset.amount_of_movement,
          direction_angle       = self.__offset.direction_angle,
          collision_flag        = self.__offset.collision_flag,
          boids_flag            = BoidsType(self.__offset.boids_flag.value),
          estimated_probability = self.__offset.estimated_probability,
          role                  = RobotRole.LEADER if index == 0 else RobotRole.FOLLOWER
        ) for index in range(self.__robot_num)]
        
        # 初期leaderを設定（固定）
        self.robots[0].set_role(RobotRole.LEADER)
        self.current_leader = self.robots[0]
        self.current_leader_index = 0
        self.leader_switch_counter = 0  # 現在は使用しない
        
        # 初期群を作成
        initial_swarm = Swarm(
            swarm_id=self.initial_swarm_id,
            leader=self.robots[0],
            followers=self.robots[1:].copy()
        )
        
        # 全ロボットに群IDを設定
        for robot in self.robots:
            robot.swarm_id = self.initial_swarm_id
        
        self.swarms = [initial_swarm]
        self.swarm_id_counter = 1

        # ----- reset drawing infos -----
        self.agent_trajectory = [self.agent_coordinate.copy()] # 奇跡の初期化
        self.result_pdf_list  = [] # 統合確率分布の初期化

        # ----- reset state -----
        self.scorer = Score()
        self.state = copy.deepcopy(self.__initial_state)
        
        # follower_mobility_scoresの初期化
        self.state['follower_mobility_scores'] = [0.0] * 10

        self.follower_collision_points = [] # 描画用のfollower衝突リスト

        # 非同期実行用のスレッドプールを再初期化
        if hasattr(self, 'follower_executor'):
            self.follower_executor.shutdown(wait=True)
        self.follower_executor = ThreadPoolExecutor(max_workers=self.__robot_num)
        self.follower_futures = []

        return self.state
  

    def _switch_leader(self):
        """
        leaderを切り替える（現在は使用されていない：群分岐時のみ新しいリーダーが増える）
        """
        # 現在のleaderをfollowerに変更
        self.current_leader.set_role(RobotRole.FOLLOWER)
        
        # 次のleaderを選択（ラウンドロビン方式）
        self.current_leader_index = (self.current_leader_index + 1) % self.__robot_num
        self.current_leader = self.robots[self.current_leader_index]
        self.current_leader.set_role(RobotRole.LEADER)
        
        print(f"Leader switched to robot_{self.current_leader_index}")


    def _create_new_swarm(self, new_leader: Red, followers: List[Red]):
        """
        新しい群を作成
        """
        new_swarm_id = self.swarm_id_counter
        self.swarm_id_counter += 1
        
        # 新しい群を作成
        new_swarm = Swarm(
            swarm_id=new_swarm_id,
            leader=new_leader,
            followers=followers
        )
        
        # ロボットの群IDを更新
        new_leader.swarm_id = new_swarm_id
        for follower in followers:
            follower.swarm_id = new_swarm_id
        
        self.swarms.append(new_swarm)
        print(f"New swarm {new_swarm_id} created with leader {new_leader.id}")


    def _merge_swarms(self, target_swarm: Swarm, source_swarm: Swarm):
        """
        群を統合
        """
        # source_swarmのfollowerをtarget_swarmに移動
        for follower in source_swarm.followers:
            target_swarm.add_follower(follower)
        
        # source_swarmのleaderもfollowerとして追加
        source_swarm.leader.set_role(RobotRole.FOLLOWER)
        target_swarm.add_follower(source_swarm.leader)
        
        # source_swarmを削除
        self.swarms.remove(source_swarm)
        print(f"Swarm {source_swarm.swarm_id} merged into swarm {target_swarm.swarm_id}")


    def _handle_swarm_mode(self, mode: int):
        """
        群分岐・統合の処理
        """
        if mode == 0:  # 通常動作
            return
        
        elif mode == 1:  # 群分岐
            # 現在の群から新しい群を作成
            current_swarm = self._find_swarm_by_leader(self.current_leader)
            if current_swarm and len(current_swarm.followers) >= 3:  # followerが3台以上の場合のみ分岐
                # 新しいleaderを選択（最初のfollower）
                new_leader = current_swarm.followers[0]
                new_leader.set_role(RobotRole.LEADER)
                
                # 新しいfollowerを選択（残りのfollowerの半分）
                remaining_followers = current_swarm.followers[1:]
                split_point = len(remaining_followers) // 2
                new_followers = remaining_followers[:split_point]
                
                # 新しい群を作成
                self._create_new_swarm(new_leader, new_followers)
                
                # 元の群からfollowerを削除
                for follower in [new_leader] + new_followers:
                    current_swarm.remove_follower(follower)
            else:
                print(f"Swarm {current_swarm.swarm_id if current_swarm else 'Unknown'} has insufficient followers ({len(current_swarm.followers) if current_swarm else 0}) for branching. Minimum required: 3")
        
        elif mode == 2:  # 群統合
            # 最も近い群に統合
            current_swarm = self._find_swarm_by_leader(self.current_leader)
            if current_swarm and len(self.swarms) > 1:
                # 最も近い群を探す
                closest_swarm = self._find_closest_swarm(current_swarm)
                if closest_swarm and closest_swarm != current_swarm:
                    self._merge_swarms(closest_swarm, current_swarm)


    def _find_swarm_by_leader(self, leader: Red) -> Optional[Swarm]:
        """
        leaderから群を検索
        """
        for swarm in self.swarms:
            if swarm.leader == leader:
                return swarm
        return None


    def _find_closest_swarm(self, target_swarm: Swarm) -> Optional[Swarm]:
        """
        最も近い群を検索
        """
        min_distance = float('inf')
        closest_swarm = None
        
        for swarm in self.swarms:
            if swarm == target_swarm:
                continue
            
            # 群の中心間距離を計算
            target_center = target_swarm.leader.coordinate
            swarm_center = swarm.leader.coordinate
            distance = np.linalg.norm(target_center - swarm_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_swarm = swarm
        
        return closest_swarm


    def _execute_follower_step_async(self, follower_index):
        """
        followerの1ステップを非同期実行する（従来方式）
        """
        try:
            robot = self.robots[follower_index]
            if robot.role != RobotRole.FOLLOWER:
                return {
                    'index': follower_index,
                    'collision_point': None,
                    'collision_data': []
                }
                
            previous_coordinate = robot.coordinate.copy()
            
            # followerの動きを実行（現在のleaderを参照）
            robot.step_motion(agent_coordinate=self.current_leader.coordinate)
            
            # 探査マップを更新
            self.update_exploration_map(previous_coordinate, robot.coordinate)
            
            # ロボットデータを記録
            self.scorer.add_robot_data(
                step=self.agent_step,
                robot_id=robot.id,
                x=robot.x,
                y=robot.y,
                collision_flag=robot.collision_flag,
                boids_flag=robot.boids_flag.value,
                distance=robot.distance
            )
            
            # 衝突フラグがTrueなら座標を追加
            collision_point = None
            if robot.collision_flag:
                cx = robot.coordinate[1]
                cy = robot.coordinate[0]
                collision_point = (cx, cy)
            
            return {
                'index': follower_index,
                'collision_point': collision_point,
                'collision_data': robot.get_collision_data()
            }
        except Exception as e:
            print(f"Error in follower {follower_index}: {e}")
            return {
                'index': follower_index,
                'collision_point': None,
                'collision_data': []
            }


    def _execute_follower_step_async_by_swarm(self, swarm: Swarm, follower: Red):
        """
        群ベースのfollowerの1ステップを非同期実行する
        """
        try:
            if follower.role != RobotRole.FOLLOWER:
                return {
                    'index': -1,
                    'collision_point': None,
                    'collision_data': []
                }
                
            previous_coordinate = follower.coordinate.copy()
            
            # followerの動きを実行（所属群のleaderを参照）
            follower.step_motion(agent_coordinate=swarm.leader.coordinate)
            
            # 探査マップを更新
            self.update_exploration_map(previous_coordinate, follower.coordinate)
            
            # ロボットデータを記録
            self.scorer.add_robot_data(
                step=self.agent_step,
                robot_id=follower.id,
                x=follower.x,
                y=follower.y,
                collision_flag=follower.collision_flag,
                boids_flag=follower.boids_flag.value,
                distance=follower.distance
            )
            
            # 衝突フラグがTrueなら座標を追加
            collision_point = None
            if follower.collision_flag:
                cx = follower.coordinate[1]
                cy = follower.coordinate[0]
                collision_point = (cx, cy)
            
            return {
                'index': -1,
                'collision_point': collision_point,
                'collision_data': follower.get_collision_data()
            }
        except Exception as e:
            print(f"Error in follower {follower.id}: {e}")
            return {
                'index': -1,
                'collision_point': None,
                'collision_data': []
            }


    def render(self, fig_size = 6, mode = 'human', ax = None):
        """
        環境のレンダリング
        マップの描画
        フォロワの描画
        エージェントの描画
        """
        if mode == 'rgb_gray':
          pass
        elif mode == 'rgb_array':
          # GIF保存用のrgb_arrayモード
          fig, ax = plt.subplots(figsize=(fig_size, fig_size))
          
          # 地図
          ax.imshow(
              self.__map,
              cmap='gray_r',
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # 探査済み領域
          cmap = mColors.ListedColormap(['white', 'gray', 'black'])
          bounds = [0, 1, self.__obstacle_value, self.__obstacle_value + 1]
          norm = mColors.BoundaryNorm(bounds, cmap.N)

          ax.imshow(
              self.explored_map,
              cmap=cmap,
              alpha=0.5,
              norm=norm,
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # === 追加: リーダー描画用の外部関数 ===
          def render_leader(ax, swarm, colors, trajectory_colors, exploration_radius):
              """
              指定したswarmのリーダー・軌跡・探査円を描画する
              """
              leader_id = int(swarm.leader.id) if str(swarm.leader.id).isdigit() else hash(swarm.leader.id) % len(colors)
              leader_color = colors[leader_id % len(colors)]
              trajectory_color = trajectory_colors[leader_id % len(trajectory_colors)]
              leader = swarm.leader
              # リーダーの位置
              ax.scatter(
                  x=leader.data['x'].iloc[-1],
                  y=leader.data['y'].iloc[-1],
                  color=leader_color,
                  s=25,
                  marker='*',
                  edgecolors='black',
                  linewidth=1,
                  label=f"Swarm {swarm.swarm_id} Leader (ID: {leader.id})"
              )
              # リーダーの軌跡
              if hasattr(leader, 'leader_trajectory_data') and len(leader.leader_trajectory_data) > 0:
                  ax.plot(
                      leader.leader_trajectory_data['x'],
                      leader.leader_trajectory_data['y'],
                      color=trajectory_color,
                      linewidth=1.5,
                      alpha=0.6,
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              else:
                  ax.plot(
                      leader.data['x'],
                      leader.data['y'],
                      color=trajectory_color,
                      linewidth=1.5,
                      alpha=0.6,
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              # 探査円
              if len(leader.data['x']) > 0:
                  current_x = leader.data['x'].iloc[-1]
                  current_y = leader.data['y'].iloc[-1]
                  circle = Circle((current_x, current_y), exploration_radius, color='blue', alpha=0.1, fill=True, linestyle='--', linewidth=1)
                  ax.add_patch(circle)
                  circle_boundary = Circle((current_x, current_y), exploration_radius, color='blue', alpha=0.3, fill=False, linestyle='--', linewidth=1.5)
                  ax.add_patch(circle_boundary)

          # --- 各群のロボットを描画 ---
          colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
          trajectory_colors = ['darkgreen', 'darkorange', 'darkred', 'darkblue', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkseagreen', 'darkslateblue']
          for swarm_idx, swarm in enumerate(self.swarms):
              # --- リーダー描画を外部関数で ---
              render_leader(ax, swarm, colors, trajectory_colors, self.exploration_radius)
              # followerの描画（従来通り）
              for i, follower in enumerate(swarm.followers):
                  if follower.role == RobotRole.FOLLOWER:
                      fx = follower.data['x'].iloc[-1]
                      fy = follower.data['y'].iloc[-1]
                      ax.scatter(
                          x=fx,
                          y=fy,
                          color=colors[swarm_idx % len(colors)],
                          s=10,
                          marker='o',
                          alpha=0.7,
                          label=f"Follower (Leader ID: {swarm.leader.id})" if i == 0 and swarm_idx == 0 else None
                      )
                      if len(follower.data) > 1:
                          ax.plot(
                              follower.data['x'],
                              follower.data['y'],
                              color=trajectory_colors[swarm_idx % len(trajectory_colors)],
                              linewidth=0.5,
                              alpha=0.3,
                              linestyle='-'
                          )

          # 軸の調整
          ax.set_xlim(0, self.__map_width)
          ax.set_ylim(0, self.__map_height)
          ax.set_title('Explore Environment')
          ax.set_xlabel('X')
          ax.set_ylabel('Y')
          ax.grid(False)

          # === フォロワによる衝突点の描画 ===
          for cx, cy in self.follower_collision_points:
              ax.plot(cx, cy, 'rx', markersize=6, zorder=5, alpha=0.3)

          # 凡例の表示（重複を避ける）
          handles, labels = ax.get_legend_handles_labels()
          by_label = dict(zip(labels, handles))
          ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

          # rgb_arrayを取得
          fig.canvas.draw()
          
          # matplotlibのバージョンに応じて適切なメソッドを使用
          try:
              # 新しいmatplotlibバージョン用
              img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
              img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
          except (AttributeError, TypeError):
              try:
                  # 代替方法1: buffer_rgbaを使用
                  img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore
                  img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                  img_array = img_array[:, :, :3]  # アルファチャンネルを除去
              except (AttributeError, TypeError):
                  # 代替方法2: savefigを使用して一時ファイルから読み込み
                  import tempfile
                  import io
                  from PIL import Image
                  
                  # 一時ファイルに保存
                  with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                      fig.savefig(tmp_file.name, format='png', bbox_inches='tight', dpi=100)
                      
                      # PILで読み込み
                      img = Image.open(tmp_file.name)
                      img_array = np.array(img)
                      
                      # 一時ファイルを削除
                      import os
                      os.unlink(tmp_file.name)
          
          plt.close(fig)  # メモリリーク防止
          plt.clf()  # 追加のクリーンアップ
          return img_array
          
        elif mode == 'human':
          # plt.ion()  # ← これがあると画面が更新されるようになる
          if ax is None:
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
          else:
            ax.clear()

          # 地図
          ax.imshow(
              self.__map,
              cmap='gray_r',
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # 探査済み領域
          cmap = mColors.ListedColormap(['white', 'gray', 'black'])
          bounds = [0, 1, self.__obstacle_value, self.__obstacle_value + 1]
          norm = mColors.BoundaryNorm(bounds, cmap.N)

          ax.imshow(
              self.explored_map,
              cmap=cmap,
              alpha=0.5,
              norm=norm,
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # === 追加: リーダー描画用の外部関数 ===
          def render_leader(ax, swarm, colors, trajectory_colors, exploration_radius):
              """
              指定したswarmのリーダー・軌跡・探査円を描画する
              """
              leader_id = int(swarm.leader.id) if str(swarm.leader.id).isdigit() else hash(swarm.leader.id) % len(colors)
              leader_color = colors[leader_id % len(colors)]
              trajectory_color = trajectory_colors[leader_id % len(trajectory_colors)]
              leader = swarm.leader
              # リーダーの位置
              ax.scatter(
                  x=leader.data['x'].iloc[-1],
                  y=leader.data['y'].iloc[-1],
                  color=leader_color,
                  s=25,
                  marker='*',
                  edgecolors='black',
                  linewidth=1,
                  label=f"Swarm {swarm.swarm_id} Leader (ID: {leader.id})"
              )
              # リーダーの軌跡
              if hasattr(leader, 'leader_trajectory_data') and len(leader.leader_trajectory_data) > 0:
                  ax.plot(
                      leader.leader_trajectory_data['x'],
                      leader.leader_trajectory_data['y'],
                      color=trajectory_color,
                      linewidth=1.5,
                      alpha=0.6,
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              else:
                  ax.plot(
                      leader.data['x'],
                      leader.data['y'],
                      color=trajectory_color,
                      linewidth=1.5,
                      alpha=0.6,
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              # 探査円
              if len(leader.data['x']) > 0:
                  current_x = leader.data['x'].iloc[-1]
                  current_y = leader.data['y'].iloc[-1]
                  circle = Circle((current_x, current_y), exploration_radius, color='blue', alpha=0.1, fill=True, linestyle='--', linewidth=1)
                  ax.add_patch(circle)
                  circle_boundary = Circle((current_x, current_y), exploration_radius, color='blue', alpha=0.3, fill=False, linestyle='--', linewidth=1.5)
                  ax.add_patch(circle_boundary)

          # --- 各群のロボットを描画 ---
          colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
          trajectory_colors = ['darkgreen', 'darkorange', 'darkred', 'darkblue', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkseagreen', 'darkslateblue']
          for swarm_idx, swarm in enumerate(self.swarms):
              # --- リーダー描画を外部関数で ---
              render_leader(ax, swarm, colors, trajectory_colors, self.exploration_radius)
              # followerの描画（従来通り）
              for i, follower in enumerate(swarm.followers):
                  if follower.role == RobotRole.FOLLOWER:
                      fx = follower.data['x'].iloc[-1]
                      fy = follower.data['y'].iloc[-1]
                      ax.scatter(
                          x=fx,
                          y=fy,
                          color=colors[swarm_idx % len(colors)],
                          s=10,
                          marker='o',
                          alpha=0.7,
                          label=f"Follower (Leader ID: {swarm.leader.id})" if i == 0 and swarm_idx == 0 else None
                      )
                      if len(follower.data) > 1:
                          ax.plot(
                              follower.data['x'],
                              follower.data['y'],
                              color=trajectory_colors[swarm_idx % len(trajectory_colors)],
                              linewidth=0.5,
                              alpha=0.3,
                              linestyle='-'
                          )

          # 軸の調整
          ax.set_xlim(0, self.__map_width)
          ax.set_ylim(0, self.__map_height)
          ax.set_title('Explore Environment')
          ax.set_xlabel('X')
          ax.set_ylabel('Y')
          ax.grid(False)

          # === フォロワによる衝突点の描画 ===
          for cx, cy in self.follower_collision_points:
              ax.plot(cx, cy, 'rx', markersize=6, zorder=5, alpha=0.3)

          # 凡例の表示（重複を避ける）
          handles, labels = ax.get_legend_handles_labels()
          by_label = dict(zip(labels, handles))
          ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)


        #   # === フォロワの現在位置 ===
        #   for i, follower in enumerate(self.follower_robots):
        #       fx = follower.data['x'].iloc[-1]
        #       fy = follower.data['y'].iloc[-1]
        # ax.plot(fx, fy, 'ro', markersize=4, label="Follower" if i == 0 else "", zorder=3)

  

    def step(self, action):
        """
        環境のステップ → 動的leader-followerシステム + 群分岐・統合
        """
        # 群分岐・統合の処理
        mode = action.get('mode', 0)
        self._handle_swarm_mode(mode)
        
        # leader切り替えチェック（無効化：群分岐時のみ新しいリーダーが増える）
        # if mode == 0:
        #     self.leader_switch_counter += 1
        #     if self.leader_switch_counter >= self.leader_switch_interval:
        #         self._switch_leader()
        #         self.leader_switch_counter = 0

        # 各群のleaderの行動を実行
        leader_rewards = {}  # 各リーダーの報酬を個別に管理
        leader_states = {}   # 各リーダーの独立した状態空間
        for swarm in self.swarms:
            if swarm.leader.role == RobotRole.LEADER:
                # 各リーダーの個別状態空間を構築
                leader_state = self._build_leader_state(swarm.leader, swarm)
                leader_states[swarm.leader.id] = leader_state
                
                # 各リーダーが独立した行動を決定
                # リーダー固有のアルゴリズムで行動決定
                action_tensor, action_dict = swarm.leader.get_leader_action(
                    algorithm=None,  # リーダー固有のアルゴリズムを使用
                    state=leader_state,
                    sampled_params=action.get('sampled_params', [0.5, 10.0, 5.0, 0.3, 0.7]),
                    episode=self.agent_step,
                    log_dir=None
                )
                
                # 各leaderの行動を実行
                previous_coordinate = swarm.leader.coordinate.copy()
                success = swarm.leader.execute_leader_action(action_dict)
                if not success:
                    print(f"Leader {swarm.leader.id} collision detected")
                else:
                    # リーダーの移動が成功した場合、探査マップを更新
                    self.update_exploration_map(previous_coordinate, swarm.leader.coordinate)
                
                # 各リーダーの個別報酬を計算
                leader_reward = self._calculate_leader_reward(swarm.leader, previous_coordinate)
                leader_rewards[swarm.leader.id] = leader_reward
        
        # エージェントのステップカウントをインクリメント
        self.agent_step += 1
        
        # 現在のleaderの状態を環境状態として設定（最初の群のleader）
        if self.swarms:
            main_leader = self.swarms[0].leader
            self.state['agent_coordinate_x'] = main_leader.x
            self.state['agent_coordinate_y'] = main_leader.y
            self.state['agent_azimuth'] = main_leader.leader_azimuth
            self.state['agent_collision_flag'] = 1.0 if main_leader.collision_flag else 0.0
            self.state['agent_step_count'] = main_leader.leader_step_count
            
            # 各リーダーの個別状態を追加
            self.state['leader_states'] = leader_states
        
        # 各群のfollowerの探査行動（非同期実行）
        follower_results = []
        for _ in range(self.__offset.one_explore_step):
            # 各群のfollowerを非同期で実行
            futures = []
            for swarm in self.swarms:
                for follower in swarm.followers:
                    if follower.role == RobotRole.FOLLOWER:
                        future = self.follower_executor.submit(self._execute_follower_step_async_by_swarm, swarm, follower)
                        futures.append(future)
            
            # 全followerの完了を待機
            for future in as_completed(futures):
                try:
                    result = future.result()
                    follower_results.append(result)
                    
                    # 衝突点を追加
                    if result['collision_point']:
                        self.follower_collision_points.append(result['collision_point'])
                except Exception as e:
                    print(f"Error in follower execution: {e}")
            
            # 探査率計算
            previous_ratio = self.scorer.exploration_rate[-1] if self.scorer.exploration_rate else 0.0
            self.exploration_ratio = self.scorer.calc_exploration_rate(
                explored_area=self.explored_area,
                total_area=self.total_area
            )
            print(f"exploration ratio | {self.exploration_ratio} | ( {self.explored_area} / {self.total_area})")

        # followerから衝突データ収集
        follower_collision_data = []
        current_follower_collision_count = 0
        follower_mobility_scores = []
        
        for result in follower_results:
            collision_data = result['collision_data']
            follower_collision_data.extend(collision_data)
            current_follower_collision_count += len(collision_data)
            self.scorer.follower_collision_count += len(collision_data)
        
        # 各群の全followerのmobility_scoreを集約
        for swarm in self.swarms:
            for follower in swarm.followers:
                follower_mobility_scores.append(follower.mobility_score)
        
        # 最大ロボット数（10個）に合わせてパディング
        MAX_ROBOT_NUM = 10
        while len(follower_mobility_scores) < MAX_ROBOT_NUM:
            follower_mobility_scores.append(0.0)
        follower_mobility_scores = follower_mobility_scores[:MAX_ROBOT_NUM]  # 最大数を超える場合は切り詰め
        
        self.state['follower_mobility_scores'] = follower_mobility_scores

        # MAX_COLLISION_NUM 個だけ使う（足りない分はゼロ埋め）
        padded_list = follower_collision_data[:self.MAX_COLLISION_NUM]
        padding_needed = self.MAX_COLLISION_NUM - len(padded_list)

        fcd_ndarray = [np.array(pair, dtype=np.float32) for pair in padded_list]
        fcd_ndarray.extend([np.array([0.0, 0.0], dtype=np.float32)] * padding_needed)

        self.state['follower_collision_data'] = fcd_ndarray
        
        # ----- calculate reward -----
        reward = self.__reward_dict.get('default', -1)

        # 探査率の上昇具合に応じた報酬計算
        from .reward import calculate_exploration_reward
        exploration_reward = calculate_exploration_reward(
            self.exploration_ratio, 
            previous_ratio, 
            self.__reward_dict
        )
        reward += exploration_reward

        # leaderが障害物に衝突した場合
        if self.current_leader.collision_flag:
            self.scorer.agent_collision_count += 1
            reward += self.__reward_dict.get('collision_penalty', 0.0)

        # ----- finish condition -----
        done = False
        turncated = False
        if self.scorer.goal_reaching_step is None and self.explored_area >= self.total_area * self.__finish_rate:
            self.scorer.goal_reaching_step = self.agent_step
            done = True
            reward += self.__reward_dict.get('clear_target_rate', 0.0)
        
        # stepによる終了
        if self.agent_step >= self.__finish_step:
            done = True
            reward += self.__reward_dict.get('none_finish_penalty', 0.0)
        
        # ステップごとのスコアデータを記録
        self.scorer.add_step_data(
            step=self.agent_step,
            exploration_rate=self.exploration_ratio,
            explored_area=self.explored_area,
            total_area=self.total_area,
            agent_collision_flag=int(self.state['agent_collision_flag']),
            follower_collision_count=current_follower_collision_count,
            reward=reward
        )

        return self.state, reward, done, turncated, {}
  

    def close(self):
        """
        環境を閉じる際のクリーンアップ
        """
        if hasattr(self, 'follower_executor'):
            self.follower_executor.shutdown(wait=True)


    def next_coordinate(self, dy: float, dx: float) -> tuple[np.ndarray, bool]:
        """
        エージェントの次の位置を計算
        - dy: y方向の移動量
        - dx: x方向の移動量
        Returns:
          新しい位置と衝突フラグ
        """
        SAMPLING_NUM = max(150, int(np.ceil(np.linalg.norm([dy, dx]) * 10)))
        SAFE_DISTANCE = 1.0  # 安全距離
        collision_flag = False

        for i in range(1, SAMPLING_NUM + 1):
           intermediate_coordinate = np.array([
              self.agent_coordinate[0] + dy * (i / SAMPLING_NUM),
              self.agent_coordinate[1] + dx * (i / SAMPLING_NUM)
           ])

           # マップ内か判断
           if (0 < intermediate_coordinate[0] < self.__map_height and
               0 < intermediate_coordinate[1] < self.__map_width):
             # サンプリング点が障害物に衝突しているか確認
              map_y = int(intermediate_coordinate[0])
              map_x = int(intermediate_coordinate[1])

              # 衝突判定
              if self.__map[map_y, map_x] == self.__obstacle_value:
                print(f"Agent collision detected at {intermediate_coordinate}")
                collision_flag = True
                
                # 障害物に衝突する事前位置の計算
                collision_coordinate = intermediate_coordinate
                direction_vector = collision_coordinate - self.agent_coordinate
                norm_direction_vector = np.linalg.norm(direction_vector)
                stop_coordinate = self.agent_coordinate + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)
                return stop_coordinate, collision_flag
              else:
                # 衝突がない場合、最終位置を更新
                continue

        return self.agent_coordinate + np.array([dy, dx]), collision_flag
         

    def save_gif(self, log_dir, episode) -> None:
        """
        保存したフレームをGIFとして保存する

        Parameters:
        - log_dir: ログディレクトリ (gifs/ 以下に保存される）
        - episode: エピソード番号
        """
        gif_dir = os.path.join(log_dir, "gifs")
        gif_name = f"episode_{episode:04d}.gif"

        os.makedirs(gif_dir, exist_ok=True)
        imageio.mimsave(
            os.path.join(gif_dir, gif_name),  # 保存パス
            self.env_frames,                  # フレームリスト
            duration=0.1,
        )

        self.env_frames = []  # フレームをリセット


    def update_exploration_map(self, previous_coordinate, current_coordinate) -> None:
        """
        探査済みマップの更新
        """
        # 補間線を取得（整数座標のペア: (int(y), int(x))）
        line_points = self.interpolate_line(previous_coordinate, current_coordinate)

        for y, x in line_points:
            # 座標がマップ内であるかを確認
            if 0 <= y < self.__map_height and 0 <= x < self.__map_width:
                # 障害物でなく、未探査であれば探査済みにする
                if self.explored_map[y, x] == 0 and self.__map[y, x] != self.__obstacle_value:
                    self.explored_map[y, x] = 1
                    self.explored_area += 1
  

    def interpolate_line(self, p1, p2):
        """
        2点間の整数座標の線分を取得(Bresenhamのアルゴリズム)
        p1: 始点 (y, x)
        p2: 終点 (y, x)
        Returns:
          線分上の全ての整数座標
        """
        y1, x1 = int(p1[0]), int(p1[1])
        y2, x2 = int(p2[0]), int(p2[1])

        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
          points.append((y1, x1))
          if x1 == x2 and y1 == y2:
            break

          e2 = err * 2

          if e2 > -dy:
            err -= dy
            x1 += sx

          if e2 < dx:
            err += dx
            y1 += sy

        return points
  

    def _calculate_leader_reward(self, leader, previous_coordinate):
        """
        各リーダーの個別報酬を計算
        """
        reward = self.__reward_dict.get('default', -1)
        
        # 探査率の上昇具合に応じた報酬計算
        from .reward import calculate_exploration_reward
        exploration_reward = calculate_exploration_reward(
            self.exploration_ratio, 
            self.exploration_ratio,  # 前回の探査率（簡略化）
            self.__reward_dict
        )
        reward += exploration_reward
        
        # リーダーが障害物に衝突した場合
        if leader.collision_flag:
            reward += self.__reward_dict.get('collision_penalty', 0.0)
        
        # 移動距離に基づく報酬（探査促進）
        if hasattr(leader, 'coordinate') and previous_coordinate is not None:
            movement_distance = np.linalg.norm(leader.coordinate - previous_coordinate)
            if movement_distance > 0:
                reward += movement_distance * 0.1  # 移動距離に応じた小さな報酬
        
        return reward
  

    def _build_leader_state(self, leader, swarm):
        """
        各リーダーの個別状態を構築
        """
        # 基本状態
        state = {
            'agent_coordinate_x': leader.x,
            'agent_coordinate_y': leader.y,
            'agent_azimuth': leader.leader_azimuth,
            'agent_collision_flag': 1.0 if leader.collision_flag else 0.0,
            'agent_step_count': leader.leader_step_count,
            'swarm_id': swarm.swarm_id,
            'leader_id': leader.id
        }
        
        # 所属群のfollowerの情報を収集
        follower_collision_data = []
        follower_mobility_scores = []
        
        for follower in swarm.followers:
            if follower.role == RobotRole.FOLLOWER:
                # followerの衝突データ
                collision_data = follower.get_collision_data()
                follower_collision_data.extend(collision_data)
                
                # followerのmobility_score
                follower_mobility_scores.append(follower.mobility_score)
        
        # follower_mobility_scoresのパディング
        MAX_ROBOT_NUM = 10
        while len(follower_mobility_scores) < MAX_ROBOT_NUM:
            follower_mobility_scores.append(0.0)
        follower_mobility_scores = follower_mobility_scores[:MAX_ROBOT_NUM]
        
        # follower_collision_dataのパディング
        MAX_COLLISION_NUM = 100
        padded_list = follower_collision_data[:MAX_COLLISION_NUM]
        padding_needed = MAX_COLLISION_NUM - len(padded_list)
        
        fcd_ndarray = [np.array(pair, dtype=np.float32) for pair in padded_list]
        fcd_ndarray.extend([np.array([0.0, 0.0], dtype=np.float32)] * padding_needed)
        
        # 状態に追加
        state['follower_collision_data'] = fcd_ndarray
        state['follower_mobility_scores'] = follower_mobility_scores
        
        return state
  

    def _create_leader_agent(self, leader_id: str):
        """
        リーダー専用のエージェントとモデルを作成
        """
        if leader_id not in self.leader_agents:
            # リーダー専用のモデルを作成
            from models.actor_critic import ModelActorCritic
            from agents.agent_a2c import A2CAgent
            
            # 状態空間の次元を計算
            from utils.utils import flatten_state
            sample_state = self.state
            input_dim = len(flatten_state(sample_state))
            
            # モデル作成
            model = ModelActorCritic(input_dim=input_dim)
            
            # オプティマイザー作成（TensorFlow 2.x対応）
            optimizer = None
            try:
                import tensorflow as tf
                # TensorFlow 2.x
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # type: ignore
            except (AttributeError, ImportError):
                try:
                    import tensorflow as tf
                    # TensorFlow 1.x
                    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)  # type: ignore
                except (AttributeError, ImportError):
                    # Fallback: 基本的なオプティマイザー
                    print("Warning: Using fallback optimizer")
                    optimizer = None
            
            # エージェント作成
            agent = A2CAgent(
                env=self,
                algorithm=None,  # リーダー固有のアルゴリズムを使用
                model=model,
                optimizer=optimizer,
                gamma=0.99,
                n_steps=10,
                max_steps_per_episode=1000,
                action_space=self.action_space
            )
            
            self.leader_models[leader_id] = model
            self.leader_agents[leader_id] = agent
            
            print(f"Created new leader agent for leader {leader_id}")
        
        return self.leader_agents[leader_id]
  

    def save_initial_state(self, log_dir):
        """
        初期状態をJSONで保存する
        """
        # 基本情報
        initial_state_dict = {
            'timestamp': datetime.now().isoformat(),
            'simulation_info': {
                'map_width': self.__map_width,
                'map_height': self.__map_height,
                'map_seed': self.__map_seed,
                'obstacle_probability': self.__obstacle_probability,
                'obstacle_max_size': self.__obstacle_max_size,
                'obstacle_value': self.__obstacle_value,
                'robot_num': self.__robot_num,
                'finish_rate': self.__finish_rate,
                'finish_step': self.__finish_step
            },
            'robot_config': {
                'movement_min': self.__movement_min,
                'movement_max': self.__movement_max,
                'boids_min': self.__boids_min,
                'boids_max': self.__boids_max,
                'avoidance_min': self.__avoidance_min,
                'avoidance_max': self.__avoidance_max,
                'offset_position': self.__offset.position,
                'offset_step': self.__offset.step,
                'offset_amount_of_movement': self.__offset.amount_of_movement,
                'offset_direction_angle': self.__offset.direction_angle,
                'offset_collision_flag': self.__offset.collision_flag,
                'offset_boids_flag': self.__offset.boids_flag.value,
                'offset_estimated_probability': self.__offset.estimated_probability
            },
            'exploration_config': {
                'boundary_inner': self.__boundary_inner,
                'boundary_outer': self.__boundary_outer,
                'mv_mean': self.__mv_mean,
                'mv_variance': self.__mv_variance,
                'initial_coordinate_x': self.__initial_coordinate_x,
                'initial_coordinate_y': self.__initial_coordinate_y,
                'exploration_radius': self.exploration_radius
            },
            'swarm_config': {
                'leader_switch_interval': self.leader_switch_interval,
                'initial_swarm_id': self.initial_swarm_id,
                'swarm_count': len(self.swarms)
            },
            'initial_state': {
                'explored_area': self.explored_area,
                'exploration_ratio': self.exploration_ratio,
                'agent_step': self.agent_step,
                'collision_flag': self.collision_flag,
                'total_area': self.total_area,
                'MAX_COLLISION_NUM': self.MAX_COLLISION_NUM
            },
            'robots_info': []
        }
        
        # 各ロボットの初期情報を保存
        for i, robot in enumerate(self.robots):
            robot_info = {
                'robot_id': robot.id,
                'initial_position': {
                    'x': robot.x,
                    'y': robot.y
                },
                'role': robot.role.value,
                'swarm_id': robot.swarm_id,
                'motion_params': {
                    'step': robot.step,
                    'amount_of_movement': robot.amount_of_movement,
                    'direction_angle': robot.direction_angle
                }
            }
            initial_state_dict['robots_info'].append(robot_info)
        
        # 群の初期情報を保存
        initial_state_dict['swarms_info'] = []
        for swarm in self.swarms:
            swarm_info = {
                'swarm_id': swarm.swarm_id,
                'leader_id': swarm.leader.id,
                'follower_ids': [follower.id for follower in swarm.followers],
                'robot_count': swarm.get_robot_count(),
                'exploration_rate': swarm.exploration_rate,
                'step_count': swarm.step_count
            }
            initial_state_dict['swarms_info'].append(swarm_info)
        
        # 報酬設定を保存
        initial_state_dict['reward_config'] = self.__reward_dict
        
        # 状態空間の情報を保存
        initial_state_dict['state_space_info'] = {
            'state_keys': list(self.state.keys()),
            'state_dimensions': {key: len(value) if hasattr(value, '__len__') else 1 
                               for key, value in self.state.items()}
        }
        
        # ファイルに保存
        initial_state_file = os.path.join(log_dir, 'initial_state.json')
        with open(initial_state_file, 'w', encoding='utf-8') as f:
            json.dump(initial_state_dict, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Initial state saved to {initial_state_file}")
        
        return initial_state_file
  