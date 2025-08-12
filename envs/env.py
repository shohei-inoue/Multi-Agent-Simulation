"""
Environment class for the swarm robot exploration simulation.
Implements the core simulation environment with rendering and logging capabilities.
"""

import gym
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # 非インタラクティブバックエンドを使用
import matplotlib.pyplot as plt
import matplotlib.colors as mColors
from matplotlib.patches import Circle
import os
import imageio
import copy
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
import pandas as pd
from envs.env_map import generate_explored_map, generate_rect_obstacle_map
from .observation_space import create_initial_state
from robots.red import Red, BoidsType, RobotRole
from dataclasses import dataclass
from params.simulation import SimulationParam
from params.robot_logging import RobotLoggingConfig
from scores.score import Score
from core.logging import get_logger
from core.interfaces import Configurable, Stateful, Loggable, Renderable
from envs.action_space import create_swarm_action_space
from envs.observation_space import create_swarm_observation_space


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
    def __init__(self, param: SimulationParam, action_space=None, observation_space=None, reward_dict=None):
        super().__init__()
        
        # Initialize logging
        self.logger = get_logger("environment")
        
        # Configuration
        self._config = {
            "map_width": param.environment.map.width if param.environment and param.environment.map else 150,
            "map_height": param.environment.map.height if param.environment and param.environment.map else 60,
            "robot_num": param.explore.robotNum if param.explore else 10,
            "finish_rate": param.explore.finishRate if param.explore else 0.8
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
        
        # GIF作成用のフレーム保存
        self.env_frames = []
        self.current_episode = 0
        self.frame_interval = 1  # 毎ステップフレームを保存
        
        # 初期群ID
        self.initial_swarm_id = 1
        
        # 統合されたleaderの軌跡を保持するための辞書
        self.integrated_leader_trajectories = {}

        # map
        self.__map_width = param.environment.map.width if param.environment and param.environment.map else 150
        self.__map_height = param.environment.map.height if param.environment and param.environment.map else 60
        self.__map_seed = param.environment.map.seed if param.environment and param.environment.map else 42
        self.__obstacle_probability = param.environment.obstacle.probability if param.environment and param.environment.obstacle else 0.005
        self.__obstacle_max_size = param.environment.obstacle.maxSize if param.environment and param.environment.obstacle else 10
        self.__obstacle_value = param.environment.obstacle.value if param.environment and param.environment.obstacle else 1000
        self.__wall_thickness = param.environment.obstacle.wall_thickness if param.environment and param.environment.obstacle else 3

        # robot
        robot_param = param.robot_params[0] if param.robot_params else None
        if robot_param:
            self.__movement_min = robot_param.movement.min if robot_param.movement else 2.0
            self.__movement_max = robot_param.movement.max if robot_param.movement else 3.0
            self.__boids_min = robot_param.boids.min if robot_param.boids else 2.0
            self.__boids_max = robot_param.boids.max if robot_param.boids else 3.0
            self.__avoidance_min = robot_param.avoidance.min if robot_param.avoidance else 90.0
            self.__avoidance_max = robot_param.avoidance.max if robot_param.avoidance else 270.0
            self.__offset = robot_param.offset if robot_param.offset else None
        else:
            self.__movement_min = 2.0
            self.__movement_max = 3.0
            self.__boids_min = 2.0
            self.__boids_max = 3.0
            self.__avoidance_min = 90.0
            self.__avoidance_max = 270.0
            self.__offset = None

        # explore
        self.__boundary_inner = param.explore.boundary.inner if param.explore and param.explore.boundary else 0.0
        self.__boundary_outer = param.explore.boundary.outer if param.explore and param.explore.boundary else 10.0
        self.__mv_mean = param.explore.mv.mean if param.explore and param.explore.mv else 0.0
        self.__mv_variance = param.explore.mv.variance if param.explore and param.explore.mv else 10.0
        self.__initial_coordinate_x = param.explore.coordinate.x if param.explore and param.explore.coordinate else 10.0
        self.__initial_coordinate_y = param.explore.coordinate.y if param.explore and param.explore.coordinate else 10.0
        self.__robot_num = param.explore.robotNum if param.explore else 10
        self.__finish_rate = param.explore.finishRate if param.explore else 0.8
        self.MAX_COLLISION_NUM = 100
        self.__finish_step = param.explore.finishStep if param.explore else 40

        # 状態空間・行動空間はswarm_agent用のみ
        self.action_space = action_space or create_swarm_action_space()
        self.__observation_space = observation_space or create_swarm_observation_space()
        self.__reward_dict = reward_dict or (param.reward.to_dict() if param.reward else {})

        # ----- set initialize map -----
        self.__map = generate_rect_obstacle_map(
          map_width=self.__map_width, 
          map_height=self.__map_height,
          obstacle_prob=self.__obstacle_probability,
          obstacle_max_size=self.__obstacle_max_size,
          obstacle_val=self.__obstacle_value,
          seed=self.__map_seed,
          wall_thickness=self.__wall_thickness
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
          x                     = self.__initial_coordinate_x + (
              (self.__offset.position if self.__offset and self.__offset.position is not None else 0.0)
              * math.cos((2 * math.pi * index / (self.__robot_num)))
          ),
          y                     = self.__initial_coordinate_y + (
              (self.__offset.position if self.__offset and self.__offset.position is not None else 0.0)
              * math.sin((2 * math.pi * index / (self.__robot_num)))
          ),
          step                  = self.__offset.step if self.__offset and self.__offset.step is not None else 0,
          amount_of_movement    = self.__offset.amount_of_movement if self.__offset and self.__offset.amount_of_movement is not None else 0.0,
          direction_angle       = self.__offset.direction_angle if self.__offset and self.__offset.direction_angle is not None else 0.0,
          collision_flag        = self.__offset.collision_flag if self.__offset and self.__offset.collision_flag is not None else False,
          boids_flag            = BoidsType(self.__offset.boids_flag.value) if self.__offset and self.__offset.boids_flag is not None else BoidsType.NONE,
          estimated_probability = self.__offset.estimated_probability if self.__offset and self.__offset.estimated_probability is not None else 0.0,
          role                  = RobotRole.LEADER if index == 0 else RobotRole.FOLLOWER
        ) for index in range(self.__robot_num)]
        
        # 初期leaderを設定
        self.robots[0].set_role(RobotRole.LEADER)
        self.current_leader = self.robots[0]
        
        # 初期群を作成
        self.logger.info(f"Creating initial swarm with ID: {self.initial_swarm_id}")
        initial_swarm = Swarm(
            swarm_id=self.initial_swarm_id,
            leader=self.robots[0],
            followers=self.robots[1:].copy()
        )
        
        # 初期leaderの軌跡データを初期化
        self.robots[0].leader_trajectory_data = pd.DataFrame(columns=[
            'step', 'x', 'y', 'azimuth', 'role', 'swarm_id'
        ])
        
        # 全ロボットに群IDを設定
        for robot in self.robots:
            robot.swarm_id = self.initial_swarm_id
        
        self.swarms = [initial_swarm]
        self.swarm_id_counter = self.initial_swarm_id + 1
        self.logger.info(f"Swarm created with ID: {initial_swarm.swarm_id}, swarm_id_counter: {self.swarm_id_counter}")

        # ----- set initial state -----
        initial_fcd = [np.array([0.0, 0.0], dtype=np.float32)] * self.MAX_COLLISION_NUM
        self.__initial_state = create_initial_state(
          self.__agent_initial_coordinate[1],
          self.__agent_initial_coordinate[0],
          azimuth=0,          
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
        # 初期群IDを再設定
        self.initial_swarm_id = 1
        
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
          x                     = self.__initial_coordinate_x + (
              (self.__offset.position if self.__offset and self.__offset.position is not None else 0.0)
              * math.cos((2 * math.pi * index / (self.__robot_num)))
          ),
          y                     = self.__initial_coordinate_y + (
              (self.__offset.position if self.__offset and self.__offset.position is not None else 0.0)
              * math.sin((2 * math.pi * index / (self.__robot_num)))
          ),
          step                  = self.__offset.step if self.__offset and self.__offset.step is not None else 0,
          amount_of_movement    = self.__offset.amount_of_movement if self.__offset and self.__offset.amount_of_movement is not None else 0.0,
          direction_angle       = self.__offset.direction_angle if self.__offset and self.__offset.direction_angle is not None else 0.0,
          collision_flag        = self.__offset.collision_flag if self.__offset and self.__offset.collision_flag is not None else False,
          boids_flag            = BoidsType(self.__offset.boids_flag.value) if self.__offset and self.__offset.boids_flag is not None else BoidsType.NONE,
          estimated_probability = self.__offset.estimated_probability if self.__offset and self.__offset.estimated_probability is not None else 0.0,
          role                  = RobotRole.LEADER if index == 0 else RobotRole.FOLLOWER
        ) for index in range(self.__robot_num)]
        
        # 初期leaderを設定（固定）
        self.robots[0].set_role(RobotRole.LEADER)
        self.current_leader = self.robots[0]
        self.current_leader_index = 0
        self.leader_switch_counter = 0  # 現在は使用しない
        
        # 初期群を作成
        self.logger.info(f"Creating initial swarm with ID: {self.initial_swarm_id}")
        initial_swarm = Swarm(
            swarm_id=self.initial_swarm_id,
            leader=self.robots[0],
            followers=self.robots[1:].copy()
        )
        
        # 初期leaderの軌跡データを初期化
        self.robots[0].leader_trajectory_data = pd.DataFrame(columns=[
            'step', 'x', 'y', 'azimuth', 'role', 'swarm_id'
        ])
        
        # 全ロボットに群IDを設定
        for robot in self.robots:
            robot.swarm_id = self.initial_swarm_id
        
        self.swarms = [initial_swarm]
        self.swarm_id_counter = self.initial_swarm_id + 1
        self.logger.info(f"Swarm created with ID: {initial_swarm.swarm_id}, swarm_id_counter: {self.swarm_id_counter}")

        # ----- reset drawing infos -----
        self.agent_trajectory = [self.agent_coordinate.copy()] # 奇跡の初期化
        self.result_pdf_list  = [] # 統合確率分布の初期化
        
        # 統合されたleaderの軌跡データをリセット
        self.integrated_leader_trajectories = {}

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

        # SystemAgentのエピソードリセット
        if hasattr(self, 'system_agent') and self.system_agent is not None:
            self.system_agent.reset_episode()

        return self.state


    def _create_new_swarm(self, new_leader: Red, followers: List[Red]):
        """新しい群を作成"""
        new_swarm_id = self.get_next_swarm_id()
        new_swarm = Swarm(
            swarm_id=new_swarm_id,
            leader=new_leader,
            followers=followers
        )
        
        # 新しいleaderの軌跡データを初期化
        new_leader.leader_trajectory_data = pd.DataFrame(columns=[
            'step', 'x', 'y', 'azimuth', 'role', 'swarm_id'
        ])
        
        # 各followerのswarm_idを更新
        for follower in followers:
            follower.swarm_id = new_swarm_id
        
        self.swarms.append(new_swarm)
        return new_swarm


    def _merge_swarms(self, target_swarm: Swarm, source_swarm: Swarm):
        """
        群を統合
        """
        # source_swarmのfollowerをtarget_swarmに移動し、leader参照を更新
        for follower in source_swarm.followers:
            target_swarm.add_follower(follower)
            # followerのleader参照を統合先のleaderに更新
            follower.agent_coordinate = target_swarm.leader.coordinate
        
        # source_swarmのleaderの軌跡を保存（統合後も表示するため）
        if (hasattr(source_swarm.leader, 'leader_trajectory_data') and 
            source_swarm.leader.leader_trajectory_data is not None and 
            len(source_swarm.leader.leader_trajectory_data) > 0):
            self.integrated_leader_trajectories[source_swarm.leader.id] = source_swarm.leader.leader_trajectory_data.copy()
        
        # source_swarmのleaderもfollowerとして追加
        source_swarm.leader.set_role(RobotRole.FOLLOWER)
        target_swarm.add_follower(source_swarm.leader)
        # 統合されたleaderのleader参照も更新
        source_swarm.leader.agent_coordinate = target_swarm.leader.coordinate
        
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
        elif mode == 'gif':
          # GIF生成用のモード
          if ax is None:
              # マップのアスペクト比に合わせてfigsizeを調整
              aspect_ratio = self.__map_width / self.__map_height
              if aspect_ratio > 1:
                  # 横幅が大きい場合（200x100など）
                  fig_width = 12  # 横幅を大きくする
                  fig_height = 12 / aspect_ratio
              else:
                  # 縦幅が大きい場合
                  fig_height = 12
                  fig_width = 12 * aspect_ratio
              
              fig, ax = plt.subplots(figsize=(fig_width, fig_height))
          else:
              fig = ax.figure
        elif mode == 'rgb_array':
          # GIF保存用のrgb_arrayモード
          fig, ax = plt.subplots(figsize=(fig_size, fig_size))
          
          # 背景を白で表示
          ax.imshow(
              np.ones((self.__map_height, self.__map_width)),
              cmap='gray',
              vmin=0, vmax=1,
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )
          
          # 地図（障害物を黒で表示）
          ax.imshow(
              self.__map,
              cmap='gray_r',
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # 探査済み領域（薄いグレーで表示）
          explored_display = np.where(self.explored_map > 0, 1, 0)
          ax.imshow(
              explored_display,
              cmap='gray',
              alpha=0.2,  # より薄い透明度
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # 各群のロボットを描画
          colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
          trajectory_colors = ['darkgreen', 'darkorange', 'darkred', 'darkblue', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkseagreen', 'darkslateblue']
          
          for swarm_idx, swarm in enumerate(self.swarms):
              # リーダーIDに基づいてカラーを決定
              leader_id = int(swarm.leader.id) if swarm.leader.id.isdigit() else hash(swarm.leader.id) % len(colors)
              leader_color = colors[leader_id % len(colors)]
              trajectory_color = trajectory_colors[leader_id % len(trajectory_colors)]
              
              # leaderの描画（探査中心として表示）
              leader = swarm.leader
              ax.scatter(
                  x=leader.data['x'].iloc[-1],
                  y=leader.data['y'].iloc[-1],
                  color=leader_color,  # リーダー固有のカラー
                  s=25,
                  marker='*',
                  edgecolors='black',
                  linewidth=1,
                  label=f"Swarm {swarm.swarm_id} Leader (ID: {leader.id})"
              )
              
              # leaderの軌跡（リーダー固有のカラーで表示）
              if hasattr(leader, 'leader_trajectory_data') and len(leader.leader_trajectory_data) > 0:
                  # リーダーになった時点からの軌跡を表示
                  ax.plot(
                      leader.leader_trajectory_data['x'],
                      leader.leader_trajectory_data['y'],
                      color=trajectory_color,  # リーダー固有の軌跡カラー
                      linewidth=1.5,
                      alpha=0.6,  # 適度な透明度
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              else:
                  # 従来の軌跡表示（フォールバック）
                  ax.plot(
                      leader.data['x'],
                      leader.data['y'],
                      color=trajectory_color,  # リーダー固有の軌跡カラー
                      linewidth=1.5,
                      alpha=0.6,  # 適度な透明度
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              
              # leaderを中心とした探査領域の表示
              if len(leader.data['x']) > 0:
                  current_x = leader.data['x'].iloc[-1]
                  current_y = leader.data['y'].iloc[-1]
                  
                  # 探査領域の円を描画
                  circle = Circle(
                      (current_x, current_y), 
                      self.exploration_radius, 
                      color='blue', 
                      alpha=0.1, 
                      fill=True,
                      linestyle='--',
                      linewidth=1
                  )
                  ax.add_patch(circle)
                  
                  # 探査領域の境界線
                  circle_boundary = Circle(
                      (current_x, current_y), 
                      self.exploration_radius, 
                      color='blue', 
                      alpha=0.3, 
                      fill=False,
                      linestyle='--',
                      linewidth=1.5
                  )
                  ax.add_patch(circle_boundary)
              
              # followerの描画
              for i, follower in enumerate(swarm.followers):
                if follower.role == RobotRole.FOLLOWER:
                  fx = follower.data['x'].iloc[-1]
                  fy = follower.data['y'].iloc[-1]
                  ax.scatter(
                      x=fx,
                      y=fy,
                      color=leader_color,  # リーダーと同じカラーを使用
                      s=10,
                      marker='o',
                      alpha=0.7,
                      label=f"Follower (Leader ID: {leader.id})" if i == 0 and swarm_idx == 0 else None
                  )
                  
                  # フォロワの軌跡（薄い線で表示）
                  if len(follower.data['x']) > 1:
                      ax.plot(
                          follower.data['x'],
                          follower.data['y'],
                          color='gray',  # グレーで統一
                          linewidth=0.5,
                          alpha=0.3,  # より薄い透明度
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
          
          # より堅牢な方法: 一時ファイルを使用して画像を生成
          import tempfile
          import os
          from PIL import Image
          
          # 一時ファイルに保存
          with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
              fig.savefig(tmp_file.name, format='png', bbox_inches='tight', dpi=100)
              
              # PILで読み込み
              img = Image.open(tmp_file.name)
              img_array = np.array(img)
              
              # 一時ファイルを削除
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

          # 背景色を白に設定
          ax.set_facecolor('white')
          fig.patch.set_facecolor('white')
          
          # 背景を白で表示（最初に白い背景を描画）
          background = np.ones((self.__map_height, self.__map_width))
          ax.imshow(
              background,
              cmap='gray',
              vmin=1, vmax=1,  # 白を表示するため
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )
          
          # 地図（障害物を黒で表示）
          ax.imshow(
              self.__map,
              cmap='gray_r',
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # 探査済み領域（薄いグレーで表示）
          explored_display = np.where(self.explored_map > 0, 1, 0)
          ax.imshow(
              explored_display,
              cmap='gray',
              alpha=0.2,  # より薄い透明度
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # 各群のロボットを描画
          colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
          trajectory_colors = ['darkgreen', 'darkorange', 'darkred', 'darkblue', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkseagreen', 'darkslateblue']
          
          for swarm_idx, swarm in enumerate(self.swarms):
              # リーダーIDに基づいてカラーを決定
              leader_id = int(swarm.leader.id) if swarm.leader.id.isdigit() else hash(swarm.leader.id) % len(colors)
              leader_color = colors[leader_id % len(colors)]
              trajectory_color = trajectory_colors[leader_id % len(trajectory_colors)]
              
              # leaderの描画（探査中心として表示）
              leader = swarm.leader
              ax.scatter(
                  x=leader.data['x'].iloc[-1],
                  y=leader.data['y'].iloc[-1],
                  color=leader_color,  # リーダー固有のカラー
                  s=25,
                  marker='*',
                  edgecolors='black',
                  linewidth=1,
                  label=f"Swarm {swarm.swarm_id} Leader (ID: {leader.id})"
              )
              
              # leaderの軌跡（リーダー固有のカラーで表示）
              if hasattr(leader, 'leader_trajectory_data') and len(leader.leader_trajectory_data) > 0:
                  # リーダーになった時点からの軌跡を表示
                  ax.plot(
                      leader.leader_trajectory_data['x'],
                      leader.leader_trajectory_data['y'],
                      color=trajectory_color,  # リーダー固有の軌跡カラー
                      linewidth=1.5,
                      alpha=0.6,  # 適度な透明度
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              else:
                  # 従来の軌跡表示（フォールバック）
                  ax.plot(
                      leader.data['x'],
                      leader.data['y'],
                      color=trajectory_color,  # リーダー固有の軌跡カラー
                      linewidth=1.5,
                      alpha=0.6,  # 適度な透明度
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              
              # leaderを中心とした探査領域の表示
              if len(leader.data['x']) > 0:
                  current_x = leader.data['x'].iloc[-1]
                  current_y = leader.data['y'].iloc[-1]
                  
                  # 探査領域の円を描画
                  circle = Circle(
                      (current_x, current_y), 
                      self.exploration_radius, 
                      color='blue', 
                      alpha=0.1, 
                      fill=True,
                      linestyle='--',
                      linewidth=1
                  )
                  ax.add_patch(circle)
                  
                  # 探査領域の境界線
                  circle_boundary = Circle(
                      (current_x, current_y), 
                      self.exploration_radius, 
                      color='blue', 
                      alpha=0.3, 
                      fill=False,
                      linestyle='--',
                      linewidth=1.5
                  )
                  ax.add_patch(circle_boundary)
              
              # followerの描画
              for i, follower in enumerate(swarm.followers):
                if follower.role == RobotRole.FOLLOWER:
                  fx = follower.data['x'].iloc[-1]
                  fy = follower.data['y'].iloc[-1]
                  ax.scatter(
                      x=fx,
                      y=fy,
                      color=leader_color,  # リーダーと同じカラーを使用
                      s=10,
                      marker='o',
                      alpha=0.7,
                      label=f"Follower (Leader ID: {leader.id})" if i == 0 and swarm_idx == 0 else None
                  )
                  
                  # フォロワの軌跡（薄い線で表示）
                  if len(follower.data['x']) > 1:
                      ax.plot(
                          follower.data['x'],
                          follower.data['y'],
                          color='gray',  # グレーで統一
                          linewidth=0.5,
                          alpha=0.3,  # より薄い透明度
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

        # GIFモードの場合は、rgb_arrayモードと同じ処理を行う
        elif mode == 'gif':
          # 背景を白で表示
          ax.imshow(
              np.ones((self.__map_height, self.__map_width)),
              cmap='gray',
              vmin=0, vmax=1,
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )
          
          # 地図（障害物を黒で表示）
          ax.imshow(
              self.__map,
              cmap='gray_r',
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # 探査済み領域（薄いグレーで表示）
          explored_display = np.where(self.explored_map > 0, 1, 0)
          ax.imshow(
              explored_display,
              cmap='gray',
              alpha=0.3,
              origin='lower',
              extent=(0, self.__map_width, 0, self.__map_height),
          )

          # 各群のロボットを描画
          colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
          trajectory_colors = ['darkgreen', 'darkorange', 'darkred', 'darkblue', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkseagreen', 'darkslateblue']
          
          for swarm_idx, swarm in enumerate(self.swarms):
              # リーダーIDに基づいてカラーを決定
              leader_id = int(swarm.leader.id) if swarm.leader.id.isdigit() else hash(swarm.leader.id) % len(colors)
              leader_color = colors[leader_id % len(colors)]
              trajectory_color = trajectory_colors[leader_id % len(trajectory_colors)]
              
              # leaderの描画（探査中心として表示）
              leader = swarm.leader
              ax.scatter(
                  x=leader.data['x'].iloc[-1],
                  y=leader.data['y'].iloc[-1],
                  color=leader_color,  # リーダー固有のカラー
                  s=25,
                  marker='*',
                  edgecolors='black',
                  linewidth=1,
                  label=f"Swarm {swarm.swarm_id} Leader (ID: {leader.id})"
              )
              
              # leaderの軌跡（リーダー固有のカラーで表示）
              if hasattr(leader, 'leader_trajectory_data') and len(leader.leader_trajectory_data) > 0:
                  # リーダーになった時点からの軌跡を表示
                  ax.plot(
                      leader.leader_trajectory_data['x'],
                      leader.leader_trajectory_data['y'],
                      color=trajectory_color,  # リーダー固有の軌跡カラー
                      linewidth=1.5,
                      alpha=0.6,  # 適度な透明度
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              else:
                  # 従来の軌跡表示（フォールバック）
                  ax.plot(
                      leader.data['x'],
                      leader.data['y'],
                      color=trajectory_color,  # リーダー固有の軌跡カラー
                      linewidth=1.5,
                      alpha=0.6,  # 適度な透明度
                      linestyle='-',
                      label=f"Leader {swarm.swarm_id} Trajectory (ID: {leader.id})"
                  )
              
              # leaderを中心とした探査領域の表示
              if len(leader.data['x']) > 0:
                  current_x = leader.data['x'].iloc[-1]
                  current_y = leader.data['y'].iloc[-1]
                  
                  # 探査領域の円を描画
                  circle = Circle(
                      (current_x, current_y), 
                      self.exploration_radius, 
                      color='blue', 
                      alpha=0.1, 
                      fill=True,
                      linestyle='--',
                      linewidth=1
                  )
                  ax.add_patch(circle)
                  
                  # 探査領域の境界線
                  circle_boundary = Circle(
                      (current_x, current_y), 
                      self.exploration_radius, 
                      color='blue', 
                      alpha=0.3, 
                      fill=False,
                      linestyle='--',
                      linewidth=1.5
                  )
                  ax.add_patch(circle_boundary)
              
              # フォロワの描画
              for follower in swarm.followers:
                  if len(follower.data['x']) > 0:
                      fx = follower.data['x'].iloc[-1]
                      fy = follower.data['y'].iloc[-1]
                      ax.scatter(
                          fx, fy,
                          color=leader_color,  # リーダーと同じカラー
                          s=15,
                          marker='o',
                          edgecolors='black',
                          linewidth=0.5,
                          alpha=0.8
                      )
                      
                      # フォロワの軌跡（薄い線で表示）
                      if len(follower.data['x']) > 1:
                          ax.plot(
                              follower.data['x'],
                              follower.data['y'],
                              color=trajectory_color,  # リーダーと同じ軌跡カラー
                              linewidth=0.5,
                              alpha=0.3,  # より薄い透明度
                              linestyle='-'
                          )

        #   # === フォロワの現在位置 ===
        #   for i, follower in enumerate(self.follower_robots):
        #       fx = follower.data['x'].iloc[-1]
        #       fy = follower.data['y'].iloc[-1]
        # ax.plot(fx, fy, 'ro', markersize=4, label="Follower" if i == 0 else "", zorder=3)

  

    def step(self, actions):
        """
        actions: {swarm_id: {"theta": ...}, ...}
        各swarm_agentのアクションをまとめて受け取り、全群のリーダーの行動を一斉に反映する。
        """
        # agent_stepを更新
        self._state["agent_step"] += 1
        
        # SystemAgentのステップカウンターを更新
        if hasattr(self, 'system_agent') and self.system_agent is not None:
            self.system_agent.update_step()
        
        for swarm_id, action in actions.items():
            theta = action.get("theta")
            swarm = self._find_swarm_by_id(swarm_id)
            if swarm is None:
                continue
            leader = getattr(swarm, "leader", None)
            if leader is not None:
                # 毎アクション開始時に衝突フラグをリセット
                leader.collision_flag = False
                
                # 1. thetaから予定座標を算出（フロンティアベース探査）
                next_coordinate = None
                if theta is not None:
                    # 移動距離を設定（outer_boundary - フロンティアベース探査）
                    movement_distance = self.__boundary_outer
                    # thetaから次の座標を計算
                    next_x = leader.x + movement_distance * np.cos(theta)
                    next_y = leader.y + movement_distance * np.sin(theta)
                    next_coordinate = np.array([next_y, next_x])
                # 2. 衝突判定（移動経路上の障害物もチェック）
                collision = False
                if next_coordinate is not None:
                    # 移動経路上の複数点をチェック
                    current_coord = np.array([leader.y, leader.x])
                    target_coord = next_coordinate
                    
                    # 移動距離に応じてサンプリング数を決定
                    distance = np.linalg.norm(target_coord - current_coord)
                    sampling_num = max(200, int(distance * 50))  # 最低200点、距離に応じて増加
                    
                    # ロボットのサイズを考慮した衝突判定
                    robot_radius = 1  # ロボットの半径を小さく調整
                    
                    for i in range(0, sampling_num + 1):  # 0から開始して開始点もチェック
                        # 中間点を計算
                        t = i / sampling_num
                        intermediate_coord = current_coord + t * (target_coord - current_coord)
                        
                        # ロボットの中心座標
                        center_y = int(intermediate_coord[0])
                        center_x = int(intermediate_coord[1])
                        
                        # ロボットの範囲内の全てのピクセルをチェック
                        collision_detected = False
                        for dy in range(-robot_radius, robot_radius + 1):
                            for dx in range(-robot_radius, robot_radius + 1):
                                # 円形の範囲内かチェック
                                if dx*dx + dy*dy <= robot_radius*robot_radius:
                                    check_y = center_y + dy
                                    check_x = center_x + dx
                                    
                                    # マップ内かチェック
                                    if (0 <= check_y < self.__map_height and 0 <= check_x < self.__map_width):
                                        # 障害物かチェック
                                        if self.__map[check_y, check_x] == self.__obstacle_value:
                                            collision = True
                                            collision_detected = True
                                            break
                                    else:
                                        collision = True  # マップ外は衝突
                                        collision_detected = True
                                        break
                            
                            if collision_detected:
                                break
                        
                        if collision:
                            break
                # 3. 移動
                if not collision and next_coordinate is not None:
                    previous_coordinate = leader.coordinate.copy()
                    # ロボットの座標を直接更新
                    leader.coordinate = next_coordinate
                    leader.x = next_coordinate[1]  # x座標
                    leader.y = next_coordinate[0]  # y座標
                    leader.agent_coordinate = next_coordinate  # 追加: agent_coordinateも更新
                    
                    # 移動が成功した場合は衝突フラグをリセット
                    leader.collision_flag = False
                    
                    # leader_trajectory_dataを更新（移動成功時のみ）
                    if hasattr(leader, 'leader_trajectory_data'):
                        # 新しいデータを追加
                        new_data = leader.get_arguments()
                        leader.leader_trajectory_data = pd.concat([
                            leader.leader_trajectory_data, 
                            new_data
                        ], ignore_index=True)
                    
                    # 探査マップを更新
                    self.update_exploration_map(previous_coordinate, leader.coordinate)
                else:
                    # 衝突時の処理
                    if collision:
                        print(f"[WARNING] Leader {leader.id} collision detected, but continuing movement")
                        leader.collision_flag = True
                        # 衝突フラグは立てるが、移動は継続（VFH-Fuzzyで回避方向を選択）
                    else:
                        # next_coordinateがNoneの場合
                        # この場合は移動を試行しないが、衝突フラグはリセット
                        leader.collision_flag = False
                # 4. followerのleader座標を更新
                followers = getattr(swarm, "followers", [])
                one_explore_step = getattr(self.__offset, "one_explore_step", 60) if hasattr(self, "__offset") and self.__offset is not None else 60
                # 5. 全followerを1stepずつ動かすループをone_explore_step回繰り返す
                for _ in range(one_explore_step):
                    for follower in followers:
                        # followerの移動前の座標を保存
                        previous_follower_coordinate = follower.coordinate.copy()
                        
                        # leaderの座標を渡してstep_motion
                        follower.step_motion(agent_coordinate=leader.coordinate)
                        
                        # followerの移動経路を探査済みにする
                        self.update_exploration_map(previous_follower_coordinate, follower.coordinate)

        # 環境の状態・報酬・done判定を更新
        reward = self._calculate_reward() if hasattr(self, "_calculate_reward") else 0.0
        done = self._check_done() if hasattr(self, "_check_done") else False
        truncated = False
        self.state = self._get_current_state() if hasattr(self, "_get_current_state") else {}
        info = {}
        return self.state, reward, done, truncated, info
  

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
           else:
               # マップ外に出た場合、衝突として処理
               print(f"Agent would move outside map at {intermediate_coordinate}")
               collision_flag = True
               
               # マップ境界に衝突する事前位置の計算
               collision_coordinate = intermediate_coordinate
               direction_vector = collision_coordinate - self.agent_coordinate
               norm_direction_vector = np.linalg.norm(direction_vector)
               stop_coordinate = self.agent_coordinate + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)
               return stop_coordinate, collision_flag

        return self.agent_coordinate + np.array([dy, dx]), collision_flag
         

    def save_gif(self, log_dir, episode) -> None:
        """
        保存したフレームをGIFとして保存する（拡張版）

        Parameters:
        - log_dir: ログディレクトリ (gifs/ 以下に保存される）
        - episode: エピソード番号
        """
        if not self.env_frames:
            self.logger.warning(f"No frames to save for episode {episode}")
            return
            
        gif_dir = os.path.join(log_dir, "gifs")
        gif_name = f"episode_{episode:04d}.gif"

        os.makedirs(gif_dir, exist_ok=True)
        
        try:
            imageio.mimsave(
                os.path.join(gif_dir, gif_name),
                self.env_frames,
                duration=0.2,  # フレーム間隔を調整
                loop=0  # 無限ループ
            )
            self.logger.info(f"GIF saved: {gif_name} with {len(self.env_frames)} frames")
        except Exception as e:
            self.logger.error(f"Failed to save GIF: {e}")
        finally:
            self.env_frames = []  # フレームをリセット

    def start_episode(self, episode: int):
        """エピソード開始時の処理"""
        self.current_episode = episode
        self.env_frames = []  # フレームをリセット
        
        # agent_stepをリセット
        self._state["agent_step"] = 0
        
        # 統合されたleaderの軌跡データをリセット
        self.integrated_leader_trajectories = {}
        
        # 全leaderの軌跡データをリセット
        for swarm in self.swarms:
            if hasattr(swarm.leader, 'leader_trajectory_data'):
                swarm.leader.leader_trajectory_data = pd.DataFrame(columns=[
                    'step', 'x', 'y', 'azimuth', 'role', 'swarm_id'
                ])
        
        self.logger.info(f"Started episode {episode}")

    def end_episode(self, log_dir: str):
        """エピソード終了時の処理"""
        if self.env_frames:
            self.save_gif(log_dir, self.current_episode)
        self.logger.info(f"Ended episode {self.current_episode}")

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
                mobility_score = follower.mobility_score
                follower_mobility_scores.append(mobility_score)
        
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
        初期状態をJSONで保存する（新しいパラメータ構造対応）
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
                'offset_position': self.__offset.position if self.__offset else None,
                'offset_step': self.__offset.step if self.__offset else None,
                'offset_amount_of_movement': self.__offset.amount_of_movement if self.__offset else None,
                'offset_direction_angle': self.__offset.direction_angle if self.__offset else None,
                'offset_collision_flag': self.__offset.collision_flag if self.__offset else None,
                'offset_boids_flag': self.__offset.boids_flag.value if self.__offset else None,
                'offset_estimated_probability': self.__offset.estimated_probability if self.__offset else None
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
  

    def get_system_agent_observation(self):
        """
        SystemAgent用の観測情報を生成
        Returns:
            Dict with keys:
            - swarm_mobility_score: アクセスしてきた群のfollower_mobility_scoreの分布や平均
            - swarm_count: 現在の全体の群数
            - swarm_id: system_agentにアクセスしてきた群のID
            - follower_count: アクセスしてきた群のfollower数
        """
        MAX_FOLLOWERS = 100
        
        # デフォルトの観測（群がアクセスしていない場合）
        obs = {
            "swarm_mobility_score": np.zeros(MAX_FOLLOWERS, dtype=np.float32),
            "swarm_count": len(self.swarms),
            "swarm_id": 0,
            "follower_count": 0
        }
        
        # 現在アクセスしている群の情報を設定
        # 実際の実装では、どの群がアクセスしているかを追跡する必要がある
        # 現在は簡易的に最初の群の情報を使用
        if self.swarms:
            current_swarm = self.swarms[0]  # 簡易的に最初の群を使用
            
            # followerのmobility_scoreを取得
            follower_scores = []
            for follower in current_swarm.followers:
                mobility_score = self._calculate_mobility_score(follower)
                follower_scores.append(mobility_score)
            
            # 配列に変換（MAX_FOLLOWERSまで）
            scores_array = np.array(follower_scores[:MAX_FOLLOWERS], dtype=np.float32)
            obs["swarm_mobility_score"][:len(scores_array)] = scores_array
            
            obs["swarm_id"] = current_swarm.swarm_id
            obs["follower_count"] = len(current_swarm.followers)
        
        return obs

    def get_swarm_agent_observation(self, swarm_id: int):
        """
        SwarmAgent用の観測を取得
        """
        swarm = self._find_swarm_by_id(swarm_id)
        if swarm is None:
            # デフォルトの観測を返す
            return {
                "agent_coordinate_x": 0.0,
                "agent_coordinate_y": 0.0,
                "agent_azimuth": 0.0,
                "agent_collision_flag": False,
                "agent_step_count": self.agent_step,
                "follower_collision_data": [0.0] * 20,  # 10個 × 2要素
                "follower_mobility_scores": [0.0] * 10,
                "swarm_count": len(self.swarms)
            }
        
        leader = swarm.leader
        followers = swarm.followers
        
        # followerの衝突データ（azimuth, distance形式）
        follower_collision_data = []
        for follower in followers:
            # leaderからの相対位置を計算
            dx = follower.x - leader.x
            dy = follower.y - leader.y
            distance = np.sqrt(dx**2 + dy**2)
            azimuth = np.arctan2(dy, dx)
            
            # 衝突フラグがある場合は距離を0に（危険として扱う）
            if follower.collision_flag:
                distance = 0.0
            
            follower_collision_data.extend([azimuth, distance])
        
        # 20個分に調整（10個 × 2要素）
        while len(follower_collision_data) < 20:  # 10個 × 2要素
            follower_collision_data.extend([0.0, 0.0])
        follower_collision_data = follower_collision_data[:20]
        
        # followerのmobility_scores
        follower_mobility_scores = []
        for follower in followers:
            score = self._calculate_mobility_score(follower)
            follower_mobility_scores.append(score)
        
        # 10個分に調整
        while len(follower_mobility_scores) < 10:
            follower_mobility_scores.append(0.0)
        follower_mobility_scores = follower_mobility_scores[:10]
        
        return {
            "agent_coordinate_x": leader.x,
            "agent_coordinate_y": leader.y,
            "agent_azimuth": leader.direction_angle if hasattr(leader, 'direction_angle') else 0.0,
            "agent_collision_flag": leader.collision_flag,
            "agent_step_count": self.agent_step,
            "follower_collision_data": follower_collision_data,
            "follower_mobility_scores": follower_mobility_scores,
            "swarm_count": len(self.swarms)
        }

    def handle_swarm_branch(self, source_swarm_id: int, new_swarm_id: int, valid_directions: List):
        """
        SystemAgentから呼び出される分岐処理
        Args:
            source_swarm_id: 元の群ID
            new_swarm_id: 新しい群ID
            valid_directions: 有効な移動方向のリスト
        """
        # 元の群を検索
        source_swarm = self._find_swarm_by_id(source_swarm_id)
        if source_swarm is None:
            print(f"Source swarm {source_swarm_id} not found for branching")
            return
        
        # 分岐条件チェック：最低6体のfollowerが必要（3体ずつに分割するため）
        if len(source_swarm.followers) < 6:
            print(f"Swarm {source_swarm_id} has insufficient followers ({len(source_swarm.followers)}) for branching. Need at least 6 followers to ensure both swarms have 3+ followers.")
            return
        
        # SystemAgentから分岐アルゴリズムタイプを取得
        branch_algorithm_type = "mobility_based"  # デフォルト値
        if hasattr(self, 'system_agent') and self.system_agent:
            if hasattr(self.system_agent, 'param') and self.system_agent.param:
                branch_algorithm_type = getattr(self.system_agent.param.branch_condition, 'branch_algorithm', 'mobility_based')
        
        # 分岐アルゴリズムを作成
        from algorithms.branch_algorithm import create_branch_algorithm
        branch_algorithm = create_branch_algorithm(branch_algorithm_type)
        
        # followerのmobility_scoreを計算
        mobility_scores = []
        for follower in source_swarm.followers:
            mobility_score = self._calculate_mobility_score(follower)
            mobility_scores.append(mobility_score)
        
        # 分岐アルゴリズムを実行
        branch_result = branch_algorithm.execute_branch(source_swarm, valid_directions, mobility_scores)
        if branch_result is None:
            print(f"Branch algorithm failed for swarm {source_swarm_id}")
            return
        
        # 分岐後の両方の群が最低3体以上のfollowerを持つことを確認
        new_follower_count = len(branch_result.new_followers)
        remaining_follower_count = len(source_swarm.followers) - new_follower_count - 1  # -1は新しいleader
        
        if new_follower_count < 3:
            print(f"Branch validation failed: new swarm would have only {new_follower_count} followers (need 3+)")
            return
        
        if remaining_follower_count < 3:
            print(f"Branch validation failed: original swarm would have only {remaining_follower_count} followers (need 3+)")
            return
        
        # 新しい群を作成
        new_swarm = Swarm(
            swarm_id=new_swarm_id,
            leader=branch_result.new_leader,
            followers=branch_result.new_followers
        )
        
        # 新しいleaderの軌跡データを初期化
        branch_result.new_leader.leader_trajectory_data = pd.DataFrame(columns=[
            'step', 'x', 'y', 'azimuth', 'role', 'swarm_id'
        ])
        
        # ロボットの群IDと役割を更新
        branch_result.new_leader.set_role(RobotRole.LEADER)
        branch_result.new_leader.swarm_id = new_swarm_id
        
        # 新しい群のfollowerのleader参照を更新
        for follower in branch_result.new_followers:
            follower.swarm_id = new_swarm_id
            # followerのleader参照を新しいleaderに更新
            follower.agent_coordinate = branch_result.new_leader.coordinate
        
        # 新しい群を環境に追加
        self.swarms.append(new_swarm)
        
        # 元の群からfollowerを削除
        for follower in [branch_result.new_leader] + branch_result.new_followers:
            source_swarm.remove_follower(follower)
        
        # 新しい群の初期アクションを設定
        self._set_initial_action_for_new_swarm(new_swarm, branch_result.initial_theta)
        
        print(f"Branch completed: swarm {source_swarm_id} -> new swarm {new_swarm_id} with {len(branch_result.new_followers)} followers, initial_theta: {np.rad2deg(branch_result.initial_theta):.1f}° (algorithm: {branch_algorithm_type})")

    def handle_swarm_integration(self, source_swarm_id: int, target_swarm_id: int):
        """
        SystemAgentから呼び出される統合処理
        Args:
            source_swarm_id: 統合元の群ID
            target_swarm_id: 統合先の群ID
        """
        # 統合元と統合先の群を検索
        source_swarm = self._find_swarm_by_id(source_swarm_id)
        target_swarm = self._find_swarm_by_id(target_swarm_id)
        
        if source_swarm is None or target_swarm is None:
            print(f"Source swarm {source_swarm_id} or target swarm {target_swarm_id} not found for integration")
            return
        
        if source_swarm == target_swarm:
            print(f"Cannot integrate swarm {source_swarm_id} into itself")
            return
        
        # SystemAgentから統合アルゴリズムタイプを取得
        integration_algorithm_type = "nearest"  # デフォルト値
        if hasattr(self, 'system_agent') and self.system_agent:
            if hasattr(self.system_agent, 'param') and self.system_agent.param:
                integration_algorithm_type = getattr(self.system_agent.param.integration_condition, 'integration_algorithm', 'nearest')
        
        # 統合アルゴリズムを作成
        from algorithms.integration_algorithm import create_integration_algorithm
        integration_algorithm = create_integration_algorithm(integration_algorithm_type)
        
        # 統合アルゴリズムを実行
        integration_result = integration_algorithm.execute_integration(source_swarm, self.swarms, source_swarm_id)
        if integration_result is None:
            print(f"Integration algorithm failed for swarm {source_swarm_id}")
            return
        
        # 統合処理を実行
        self._merge_swarms(target_swarm, source_swarm)
        
        print(f"Integration completed: swarm {source_swarm_id} merged into swarm {target_swarm_id} using {integration_result.integration_method} method (algorithm: {integration_algorithm_type})")

    def _set_initial_action_for_new_swarm(self, new_swarm, initial_theta: float):
        """
        新しい群の初期アクションを設定
        """
        # 新しい群のleaderに初期アクションを設定
        if hasattr(new_swarm.leader, 'set_initial_action'):
            new_swarm.leader.set_initial_action(initial_theta)
        else:
            # 簡易的な初期アクション設定
            if hasattr(new_swarm.leader, 'azimuth'):
                new_swarm.leader.azimuth = initial_theta

    def _find_swarm_by_id(self, swarm_id: int) -> Optional[Swarm]:
        """
        群IDから群を検索
        """
        for swarm in self.swarms:
            if swarm.swarm_id == swarm_id:
                return swarm
        return None

    def _select_branch_leader(self, source_swarm: Swarm, valid_directions: List) -> Optional[Red]:
        """
        分岐時の新しいleaderを選択
        Args:
            source_swarm: 元の群
            valid_directions: 有効な移動方向のリスト
        Returns:
            選択された新しいleader
        """
        if not source_swarm.followers:
            return None
        
        # 最も高いmobility_scoreを持つfollowerを選択
        best_follower = None
        best_score = -1.0
        
        for follower in source_swarm.followers:
            # mobility_scoreを計算（簡易版）
            mobility_score = self._calculate_mobility_score(follower)
            if mobility_score > best_score:
                best_score = mobility_score
                best_follower = follower
        
        return best_follower

    def _calculate_mobility_score(self, robot: Red) -> float:
        """
        ロボットのmobility_scoreを計算
        Args:
            robot: 対象ロボット
        Returns:
            mobility_score (0.0-1.0)
        """
        # 簡易的なmobility_score計算
        # 実際の実装では、より複雑な計算が必要
        
        # 1. 周囲の障害物密度
        obstacle_density = self._calculate_obstacle_density(robot.coordinate)
        
        # 2. 他のロボットとの距離
        robot_distance = self._calculate_robot_distance(robot)
        
        # 3. 探査済み領域との距離
        exploration_distance = self._calculate_exploration_distance(robot.coordinate)
        
        # スコアを統合（0.0-1.0）
        mobility_score = (
            (1.0 - obstacle_density) * 0.4 +
            robot_distance * 0.3 +
            exploration_distance * 0.3
        )
        
        return max(0.0, min(1.0, mobility_score))

    def _calculate_obstacle_density(self, coordinate: np.ndarray) -> float:
        """
        座標周囲の障害物密度を計算
        """
        x, y = int(coordinate[0]), int(coordinate[1])
        radius = 5
        obstacle_count = 0
        total_count = 0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.__map_width and 
                    0 <= check_y < self.__map_height):
                    total_count += 1
                    if self.__map[check_y, check_x] > 0:
                        obstacle_count += 1
        
        return obstacle_count / total_count if total_count > 0 else 0.0

    def _calculate_robot_distance(self, robot: Red) -> float:
        """
        他のロボットとの距離を計算
        """
        min_distance = float('inf')
        
        for swarm in self.swarms:
            for other_robot in swarm.get_all_robots():
                if other_robot != robot:
                    distance = np.linalg.norm(robot.coordinate - other_robot.coordinate)
                    min_distance = min(min_distance, distance)
        
        # 距離を0.0-1.0に正規化（距離が遠いほど高いスコア）
        if min_distance == float('inf'):
            return 1.0
        return min(1.0, min_distance / 20.0)  # 20.0を最大距離とする

    def _calculate_exploration_distance(self, coordinate: np.ndarray) -> float:
        """
        探査済み領域との距離を計算
        """
        x, y = int(coordinate[0]), int(coordinate[1])
        radius = 10
        explored_count = 0
        total_count = 0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.__map_width and 
                    0 <= check_y < self.__map_height):
                    total_count += 1
                    if self.explored_map[check_y, check_x] > 0:
                        explored_count += 1
        
        # 未探査領域が多いほど高いスコア
        unexplored_ratio = 1.0 - (explored_count / total_count if total_count > 0 else 0.0)
        return unexplored_ratio

    def _check_follower_collision(self, follower, target_coordinate) -> bool:
        """
        followerの移動経路上の衝突判定
        """
        current_coord = follower.coordinate.copy()
        
        # followerの移動距離を計算（leaderへの移動）
        movement_vector = target_coordinate - current_coord
        movement_distance = np.linalg.norm(movement_vector)
        
        # 移動距離が非常に小さい場合は衝突なし
        if movement_distance < 0.1:
            return False
        
        # 移動方向を正規化
        movement_direction = movement_vector / movement_distance
        
        # 移動経路上の複数点をチェック（細かいサンプリング）
        sampling_num = max(30, int(movement_distance * 8))  # 最低30点、距離に応じて増加
        
        for i in range(1, sampling_num + 1):
            # 中間点を計算
            t = i / sampling_num
            intermediate_coord = current_coord + t * movement_vector
            
            map_y = int(intermediate_coord[0])
            map_x = int(intermediate_coord[1])
            
            # マップ内かチェック
            if (0 <= map_y < self.__map_height and 0 <= map_x < self.__map_width):
                # 障害物かチェック
                if self.__map[map_y, map_x] == self.__obstacle_value:
                    return True
            else:
                return True  # マップ外は衝突
        
        return False



    def set_system_agent(self, system_agent):
        """
        SystemAgentを環境に設定
        Args:
            system_agent: SystemAgentインスタンス
        """
        self.system_agent = system_agent
        print(f"SystemAgent set in environment")



    def get_exploration_rate(self) -> float:
        """探査率を取得"""
        explored_cells = np.sum(self.explored_map > 0)
        total_cells = self.__map_width * self.__map_height
        return explored_cells / total_cells if total_cells > 0 else 0.0
    
    def get_exploration_info(self) -> Dict[str, Any]:
        """探査に関する詳細情報を取得"""
        explored_cells = np.sum(self.explored_map > 0)
        total_cells = self.__map_width * self.__map_height
        exploration_rate = explored_cells / total_cells if total_cells > 0 else 0.0
        
        # 新しく探査されたエリア数を計算（前ステップとの差分）
        if hasattr(self, '_previous_explored_cells'):
            new_explored_cells = explored_cells - self._previous_explored_cells
        else:
            new_explored_cells = 0
        
        # 現在の探査済みセル数を保存
        self._previous_explored_cells = explored_cells
        
        return {
            'explored_area': explored_cells,
            'total_area': total_cells,
            'exploration_rate': exploration_rate,
            'new_explored_area': new_explored_cells
        }
    
    def get_collision_info(self) -> Dict[str, Any]:
        """衝突に関する情報を取得"""
        agent_collision_flag = 0
        follower_collision_count = 0
        
        # 全ロボットの衝突情報を集計
        for swarm in self.swarms:
            # leaderの衝突
            if hasattr(swarm.leader, 'collision_flag') and swarm.leader.collision_flag:
                agent_collision_flag = 1
            
            # followersの衝突
            for follower in swarm.followers:
                if hasattr(follower, 'collision_flag') and follower.collision_flag:
                    follower_collision_count += 1
        
        return {
            'agent_collision_flag': agent_collision_flag,
            'follower_collision_count': follower_collision_count
        }
    
    def get_robot_positions(self) -> List[Dict[str, Any]]:
        """全ロボットの位置情報を取得"""
        robot_positions = []
        
        for swarm in self.swarms:
            # leaderの位置
            if hasattr(swarm.leader, 'coordinate'):
                robot_positions.append({
                    'robot_id': f"leader_{swarm.swarm_id}",
                    'swarm_id': swarm.swarm_id,
                    'role': 'leader',
                    'x': float(swarm.leader.coordinate[1]),
                    'y': float(swarm.leader.coordinate[0]),
                    'collision_flag': getattr(swarm.leader, 'collision_flag', False)
                })
            
            # followersの位置
            for i, follower in enumerate(swarm.followers):
                if hasattr(follower, 'coordinate'):
                    robot_positions.append({
                        'robot_id': f"follower_{swarm.swarm_id}_{i}",
                        'swarm_id': swarm.swarm_id,
                        'role': 'follower',
                        'x': float(follower.coordinate[1]),
                        'y': float(follower.coordinate[0]),
                        'collision_flag': getattr(follower, 'collision_flag', False)
                    })
        
        return robot_positions

    # Loggableインターフェースの実装
    def get_log_data(self):
        """ログデータを取得"""
        return {
            "exploration_rate": self.get_exploration_rate(),
            "swarm_count": len(self.swarms),
            "robot_count": sum(len(swarm.get_all_robots()) for swarm in self.swarms)
        }
    
    def save_log(self, log_dir: str):
        """ログを保存"""
        # ログ保存の実装は必要に応じて追加
        pass
    
    # Configurableインターフェースの実装
    def get_config(self):
        """設定を取得"""
        return {
            "map_size": self.explored_map.shape if self.explored_map is not None else None,
            "swarm_count": len(self.swarms)
        }
    
    def set_config(self, config: dict):
        """設定を設定"""
        # 設定の実装は必要に応じて追加
        pass
    
    def load_config(self, config: dict):
        """設定を読み込み"""
        # 設定読み込みの実装は必要に応じて追加
        pass
    
    # Statefulインターフェースの実装
    def get_state(self):
        """状態を取得"""
        return {
            "explored_map": self.explored_map.copy() if self.explored_map is not None else None,
            "swarms": [swarm.swarm_id for swarm in self.swarms]
        }
    
    def set_state(self, state: dict):
        """状態を設定"""
        if state.get("explored_map") is not None:
            self.explored_map = state["explored_map"].copy()
    
    def reset_state(self):
        """状態をリセット"""
        # 環境のリセット処理は既にreset()メソッドで実装済み
        pass
  
    def render_gif_frame(self, fig_size=8):
        """
        GIF作成用のフレームをレンダリング
        main.pyと同じように環境の状態を表示
        """
        # 固定サイズでフレームを生成（GIF生成の安定性のため）
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        
        # マップのアスペクト比に合わせてfigsizeを調整
        aspect_ratio = self.__map_width / self.__map_height
        if aspect_ratio > 1:
            # 横幅が大きい場合（200x100など）
            fig_width = 12  # 横幅を大きくする
            fig_height = 12 / aspect_ratio
        else:
            # 縦幅が大きい場合
            fig_height = 12
            fig_width = 12 * aspect_ratio
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.set_size_inches(fig_width, fig_height)
        fig.set_dpi(100)
        
        # 背景色を白に設定
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 地図（障害物を黒で表示）
        ax.imshow(
            self.__map,
            cmap='gray_r',
            origin='lower',
            extent=(0, self.__map_width, 0, self.__map_height),
        )
        
        # 探査済み領域（薄いグレーで表示）
        explored_display = np.where(self.explored_map > 0, 1, 0)
        ax.imshow(
            explored_display,
            cmap='gray',
            alpha=0.2,  # より薄い透明度
            origin='lower',
            extent=(0, self.__map_width, 0, self.__map_height),
        )
        
        # 各群の色定義
        swarm_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        trajectory_colors = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkseagreen', 'darkslateblue']
        
        # 各群を描画
        for swarm_idx, swarm in enumerate(self.swarms):
            color_idx = swarm.swarm_id % len(swarm_colors)
            swarm_color = swarm_colors[color_idx]
            trajectory_color = trajectory_colors[color_idx]
            
            # Leaderの描画
            leader = swarm.leader
            # leaderの現在位置を直接取得
            current_x = leader.x
            current_y = leader.y
            
            # Leader（星マーカー）
            ax.scatter(
                current_x, current_y,
                color=swarm_color,
                s=100,
                marker='*',
                edgecolors='black',
                linewidth=2,
                label=f'Swarm {swarm.swarm_id} Leader'
            )
            
            # Leaderの軌跡（leaderごとに色を変えて表示）
            if (hasattr(leader, 'leader_trajectory_data') and 
                leader.leader_trajectory_data is not None and 
                len(leader.leader_trajectory_data) > 1 and
                'x' in leader.leader_trajectory_data.columns and 
                'y' in leader.leader_trajectory_data.columns):
                
                # leaderのIDに基づいて色を決定（文字列IDの場合はhashを使用）
                try:
                    leader_color_idx = int(leader.id) % len(trajectory_colors)
                except (ValueError, TypeError):
                    # 文字列IDの場合はhashを使用
                    leader_color_idx = hash(leader.id) % len(trajectory_colors)
                leader_trajectory_color = trajectory_colors[leader_color_idx]
                
                # 軌跡データが有効な場合のみ描画
                x_data = leader.leader_trajectory_data['x'].dropna()
                y_data = leader.leader_trajectory_data['y'].dropna()
                
                if len(x_data) > 1 and len(y_data) > 1:
                    ax.plot(
                        x_data,
                        y_data,
                        color=leader_trajectory_color,
                        linewidth=2,
                        alpha=0.7,
                        linestyle='-'
                    )
            
            # フォロワの描画
            for follower in swarm.followers:
                if len(follower.data['x']) > 0:
                    fx = follower.data['x'].iloc[-1]
                    fy = follower.data['y'].iloc[-1]
                    ax.scatter(
                        fx, fy,
                        color=swarm_color,  # リーダーと同じカラー
                        s=15,
                        marker='o',
                        edgecolors='black',
                        linewidth=0.5,
                        alpha=0.8
                    )
                    
                    # フォロワの軌跡（薄い線で表示）
                    if len(follower.data['x']) > 1:
                        ax.plot(
                            follower.data['x'],
                            follower.data['y'],
                            color='gray',  # グレーで統一
                            linewidth=0.5,
                            alpha=0.3,  # より薄い透明度
                            linestyle='-'
                        )
        
        # 軸の設定
        ax.set_xlim(0, self.__map_width)
        ax.set_ylim(0, self.__map_height)
        ax.set_aspect('equal')
        
        # メモリを削除
        ax.set_xticks([])
        ax.set_yticks([])
        
        # グリッド線
        ax.grid(True, alpha=0.3)
        
        # フレームを画像として保存
        fig.canvas.draw()
        
        # マップサイズに合わせてフレームサイズを調整
        # 200x100のマップに対して、800x400のフレームを生成
        target_width = 800
        target_height = int(target_width * (self.__map_height / self.__map_width))
        
        try:
            # より堅牢な方法: 一時ファイルを使用して画像を生成
            import tempfile
            import os
            from PIL import Image
            
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                fig.savefig(tmp_file.name, dpi=100, bbox_inches='tight', pad_inches=0)
                tmp_path = tmp_file.name
            
            # PILで読み込み
            img = Image.open(tmp_path)
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            frame = np.array(img)
            
            # 一時ファイルを削除
            os.unlink(tmp_path)
                
        except Exception as e:
            print(f"Error in frame generation: {e}")
            # エラーが発生した場合は、空のフレームを生成
            frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            # デバッグ用に赤い枠を追加
            frame[0:10, :, 0] = 255  # 上辺を赤に
            frame[-10:, :, 0] = 255  # 下辺を赤に
            frame[:, 0:10, 0] = 255  # 左辺を赤に
            frame[:, -10:, 0] = 255  # 右辺を赤に
        
        plt.close(fig)
        return frame

    def capture_frame(self):
        """現在の状態をフレームとしてキャプチャ"""
        if self._state["agent_step"] % self.frame_interval == 0:
            frame = self.render_gif_frame()
            self.env_frames.append(frame)
  
    def check_exploration_area_overlap(self, swarm1_id: int, swarm2_id: int) -> bool:
        """
        2つの群の探査領域が重複しているかチェック
        Args:
            swarm1_id: 1つ目の群ID
            swarm2_id: 2つ目の群ID
        Returns:
            探査領域が重複している場合True
        """
        swarm1 = self._find_swarm_by_id(swarm1_id)
        swarm2 = self._find_swarm_by_id(swarm2_id)
        
        if swarm1 is None or swarm2 is None:
            return False
        
        # 2つの群のleader間の距離を計算
        leader1_pos = swarm1.leader.coordinate
        leader2_pos = swarm2.leader.coordinate
        distance = np.linalg.norm(leader1_pos - leader2_pos)
        
        # 探査領域の半径（outer_boundary）の2倍以内なら重複とみなす
        # これにより、探査領域が重複している場合のみ統合が実行される
        overlap_threshold = self.exploration_radius * 2.0
        
        return distance <= overlap_threshold

    def _is_movement_possible(self, current_coord: np.ndarray, target_coord: np.ndarray) -> bool:
        """
        指定された移動が可能かどうかをチェック
        Args:
            current_coord: 現在の座標
            target_coord: 目標座標
        Returns:
            移動可能な場合True
        """
        # 移動経路上の複数点をチェック
        distance = np.linalg.norm(target_coord - current_coord)
        sampling_num = max(20, int(distance * 5))  # 軽量なサンプリング
        
        # ロボットのサイズを考慮した衝突判定
        robot_radius = 1  # ロボットの半径
        
        for i in range(0, sampling_num + 1):
            # 中間点を計算
            t = i / sampling_num
            intermediate_coord = current_coord + t * (target_coord - current_coord)
            
            # ロボットの中心座標
            center_y = int(intermediate_coord[0])
            center_x = int(intermediate_coord[1])
            
            # ロボットの範囲内の全てのピクセルをチェック
            for dy in range(-robot_radius, robot_radius + 1):
                for dx in range(-robot_radius, robot_radius + 1):
                    # 円形の範囲内かチェック
                    if dx*dx + dy*dy <= robot_radius*robot_radius:
                        check_y = center_y + dy
                        check_x = center_x + dx
                        
                        # マップ内かチェック
                        if (0 <= check_y < self.__map_height and 0 <= check_x < self.__map_width):
                            # 障害物かチェック
                            if self.__map[check_y, check_x] == self.__obstacle_value:
                                return False
                        else:
                            return False  # マップ外は移動不可能
        
        return True

    def get_next_swarm_id(self) -> int:
        """
        新しい群IDを取得
        Returns:
            新しい群ID
        """
        new_id = self.swarm_id_counter
        self.swarm_id_counter += 1
        return new_id
  