"""
Robot class for swarm exploration simulation.
Implements individual robot behavior with role-based actions.
"""

import numpy as np
import math
import pandas as pd
import os
import random
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List

from core.interfaces import Configurable, Stateful, Loggable
from core.logging import get_component_logger

class BoidsType(Enum):
    """
    boids判断用のタイプ
    """
    NONE  = 0
    OUTER = 1
    INNER = 2


class RobotRole(Enum):
    """
    ロボットの役割
    """
    FOLLOWER = 0
    LEADER = 1


class Red():
  """
  REDの確率密度制御を模倣したクラス
  """
  def __init__(
    self,
    id: str,
    movement_min: float,
    movement_max:float,
    boids_min: float,
    boids_max: float,
    avoidance_min: float,
    avoidance_max: float,
    inner_boundary: float,
    outer_boundary: float,
    map: np.ndarray,
    obstacle_value: int,
    mean: float,
    variance: float,
    agent_coordinate: np.array,
    x: float,
    y: float,
    step: int = 0,
    amount_of_movement: float = 0.0,
    direction_angle: float = 0.0,
    collision_flag: bool = False,
    boids_flag: BoidsType = BoidsType.NONE,
    estimated_probability: float = 0.0,
    role: RobotRole = RobotRole.FOLLOWER,
  ):
    """
    initialize RED
    """
    self.id = id
    # ----- robot control request parameter -----
    self.__movement_min   = movement_min
    self.__movement_max   = movement_max
    self.__boids_min      = boids_min
    self.__boids_max      = boids_max
    self.__avoidance_min  = avoidance_min
    self.__avoidance_max  = avoidance_max
    self.__boundary_inner = inner_boundary
    self.__boundary_outer = outer_boundary
    self.__mv_mean        = mean
    self.__mv_variance    = variance

    # ----- map parameter -----
    self.__map            = map
    self.__map_height     = map.shape[0]
    self.__map_width      = map.shape[1]
    self.__obstacle_value = obstacle_value

    # ----- robot initialize -----
    self.agent_coordinate             = agent_coordinate
    self.x: float                     = x
    self.y: float                     = y
    self.coordinate: np.array         = np.array([self.y, self.x])
    self.step: int                    = step
    self.amount_of_movement: float    = amount_of_movement
    self.direction_angle: float       = direction_angle
    self.distance: float              = np.linalg.norm(self.coordinate - self.agent_coordinate)
    self.azimuth: float               = self.azimuth_adjustment()
    self.collision_flag: bool         = collision_flag
    self.boids_flag: BoidsType        = boids_flag
    self.estimated_probability: float = estimated_probability
    self.role: RobotRole              = role
    
    # ----- leader specific attributes -----
    self.leader_azimuth: float        = 0.0  # leaderとしての方位角
    self.leader_step_count: int       = 0    # leaderとしてのステップ数
    
    # ----- swarm management -----
    self.swarm_id: int                = 0    # 所属群ID
    
    # ----- mobility metrics -----
    self.mobility_score: float        = 0.0  # 動きやすさスコア
    self.collision_density: float     = 0.0  # 衝突密度
    self.space_availability: float    = 0.0  # 空間利用可能性

    # ----- data set -----
    self.data                         = self.get_arguments()
    self.one_explore_data             = self.get_arguments()

    # ----- buffer -----
    self._data_buffer = []

  
  def azimuth_adjustment(self) -> float:
    """
    自身から見た agent の方向（正しく agent に向かうベクトルの方位角）[0, 2π)
    """
    dy = self.agent_coordinate[0] - self.y
    dx = self.agent_coordinate[1] - self.x
    azimuth_rad = np.arctan2(dy, dx)
    return azimuth_rad if azimuth_rad >= 0 else (2 * np.pi + azimuth_rad)

  def get_arguments(self):
    """
    データをデータフレーム形式で取得
    """
    return pd.DataFrame({
      'id':                     [self.id],
      'step':                   [self.step],
      'x':                      [self.x],
      'y':                      [self.y],
      'coordinate':             [self.coordinate],
      'amount_of_movement':     [self.amount_of_movement],
      'direction_angle':        [self.direction_angle],
      'distance':               [self.distance],
      'azimuth':                [self.azimuth],
      'collision_flag':         [self.collision_flag],
      'boids_flag':             [self.boids_flag],
      'estimated_probability':  [self.estimated_probability],
    })
  

  def get_arguments_dict(self):
    return {
        'step': self.step,
        'x': self.x,
        'y': self.y,
        'coordinate': self.coordinate.copy(),
        'amount_of_movement': self.amount_of_movement,
        'direction_angle': self.direction_angle,
        'distance': self.distance,
        'azimuth': self.azimuth,
        'collision_flag': self.collision_flag,
        'boids_flag': self.boids_flag.value,
        'estimated_probability': self.estimated_probability,
    }


  def calculate_mobility_metrics(self) -> dict:
    """
    followerの動きやすさ指標を計算
    """
    # 衝突密度の計算（近傍の障害物密度）
    collision_density = 0.0
    space_availability = 0.0
    
    # 8方向の空間利用可能性をチェック
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    
    available_directions = 0
    total_directions = len(directions)
    
    for dy, dx in directions:
        check_y = int(self.y + dy)
        check_x = int(self.x + dx)
        
        # マップ内かチェック
        if (0 <= check_y < self.__map_height and 
            0 <= check_x < self.__map_width):
            
            # 障害物でない場合
            if self.__map[check_y, check_x] != self.__obstacle_value:
                available_directions += 1
            else:
                collision_density += 1.0 / total_directions
    
    # 空間利用可能性（0-1の値）
    space_availability = available_directions / total_directions
    
    # 動きやすさスコアの計算
    # 空間利用可能性が高く、衝突密度が低いほど高いスコア
    mobility_score = space_availability * (1.0 - collision_density)
    
    # 距離による重み付け（leaderから遠いほど動きにくい）
    distance_weight = max(0.1, 1.0 - (self.distance / self.__boundary_outer))
    mobility_score *= distance_weight
    
    # 値を更新
    self.mobility_score = mobility_score
    self.collision_density = collision_density
    self.space_availability = space_availability
    
    return {
        'mobility_score': mobility_score,
        'collision_density': collision_density,
        'space_availability': space_availability,
        'distance_weight': distance_weight
    }
  

  def get_csv(self, episode, data_time):
    """
    TODO 上位層からlog_dirを取得して同じ階層に入れたい
    データをcsv形式で取得
    """
    directory = f"csv/{data_time}_episode{episode}"
    os.makedirs(directory, exist_ok=True)
    self.data.to_csv(f"{directory}/{self.id}.csv")
  

  def step_motion(self, agent_coordinate=None) -> None:
    """
    行動制御
    """
    # agentの情報を更新
    if agent_coordinate is not None:
      self.agent_coordinate = agent_coordinate 

    # distanceとazimuthを更新してからboids判定
    self.distance = np.linalg.norm(self.coordinate - self.agent_coordinate)
    self.azimuth = self.azimuth_adjustment()

    if self.collision_flag:
      prediction_coordinate = self.avoidance_behavior()
    else:
      self.boids_judgement()
      if self.boids_flag != BoidsType.NONE:
        prediction_coordinate = self.boids_behavior()
      else:
        prediction_coordinate = self.rejection_decision()
    
    self.coordinate = self.forward_behavior(
      prediction_coordinate[0] - self.coordinate[0],
      prediction_coordinate[1] - self.coordinate[1]
    )

    self.y = self.coordinate[0]
    self.x = self.coordinate[1]
    self.distance = np.linalg.norm(self.coordinate - self.agent_coordinate)
    self.azimuth = self.azimuth_adjustment()
    self.step += 1

    # mobility_scoreを更新
    if self.role == RobotRole.FOLLOWER:
        self.calculate_mobility_metrics()

    self.data = pd.concat([self.data, self.get_arguments()])
    self.one_explore_data = pd.concat([self.one_explore_data, self.get_arguments()])

    self._data_buffer.append([
        self.step,
        self.x, self.y,
        self.coordinate.copy(),
        self.amount_of_movement,
        self.direction_angle,
        self.distance,
        self.azimuth,
        self.collision_flag,
        self.boids_flag.value,
        self.estimated_probability
    ])

    if np.isnan(self.coordinate).any():
      print(f"[ERROR] Red({self.id}) has NaN coordinate after move: {self.coordinate}")


  def set_role(self, role: RobotRole):
    """
    ロボットの役割を設定
    """
    self.role = role
    if role == RobotRole.LEADER:
        self.leader_step_count = 0
        self.leader_azimuth = self.direction_angle
        
        # リーダーになった時点で軌跡データをリセット
        # 新しいリーダーとしての軌跡を開始
        self.leader_trajectory_start_step = self.step
        self.leader_trajectory_data = pd.DataFrame(columns=[
            'step', 'x', 'y', 'coordinate', 'amount_of_movement',
            'direction_angle', 'distance', 'azimuth',
            'collision_flag', 'boids_flag', 'estimated_probability'
        ])
        # 現在位置を最初の軌跡点として追加
        self.leader_trajectory_data = pd.concat([
            self.leader_trajectory_data, 
            self.get_arguments()
        ])


  def get_leader_action(self, algorithm, state, sampled_params, episode, log_dir):
    """
    leaderとしての行動を取得
    """
    if self.role != RobotRole.LEADER:
        raise ValueError(f"Robot {self.id} is not a leader")
    
    # 各リーダーが独立したアルゴリズムインスタンスを使用
    # リーダー固有のアルゴリズムインスタンスを取得または作成
    if not hasattr(self, 'leader_algorithm'):
        # 新しいアルゴリズムインスタンスを作成
        from algorithms.algorithm_factory import select_algorithm
        self.leader_algorithm = select_algorithm('vfh_fuzzy', env=None)
    
    # リーダー固有の状態を使用
    leader_state = state
    
    # アルゴリズムで行動決定
    action_tensor, action_dict = self.leader_algorithm.policy(leader_state, sampled_params, episode, log_dir)
    
    return action_tensor, action_dict


  def execute_leader_action(self, action_dict):
    """
    leaderとしての行動を実行
    """
    if self.role != RobotRole.LEADER:
        raise ValueError(f"Robot {self.id} is not a leader")
    
    # 行動を実行
    theta = action_dict.get('theta', 0.0)
    self.leader_azimuth = theta
    
    # 移動量を計算
    dx = self.__boundary_outer * math.cos(theta)
    dy = self.__boundary_outer * math.sin(theta)
    
    # 次の位置を計算
    next_coordinate = self.coordinate + np.array([dy, dx])
    
    # 衝突判定
    if (0 < next_coordinate[0] < self.__map_height and 
        0 < next_coordinate[1] < self.__map_width):
        map_y = int(next_coordinate[0])
        map_x = int(next_coordinate[1])
        
        if self.__map[map_y, map_x] == self.__obstacle_value:
            self.collision_flag = True
            # 衝突時は現在位置を維持
            return False
        else:
            self.collision_flag = False
            self.coordinate = next_coordinate
            self.x = self.coordinate[1]
            self.y = self.coordinate[0]
            self.direction_angle = theta
            self.leader_step_count += 1
            
            # データの更新
            self.step += 1
            self.distance = np.linalg.norm(self.coordinate - self.agent_coordinate)
            self.azimuth = self.azimuth_adjustment()
            
            # データフレームに新しい位置を追加
            self.data = pd.concat([self.data, self.get_arguments()])
            self.one_explore_data = pd.concat([self.one_explore_data, self.get_arguments()])
            
            # リーダー軌跡データにも追加
            if hasattr(self, 'leader_trajectory_data'):
                self.leader_trajectory_data = pd.concat([
                    self.leader_trajectory_data, 
                    self.get_arguments()
                ])
            
            # バッファにも追加
            self._data_buffer.append([
                self.step,
                self.x, self.y,
                self.coordinate.copy(),
                self.amount_of_movement,
                self.direction_angle,
                self.distance,
                self.azimuth,
                self.collision_flag,
                self.boids_flag.value,
                self.estimated_probability
            ])
            
            return True
    else:
        self.collision_flag = True
        return False


  def finalize_data(self):
    columns = [
        'step', 'x', 'y', 'coordinate', 'amount_of_movement',
        'direction_angle', 'distance', 'azimuth',
        'collision_flag', 'boids_flag', 'estimated_probability'
    ]
    self.data = pd.DataFrame(self._data_buffer, columns=columns)
  

  def change_agent_state(self, agent_coordinate) -> None:
    """
    エージェントの状態が変化した場合の処理
    """  
    self.agent_coordinate = agent_coordinate
    self.one_explore_data = self.get_arguments()


  def avoidance_behavior(self) -> np.array:
    """
    障害物回避行動
    """
    self.direction_angle = self.direction_angle + random.uniform(
      math.radians(self.__avoidance_min), 
      math.radians(self.__avoidance_max)
    )
    amount_of_movement = random.uniform(self.__movement_min, self.__movement_max)
    dx = amount_of_movement * math.cos(self.direction_angle)
    dy = amount_of_movement * math.sin(self.direction_angle)
    return np.array([self.y + dy, self.x + dx])
  

  def boids_judgement(self):
    """
    boids行動の判断
    """
    self.distance = np.linalg.norm(self.coordinate - self.agent_coordinate)
    if self.distance > self.__boundary_outer:
      self.boids_flag = BoidsType.OUTER
    elif self.distance < self.__boundary_inner:
      self.boids_flag = BoidsType.INNER
    else:
      self.boids_flag = BoidsType.NONE
  

  def boids_behavior(self) -> np.array:
    """
    boids行動
    - OUTER: agent方向に近づく（収束）
    - INNER: agent方向から離れる（発散）
    """
    if self.boids_flag == BoidsType.OUTER:
        # agent方向へ進む（azimuth）
        self.direction_angle = self.azimuth
    elif self.boids_flag == BoidsType.INNER:
        # agent方向の反対（180度反転）
        self.direction_angle = (self.azimuth + math.pi) % (2 * math.pi)
    else:
        # NONEの場合は現状維持
        self.direction_angle = self.azimuth

    amount_of_movement = random.uniform(
        self.__boids_min,
        self.__boids_max
    )

    dx = amount_of_movement * math.cos(self.direction_angle)
    dy = amount_of_movement * math.sin(self.direction_angle)
    return np.array([self.y + dy, self.x + dx])
  

  def rejection_decision(self) -> np.array:
    """
    メトロポリス法による棄却決定
    """
    def distribution(distance, mean, variance) -> float:
      """
      正規分布
      """
      return np.exp(-(distance - mean) ** 2 / (2 * variance ** 2)) / np.sqrt(2 * np.pi)
    

    while True:
      direction_angle = random.uniform(0.0, 2.0 * math.pi)
      amount_of_movement = random.uniform(self.__movement_min, self.__movement_max)
      dx = amount_of_movement * math.cos(direction_angle)
      dy = amount_of_movement * math.sin(direction_angle)
      prediction_position = np.array([self.y + dy, self.x + dx])
      distance = np.linalg.norm(prediction_position - self.agent_coordinate)
      estimated_probability = distribution(distance, self.__mv_mean, self.__mv_variance)
      if self.estimated_probability == 0.0:
        self.estimated_probability = estimated_probability
        self.direction_angle = direction_angle
        return prediction_position
      else:
        if estimated_probability / self.estimated_probability > np.random.rand():
          self.estimated_probability = estimated_probability
          self.direction_angle = direction_angle
          return prediction_position
        else:
          continue
  

  # def forward_behavior(self, dy, dx) -> np.array:
  #   steps = np.linspace(1, 0, num=max(150, int(np.ceil(np.hypot(dy, dx) * 10))))[:, None]
  #   deltas = np.array([dy, dx])[None, :] * steps
  #   positions = self.coordinate + deltas

  #   ys = positions[:, 0].astype(int)
  #   xs = positions[:, 1].astype(int)

  #   valid = (0 < ys) & (ys < self.__map_height) & (0 < xs) & (xs < self.__map_width)

  #   for pos, y, x in zip(positions[valid], ys[valid], xs[valid]):
  #       if self.__map[y, x] == self.__obstacle_value:
  #           direction = pos - self.coordinate
  #           norm = np.linalg.norm(direction)
  #           stop = self.coordinate + (direction / norm) * (norm - 1.0)
  #           self.collision_flag = True
  #           return stop

  #   self.collision_flag = False
  #   return self.coordinate + np.array([dy, dx])


  def forward_behavior(self, dy, dx) -> np.array:
    """
    直進行動処理
    """
    SAMPLING_NUM = max(150, int(np.ceil(np.linalg.norm([dy, dx]) * 10)))
    SAFE_DISTANCE = 1.0 # マップの安全距離

    for i in range(1, SAMPLING_NUM + 1):
      intermediate_position = np.array([
        self.coordinate[0] + (dy * i / SAMPLING_NUM),
        self.coordinate[1] + (dx * i / SAMPLING_NUM)
      ])

      if (0 < intermediate_position[0] < self.__map_height) and (0 < intermediate_position[1] < self.__map_width):
        map_y = int(intermediate_position[0])
        map_x = int(intermediate_position[1])

        if self.__map[map_y, map_x] == self.__obstacle_value:
          # 障害物に衝突する事前位置を計算
          collision_position = intermediate_position
          direction_vector = collision_position - self.coordinate
          norm_direction_vector = np.linalg.norm(direction_vector)

          stop_position = self.coordinate + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)
          self.collision_flag = True
          return stop_position
      else:
        continue

    self.collision_flag = False
    return self.coordinate + np.array([dy, dx])


  def get_collision_data(self) -> list:
    """
    直近の探索ステップ（agentが移動するまで）の衝突情報（agent → follower 向きの azimuth, distance）をリストで返す
    """
    if hasattr(self, 'one_explore_data') and not self.one_explore_data.empty:
        mask = self.one_explore_data['collision_flag'].values.astype(bool)
        if not mask.any():
            return []
        
        azimuths_f2a = self.one_explore_data['azimuth'].values[mask]
        distances = self.one_explore_data['distance'].values[mask]

        # 反転：agent → follower に変換
        azimuths_a2f = (azimuths_f2a + np.pi) % (2 * np.pi)

        return list(zip(azimuths_a2f, distances))
    else:
        return []




