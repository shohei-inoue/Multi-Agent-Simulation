import numpy as np
import math
import pandas as pd
import os
import random
from enum import Enum

class BoidsType(Enum):
    """
    boids判断用のタイプ
    """
    NONE  = 0
    OUTER = 1
    INNER = 2


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

    # ----- data set -----
    self.data                         = self.get_arguments()
    self.one_explore_data             = self.get_arguments()

  
  def azimuth_adjustment(self) -> float:
    """
    エージェントから見た自身の方向（方位角）[0, 2π)
    """
    dy = self.agent_coordinate[0] - self.y
    dx = self.agent_coordinate[1] - self.x
    azimuth_rad = np.arctan2(dy, dx)
    azimuth_rad = azimuth_rad if azimuth_rad >= 0 else (2 * np.pi + azimuth_rad)
    return np.rad2deg(azimuth_rad)
  

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
  

  def get_csv(self, episode, data_time):
    """
    TODO 上位層からlog_dirを取得して同じ階層に入れたい
    データをcsv形式で取得
    """
    directory = f"csv/{data_time}_episode{episode}"
    os.makedirs(directory, exist_ok=True)
    self.data.to_csv(f"{directory}/{self.id}.csv")
  

  def step_motion(self) -> None:
    """
    行動制御
    """
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

    self.data = pd.concat([self.data, self.get_arguments()])
    self.one_explore_data = pd.concat([self.one_explore_data])
  

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
    self.direction_angle = (self.direction_angle + random.uniform(self.__avoidance_min, self.__avoidance_max))
    amount_of_movement = random.uniform(self.__movement_min, self.__movement_max)
    dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
    dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
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
        self.direction_angle = (self.azimuth + 180.0) % 360.0
    else:
        # NONEの場合は現状維持
        self.direction_angle = self.azimuth

    amount_of_movement = random.uniform(
        self.__boids_min,
        self.__boids_max
    )
    angle_rad = math.radians(self.direction_angle)

    dx = amount_of_movement * math.cos(angle_rad)
    dy = amount_of_movement * math.sin(angle_rad)
    return np.array([self.y + dy, self.x + dx])
  

  def rejection_decision(self) -> np.array:
    """
    メトロポリス法による棄却決定
    """
    def distribution(distance, mean, variance) -> float:
      """
      正規分布
      """
      return 1 / math.sqrt(2 * math.pi) * math.exp(-(distance - mean) ** 2 / (2 * variance ** 2))
    

    while True:
      direction_angle = np.rad2deg(random.uniform(0.0, 2.0 * math.pi))
      amount_of_movement = random.uniform(self.__movement_min, self.__movement_max)
      dx = amount_of_movement * math.cos(math.radians(direction_angle))
      dy = amount_of_movement * math.sin(math.radians(direction_angle))
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
    衝突情報を取得
    """
    # 衝突のある情報のみを抽出
    collision_data = self.data[self.data['collision_flag'] == True]

    # 衝突情報がない場合は空のリストを返す
    if len(collision_data) == 0:
      return []
    
    return [
      (float(row["azimuth"]), float(row['distance']))
      for _, row in collision_data.iterrows()
    ]
    


