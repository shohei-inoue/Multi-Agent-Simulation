import numpy as np
import math
import pandas as pd
import os
import random
from robots.red_parameter import RedParam, BoidsType

class Red():
  """
  REDの確率密度制御を模倣したクラス
  """
  def __init__(
    self,
    id: str,
    env,
    agent_position: np.array,
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
    self.id: str                      = id
    self.env                          = env
    self.agent_position: np.array     = agent_position
    self.x: float                     = x
    self.y: float                     = y
    self.position: np.array           = np.array([self.y, self.x])
    self.param: RedParam              = RedParam
    self.step: int                    = step
    self.amount_of_movement: float    = amount_of_movement
    self.direction_angle: float       = direction_angle
    self.distance: float              = np.linalg.norm(self.position - self.agent_position)
    self.azimuth: float               = self.azimuth_adjustment()
    self.collision_flag: bool         = collision_flag
    self.boids_flag: BoidsType        = boids_flag
    self.estimated_probability: float = estimated_probability
    self.data                         = self.get_arguments()
    self.one_explore_data              = self.get_arguments()

  
  def azimuth_adjustment(self) -> float:
    """
    エージェントから見た自身の方向（方位角）[0, 2π)
    """
    dy = self.agent_position[0] - self.y
    dx = self.agent_position[1] - self.x
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
      'position':               [self.position],
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
      prediction_position = self.avoidance_behavior()
    else:
      self.boids_judgement()
      if self.boids_flag != BoidsType.NONE:
        prediction_position = self.boids_behavior()
      else:
        prediction_position = self.rejection_decision()
    
    self.position = self.forward_behavior(
      prediction_position[0] - self.position[0],
      prediction_position[1] - self.position[1]
    )

    self.y = self.position[0]
    self.x = self.position[1]
    self.distance = np.linalg.norm(self.position - self.agent_position)
    self.azimuth = self.azimuth_adjustment()
    self.step += 1

    self.data = pd.concat([self.data, self.get_arguments()])
    self.one_explore_data = pd.concat([self.one_explore_data])
  

  def change_agent_state(self, agent_position) -> None:
    """
    エージェントの状態が変化した場合の処理
    """  
    self.agent_position = agent_position
    self.one_explore_data = self.get_arguments()


  def avoidance_behavior(self) -> np.array:
    """
    障害物回避行動
    """
    self.direction_angle = (self.direction_angle + random.uniform(self.param.MIN_AVOIDANCE_BEHAVIOR, self.param.MAX_AVOIDANCE_BEHAVIOR))
    amount_of_movement = random.uniform(self.param.MIN_MOVEMENT, self.param.MAX_MOVEMENT)
    dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
    dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
    return np.array([self.y + dy, self.x + dx])
  

  def boids_judgement(self):
    """
    boids行動の判断
    """
    self.distance = np.linalg.norm(self.position - self.agent_position)
    if self.distance > self.env.param.OUTER_BOUNDARY:
      self.boids_flag = BoidsType.OUTER
    elif self.distance < self.env.param.INNER_BOUNDARY:
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
        self.param.MIN_BOIDS_MOVEMENT,
        self.param.MAX_BOIDS_MOVEMENT
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
      amount_of_movement = random.uniform(self.param.MIN_MOVEMENT, self.param.MAX_MOVEMENT)
      dx = amount_of_movement * math.cos(math.radians(direction_angle))
      dy = amount_of_movement * math.sin(math.radians(direction_angle))
      prediction_position = np.array([self.y + dy, self.x + dx])
      distance = np.linalg.norm(prediction_position - self.agent_position)
      estimated_probability = distribution(distance, self.env.param.MEAN, self.env.param.VARIANCE)
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
        self.position[0] + (dy * i / SAMPLING_NUM),
        self.position[1] + (dx * i / SAMPLING_NUM)
      ])

      if (0 < intermediate_position[0] < self.env.param.ENV_HEIGHT) and (0 < intermediate_position[1] < self.env.param.ENV_WIDTH):
        map_y = int(intermediate_position[0])
        map_x = int(intermediate_position[1])

        if self.env.map[map_y, map_x] == self.env.param.OBSTACLE_VALUE:
          # 障害物に衝突する事前位置を計算
          collision_position = intermediate_position
          direction_vector = collision_position - self.position
          norm_direction_vector = np.linalg.norm(direction_vector)

          stop_position = self.position + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)
          self.collision_flag = True
          return stop_position
      else:
        continue
    
    self.collision_flag = False

    return self.position + np.array([dy, dx])
  
  
  def sampling_collision_data(self, sample_num: int = 5, seed: int = 42):
    """
    衝突情報をサンプリングした形で取得
    return:
        {
          "count": int,
          "collision_samples": np.array([distance, azimuth] * sample_num),
          "mask": 0 or 1
        }
    """
    # 衝突のある情報のみを抽出
    collision_data = self.data[self.data['collision_flag'] == True]
    collision_count = len(collision_data)

    if collision_count == 0:
      # 衝突がなかった場合, maskでスキップ判定に使う
      return {
        "count": 0,
        "collision_samples": np.array(
          [[0.0, 0.0]] * sample_num,
          dtype=np.float32
        ),
        "mask": 1
      }
    
    # ランダムサンプリング(seed固定)
    sampled_data = collision_data.sample(
      n=min(sample_num, collision_count), 
      replace=collision_count < sample_num, 
      random_state=seed
    )

    # 必要な情報を取り出してリスト化
    collision_samples = np.array([
        [float(row["distance"]), float(row["azimuth"])]
        for _, row in sampled_data.iterrows()
    ], dtype=np.float32)

    # サンプルが不足していたら padding
    if len(collision_samples) < sample_num:
        padding = np.array([[0.0, 0.0]] * (sample_num - len(collision_samples)), dtype=np.float32)
        collision_samples = np.concatenate([collision_samples, padding], axis=0)
    
    return {
        "count": collision_count,
        "collision_samples": collision_samples,
        "mask": 0
    }


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
      (float(row["distance"]), float(row['azimuth']))
      for _, row in collision_data.iterrows()
    ]
    


