import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mColors
from matplotlib.patches import Circle
import os
import imageio
import copy

from envs.env_map import generate_explored_map, generate_rect_obstacle_map
from .action_space import create_action_space
from .observation_space import create_observation_space, create_initial_state
from .reward import create_reward
from robots.red import Red
from params.simulation import Param
from scores.score import Score

class Env(gym.Env):
  """ 
  環境
  """
  def __init__(self, param=Param, action_space=None, observation_space=None, reward_dict=None):
    super().__init__()

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
    self.scorer = Score()

    # ----- set explore infos -----
    self.total_area = np.count_nonzero(self.__map != self.__obstacle_value)  # 探査可能エリア
    self.explored_area          = 0    # 探査したセル数
    self.exploration_ratio      = 0.0  # 1ステップ前の探査率
    self.agent_step             = 0    # エージェントのステップ
    self.collision_flag         = False

    self.__agent_initial_coordinate = np.array([self.__initial_coordinate_y, self.__initial_coordinate_x])
    self.agent_coordinate = self.__agent_initial_coordinate
    
    # 1ステップ前のエージェント位置情報
    self.__agent_initial_previous_coordinate = None 
    self.agent_previous_coordinate           = self.__agent_initial_previous_coordinate 

    # ----- set drawing infos -----
    self.agent_trajectory = [self.agent_coordinate.copy()] # 奇跡の初期化
    self.env_frames = [] # 描画用フレームの初期化

    # -----set follower robots -----
    self.follower_robots = [Red(
      id                    = f"follower_{index}",
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
      boids_flag            = self.__offset.boids_flag,
      estimated_probability = self.__offset.estimated_probability
    ) for index in range(self.__robot_num)]

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
    self.follower_robots = [Red(
      id                    = f"follower_{index}",
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
      boids_flag            = self.__offset.boids_flag,
      estimated_probability = self.__offset.estimated_probability
    ) for index in range(self.__robot_num)]

    # ----- reset drawing infos -----
    self.agent_trajectory = [self.agent_coordinate.copy()] # 奇跡の初期化
    self.result_pdf_list  = [] # 統合確率分布の初期化

    # ----- reset state -----
    self.scorer = Score()
    self.state = copy.deepcopy(self.__initial_state)

    self.follower_collision_points = [] # 描画用のfollower衝突リスト

    return self.state
  

  def render(self, fig_size = 6, mode = 'human', ax = None):
    """
    環境のレンダリング
    マップの描画
    フォロワの描画
    エージェントの描画
    """
    if mode == 'rgb_gray':
      pass
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
          extent=[0, self.__map_width, 0, self.__map_height],
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
          extent=[0, self.__map_width, 0, self.__map_height],
      )

      # 探査中心
      ax.scatter(
          x=self.agent_coordinate[1],
          y=self.agent_coordinate[0],
          color='blue',
          s=100,
          label="agent"
      )

      # 軌跡
      trajectory = np.array(self.agent_trajectory, dtype=np.float32)
      ax.plot(
          trajectory[:, 1],
          trajectory[:, 0],
          color='blue',
          linewidth=2,
      )

      # 探査エリアの円
      explore_area = Circle(
          (self.agent_coordinate[1], self.agent_coordinate[0]),
          self.__boundary_outer,
          color='black',
          fill=False,
          linewidth=1,
      )
      ax.add_patch(explore_area)

      # フォロワの描画
      for i, follower in enumerate(self.follower_robots):
          ax.scatter(
              x=follower.data['x'].iloc[-1],
              y=follower.data['y'].iloc[-1],
              color='red',
              s=10,
              label="follower" if i == 0 else None
          )
          ax.plot(
              follower.data['x'],
              follower.data['y'],
              color='gray',
              linewidth=0.5,
              alpha=0.5,
          )

      # 軸の調整
      ax.set_xlim(0, self.__map_width)
      ax.set_ylim(0, self.__map_height)
      ax.set_title('Explore Environment')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.grid(False)

      # === 探索範囲円とフォロワ・衝突位置の可視化 ===

      # エージェント中心と探索範囲
      center_x = self.agent_coordinate[1]
      center_y = self.agent_coordinate[0]
      explore_radius = self.__boundary_outer

      # 探索円を描画（点線で見やすく）
      explore_circle = Circle(
          (center_x, center_y),
          explore_radius,
          color='black',
          fill=False,
          linestyle='dashed',
          linewidth=1.5,
          alpha=0.6,
          zorder=2
      )
      ax.add_patch(explore_circle)


      # === フォロワによる衝突点の描画 ===
      for cx, cy in self.follower_collision_points:
          ax.plot(cx, cy, 'rx', markersize=6, zorder=5)


    #   # === フォロワの現在位置 ===
    #   for i, follower in enumerate(self.follower_robots):
    #       fx = follower.data['x'].iloc[-1]
    #       fy = follower.data['y'].iloc[-1]
    # ax.plot(fx, fy, 'ro', markersize=4, label="Follower" if i == 0 else "", zorder=3)

  

  def step(self, action):
    """
    環境のステップ → 環境情報を出力する
    """
    # TODO modeの動作導入

    # 移動先の距離計算
    # エージェントのステップ処理の先頭または中で
    self.state['agent_azimuth'] = action['theta']  
    dx = self.__boundary_outer * math.cos(action['theta'])
    dy = self.__boundary_outer * math.sin(action['theta'])

    self.previous_agent_coordinate = self.agent_coordinate.copy()         # 1ステップ前の位置を保存
    self.agent_coordinate, self.collision_flag = self.next_coordinate(dy, dx)  # エージェントの次の位置を計算
    self.agent_trajectory.append(self.agent_coordinate.copy())          # エージェントの軌跡を更新

    # フォロワ機の位置を更新
    for index in range(len(self.follower_robots)):
      self.follower_robots[index].change_agent_state(self.agent_coordinate)
    
    # リーダー機の衝突情報を取得
    if self.collision_flag:
      self.state['agent_collision_flag'] = 1
    else:
      self.state['agent_collision_flag'] = 0
    
    # エージェントのステップカウントをインクリメント
    self.agent_step += 1
    self.state['agent_step_count'] = self.agent_step
    
    # フォロワの探査行動
    for _ in range(self.__offset.one_explore_step):
      for index in range(len(self.follower_robots)):
        previous_coordinate = self.follower_robots[index].coordinate
        self.follower_robots[index].step_motion(agent_coordinate=self.agent_coordinate)
        self.update_exploration_map(previous_coordinate, self.follower_robots[index].coordinate)

         # === 衝突フラグがTrueなら座標を追加 ===
        if self.follower_robots[index].collision_flag:
            cx = self.follower_robots[index].coordinate[1]
            cy = self.follower_robots[index].coordinate[0]
            self.follower_collision_points.append((cx, cy))

            # 検証: get_collision_data() の出力と一致するか
            agent_y, agent_x = self.agent_coordinate
            dy = cy - agent_y
            dx = cx - agent_x
            distance = np.sqrt(dx**2 + dy**2)
            azimuth = np.arctan2(dy, dx)

            debug_data = self.follower_robots[index].get_collision_data()
            if debug_data:
                a, d = debug_data[-1]  # 最新の1件
                est_x = agent_x + d * np.cos(a)
                est_y = agent_y + d * np.sin(a)
                # print(f"Follower {index} collision: actual=({cx:.2f},{cy:.2f}), est=({est_x:.2f},{est_y:.2f})")
      
      # レンダリング
      if hasattr(self, "_render_flag") and self._render_flag:
        self.render(ax=self._render_ax)
        self._render_ax.figure.canvas.draw()
        self._render_ax.figure.canvas.flush_events()
        if hasattr(self, "capture_frame"):
            self.capture_frame()
      
      # 探査率計算
      previous_ratio = self.scorer.exploration_rate[-1] if self.scorer.exploration_rate else 0.0
      self.exploration_ratio = self.scorer.calc_exploration_rate(
        explored_area=self.explored_area,
        total_area=self.total_area
      )
      print(f"exploration ratio | {self.exploration_ratio} | ( {self.explored_area} / {self.total_area})")

    # フォロワから衝突データ収集
    follower_collision_data = []
    for follower in self.follower_robots:
        follower_collision_data.extend(follower.get_collision_data())  # List[Tuple[float, float]] # distance, azimuth
        self.scorer.follower_collision_count += len(follower_collision_data)

    # MAX_COLLISION_NUM 個だけ使う（足りない分はゼロ埋め）
    padded_list = follower_collision_data[:self.MAX_COLLISION_NUM]
    padding_needed = self.MAX_COLLISION_NUM - len(padded_list)

    fcd_ndarray = [np.array(pair, dtype=np.float32) for pair in padded_list]
    fcd_ndarray.extend([np.array([0.0, 0.0], dtype=np.float32)] * padding_needed)

    self.state['follower_collision_data'] = fcd_ndarray
    
    # ----- calculate reward -----
    # デフォルト報酬
    reward = self.__reward_dict.get('default', -1)

    # 探査率が上昇した時
    if self.exploration_ratio > previous_ratio:
      reward += self.__reward_dict.get('exploration_gain', 0.0)
    else:
    # 上がらなかった場合
      reward += self.__reward_dict.get('revisit_penalty', 0.0)

    # リーダーが障害物に衝突した場合
    if self.collision_flag:
      self.scorer.agent_collision_count += 1
      reward += self.__reward_dict.get('collision_penalty', 0.0)

    # ----- finish condition -----
    done = False
    turncated = False # 途中終了フラグ
    if self.scorer.goal_reaching_step is None and self.explored_area >= self.total_area * self.__finish_rate:
      self.scorer.goal_reaching_step = self.agent_step
      done = True
      reward += self.__reward_dict.get('clear_target_rate', 0.0)
    
    # stepによる終了
    if self.agent_step >= self.__finish_step:
      done = True
      reward += self.__reward_dict.get('none_finish_penalty', 0.0)
    

    distance = np.linalg.norm(self.agent_coordinate - self.previous_agent_coordinate)
    self.scorer.total_distance_traveled += distance

    return self.state, reward, done, turncated, {}
  

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
    - log_dir: ログディレクトリ（gifs/ 以下に保存される）
    - episode: エピソード番号
    """
    gif_dir = os.path.join(log_dir, "gifs")
    gif_name = f"episode_{episode:04d}.gif"

    os.makedirs(gif_dir, exist_ok=True)
    imageio.mimsave(
        os.path.join(gif_dir, gif_name),  # 保存パス
        self.env_frames,                  # フレームリスト
        format='GIF',
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
  