import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridSpec
import matplotlib.colors as mColors
from matplotlib.patches import Circle
from PIL import Image
import io
import os
import imageio

from .env_parameter import EnvParam
from .action_space import create_action_space
from .observation_space import create_observation_space
from robots.red import Red

class Env(gym.Env):
  """
  環境
  """
  def __init__(self, param=EnvParam):
    """
    移動方向 → VFHにより決定
    走行可能性
    探査向上性
    """
    super().__init__()
    self.param              = param
    self.action_space       = create_action_space()
    self.observation_space  = create_observation_space()

    # TODO 報酬レンジを定義

    # ----- set initialize -----
    self.map          = self.generate_rect_obstacle_map()
    self.explored_map = self.generate_explored_map()

    # ----- set explore infos -----
    self.total_area             = np.prod((self.map.shape) - np.sum(self.map == self.param.OBSTACLE_VALUE)) # 探査可能エリア
    self.explored_map           = 0 # 探査済みエリア
    self.previous_explored_area = 0.0 # # 1ステップ前の探査率 TODO 学習には使用したくない
    self.agent_step             = 0 # エージェントのステップ

    self.agent_position = self.param.INITIAL_POSITION # エージェントの位置情報の追加
    self.previous_agent_position = None # 1ステップ前のエージェント位置情報

    # ----- set drawing infos -----
    self.agent_trajectory = [self.agent_position.copy()] # 奇跡の初期化
    self.env_frames = [] # 描画用フレームの初期化

    # -----set follower robots -----
    self.follower_robots = [Red(
      id=f'follower_{index}',
      env=self,
      agent_position=self.agent_position,
      x=self.agent_position[1] + self.param.FOLLOWER_POSITION_OFFSET * math.cos((2 * math.pi * index / (self.param.FOLLOWER_NUM))),
      y=self.agent_position[0] + self.param.FOLLOWER_POSITION_OFFSET * math.sin((2 * math.pi * index / (self.param.FOLLOWER_NUM)))
    ) for index in range(self.param.FOLLOWER_NUM)]

    # ----- set initial state -----
    leader_info = {
      "azimuth" : 0.0,
      "mask"    : 1 # None flag
    }

    # 初期状態を生成
    self.state = {
      "leader_info" : leader_info,
      "k_e"         : self.param.INITIAL_K,
      "k_c"         : self.param.INITIAL_K,
      "th"          : self.param.INITIAL_TH
    }
  

  def reset(self):
    """
    環境の初期化関数
    """
    # ----- reset explore infos -----
    self.agent_position = self.param.INITIAL_POSITION # エージェントの位置情報の初期化
    self.previous_agent_position = None # 1ステップ前の位置情報の初期化
    self.explored_area = 0 # 探査済みエリアの初期化
    self.explored_map = self.generate_explored_map() # 探査済みマップの初期化
    self.previous_explored_area = 0.0 # 1ステップ前の探査率の初期化

    # ----- reset robot info -----
    self.follower_robots = [Red(
      id=f'follower_{index}',
      env=self,
      agent_position=self.agent_position,
      x=self.agent_position[1] + self.param.FOLLOWER_POSITION_OFFSET * math.cos((2 * math.pi * index / (self.param.FOLLOWER_NUM))),
      y=self.agent_position[0] + self.param.FOLLOWER_POSITION_OFFSET * math.sin((2 * math.pi * index / (self.param.FOLLOWER_NUM)))
    ) for index in range(self.param.FOLLOWER_NUM)] # フォロワーの初期化

    # ----- reset drawing infos -----
    self.agent_trajectory = [self.agent_position.copy()] # 奇跡の初期化
    self.result_pdf_list  = [] # 統合確率分布の初期化

    # ----- reset state -----
    leader_info = {
      "azimuth" : 0.0,
      "mask"    : 1 # None flag
    }

    # 初期状態を生成
    self.state = {
      "leader_info" : leader_info,
      "k_e"         : self.param.INITIAL_K,
      "k_c"         : self.param.INITIAL_K,
      "th"          : self.param.INITIAL_TH
    }

    return self.state
  

  def render(self, fig_size, save_frames = False, mode='human'):
    """
    環境のレンダリング
    マップの描画
    フォロワの描画
    エージェントの描画
    """
    if mode == 'rgb_gray':
      pass
    elif mode == 'human':
      # plt.ion()  # インタラクティブモードを有効化 コメントアウトでoff, 

      fig = plt.figure("Environment", figsize=(fig_size, fig_size))
      fig.clf()

      # 全体のグリッドを設定
      gs_master = gridSpec.GridSpec(
        nrows=3,
        ncols=3,
        height_ratios=[1, 1, 1],
      )

      # マップのグリッドを設定
      gs_env = gridSpec.GridSpecFromSubplotSpec(
        nrows=1,
        ncols=3,
        subplot_spec=gs_master[0, :],
      )

      # gs_envを全体を1つのプロットとして設定
      ax_env = fig.add_subplot(gs_env[:, :])
      ax_env.set_title('Environment Map')
      ax_env.set_xlabel('X-axis')
      ax_env.set_ylabel('Y-axis')

      # ----- Environment -----
      cmap = mColors.ListedColormap(['white', 'gray', 'black'])
      bounds = [0, 1, self.param.OBSTACLE_VALUE, self.param.OBSTACLE_VALUE + 1]
      norm = mColors.BoundaryNorm(bounds, cmap.N)

      # マップの描画
      ax_env.imshow(
        self.map,
        cmap='gray_r',
        origin='lower',
        extent=[0, self.param.ENV_WIDTH, 0, self.param.ENV_HEIGHT]
      )

      # 探査済みマップの描画
      ax_env.imshow(
        self.explored_map,
        cmap=cmap,
        alpha=0.5,
        norm=norm,
        origin='lower',
        extent=[0, self.param.ENV_WIDTH, 0, self.param.ENV_HEIGHT],
      )

      # 探査中心位置の描画
      ax_env.scatter(
        x=self.agent_position[1],
        y=self.agent_position[0],
        color='blue',
        s=100,
        label='Explore Center',
      )

      # 軌跡の描画
      trajectory = np.array(self.agent_trajectory, dtype=np.float32)
      ax_env.plot(
        trajectory[:, 1],
        trajectory[:, 0],
        color='blue',
        linewidth=2,
        label='Explore Center Trajectory',
      )

      # 探査領域の描画
      explore_area = Circle(
        (self.agent_position[1], self.agent_position[0]),
        self.param.OUTER_BOUNDARY,
        color='black',
        fill=False,
        linewidth=1,
        label='Explore Area',
      )
      ax_env.add_patch(explore_area)

      # followerの描画
      for follower in self.follower_robots:
        ax_env.scatter(
          x=follower.data['x'].iloc[-1],
          y=follower.data['y'].iloc[-1],
          color='red',
          s=10,
          # label=follower.id,
        )
        ax_env.plot(
          follower.data['x'],
          follower.data['y'],
          color='gray',
          linewidth=0.5,
          alpha=0.5,
          # label=f"{follower.id} Trajectory",
        )

      # ----- env draw settings ------
      ax_env.set_xlim(0, self.param.ENV_WIDTH)
      ax_env.set_ylim(0, self.param.ENV_HEIGHT)
      ax_env.set_title('Explore Environment')
      ax_env.set_xlabel('X')
      ax_env.set_ylabel('Y')
      ax_env.grid(False)
      ax_env.legend()

      # ----- set histogram drawing -----
      gs_histogram = gridSpec.GridSpecFromSubplotSpec(
        nrows=2,
        ncols=3,
        subplot_spec=gs_master[1:, :],
      )

      # histogramのグリッドを6分割
      result_axes: list[plt.Axes] = []
      titles = [
        "drivability of vfh",
        "exploration improvement",
        "result",
        "drive possible",
        "exploration improvement possible",
        "result possible"
      ]
      for i in range(2):
        for j in range(3):
          ax = fig.add_subplot(gs_histogram[i, j], projection='polar')
          result_axes.append(ax)
      
      # 走行可能性の分布を描画
      result_axes[0].set_title(titles[0])
      result_axes[0].bar(
        self.param.ANGLES,
        self.drivability_histogram,
        color='blue',
        linewidth=2,
      )
      result_axes[0].set_theta_zero_location('E')  # 0度の位置を東に設定
      result_axes[0].set_theta_direction(1)  # 反時計回りに設定
      result_axes[0].set_xlabel('Azimuth (rad)')
      result_axes[0].set_ylabel('score')
      result_axes[0].legend(['Drivability of VFH'], loc='upper right')

      # 探査向上性の分布を描画
      result_axes[1].set_title(titles[1])
      result_axes[1].bar(
        self.param.ANGLES,
        self.exploration_improvement_histogram,
        color='orange',
        linewidth=2,
      )
      result_axes[1].set_theta_zero_location('E')  # 0度の位置を東に設定
      result_axes[1].set_theta_direction(1)
      result_axes[1].set_xlabel('Azimuth (rad)')
      result_axes[1].set_ylabel('score')
      result_axes[1].legend(['Exploration Improvement'], loc='upper right')

      # 結合分布の描画
      result_axes[2].set_title(titles[2])
      result_axes[2].bar(
        self.param.ANGLES,
        self.result_histogram,
        color='green',
        linewidth=2,
      )
      result_axes[2].set_theta_zero_location('E')  # 0度の位置を東に設定
      result_axes[2].set_theta_direction(1)
      result_axes[2].set_xlabel('Azimuth (rad)')
      result_axes[2].set_ylabel('score')
      result_axes[2].legend(['Final Result'], loc='upper right')

      # 走行可能性の分布を描画(th以上の方向を強調)
      result_axes[3].set_title(titles[3])
      colors = ['tab:blue' if val >= self.state['th'] else 'lightgray' for val in self.drivability_histogram]
      result_axes[3].bar(
        self.param.ANGLES,
        self.drivability_histogram,
        width=2 * np.pi / self.param.BIN_NUM,
        color=colors,
        alpha=0.7
      )
      result_axes[3].set_theta_zero_location('E')
      result_axes[3].set_theta_direction(1)
      result_axes[3].set_xlabel('Azimuth (rad)')
      result_axes[3].set_ylabel('score')
      result_axes[3].legend(['Drivability (th >= 0.5)'], loc='upper right')

      # 探査向上性の分布を描画(平均以上の方向を強調)
      result_axes[4].set_title(titles[4])
      exploration_mean = np.mean(self.exploration_improvement_histogram)
      colors = ['tab:orange' if val >= exploration_mean else 'lightgray' for val in self.exploration_improvement_histogram]
      result_axes[4].bar(
          self.param.ANGLES, 
          self.exploration_improvement_histogram,
          width=2 * np.pi / self.param.BIN_NUM,
          color=colors,
          alpha=0.7
      )
      result_axes[4].set_theta_zero_location('E')
      result_axes[4].set_theta_direction(1)
      result_axes[4].set_xlabel('Azimuth (rad)')
      result_axes[4].set_ylabel('score')
      result_axes[4].legend(['Exploration Improvement (mean >= 0.5)'], loc='upper right')

      # 結合分布の描画(th以上の方向を強調)
      result_axes[5].set_title(titles[5])

      # 閾値で分類（例：4分位）
      q1, q2, q3 = np.quantile(self.result_histogram, [0.25, 0.5, 0.75])
      colors = []
      for val in self.result_histogram:
          if val < q1:
              colors.append('lightgray')  # 悪い
          elif val < q2:
              colors.append('tab:red')  # あまり良くない
          elif val < q3:
              colors.append('tab:green')  # 良い
          else:
              colors.append('tab:purple')  # 非常に良い

      result_axes[5].bar(
          self.param.ANGLES,
          self.result_histogram,
          width=2 * np.pi / self.param.BIN_NUM,
          color=colors,
          alpha=0.7
      )
      result_axes[5].set_theta_zero_location('E')
      result_axes[5].set_theta_direction(1)
      result_axes[5].set_xlabel('Azimuth (rad)')
      result_axes[5].set_ylabel('score')
      result_axes[5].legend(['Final Result (Q1, Q2, Q3)'], loc='upper right')

      # ----- adjust layout -----
      plt.tight_layout()
      # plt.show()           # コメントアウトでoff
      # plt.pause(0.01)      # コメントアウトでoff

      # ----- save frames -----
      if save_frames:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        self.env_frames.append(np.array(img))
        buf.close()
      
      plt.close(fig)
  

  def step(self, action):
    """
    acton:
      parameter k_e of the policy (continuity)  | (0 <= k_e < inf): float
      parameter k_c of the policy (continuity)  | (0 <= k_c < inf): float
      parameter th of the policy (continuity)   | (0 <= th < inf): float
    
    ファジィルール
    積和平均（product inference + defuzzify）」**として定量化
    ---------------------------------------
    走行可能性    | 探査向上性 | 結果             
    ---------------------------------------
    | 低        | 低        | 悪い          |
    | 低        | 高        | あまり良くない  |
    | 高        | 低        | 良い          |
    | 高        | 高        | 非常に良い     |
    ---------------------------------------
    """

    K_E   = action["k_e"]
    K_C   = action["k_c"]
    TH    = action["th"]

    print(f"action space [k_e : {K_E}] | [k_c : {K_C}] | [th: {TH}]")

    self.result_list = []

    # 走行可能性 (VFH)
    self.drivability_histogram = self.get_obstacle_density_histogram(th=TH)
    # self.result_list.append({
    #   "id": "drivability of vfh",
    #   "result": self.drivability_histogram,
    #   "lineStyle": "--",
    #   "lineWidth": 1,
    # })

    # 探査向上性
    self.exploration_improvement_histogram = self.get_exploration_improvement_histogram(k_e=K_E, k_c=K_C)
    # self.result_list.append({
    #   "id": "exploration improvement histogram",
    #   "result": self.exploration_improvement_histogram,
    #   "lineStyle": "--",
    #   "lineWidth": 1,
    # })

    # ファジィ推論に基づく最終結果のヒストグラムを取得
    self.result_histogram = self.get_final_result_histogram(th=TH)
    # self.result_list.append({
    #   "id": "final result histogram",
    #   "result": self.result_histogram,
    #   "lineStyle": "-",
    #   "lineWidth": 2,
    # })

    # 最終的な方向を選択
    theta = self.select_final_direction_by_weighted_sampling(self.result_histogram)

    # 移動先の距離計算
    dx = self.param.OUTER_BOUNDARY * math.cos(theta)
    dy = self.param.OUTER_BOUNDARY * math.sin(theta)

    self.previous_agent_position = self.agent_position.copy()         # 1ステップ前の位置を保存
    self.agent_position, collision_flag = self.next_position(dy, dx)  # エージェントの次の位置を計算
    self.agent_trajectory.append(self.agent_position.copy())          # エージェントの軌跡を更新

    # フォロワ機の位置を更新
    for index in range(len(self.follower_robots)):
      self.follower_robots[index].change_agent_state(self.agent_position)
    
    # リーダー機の衝突情報を取得
    if collision_flag:
      leader_info = {
        "azimuth": theta,
        "mask": 0  # 衝突があった場合はマスクを0に設定
       }
    else:
      leader_info = {
          "azimuth": theta,
          "mask": 1  # 衝突がなかった場合はマスクを1に設定
      }
    
    # フォロワの探査行動
    for _ in range(self.param.FOLLOWER_STEP):
      for index in range(len(self.follower_robots)):
        previous_position = self.follower_robots[index].position
        self.follower_robots[index].step_motion()
        self.update_exploration_map(previous_position, self.follower_robots[index].position)
      
      # レンダリング
      self.render(self.param.FIG_SIZE, save_frames=self.param.SAVE_FRAMES, mode='human')
    
    # ----- calculate reward -----
    # デフォルト報酬
    reward = self.param.REWARD_DEFAULT

    # リーダーが障害物に衝突した場合
    if collision_flag:
      reward += self.param.REWARD_AGENT_COLLISION
    
    # TODO 2025/05/27 報酬の計算を追加


    # ----- finish condition -----
    done = False
    turncated = False # 途中終了フラグ
    if self.explored_area >= self.total_area * self.param.FINISH_EXPLORED_RATE:
      done = True
      reward += self.param.REWARD_FINISH
    
    # 状態の構成
    self.state = {
      "leader_info" : leader_info,
      "k_e"         : K_E,
      "k_c"         : K_C,
      "th"          : TH
    }

    return self.state, reward, done, turncated, {
      "collision_flag": collision_flag,
      "explored_area": self.explored_area,
      "total_area": self.total_area,
      "agent_position": self.agent_position,
    }
  

  def next_position(self, dy: float, dx: float) -> tuple[np.array, bool]:
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
       intermediate_position = np.array([
          self.agent_position[0] + dy * (i / SAMPLING_NUM),
          self.agent_position[1] + dx * (i / SAMPLING_NUM)
       ])

       # マップ内か判断
       if (0 < intermediate_position[0] < self.param.ENV_HEIGHT and
           0 < intermediate_position[1] < self.param.ENV_WIDTH):
         # サンプリング点が障害物に衝突しているか確認
          map_y = int(intermediate_position[0])
          map_x = int(intermediate_position[1])

          # 衝突判定
          if self.map[map_y, map_x] == self.param.OBSTACLE_VALUE:
            print(f"Collision detected at {intermediate_position} with map value {self.map[map_y, map_x]}")
            collision_flag = True
            
            # 障害物に衝突する事前位置の計算
            collision_position = intermediate_position
            direction_vector = collision_position - self.agent_position
            norm_direction_vector = np.linalg.norm(direction_vector)
            stop_position = self.agent_position + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)
            return stop_position, collision_flag
          else:
            # 衝突がない場合、最終位置を更新
            continue

    return self.agent_position + np.array([dy, dx]), collision_flag
         

  def save_gif(self, episode, date_time) -> None:
    """
    保存したフレームをGIFとして保存
    """
    path_name = f"{date_time}_episode_{episode}"
    gif_dir = f"results/gif/{path_name}"
    gif_name = f"{path_name}.gif"

    os.makedirs(gif_dir, exist_ok=True)
    imageio.mimsave(
      os.path.join(gif_dir, gif_name),  # GIFの保存先
      self.env_frames,                  # 保存するフレーム
      format='GIF',                     # GIF形式で保存
      duration=0.1,                     # 各フレームの表示時間
    )

    self.env_frames = []  # フレームのリセット


  def generate_rect_obstacle_map(
      self,
      size=(EnvParam.ENV_HEIGHT, EnvParam.ENV_WIDTH), 
      obstacle_prob=EnvParam.OBSTACLE_PROB,
      obstacle_max_size=EnvParam.OBSTACLE_MAX_SIZE,  
      obstacle_val=EnvParam.OBSTACLE_VALUE,
      seed=EnvParam.MAP_SEED
  ) -> np.array:
    """
    矩形障害物を配置した地図を作成
    """
    np.random.seed(seed) # シードの設定
    H, W = size # マップの領域を取得

    # 初期値
    obstacle_map = np.zeros((H, W), dtype=np.uint16)

    # 壁の生成
    obstacle_map[0, :]  = obstacle_val
    obstacle_map[-1, :] = obstacle_val
    obstacle_map[:, 0]  = obstacle_val
    obstacle_map[:, -1] = obstacle_val

    # 内部にランダムに障害物を配置
    for y in range(1, H):
      for x in range( 1, W - 1):
        if np.random.rand() < obstacle_prob:
          rect_h = np.random.randint(2, obstacle_max_size)
          rect_w = np.random.randint(2, obstacle_max_size)
          y2 = min(y + rect_h, H - 1)
          x2 = min(x + rect_w, W - 1)
          obstacle_map[y:y2, x:x2] = obstacle_val

    return obstacle_map


  def generate_explored_map(
      self,
      size=(EnvParam.ENV_HEIGHT, EnvParam.ENV_WIDTH)
    ) -> np.array:
    """
    探査済み地図の作成
    """
    H, W = size
    return np.zeros((H, W), dtype=np.uint16)


  def update_exploration_map(self, previous_position, current_position) -> None:
    """
    探査済みマップの更新
    - 前回の位置と現在の位置を基に、探査済みマップを更新
    """
    # 前回の位置と現在の位置を整数座標に変換
    line_points = self.interpolate_line(previous_position, current_position)

    for y, x in line_points:
      # マップの範囲内であることを確認
      if 0 <= y < self.param.ENV_HEIGHT and 0 <= x < self.param.ENV_WIDTH:
        if self.explored_map[y, x] == 0 and self.map[y, x] != self.param.OBSTACLE_VALUE:
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
  

  def get_obstacle_density_histogram(self, th: np.float32):
    """
    vfhにより走行可能な方向を算出
    - 各フォロワの衝突情報を全取得
    - ヒストグラム形式に分割した領域ごとに衝突情報を振り分け
    - thを上回る領域を選択
    """
    collision_data = []

    # 衝突情報を取得
    for follower in self.follower_robots:
      collision_data += follower.get_collision_data()
    
    # ヒストグラムを作成
    histogram = np.zeros(self.param.BIN_NUM, dtype=np.float32)

    for distance, azimuth in collision_data:
      azimuth_deg = np.rad2deg(azimuth) % 360.0
      bin_index = int(azimuth_deg // self.param.BIN_SIZE_DEG)
      histogram[bin_index] += 1.0 / (distance + 1e-3) # 
    
    return histogram
  
  def get_exploration_improvement_histogram(self, k_e: float, k_c: float) -> np.ndarray:
    """
    探査向上性の高い方向を算出
    - 一つ前の探査方向を取得し次の方向から除外
    - リーダー機が衝突したのかを確認
    - 衝突がある場合、衝突方向を除外
    """
    def apply_direction_weight(base_azimuth: float, k: float, sharpness: float = 10.0):
      """
      指定された方位に基づいて、ヒストグラムの各方向に重みを適用
      - k: 最小スコア（中心方向の抑制強さ）
	    - sharpness: 分布の鋭さ（値が大きいほど、中心に強くペナルティがかかる）
	    - angle_diff: base_azimuth からの角度差
	    - decay: exp(-sharpness * (angle_diff)^2) を用いた減衰関数（ガウス的）
      """
      for i in range(self.param.BIN_NUM):
        theta = 2 * math.pi * i / self.param.BIN_NUM
        angle_diff = min(abs(theta - base_azimuth), 2 * math.pi - abs(theta - base_azimuth))
        decay = 1 - (1 - k) * math.exp(-sharpness * (angle_diff ** 2))
        histogram[i] *= decay

    # ヒストグラムを初期化
    histogram = np.ones(self.param.BIN_NUM, dtype=np.float32)

    # 前回探査方向の影響
    previous_azimuth = self.calculate_previous_azimuth()
    if previous_azimuth is not None:
      apply_direction_weight(previous_azimuth, k_e, sharpness=10.0)
    
    # 衝突方向の影響
    if self.state['leader_info']['mask'] == 0: # 衝突があった場合
        collision_azimuth = self.state['leader_info']['azimuth']
        apply_direction_weight(collision_azimuth, k_c, sharpness=20.0)
    
    # 未探査領域に向けてスコアを上げる
    for bin in range(self.param.BIN_NUM):
      angle = 2 * math.pi * bin / self.param.BIN_NUM
      dx = int(round(math.cos(angle)))
      dy = int(round(math.sin(angle)))
      next_y = self.agent_position[0] + dy
      next_x = self.agent_position[1] + dx

      iy = int(next_y)
      ix = int(next_x)

      if 0 <= iy < self.explored_map.shape[0] and 0 <= ix < self.explored_map.shape[1]:
        if self.explored_map[iy, ix] == 0:
          histogram[bin] *= 1.5  # 未探査方向強化

    histogram += 1e-6  # ゼロ割り防止
    histogram /= np.sum(histogram)
    return histogram


  def calculate_previous_azimuth(self) -> float | None:
    """
    前回探査した方向(戻る方向)を計算
    """
    if self.previous_agent_position is None:
      return None
    
    dx = self.previous_agent_position[1] - self.agent_position[1]
    dy = self.previous_agent_position[0] - self.agent_position[0]

    azimuth = math.atan2(dy, dx)
    if azimuth < 0:
      azimuth += 2 * math.pi

    return azimuth
  
  
  def get_final_result_histogram(self, th: float) -> np.ndarray:
    """
    ファジィ推論に基づき、方向ごとのスコアを統合したヒストグラムを返す。
    - drivability_histogram: 走行可能性ヒストグラム
    - exploration_improvement_histogram: 探査向上性ヒストグラム
    - th: 走行可能性の閾値
    Returns:
        fused_score: 各binの最終スコアヒストグラム (np.ndarray)
    """
    # 走行可能性と探査向上性のヒストグラムが同じ長さであることを確認
    assert len(self.drivability_histogram) == len(self.exploration_improvement_histogram)

    # ファジィ推論の結果を格納する配列
    # 走行可能性が閾値以下の方向は無視するため、初期化
    fused_score = np.zeros(self.param.BIN_NUM, dtype=np.float32)

    # 各方向の走行可能性と探査向上性を組み合わせる
    for bin in range(self.param.BIN_NUM):
        drive_val   = self.drivability_histogram[bin]
        explore_val = self.exploration_improvement_histogram[bin]

        # 閾値以下の方向は無視（または小さな値に抑制）
        if drive_val < th:
            continue

        # 積集合的なファジィ推論：drive × explore
        fused_score[bin] = drive_val * explore_val

    # 正規化（確率解釈や描画用に使う場合）
    fused_score += 1e-6  # ゼロ除け
    fused_score /= np.sum(fused_score)

    return fused_score


  def select_final_direction_by_weighted_sampling(self, result: np.ndarray) -> float:
    """
    ファジィ推論ヒストグラムのスコアに応じて方向選択を行う。
    ・Q3以上: 非常に良い
    ・Q2〜Q3: 良い
    ・Q1〜Q2: あまり良くない
    （Q1未満は除外）
    """
    q1, q2, q3 = np.quantile(result, [0.25, 0.5, 0.75])

    bins_very_good = [i for i, v in enumerate(result) if v >= q3]
    bins_good      = [i for i, v in enumerate(result) if q2 <= v < q3]
    bins_okay      = [i for i, v in enumerate(result) if q1 <= v < q2]

    # 重み付け確率（合計1.0にする）
    weights = {
        "very_good": 0.6,
        "good": 0.3,
        "okay": 0.1
    }

    # 候補とそのカテゴリ別の重みを格納
    candidate_bins = bins_very_good * int(weights["very_good"] * 100) + \
                     bins_good      * int(weights["good"] * 100) + \
                     bins_okay      * int(weights["okay"] * 100)

    if not candidate_bins:
        # fallback（全方向から選択 or 以前の方向）
        return self.calculate_previous_azimuth() or 0.0

    # 確率的に選択
    selected_bin = np.random.choice(candidate_bins)
    selected_angle = 2 * np.pi * selected_bin / self.param.BIN_NUM
    return selected_angle
  