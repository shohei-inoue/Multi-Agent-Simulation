import numpy as np
import math
import tensorflow as tf
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class AlgorithmVfhFuzzy():
  KAPPA         = 1.0                                                 # 逆温度
  BIN_SIZE_DEG  = 20                                                  # ビンのサイズ(度)
  BIN_NUM       = int(360 // BIN_SIZE_DEG)                            # ビン数
  ANGLES        = np.linspace(0, 2 * np.pi, BIN_NUM, endpoint=False)  # 角度


  def __init__(self, th=1.0, k_e=1.0, k_c=1.0):
    """
    使用パラメータを初期化
    """
    self.th = th
    self.k_e = k_e
    self.k_c = k_c
    self.th_history = []  # thの履歴
    self.k_e_history = []  # k_eの履歴
    self.k_c_history = []  # k_cの履歴
    self.selected_theta = []
    self.update_params(self.th, self.k_e, self.k_c)


  def update_params(self, th, k_e, k_c):
    """
    パラメータの更新
    """
    self.th_history.append(th)
    self.k_e_history.append(k_e)
    self.k_c_history.append(k_c)

    self.th = th
    self.k_e = k_e
    self.k_c = k_c
  

  def render(self, ax_polar=None, ax_params=None):
    """
    policyの可視化（外部のsubplotが渡された場合はそれを使用）
    - パラメータ履歴の折れ線グラフ
    - 極座標ヒストグラム（drivability, exploration, result）
    """
    angles = self.ANGLES

    # サブプロットが渡されていなければ新規作成（単体でのdebug用）
    if ax_polar is None or ax_params is None:
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 3)
        ax_params = [fig.add_subplot(gs[0, 2])]
        ax_polar = [fig.add_subplot(gs[1, 2], polar=True)]
        show_plot = True
    else:
        show_plot = False

    # --- パラメータ履歴 ---
    param_lists = [self.th_history, self.k_e_history, self.k_c_history]
    param_names = ["th", "k_e", "k_c"]
    for i, ax in enumerate(ax_params):
        if i < len(param_lists):
            ax.clear()
            ax.plot(param_lists[i], marker='o')
            ax.set_title(param_names[i])
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.set_xlim(left=0)

    # --- 極座標ヒストグラム（3種類）---
    histograms = [
        self.drivability_histogram,
        self.exploration_improvement_histogram,
        self.result_histogram
    ]
    titles = ["Drivability", "Exploration", "Result"]

    for i, ax in enumerate(ax_polar):
      if i >= len(histograms): continue
      ax.clear()
      ax.set_title(titles[i])

      colors = ['tab:blue'] * self.BIN_NUM
      if i == 2 and hasattr(self, "selected_bin"):
          colors[self.selected_bin] = 'tab:red'

      # ヒストグラム表示
      ax.bar(angles, histograms[i], width=np.deg2rad(self.BIN_SIZE_DEG), alpha=0.7, color=colors)

      # 衝突点（drivabilityのみ）
      if i == 0:
          for d, a in self.follower_collision_data:
              ax.plot(a, d, 'ro')  # or ax.scatter(a, d, color='red', s=20)

          # y軸スケーリング調整
          collision_max = max([d for d, _ in self.follower_collision_data], default=1.0)
          ax.set_ylim(0, max(histograms[i].max(), collision_max) * 1.2)
      
      # 選択方向の矢印（Resultのみに描画）
      if i == 2 and hasattr(self, "theta") and self.theta is not None:
          r_max = max(histograms[i]) * 1.1
          ax.annotate("",
              xy=(self.theta, r_max),
              xytext=(0, 0),
              arrowprops=dict(facecolor='red', edgecolor='red', width=2, headwidth=8),
          )


  def policy(self, state, sampled_params, episode, log_dir: str = None):
    """
    行動決定ポリシー
    - 学習ありなら sampled_params を用いてパラメータを更新
    - 学習なしなら初期化時の self.th, self.k_e, self.k_c を使用

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
    アルゴリズムにおけるactionの決定
    """
    print("start policy calc")
    # stateから必要情報を取得
    self.get_state_info(state, sampled_params)

    # 走行可能性
    self.drivability_histogram = self.get_obstacle_density_histogram()

    # 探査向上生
    self.exploration_improvement_histogram = self.get_exploration_improvement_histogram()

    # ファジィ推論に基づく最終結果のヒストグラムを取得
    self.result_histogram = self.get_final_result_histogram(self.drivability_histogram, self.exploration_improvement_histogram)

    # 最終方向
    self.theta = self.select_final_direction_by_weighted_sampling(result=self.result_histogram)

    # TODO ブランチパラメータを作成
    self.mode = 0

    # 行動の出力を作成
    action = {
      "theta": self.theta,
      "mode": self.mode
    }
    print(f"result action | theta : {self.theta}({np.rad2deg(self.theta)}[deg]) | mode : {self.mode}")

    # CSVログ保存（オプション）
    if log_dir is not None:
      self.save_algorithm_parameter_to_csv(
        log_dir=log_dir,
        episode=episode,
        step=state.get("agent_step_count", 0),
        params=(self.th, self.k_e, self.k_c),
        drivability=self.drivability_histogram,
        exploration=self.exploration_improvement_histogram,
        result=self.result_histogram
      )

    self.selected_bin = int((np.rad2deg(self.theta) % 360) // self.BIN_SIZE_DEG)  # 可視化用に保存
    self.selected_theta.append(self.theta)
    action_tensor = tf.convert_to_tensor(sampled_params, dtype=tf.float32)

    # algorithm.policy() の中
    if getattr(self, "_render_flag", False):
        self.render(ax_params=self._ax_params, ax_polar=self._ax_polar)
        self._ax_polar[0].figure.canvas.draw()
        self._ax_polar[0].figure.canvas.flush_events()
        self._ax_params[0].figure.canvas.draw()
        self._ax_params[0].figure.canvas.flush_events()

    self.update_params(*sampled_params)  # th, k_e, k_c を更新

    return action_tensor, action
  

  def save_algorithm_parameter_to_csv(self, log_dir: str, step: int,
        episode: int,
        params: tuple,
        drivability: np.ndarray,
        exploration: np.ndarray,
        result: np.ndarray):
    """
    パラメータと各ヒストグラムをCSVに記録
    """
    csv_dir = os.path.join(log_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, f"algorithm_logs_{episode:04d}.csv")

    # ヘッダ（最初のステップであれば追加）
    if step == 0 and not os.path.exists(csv_path):
      with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["step", "th", "k_e", "k_c"]
        header += [f"drivability_{i}" for i in range(self.BIN_NUM)]
        header += [f"exploration_{i}" for i in range(self.BIN_NUM)]
        header += [f"result_{i}" for i in range(self.BIN_NUM)]
        writer.writerow(header)

    # データ書き込み
    with open(csv_path, mode="a", newline="") as f:
      writer = csv.writer(f)
      row = [step, *params, *drivability.tolist(), *exploration.tolist(), *result.tolist()]
      writer.writerow(row)
  
  
  def get_state_info(self, state, sampled_params):
    """
    state内からこのpolicyで必要な情報を取得
    """
    keys = [
       "follower_collision_data", 
       "agent_azimuth",
       "agent_collision_flag",
       "agent_coordinate_x",
       "agent_coordinate_y",
       ]

    self.agent_azimuth           = state["agent_azimuth"]
    self.agent_collision_flag    = state["agent_collision_flag"]  # agentの衝突
    self.agent_coordinate_x      = state["agent_coordinate_x"] # agentの座標
    self.agent_coordinate_y      = state["agent_coordinate_y"] # agentの座標

    value = state['follower_collision_data']
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape == (2,):
        # [azimuth, distance] 単体のndarray → [[azimuth, distance]] のリストに変換
        self.follower_collision_data = [value]
    elif isinstance(value, list):
        self.follower_collision_data = value
    else:
        raise ValueError(f"Invalid follower_collision_data: {value}")
    


  def get_obstacle_density_histogram(self):
    """
    vfhにより走行可能な方向を算出
    - 各フォロワの衝突情報を全取得
    - ヒストグラム形式に分割した領域ごとに衝突情報を振り分け
    - thを上回る領域を選択
    """
    # ヒストグラムを作成
    histogram = np.zeros(self.BIN_NUM, dtype=np.float32)

    for azimuth, distance in self.follower_collision_data:
      azimuth_deg = azimuth % 360.0
      bin_index = int(azimuth_deg) // self.BIN_SIZE_DEG
      histogram[bin_index] += 1.0 / (distance + 1e-3) # 距離が遠いほど影響は小さく
    
    return histogram


  def get_exploration_improvement_histogram(self) -> np.ndarray:
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
      for i in range(self.BIN_NUM):
        theta = 2 * math.pi * i / self.BIN_NUM
        angle_diff = min(abs(theta - base_azimuth), 2 * math.pi - abs(theta - base_azimuth))
        decay = 1 - (1 - k) * math.exp(-sharpness * (angle_diff ** 2))
        histogram[i] *= decay

    # ヒストグラムを初期化
    histogram = np.ones(self.BIN_NUM, dtype=np.float32)

    # 前回探査方向の影響
    if self.agent_azimuth is not None:
      reverse_azimuth = (self.agent_azimuth + np.pi) % (2 * np.pi)  # 逆方向を計算
      apply_direction_weight(reverse_azimuth, self.k_e, sharpness=10.0)
    
    # 衝突方向の影響
    if self.agent_collision_flag: # 衝突があった場合
        collision_azimuth = self.agent_azimuth
        apply_direction_weight(collision_azimuth, self.k_c, sharpness=20.0)
    
    # ----- 周辺環境をstateで送ることになった場合に使用 ----
    # # 未探査領域に向けてスコアを上げる 
    # for bin in range(self.BIN_NUM):
    #   angle = 2 * math.pi * bin / self.BIN_NUM
    #   dx = int(round(math.cos(angle)))
    #   dy = int(round(math.sin(angle)))
    #   next_y = self.agent_coordinate_y + dy
    #   next_x = self.agent_coordinate_x + dx

    #   iy = int(next_y)
    #   ix = int(next_x)

    #   # 未探査のマスに向かう方向はスコアを強化（未探索促進）
    #   if 0 <= iy < self.explored_map.shape[0] and 0 <= ix < self.explored_map.shape[1]:
    #     if self.explored_map[iy, ix] == 0:
    #       histogram[bin] *= 1.5  # 未探査方向強化

    histogram += 1e-6  # ゼロ割り防止
    histogram /= np.sum(histogram)
    return histogram
  

  def get_final_result_histogram(self, drivability, exploration_improvement) -> np.ndarray:
    """
    ファジィ推論に基づき、方向ごとのスコアを統合したヒストグラムを返す。
    - drivability_histogram: 走行可能性ヒストグラム
    - exploration_improvement_histogram: 探査向上性ヒストグラム
    - th: 走行可能性の閾値
    Returns:
        fused_score: 各binの最終スコアヒストグラム (np.ndarray)
    """
    # 走行可能性と探査向上性のヒストグラムが同じ長さであることを確認
    assert len(drivability) == len(exploration_improvement)

    # ファジィ推論の結果を格納する配列
    # 走行可能性が閾値以下の方向は無視するため、初期化
    fused_score = np.zeros(self.BIN_NUM, dtype=np.float32)

    # 各方向の走行可能性と探査向上性を組み合わせる
    for bin in range(self.BIN_NUM):
        drive_val   = drivability[bin]
        explore_val = exploration_improvement[bin]

        # 閾値以下の方向は無視（または小さな値に抑制）
        if drive_val < self.th:
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
    （Q1未満やスコア0は除外）
    """
    q1, q2, q3 = np.quantile(result, [0.25, 0.5, 0.75])

    bins_very_good = [i for i, v in enumerate(result) if v >= q3 and v > 0.0]
    bins_good      = [i for i, v in enumerate(result) if q2 <= v < q3 and v > 0.0]
    bins_okay      = [i for i, v in enumerate(result) if q1 <= v < q2 and v > 0.0]

    bins = bins_very_good + bins_good + bins_okay

    if not bins:
        print("return fallback")
        return self.agent_azimuth or 0.0

    # 安全に重みを構築（カテゴリが空ならその重み0に）
    weights = []
    total_weight = 0

    score_q1 = 0.7
    score_q2 = 0.25
    score_q3 = 1.0 - (score_q1 + score_q2)

    if bins_very_good:
        weights += [score_q1 / len(bins_very_good)] * len(bins_very_good)
        total_weight += score_q1
    if bins_good:
        weights += [score_q2 / len(bins_good)] * len(bins_good)
        total_weight += score_q2
    if bins_okay:
        weights += [score_q3 / len(bins_okay)] * len(bins_okay)
        total_weight += score_q3

    # 正規化（合計が1.0になるように）
    weights = [w / total_weight for w in weights]

    selected_bin = np.random.choice(bins, p=weights)
    selected_angle = 2 * np.pi * selected_bin / self.BIN_NUM

    return selected_angle