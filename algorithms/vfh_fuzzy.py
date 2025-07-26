import numpy as np
import math
import tensorflow as tf
import os
import csv
import matplotlib.pyplot as plt
from scipy.special import i0


class AlgorithmVfhFuzzy():
  KAPPA         = 1.0                                                 # 逆温度
  BIN_SIZE_DEG  = 20                                                  # ビンのサイズ(度)
  BIN_NUM       = int(360 // BIN_SIZE_DEG)                            # ビン数
  ANGLES        = np.linspace(0, 2 * np.pi, BIN_NUM, endpoint=False)  # 角度


  def __init__(self, env=None):
    """
    初期化
    """
    # 環境への参照を保存
    self.env = env
    
    # 可視化用の軸
    self._render_flag = False
    self._ax_params = None
    self._ax_polar = None
    
    # パラメータ
    self.th = 0.5
    self.k_e = 10.0
    self.k_c = 5.0
    
    # 履歴
    self.th_history = []
    self.k_e_history = []
    self.k_c_history = []
    self.selected_theta = []
    
    # 状態情報
    self.agent_coordinate_x = 0.0
    self.agent_coordinate_y = 0.0
    self.agent_azimuth = None
    self.agent_collision_flag = False
    self.agent_step_count = 0
    self.follower_collision_data = []
    
    # 衝突回避用
    self.last_theta = None
    self.last_collision_theta = None  # 前回衝突した方向
    
    # ヒストグラム
    self.BIN_NUM = 72
    self.BIN_SIZE_DEG = 360 // self.BIN_NUM
    self.ANGLES = np.linspace(0, 2 * np.pi, self.BIN_NUM, endpoint=False)
    
    # 結果
    self.drivability_histogram = None
    self.exploration_improvement_histogram = None
    self.result_histogram = None
    self.theta = 0.0
    self.mode = 0


  def update_params(self, th, k_e, k_c, branch_threshold=None, merge_threshold=None):
    """
    パラメータの更新
    """
    self.th = th
    self.k_e = k_e
    self.k_c = k_c
    
    # 分岐・統合の閾値も保存（オプション）
    if branch_threshold is not None:
        self.branch_threshold = branch_threshold
    if merge_threshold is not None:
        self.merge_threshold = merge_threshold

  def render(self, ax_polar=None, ax_params=None):
    """
    policyの可視化（外部のsubplotが渡された場合はそれを使用）
    - パラメータ履歴の折れ線グラフ
    - 極座標ヒストグラム（drivability, exploration, result）
    """
    angles = self.ANGLES

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

    # --- 極座標ヒストグラム ---
    histograms = [
        self.drivability_histogram,
        self.exploration_improvement_histogram,
        self.result_histogram
    ]
    titles = ["Drivability", "Exploration", "Result"]

    for i, ax in enumerate(ax_polar):
        if i >= len(histograms):
            continue

        ax.clear()
        ax.set_title(titles[i])
        values = histograms[i]

        # --- カラー設定 ---
        if i == 2:
            # Resultヒストグラム：四分位数で色分け
            q1, q2, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            colors = []
            for v in values:
                if v >= q3:
                    colors.append('tab:red')     # very good
                elif v >= q2:
                    colors.append('tab:orange')  # good
                elif v >= q1:
                    colors.append('tab:green')   # okay
                else:
                    colors.append('tab:gray')    # bad
        else:
            colors = ['tab:blue'] * self.BIN_NUM

        # --- ヒストグラムバー描画 ---
        ax.bar(angles, values, width=np.deg2rad(self.BIN_SIZE_DEG), alpha=0.7, color=colors)

        # --- KDE線の追加（drivabilityのみ）---
        if i == 0:
            extended_angles = np.linspace(0, 2 * np.pi, 360)
            kde_values = np.interp(extended_angles, angles, values, period=2 * np.pi)
            ax.plot(extended_angles, kde_values, color='black', linewidth=2, label="KDE-like")
            ax.legend(loc="upper right")

        # --- 選択方向の矢印（Resultのみ）---
        if i == 2 and hasattr(self, "theta") and self.theta is not None:
            r_max = max(values) * 1.1
            ax.annotate("",
                        xy=(self.theta, r_max),
                        xytext=(0, 0),
                        arrowprops=dict(facecolor='red', edgecolor='red', width=2, headwidth=8))



  def policy(self, state, sampled_params, episode=None, log_dir=None):
    """
    SwarmAgent用の行動決定ポリシー: thetaとvalid_directionsを返す。
    """
    theta, valid_directions = self.select_direction_with_candidates(state, sampled_params)
    return theta, valid_directions
  

  def _determine_swarm_mode(self, state, sampled_params):
    """
    群分岐・統合のmodeを決定
    - mode 0: 通常動作（単一群）
    - mode 1: 群分岐（新しいleaderを作成）
    - mode 2: 群統合（他の群に統合）
    分岐条件：
      ・有効な移動方向binが2つ以上
      ・followerのmobility_score平均がbranch_threshold以上
    """
    # sampled_params: [th, k_e, k_c, branch_threshold, merge_threshold]
    if len(sampled_params) >= 5:
        branch_threshold = sampled_params[3]
        merge_threshold = sampled_params[4]
    else:
        branch_threshold = 0.3
        merge_threshold = 0.7

    mobility_scores = state.get('follower_mobility_scores', [])
    if len(mobility_scores) == 0:
        return 0
    
    # followerの数を確認（mobility_scoresの非ゼロ要素数）
    follower_count = sum(1 for score in mobility_scores if score > 0.0)
    avg_mobility = np.mean(mobility_scores)

    # === 有効な移動方向bin数を計算 ===
    # result_histogramはpolicy()で計算済み
    if self.result_histogram is not None:
        # 有効bin: スコアが全体平均以上のbin
        mean_score = np.mean(self.result_histogram)
        valid_bins = [i for i, v in enumerate(self.result_histogram) if v >= mean_score and v > 0.0]
        valid_bin_count = len(valid_bins)
    else:
        valid_bin_count = 0

    # 分岐判定（AND条件）
    if follower_count >= 3 and valid_bin_count >= 2 and avg_mobility >= branch_threshold:
        return 1  # 群分岐
    # 統合判定
    if avg_mobility > merge_threshold:
        return 2  # 群統合
    return 0  # 通常動作


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
    state内からこのpolicyで必要な情報を取得（群ロボット探査最適化版）
    """
    self.agent_azimuth           = state.get("agent_azimuth", 0.0)
    self.agent_collision_flag    = state.get("agent_collision_flag", False)
    self.agent_coordinate_x      = state.get("agent_coordinate_x", 0.0)
    self.agent_coordinate_y      = state.get("agent_coordinate_y", 0.0)
    value = state.get('follower_collision_data', [])
    if isinstance(value, list) and len(value) >= 2:
        # [azimuth, distance, azimuth, distance, ...] の形式を
        # [(azimuth, distance), (azimuth, distance), ...] の形式に変換
        self.follower_collision_data = []
        for i in range(0, len(value), 2):
            if i + 1 < len(value):
                self.follower_collision_data.append((value[i], value[i + 1]))
    
    # 衝突フラグが立っている場合、前回の方向を衝突方向として記録
    if self.agent_collision_flag and self.last_theta is not None:
        self.last_collision_theta = self.last_theta
        print(f"[COLLISION] Recording collision direction: {np.rad2deg(self.last_collision_theta):.1f}°")


  # def get_obstacle_density_histogram(self):
  #   histogram = np.zeros(self.BIN_NUM, dtype=np.float32)

  #   for azimuth, distance in self.follower_collision_data:
  #       if distance < 1e-6:
  #           continue  # 無効データ（0埋めなど）はスキップ

  #       azimuth_deg = azimuth % 360.0
  #       bin_index = int(azimuth_deg) // self.BIN_SIZE_DEG
  #       histogram[bin_index] += 1.0 / (distance + 1e-3)

  #   histogram += 1e-6  # 0除算対策
  #   histogram /= np.sum(histogram)

  #   return histogram


  def get_obstacle_density_histogram(self):
    """
    障害物密度ヒストグラムを計算
    """
    histogram = np.zeros(self.BIN_NUM)
    
    # 前回衝突した方向を避ける
    if self.last_collision_theta is not None:
        # 前回衝突した方向の周辺を低評価
        collision_bin = int(self.last_collision_theta * self.BIN_NUM / (2 * np.pi)) % self.BIN_NUM
        for i in range(self.BIN_NUM):
            # 前回衝突した方向から離れるほど高評価
            distance = min(abs(i - collision_bin), abs(i - collision_bin + self.BIN_NUM), abs(i - collision_bin - self.BIN_NUM))
            if distance < 8:  # 前回衝突した方向の周辺8binを低評価
                histogram[i] = 0.05  # より低い評価
            else:
                histogram[i] = 1.0
        print(f"[AVOIDANCE] Avoiding collision direction: {np.rad2deg(self.last_collision_theta):.1f}°")
    
    # 各方向の障害物密度を計算
    if self.env is not None and hasattr(self.env, 'explored_map') and self.env.explored_map is not None:
        for i in range(self.BIN_NUM):
            angle = 2 * np.pi * i / self.BIN_NUM
            obstacle_count = 0
            total_count = 0
            
            # スキャン範囲を設定
            scan_distance = 10
            
            for distance in range(1, scan_distance + 1):
                check_x = int(self.agent_coordinate_x + distance * np.cos(angle))
                check_y = int(self.agent_coordinate_y + distance * np.sin(angle))
                
                if (0 <= check_y < self.env.explored_map.shape[0] and 
                    0 <= check_x < self.env.explored_map.shape[1]):
                    total_count += 1
                    if self.env.explored_map[check_y, check_x] == 1000:  # 障害物値
                        obstacle_count += 1
            
            if total_count > 0:
                obstacle_ratio = obstacle_count / total_count
                # 障害物密度が高いほど低評価
                histogram[i] *= (1.0 - obstacle_ratio)
    
    return histogram


  def get_exploration_improvement_histogram(self) -> np.ndarray:
    """
    探査向上性の高い方向を算出
    - 一つ前の探査方向を取得し次の方向から除外
    - リーダー機が衝突したのかを確認
    - 衝突がある場合、衝突方向を除外
    - パラメータベースの探査戦略
    - 未探査領域への誘導強化
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
        theta = 2 * np.pi * i / self.BIN_NUM
        angle_diff = min(abs(theta - base_azimuth), 2 * np.pi - abs(theta - base_azimuth))
        decay = 1 - (1 - k) * np.exp(-sharpness * (angle_diff ** 2))
        histogram[i] *= decay
    

    def apply_direction_weight_von(base_azimuth: float, kappa: float):
      """
      base_azimuth方向を中心にフォン・ミーゼス分布で【抑制】をかける
      - kappa: フォン・ミーゼス分布の集中度（大きいほど鋭い抑制）
      """
      vm_pdf_max = np.exp(kappa) / (2 * np.pi * i0(kappa))  # 中心方向の最大値

      for i in range(self.BIN_NUM):
          theta = 2 * np.pi * i / self.BIN_NUM
          angle_diff = np.arctan2(np.sin(theta - base_azimuth), np.cos(theta - base_azimuth))
          vm_pdf = np.exp(kappa * np.cos(angle_diff)) / (2 * np.pi * i0(kappa))
          
          # 中心方向に向かうほど値が小さくなるように：1.0 - 正規化された vm_pdf
          suppress_factor = 1.0 - (vm_pdf / vm_pdf_max)  # ← 中心方向≒0、周囲≒1
          histogram[i] *= suppress_factor  # 凹ませる（中心方向を弱くする）

    # ヒストグラムを初期化
    histogram = np.ones(self.BIN_NUM, dtype=np.float32)

    # 探査済み方向（前回進んだ方向）を抑制
    if self.agent_azimuth is not None:
        reverse_azimuth = (self.agent_azimuth + np.pi) % (2 * np.pi)  # 逆方向を計算
        apply_direction_weight_von(reverse_azimuth, self.k_e)

    # 衝突方向を抑制
    if self.agent_collision_flag and self.agent_azimuth is not None:
        apply_direction_weight_von(self.agent_azimuth, self.k_c)

    # === 未探査領域への誘導強化 ===
    # 環境から探査マップを取得して未探査領域の方向を強化
    self._apply_unexplored_area_guidance(histogram)

    # === 群ロボット探査最適化: パラメータベースの探査戦略 ===
    self._apply_parameter_based_exploration(histogram)

    histogram += 1e-6  # ゼロ割り防止
    histogram /= np.sum(histogram)
    return histogram

  def _apply_parameter_based_exploration(self, histogram):
    """
    パラメータベースの探査戦略（環境情報に依存しない）
    - k_eに応じた探査行動の調整
    - ランダム性と方向性のバランス
    """
    # k_eに応じた探査行動の調整
    exploration_intensity = self.k_e / 50.0  # 0-1に正規化
    
    # 1. ランダム性の増加（探査促進）
    randomness_factor = 1.0 + exploration_intensity * 0.3
    for i in range(self.BIN_NUM):
        histogram[i] *= (1.0 + np.random.uniform(-0.1, 0.1) * randomness_factor)
    
    # 2. 方向性の強化（k_eが大きいほど特定方向を重視）
    if exploration_intensity > 0.5:
        # 高探査モード：より積極的な方向選択
        for i in range(self.BIN_NUM):
            # 現在の方位から離れた方向を強化
            if self.agent_azimuth is not None:
                angle_diff = abs(2 * np.pi * i / self.BIN_NUM - self.agent_azimuth)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                if angle_diff > np.pi / 2:  # 90度以上離れた方向
                    histogram[i] *= (1.0 + exploration_intensity * 0.5)
    
    # 3. 群ロボットの分散促進
    # フォロワーの衝突データを活用して分散を促進
    if len(self.follower_collision_data) > 0:
        for azimuth, distance in self.follower_collision_data:
            if distance < 1e-6:
                continue
            
            # フォロワーがいる方向を避ける（分散促進）
            for i in range(self.BIN_NUM):
                angle = 2 * np.pi * i / self.BIN_NUM
                angle_diff = min(abs(angle - azimuth), 2 * np.pi - abs(angle - azimuth))
                if angle_diff < np.pi / 4:  # 45度以内の方向を抑制
                    histogram[i] *= (1.0 - exploration_intensity * 0.3)

  def get_final_result_histogram(self, drivability, exploration_improvement) -> np.ndarray:
      """
      ファジィ推論に基づき、方向ごとのスコアを統合したヒストグラムを返す（群ロボット探査最適化版）
      - drivability_histogram: 走行可能性ヒストグラム
      - exploration_improvement_histogram: 探査向上性ヒストグラム
      Returns:
          fused_score: 各binの最終スコアヒストグラム (np.ndarray)
      """
      assert len(drivability) == len(exploration_improvement)

      fused_score = np.zeros(self.BIN_NUM, dtype=np.float32)

      # 群ロボット探査最適化のためのパラメータ調整
      alpha = 15.0  # 抑制の鋭さを増加（10.0 → 15.0）
      exploration_weight = 1.5  # 探査向上性の重みを増加

      for bin in range(self.BIN_NUM):
          drive_val = drivability[bin]
          explore_val = exploration_improvement[bin]

          # 探査率向上を重視したソフト抑制係数
          suppression = 1 / (1 + np.exp(-alpha * (drive_val - self.th)))

          # 探査向上性を重視したファジィ積推論
          # 探査向上性の重みを増加し、探査効率を向上
          fused_score[bin] = suppression * drive_val * (explore_val ** exploration_weight)

      fused_score += 1e-6  # ゼロ除け
      fused_score /= np.sum(fused_score)

      return fused_score

  
  
  def select_final_direction_by_weighted_sampling(self, result: np.ndarray) -> float:
    """
    群ロボット探査最適化版の方向選択アルゴリズム
    - 探査効率を重視した重み付け
    - より積極的な探査行動の促進
    """
    q1, q2, q3 = np.quantile(result, [0.25, 0.5, 0.75])

    bins_very_good = [i for i, v in enumerate(result) if v >= q3 and v > 0.0]
    bins_good      = [i for i, v in enumerate(result) if q2 <= v < q3 and v > 0.0]
    bins_okay      = [i for i, v in enumerate(result) if q1 <= v < q2 and v > 0.0]

    bins = bins_very_good + bins_good + bins_okay

    if not bins:
        if self.agent_azimuth is not None:
            fallback_theta = (self.agent_azimuth + np.pi) % (2 * np.pi)
            print(f"[Fallback] No valid bin found. Returning reverse direction: {np.rad2deg(fallback_theta)} deg")
            return fallback_theta
        else:
            fallback_theta = np.random.uniform(0, 2 * np.pi)
            print(f"[Fallback] No azimuth available. Returning random direction: {np.rad2deg(fallback_theta)} deg")
            return fallback_theta

    # 群ロボット探査最適化のための重み付け調整
    # より積極的な探査行動を促進
    score_q1 = 0.7 if bins_very_good else 0.0  # 0.6 → 0.7
    score_q2 = 0.2 if bins_good else 0.0       # 0.25 → 0.2
    score_q3 = 0.1 if bins_okay else 0.0       # 0.15 → 0.1
    total_score = score_q1 + score_q2 + score_q3

    # 正規化
    score_q1 /= total_score if total_score > 0 else 1.0
    score_q2 /= total_score if total_score > 0 else 1.0
    score_q3 /= total_score if total_score > 0 else 1.0

    # 重みに従って構築
    weights = []
    if bins_very_good:
        weights += [score_q1 / len(bins_very_good)] * len(bins_very_good)
    if bins_good:
        weights += [score_q2 / len(bins_good)] * len(bins_good)
    if bins_okay:
        weights += [score_q3 / len(bins_okay)] * len(bins_okay)

    selected_bin = np.random.choice(bins, p=weights)
    selected_angle = 2 * np.pi * selected_bin / self.BIN_NUM

    return selected_angle

  def _apply_unexplored_area_guidance(self, histogram):
    """
    未探査領域への誘導を強化する
    - 探査マップから未探査領域の方向を特定
    - 未探査領域の方向を強化
    """
    if self.env is not None and hasattr(self.env, 'explored_map') and self.env.explored_map is not None:
        current_x = int(self.agent_coordinate_x)
        current_y = int(self.agent_coordinate_y)
        scan_radius = 10
        for i in range(self.BIN_NUM):
            angle = 2 * np.pi * i / self.BIN_NUM
            unexplored_count = 0
            total_count = 0
            for distance in range(1, scan_radius + 1):
                check_x = int(current_x + distance * np.cos(angle))
                check_y = int(current_y + distance * np.sin(angle))
                if (0 <= check_y < self.env.explored_map.shape[0] and 
                    0 <= check_x < self.env.explored_map.shape[1]):
                    total_count += 1
                    if self.env.explored_map[check_y, check_x] == 0:
                        unexplored_count += 1
            if total_count > 0:
                unexplored_ratio = unexplored_count / total_count
                histogram[i] *= (1.0 + unexplored_ratio * 2.0)

  def select_direction_with_candidates(self, state, sampled_params):
    """
    theta（選択方向）とvalid_directions（有効方向リスト）を返す。
    隣接binを1経路グループとしてまとめ、各グループの中心角度とスコアを返す。
    """
    self.get_state_info(state, sampled_params)
    self.drivability_histogram = self.get_obstacle_density_histogram()
    self.exploration_improvement_histogram = self.get_exploration_improvement_histogram()
    self.result_histogram = self.get_final_result_histogram(self.drivability_histogram, self.exploration_improvement_histogram)
    mean_score = np.mean(self.result_histogram)
    valid_bins = [i for i, score in enumerate(self.result_histogram) if score >= mean_score and score > 0.0]

    # 隣接binをグループ化
    groups = []
    current_group = []
    for idx in range(len(valid_bins)):
        if not current_group:
            current_group.append(valid_bins[idx])
        else:
            prev = current_group[-1]
            if (valid_bins[idx] == (prev + 1) % self.BIN_NUM):
                current_group.append(valid_bins[idx])
            else:
                groups.append(current_group)
                current_group = [valid_bins[idx]]
    if current_group:
        # 先頭と末尾がつながっている場合
        if groups and (current_group[0] == 0 and groups[0][-1] == self.BIN_NUM - 1):
            groups[0] = current_group + groups[0]
        else:
            groups.append(current_group)

    # 各グループの中心角度・代表スコア
    valid_directions = []
    for group in groups:
        angles = [2 * np.pi * i / self.BIN_NUM for i in group]
        center_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        group_score = np.mean([self.result_histogram[i] for i in group])
        valid_directions.append({"angle": center_angle, "score": group_score})

    # thetaはグループのスコアで重み付けサンプリング
    scores = np.array([d["score"] for d in valid_directions])
    if len(scores) > 0:
        probs = scores / scores.sum()
        selected_idx = np.random.choice(len(valid_directions), p=probs)
        theta = valid_directions[selected_idx]["angle"]
    else:
        theta = np.random.uniform(0, 2 * np.pi)

    # 選択した方向を記録（衝突回避用）
    self.last_theta = theta

    return theta, valid_directions
