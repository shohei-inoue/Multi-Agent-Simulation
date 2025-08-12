import numpy as np
import math
import tensorflow as tf
import os
import csv
import matplotlib.pyplot as plt
from scipy.special import i0


class AlgorithmVfhFuzzy():
  KAPPA         = 1.0                                                 # 逆温度
  BIN_SIZE_DEG  = 10                                                  # ビンのサイズ(度) - 360/36=10度
  BIN_NUM       = 36                                                  # ビン数
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
    self.BIN_NUM = 36
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
                azimuth = value[i]
                distance = value[i + 1]
                
                # データ型チェック
                if (isinstance(azimuth, (int, float, np.number)) and 
                    isinstance(distance, (int, float, np.number))):
                    self.follower_collision_data.append((azimuth, distance))
                else:
                    print(f"[WARNING] Skipping invalid collision data: azimuth={azimuth} ({type(azimuth)}), distance={distance} ({type(distance)})")
    else:
        self.follower_collision_data = []
    
    # 衝突フラグが立っている場合、前回の方向を衝突方向として記録
    if self.agent_collision_flag:
        if self.last_theta is not None:
            self.last_collision_theta = self.last_theta
            print(f"[COLLISION] Recording collision direction: {np.rad2deg(self.last_collision_theta):.1f}°")
        else:
            # last_thetaがまだない場合は、現在の方位を衝突方向として記録
            if self.agent_azimuth is not None:
                self.last_collision_theta = self.agent_azimuth
                print(f"[COLLISION] Recording collision direction (from azimuth): {np.rad2deg(self.last_collision_theta):.1f}°")
    else:
        # 衝突フラグが立っていない場合は、数ステップ後に衝突方向をリセット
        # これにより、一時的な回避から徐々に通常の行動に戻る
        if self.last_collision_theta is not None:
            # 衝突方向の記憶を徐々に薄くする（時間経過による忘却）
            if not hasattr(self, '_collision_reset_counter'):
                self._collision_reset_counter = 0
            self._collision_reset_counter += 1
            
            # 10ステップ後に衝突方向をリセット
            if self._collision_reset_counter >= 10:
                self.last_collision_theta = None
                self._collision_reset_counter = 0
                print(f"[RESET] Collision direction reset after 10 steps")
        else:
            # 衝突方向がリセットされている場合はカウンターもリセット
            if hasattr(self, '_collision_reset_counter'):
                self._collision_reset_counter = 0


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
    障害物密度ヒストグラムを計算（改善版）
    フォロワーの衝突データを使用し、衝突方向は一時的に回避するが完全には遮断しない
    """
    # 基本ヒストグラムを初期化（全方向1.0）
    histogram = np.ones(self.BIN_NUM, dtype=np.float32)
    
    # 前回衝突した方向を一時的に回避（完全遮断ではなく確率調整）
    if self.last_collision_theta is not None:
        collision_bin = int(self.last_collision_theta * self.BIN_NUM / (2 * np.pi)) % self.BIN_NUM
        avoided_bins = 0
        
        for i in range(self.BIN_NUM):
            # 前回衝突した方向からの距離を計算
            distance = min(abs(i - collision_bin), abs(i - collision_bin + self.BIN_NUM), abs(i - collision_bin - self.BIN_NUM))
            
            if distance < 4:  # 衝突方向周辺4bin（40度範囲）
                # 完全に遮断せず、確率を下げる
                histogram[i] = 0.3  # 0.05から0.3に改善（完全遮断を回避）
                avoided_bins += 1
            elif distance < 8:  # 衝突方向周辺8bin（80度範囲）
                # 中程度の確率調整
                histogram[i] = 0.7
            else:
                # 遠い方向は通常の確率
                histogram[i] = 1.0
        
        print(f"[AVOIDANCE] Temporarily reducing probability for collision direction: {np.rad2deg(self.last_collision_theta):.1f}° (reduced {avoided_bins} bins)")
    
    # フォロワーの衝突データによる障害物密度調整
    for azimuth, distance in self.follower_collision_data:
        # データ型チェックとエラーハンドリング
        try:
            # distanceが配列の場合は最初の要素を使用
            if hasattr(distance, '__len__') and len(distance) > 0:
                distance = distance[0]
            
            # 数値チェック
            if not isinstance(distance, (int, float, np.number)) or distance < 1e-6:
                continue  # 無効データ（0埋めなど）はスキップ

            azimuth_deg = azimuth % 360.0
            bin_index = int(azimuth_deg) // self.BIN_SIZE_DEG
            
            # 距離に基づいて障害物密度を計算（近いほど危険）
            obstacle_weight = 1.0 / (distance + 1e-3)
            histogram[bin_index] = max(0.1, histogram[bin_index] - obstacle_weight)  # 最低値0.1を保証
        except (TypeError, ValueError, IndexError) as e:
            print(f"[WARNING] Invalid collision data: azimuth={azimuth}, distance={distance}, error={e}")
            continue

    # 正規化
    histogram += 1e-6  # 0除算対策
    histogram /= np.sum(histogram)

    return histogram
    
    # 正規化
    histogram += 1e-6  # 0除算対策
    histogram /= np.sum(histogram)

    return histogram


  def get_exploration_improvement_histogram(self) -> np.ndarray:
    """
    探査向上性の高い方向を算出（元々のVFH-Fuzzy設計）
    - 一つ前の探査方向を取得し次の方向から除外
    - リーダー機が衝突したのかを確認
    - 衝突がある場合、衝突方向を除外
    """
    def apply_direction_weight_gauss(base_azimuth: float, k: float, sharpness: float = 10.0):
      """
      base_azimuth方向を中心にガウス分布で【抑制】をかける
      - k: 抑制の強さ（0に近いほど強く抑制、1で抑制なし）
      - sharpness: 抑制の鋭さ（大きいほど鋭く抑制）
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

    histogram += 1e-6  # ゼロ割り防止
    histogram /= np.sum(histogram)
    return histogram



  def get_final_result_histogram(self, drivability, exploration_improvement) -> np.ndarray:
      """
      ファジィ推論に基づき、方向ごとのスコアを統合したヒストグラムを返す（元々のVFH-Fuzzy設計）
      - drivability_histogram: 走行可能性ヒストグラム
      - exploration_improvement_histogram: 探査向上性ヒストグラム
      Returns:
          fused_score: 各binの最終スコアヒストグラム (np.ndarray)
      """
      assert len(drivability) == len(exploration_improvement)

      fused_score = np.zeros(self.BIN_NUM, dtype=np.float32)

      # 元々の設計パラメータ
      alpha = 10.0  # ソフト抑制の鋭さ（元々の設計値）

      for bin in range(self.BIN_NUM):
          drive_val = drivability[bin]
          explore_val = exploration_improvement[bin]

          # ソフト抑制係数（元々の設計）
          suppression = 1 / (1 + np.exp(-alpha * (drive_val - self.th)))

          # 元々のファジィ積推論（探査向上性の重みは1.0）
          fused_score[bin] = suppression * drive_val * explore_val

      fused_score += 1e-6  # ゼロ除け
      fused_score /= np.sum(fused_score)

      return fused_score

  
  
  def select_final_direction_by_weighted_sampling(self, result: np.ndarray) -> float:
    """
    元々のVFH-Fuzzy設計の方向選択アルゴリズム
    - 四分位数ベースの重み付けサンプリング
    - 適度な探査行動の促進
    """
    q1, q2, q3 = np.quantile(result, [0.25, 0.5, 0.75])

    bins_very_good = [i for i, v in enumerate(result) if v >= q3 and v > 0.0]
    bins_good      = [i for i, v in enumerate(result) if q2 <= v < q3 and v > 0.0]
    bins_okay      = [i for i, v in enumerate(result) if q1 <= v < q2 and v > 0.0]

    bins = bins_very_good + bins_good + bins_okay

    if not bins:
        # 境界近くでの適切なフォールバック処理
        if self.env is not None and hasattr(self.env, 'explored_map'):
            map_height, map_width = self.env.explored_map.shape
            current_x = int(self.agent_coordinate_x)
            current_y = int(self.agent_coordinate_y)
            
            # 境界からの距離を計算
            distance_to_left = current_x
            distance_to_right = map_width - current_x - 1
            distance_to_top = current_y
            distance_to_bottom = map_height - current_y - 1
            
            # 最も遠い境界の方向を選択（中央方向への復帰）
            distances = [distance_to_right, distance_to_bottom, distance_to_left, distance_to_top]
            directions = [0.0, np.pi/2, np.pi, 3*np.pi/2]  # 右、下、左、上
            
            max_distance_idx = np.argmax(distances)
            fallback_theta = directions[max_distance_idx]
            
            # ランダム性を追加（±30度の範囲）
            fallback_theta += np.random.uniform(-np.pi/6, np.pi/6)
            fallback_theta = fallback_theta % (2 * np.pi)
            
            print(f"[Fallback-Boundary] No valid bin found. Moving toward center: {np.rad2deg(fallback_theta):.1f}°")
            return fallback_theta
        
        # 従来のフォールバック処理
        if self.agent_azimuth is not None:
            fallback_theta = (self.agent_azimuth + np.pi) % (2 * np.pi)
            print(f"[Fallback] No valid bin found. Returning reverse direction: {np.rad2deg(fallback_theta)} deg")
            return fallback_theta
        else:
            fallback_theta = np.random.uniform(0, 2 * np.pi)
            print(f"[Fallback] No azimuth available. Returning random direction: {np.rad2deg(fallback_theta)} deg")
            return fallback_theta

    # 元々の設計の重み付け
    score_q1 = 0.6 if bins_very_good else 0.0  # Very Good: 60%
    score_q2 = 0.25 if bins_good else 0.0      # Good: 25%
    score_q3 = 0.15 if bins_okay else 0.0      # Okay: 15%
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

    # 全方向が有効な場合でも、より確率的な動作を促進
    if len(valid_bins) >= self.BIN_NUM * 0.8:  # 80%以上のbinが有効な場合（29個以上）
        # 隣接binをグループ化して確率分布を改善
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
        print(f"[GROUPING] All directions valid ({len(valid_bins)}/36 bins), using grouped bins for better probability distribution")
    else:
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
        if len(group) == 1:
            # 個別binの場合は直接角度を計算
            center_angle = 2 * np.pi * group[0] / self.BIN_NUM
        else:
            # グループの場合は平均角度を計算
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
        
        # デバッグ情報出力（方向選択の詳細）
        if hasattr(self, 'debug_direction_selection') and self.debug_direction_selection:
            print(f"[VFH-Fuzzy Direction Selection]")
            print(f"  Valid groups: {len(valid_directions)}")
            for i, vd in enumerate(valid_directions):
                print(f"    Group {i}: angle={np.rad2deg(vd['angle']):.1f}°, score={vd['score']:.3f}, prob={probs[i]:.3f}")
            print(f"  Selected: Group {selected_idx}, theta={np.rad2deg(theta):.1f}°")
    else:
        theta = np.random.uniform(0, 2 * np.pi)
        if hasattr(self, 'debug_direction_selection') and self.debug_direction_selection:
            print(f"[VFH-Fuzzy] No valid directions, random theta={np.rad2deg(theta):.1f}°")

    # 選択した方向を記録（衝突回避用）
    self.last_theta = theta

    return theta, valid_directions
