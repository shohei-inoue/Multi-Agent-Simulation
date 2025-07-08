def create_reward():
  """
     reward
     default                  : デフォルトの時間ペナルティ（小さく探索を促進）
     exploration_gain         : 新しい未踏エリアに到達したときの報酬（基本値）
     exploration_gain_multiplier : 探査率上昇量に応じた倍率
     revisit_penalty          : 探索済み領域を再訪したときのペナルティ
     collision_penalty        : エージェントが衝突したときのペナルティ
     clear_target_rate        : 探査率が目的の値以上になった場合
     none_finnish_penalty     : 探査が最終ステップまでに終わらなかった場合
  """
  return {
    'default'                   : -1,
    'exploration_gain'          : +5,  # 基本報酬
    'exploration_gain_multiplier': 10.0,  # 上昇量に応じた倍率
    'revisit_penalty'           : -2,
    'collision_penalty'         : -10,
    'clear_target_rate'         : +50,
    'none_finish_penalty'       : -50,
  }


def calculate_exploration_reward(exploration_ratio, previous_ratio, reward_dict):
  """
  探査率の上昇具合に応じて報酬を計算
  
  Args:
    exploration_ratio: 現在の探査率
    previous_ratio: 前回の探査率
    reward_dict: 報酬設定辞書
  
  Returns:
    float: 計算された報酬
  """
  if exploration_ratio > previous_ratio:
    # 探査率が上昇した場合
    improvement = exploration_ratio - previous_ratio
    
    # 基本報酬 + 上昇量に応じた追加報酬
    base_reward = reward_dict.get('exploration_gain', 5.0)
    multiplier = reward_dict.get('exploration_gain_multiplier', 10.0)
    
    # 上昇量が大きいほど報酬も大きくなる（非線形）
    additional_reward = multiplier * (improvement ** 0.5)  # 平方根で非線形化
    
    total_reward = base_reward + additional_reward
    
    print(f"Exploration reward: base={base_reward:.2f}, improvement={improvement:.4f}, additional={additional_reward:.2f}, total={total_reward:.2f}")
    
    return total_reward
  else:
    # 探査率が上昇しなかった場合
    return reward_dict.get('revisit_penalty', -2.0)


def calculate_swarm_exploration_reward(exploration_ratio, previous_ratio, reward_config, 
                                     step_count, total_steps, explored_area, total_area):
  """
  群ロボット探査専用の報酬計算（探査率向上を最優先）
  
  Args:
    exploration_ratio: 現在の探査率
    previous_ratio: 前回の探査率
    reward_config: SwarmExplorationRewardConfigオブジェクト
    step_count: 現在のステップ数
    total_steps: 最大ステップ数
    explored_area: 探査済みエリア
    total_area: 総エリア
  
  Returns:
    float: 計算された報酬
  """
  total_reward = 0.0
  
  # 1. 探査率向上の報酬（最重要）
  if exploration_ratio > previous_ratio:
    improvement = exploration_ratio - previous_ratio
    
    # 基本報酬 + 上昇量に応じた重み付き報酬
    base_reward = reward_config.exploration_base_reward
    improvement_reward = reward_config.exploration_improvement_weight * improvement
    
    # 非線形報酬（探査率が高いほど報酬を増加）
    nonlinear_bonus = improvement_reward * (1 + exploration_ratio)
    
    exploration_reward = base_reward + nonlinear_bonus
    total_reward += exploration_reward
    
    print(f"Swarm exploration reward: base={base_reward:.2f}, improvement={improvement:.4f}, "
          f"nonlinear_bonus={nonlinear_bonus:.2f}, total_exploration={exploration_reward:.2f}")
  else:
    # 探査率が上昇しなかった場合のペナルティ
    total_reward += reward_config.revisit_penalty_weight * (previous_ratio - exploration_ratio)
  
  # 2. 時間効率報酬（早期完了ボーナス）
  if exploration_ratio > 0.8:  # 80%以上探査済みの場合
    time_efficiency = 1.0 - (step_count / total_steps)
    time_bonus = reward_config.time_efficiency_bonus * time_efficiency * exploration_ratio
    total_reward += time_bonus
    
    print(f"Time efficiency bonus: {time_bonus:.2f}")
  
  # 3. 目標達成報酬（全探索完了）
  if exploration_ratio >= 0.95:  # 95%以上探査済みの場合
    completion_bonus = reward_config.completion_bonus * exploration_ratio
    total_reward += completion_bonus
    
    print(f"Completion bonus: {completion_bonus:.2f}")
  
  return total_reward


def calculate_exploration_reward_advanced(exploration_ratio, previous_ratio, reward_config):
  """
  より高度な探査報酬計算（パラメータ設定ファイル対応）
  
  Args:
    exploration_ratio: 現在の探査率
    previous_ratio: 前回の探査率
    reward_config: RewardConfigオブジェクト
  
  Returns:
    float: 計算された報酬
  """
  if exploration_ratio > previous_ratio:
    # 探査率が上昇した場合
    improvement = exploration_ratio - previous_ratio
    
    # 設定から値を取得
    base_reward = reward_config.exploration.base_reward
    multiplier = reward_config.exploration.improvement_multiplier
    nonlinearity = reward_config.exploration.nonlinearity
    max_reward = reward_config.exploration.max_reward
    min_reward = reward_config.exploration.min_reward
    
    # 非線形報酬計算
    if nonlinearity == 0:
      # 線形
      additional_reward = multiplier * improvement
    else:
      # 非線形（平方根、二乗など）
      additional_reward = multiplier * (improvement ** nonlinearity)
    
    total_reward = base_reward + additional_reward
    
    # 報酬の範囲制限
    total_reward = max(min_reward, min(max_reward, total_reward))
    
    print(f"Advanced exploration reward: base={base_reward:.2f}, improvement={improvement:.4f}, "
          f"additional={additional_reward:.2f}, total={total_reward:.2f}")
    
    return total_reward
  else:
    # 探査率が上昇しなかった場合
    return reward_config.revisit_penalty


def calculate_exploration_reward_with_momentum(exploration_ratio, previous_ratio, reward_config, 
                                             momentum_factor=0.8, consecutive_improvements=0):
  """
  モメンタムを考慮した探査報酬計算（連続改善ボーナス）
  
  Args:
    exploration_ratio: 現在の探査率
    previous_ratio: 前回の探査率
    reward_config: RewardConfigオブジェクト
    momentum_factor: モメンタム係数（0-1）
    consecutive_improvements: 連続改善回数
  
  Returns:
    tuple: (計算された報酬, 更新された連続改善回数)
  """
  if exploration_ratio > previous_ratio:
    # 探査率が上昇した場合
    improvement = exploration_ratio - previous_ratio
    
    # 基本報酬計算
    base_reward = reward_config.exploration.base_reward
    multiplier = reward_config.exploration.improvement_multiplier
    nonlinearity = reward_config.exploration.nonlinearity
    
    # 非線形報酬計算
    if nonlinearity == 0:
      additional_reward = multiplier * improvement
    else:
      additional_reward = multiplier * (improvement ** nonlinearity)
    
    # モメンタムボーナス（連続改善による追加報酬）
    momentum_bonus = consecutive_improvements * momentum_factor * base_reward
    
    total_reward = base_reward + additional_reward + momentum_bonus
    
    # 報酬の範囲制限
    total_reward = max(reward_config.exploration.min_reward, 
                      min(reward_config.exploration.max_reward, total_reward))
    
    # 連続改善回数を更新
    new_consecutive_improvements = consecutive_improvements + 1
    
    print(f"Momentum exploration reward: base={base_reward:.2f}, improvement={improvement:.4f}, "
          f"additional={additional_reward:.2f}, momentum_bonus={momentum_bonus:.2f}, "
          f"consecutive={new_consecutive_improvements}, total={total_reward:.2f}")
    
    return total_reward, new_consecutive_improvements
  else:
    # 探査率が上昇しなかった場合
    return reward_config.revisit_penalty, 0  # 連続改善回数をリセット