class Score:
  # TODO スコアの取得がが未実装
  def __init__(self):
    self.goal_reaching_step = None # 目標の探査率に到達したステップ
    self.revisit_ratio = 0.0 # 同じセルへの再訪問率
    self.revisit_count = 0 # 再訪問回数
    self.follower_collision_count = 0 # フォロワの合計衝突回数
    self.agent_collision_count = 0 # エージェントの合計衝突回数
    self.total_distance_traveled = 0 # 合計走行距離

    self.exploration_rate = [] # 探査率格納用リスト
    
  
  def calc_exploration_rate(self, explored_area: int, total_area: int) -> float:
    """
    探査率 = 探査済みセル数 / 探査可能セル数
    計算して履歴に追加
    """
    if total_area == 0:
      rate = 0.0
    else:
      rate = explored_area / total_area
    
    self.exploration_rate.append(rate)
    return rate
  

  def export_metrics(self) -> dict:
    return {
        "goal_reaching_step": self.goal_reaching_step,
        "revisit_ratio": self.revisit_ratio,
        "revisit_count": self.revisit_count,
        "follower_collision_count": self.follower_collision_count,
        "agent_collision_count": self.agent_collision_count,
        "total_distance_traveled": self.total_distance_traveled,
        "final_exploration_rate": self.exploration_rate[-1] if self.exploration_rate else 0.0,
        "exploration_rate_curve": self.exploration_rate,
    }


