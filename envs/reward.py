def create_reward():
  """
     reward
     default                  : デフォルトの時間ペナルティ（小さく探索を促進）
     exploration_gain         : 新しい未踏エリアに到達したときの報酬
     revisit_penalty          : 探索済み領域を再訪したときのペナルティ
     collision_penalty        : エージェントが衝突したときのペナルティ
     clear_target_rate        : 探査率が目的の値以上になった場合
     none_finnish_penalty     : 探査が最終ステップまでに終わらなかった場合
  """
  return {
    'default'                   : -1,
    'exploration_gain'          : +5,
    'revisit_penalty'           : -2,
    'collision_penalty'         : -10,
    'clear_target_rate'         : +50,
    'none_finish_penalty'       : -50,
  }