import numpy as np

class EnvParam:
  """
  環境用パラメータを定義
  """
  # ----- Metadata: rendering mode -----
  METADATA = {'render.modes': ['human', 'rgb_gray']}

  # ----- save data -----
  SAVE_FRAMES = True # フレームの保存
  FIG_SIZE    = 12 # プロット画像のサイズ

  # ----- size of drawing -----
  ENV_WIDTH         = 150   # マップ横幅
  ENV_HEIGHT        = 60    # マップ高さ
  OBSTACLE_VALUE    = 1000  # 障害物値
  OBSTACLE_PROB     = 0.01  # 障害物の生成率
  OBSTACLE_MAX_SIZE = 10    # 矩形障害物のサイズ
  MAP_SEED          = 42    # マップのシード値

  # ----- exploration parameter -----
  OUTER_BOUNDARY  = 10.0  # 探査半径(外側)
  INNER_BOUNDARY  = 0.0   # 探査半径(内側)
  MEAN            = 0.0   # 探査中心の平均
  VARIANCE        = 10.0  # 探査中心の分散
  FOLLOWER_STEP   = 100   # フォロワのステップ数

  # ----- reward parameter ----- # TODO 変更必須
  REWARD_DEFAULT          = -1  # デフォルト報酬
  REWARD_AGENT_COLLISION  = -10 # エージェント衝突時の報酬
  REWARD_ROBOT_COLLISION  = -1  # ロボット衝突時の報酬
  REWARD_AVOID_BACKWARD   = 1   # 後退回避時の報酬
  REWARD_FINISH           = 10  # 終了時の探査報酬

  # ----- finish parameter -----
  FINISH_EXPLORED_RATE = 0.95 # 探査終了条件
  
  # ----- distribution parameter -----
  KAPPA         = 1.0                                                 # 逆温度
  BIN_SIZE_DEG  = 20                                                  # ビンのサイズ(度)
  BIN_NUM       = int(360 // BIN_SIZE_DEG)                            # ビン数
  ANGLES        = np.linspace(0, 2 * np.pi, BIN_NUM, endpoint=False)  # 角度

  # ----- initial learning parameter ------
  INITIAL_K                 = 1.0                     # 探査向上性の集中度パラメータ
  INITIAL_TH                = 1.0                     # 走行可能性における閾値
  INITIAL_POSITION          = [10.0, 10.0]            # エージェント初期位置(y, x)
  FOLLOWER_NUM              = 10                      # ロボットの機体数
  FOLLOWER_POSITION_OFFSET  = 5.0                     # ロボットの初期位置のオフセット
  # SAMPLE_NUM                = 5                     # 取得する障害物のサンプリング数
  # DUMMY_DISTANCE            = OUTER_BOUNDARY * 2.0  # ダミー距離値
  # DUMMY_AZIMUTH             = 0.0                   # ダミー用のazimuth
