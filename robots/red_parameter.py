from enum import Enum

class RedParam:
  MIN_MOVEMENT = 2.0              # 最小直進量
  MAX_MOVEMENT = 3.0              # 最大直進量
  MIN_BOIDS_MOVEMENT = 2.0        # boids行動時最小直進量
  MAX_BOIDS_MOVEMENT = 3.0        # boids行動時最大直進量
  MIN_AVOIDANCE_BEHAVIOR = 90.0   # 回避行動最小角度
  MAX_AVOIDANCE_BEHAVIOR = 270.0  # 回避行動最大角度 


class BoidsType(Enum):
    """
    boids判断用のタイプ
    """
    NONE  = 0
    OUTER = 1
    INNER = 2