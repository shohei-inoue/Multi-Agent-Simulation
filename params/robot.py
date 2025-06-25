from enum import Enum

class Movement:
  min: float = 2.0
  max: float = 3.0


class Boids:
  min: float = 2.0
  max: float = 3.0

class Avoidance:
  min: float = 90.0
  max: float = 270.0


class BoidsType(Enum):
    """
    boids判断用のタイプ
    """
    NONE  = 0
    OUTER = 1
    INNER = 2


class Offset:
  position              : float     = 5.0
  step                  : int       = 0
  amount_of_movement    : float     = 0.0
  direction_angle       : float     = 0.0
  collision_flag        : bool      = False
  boids_flag            : BoidsType = BoidsType.NONE
  estimated_probability : float     = 0.0
  one_explore_step      : int       = 100



class RobotParam:
  movement  : Movement  = Movement()
  boids     : Boids     = Boids()
  avoidance : Avoidance = Avoidance()
  offset    : Offset    = Offset()