import gym
import numpy as np

def create_observation_space():
  """
  observation_space:
    leader_collision_point: theta             | (0 <= theta < 2 * pi: float | None
    parameter k_e of the policy (continuity)  | (0 <= k_e < inf): float
    parameter k_c of the policy (continuity)  | (0 <= k_c < inf): float
    parameter th of the policy (continuity)   | (0 <= th < inf): float
  """
  leader_info_space = gym.spaces.Dict({
    "azimuth": gym.spaces.Box(low=0.0, high=2 * np.pi, shape=(), dtype=np.float32),
    "mask": gym.spaces.Discrete(2) # 0 if not None, 1 if None
  })

  return gym.spaces.Dict({
    "leader_info" : leader_info_space,
    "k_e"         : gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
    "k_c"         : gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
    "th"          : gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
  })

  