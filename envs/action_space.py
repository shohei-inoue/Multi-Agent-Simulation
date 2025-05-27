import gym
import numpy as np

def create_action_space():
  """
  action_space:
    parameter k_e of the policy (continuity)  | (0 <= k_e < inf): float
    parameter k_c of the policy (continuity)  | (0 <= k_c < inf): float
    parameter th of the policy (continuity)   | (0 <= th < inf): float
  """
  return gym.spaces.Dict({
    "k_e"   : gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
    "k_c"   : gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
    "th"    : gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
  })