import gym
import numpy as np
import math

def create_action_space():
  """
  action_space:
    - parameter theta of the policy (continuous) ∈ [0, 2π): float
    - parameter mode of the policy (discrete) ∈ {0, 1, 2}: int
  """
  return gym.spaces.Dict({
      "theta": gym.spaces.Box(low=0.0, high=2 * math.pi, shape=(), dtype=np.float32),
      "mode": gym.spaces.Discrete(3)
  })