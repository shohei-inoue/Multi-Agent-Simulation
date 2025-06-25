import gym
import numpy as np
import math

def create_observation_space() -> gym.spaces.Dict:
  """
  observation_space:
    - agent_coordinate_x      ∈ [0, ∞): float (scalar)
    - agent_coordinate_y      ∈ [0, ∞): float (scalar)
    - agent_azimuth           ∈ [0, 2π): float (scalar)
    - agent_collision_flag    ∈ {0, 1}: float (scalar)
    - agent_step_count        ∈ [0, ∞): float (scalar)
    - follower_collision_data ∈ [0, ∞): float (vector) TODO 追加必要
  """
  MAX_COLLISION_NUM = 100  # 最大フォロワー数 × 各フォロワの最大衝突数
  return gym.spaces.Dict({
    "agent_coordinate_x"     : gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),  
    "agent_coordinate_y"     : gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),  
    "agent_azimuth"          : gym.spaces.Box(low=0.0, high=2 * math.pi, shape=(), dtype=np.float32),  
    "agent_collision_flag"   : gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
    "agent_step_count"       : gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
    "follower_collision_data": gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(MAX_COLLISION_NUM * 2,),  # distance と azimuth
            dtype=np.float32
        ),
  })


def create_initial_state(coordinate_x, coordinate_y, azimuth, collision_flag, agent_step_count, follower_collision_data=None):
    """
    初期状態を構成
    """
    state = {
        "agent_coordinate_x"  : coordinate_x,
        "agent_coordinate_y"  : coordinate_y,
        "agent_azimuth"       : azimuth,
        "agent_collision_flag": collision_flag,
        "agent_step_count"    : agent_step_count 
    }

    MAX_COLLISION_NUM = 100
    if follower_collision_data is None:
        # [[0.0, 0.0], [0.0, 0.0], ..., 100個]
        padded_list = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(MAX_COLLISION_NUM)]
    else:
        padded_list = [np.array([a, d], dtype=np.float32) for a, d in follower_collision_data[:MAX_COLLISION_NUM]]
        # 足りない分を補完
        while len(padded_list) < MAX_COLLISION_NUM:
            padded_list.append(np.array([0.0, 0.0], dtype=np.float32))

    state["follower_collision_data"] = padded_list

    return state