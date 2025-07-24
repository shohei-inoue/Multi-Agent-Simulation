import gymnasium as gym
import numpy as np
import math

def create_swarm_action_space():
    """
    SwarmAgent用の行動空間:
      - theta: [0, 2π) 移動方向のみ
    """
    return gym.spaces.Dict({
        "theta": gym.spaces.Box(low=0.0, high=2 * math.pi, shape=(), dtype=np.float32)
    })

def create_system_action_space():
    """
    SystemAgent用の行動空間:
      - action_type: {0, 1, 2} (0: 何もしない, 1: 分岐, 2: 統合)
      - target_swarm: 群ID（分岐/統合対象の群を指定、最大群数分の離散値）
    """
    MAX_SWARMS = 10
    return gym.spaces.Dict({
        "action_type": gym.spaces.Discrete(3),  # 0: 何もしない, 1: 分岐, 2: 統合
        "target_swarm": gym.spaces.Discrete(MAX_SWARMS)
    })