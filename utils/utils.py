"""
Utility functions for the swarm robot exploration simulation.
Provides common helper functions and data processing utilities.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from core.logging import get_component_logger

logger = get_component_logger("utils")


def flatten_state(state: Dict[str, Any]) -> np.ndarray:
    base = np.array([
        state["agent_coordinate_x"],
        state["agent_coordinate_y"],
        state["agent_azimuth"],
        state["agent_collision_flag"],
        state["agent_step_count"],
    ], dtype=np.float32)

    try:
        follower_data = np.array(state["follower_collision_data"], dtype=np.float32).flatten()
    except Exception as e:
        raise ValueError(f"Invalid follower_collision_data format: {state['follower_collision_data']}\n{e}")

    # follower_mobility_scoresを追加
    mobility_scores = state.get("follower_mobility_scores", [])
    if len(mobility_scores) == 0:
        # 空の場合は0で埋める（最大ロボット数分）
        mobility_scores = [0.0] * 10  # デフォルトで10個分
    mobility_array = np.array(mobility_scores, dtype=np.float32)

    return np.concatenate([base, follower_data, mobility_array])