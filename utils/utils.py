import numpy as np

def flatten_state(state: dict) -> np.ndarray:
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

    return np.concatenate([base, follower_data])