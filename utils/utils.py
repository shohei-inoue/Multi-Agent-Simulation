import numpy as np

def flatten_state(state_dict):
    azimuth = state_dict["leader_info"]["azimuth"]
    mask = state_dict["leader_info"]["mask"]
    k_e = state_dict["k_e"]
    k_c = state_dict["k_c"]
    th = state_dict["th"]

    return np.array([azimuth, mask, k_e, k_c, th], dtype=np.float32)