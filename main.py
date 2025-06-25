import uuid
import os
import pandas as pd
from datetime import datetime

from params.simulation import Param
from agents.agent_config import AgentConfig
from envs.env import Env

def main():
    # === param を最初に定義 ===
    param = Param()

    # ----- 保存ディレクトリ作成 -----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_id = getattr(param, "simulationId", None) or f"sim_{timestamp}_{uuid.uuid4().hex[:6]}"
    log_dir = f"./logs/{sim_id}"
    os.makedirs(log_dir, exist_ok=True)

    # --- サブディレクトリ作成 ---
    subdirs = ["metrics", "gifs", "models", "tensorboard"]
    for sub in subdirs:
        os.makedirs(os.path.join(log_dir, sub), exist_ok=True)

    # --- 環境の初期化 ---
    env = Env(param=param)

    # --- 環境パラメータ保存 ---
    env_info_path = os.path.join(log_dir, "env_info.csv")
    pd.DataFrame({
        "map_width": [param.environment.map.width],
        "map_height": [param.environment.map.height],
        "map_seed": [param.environment.map.seed],
        "obstacle_prob": [param.environment.obstacle.probability],
        "obstacle_max_size": [param.environment.obstacle.maxSize],
        "robot_num": [param.explore.robotNum],
        "finish_rate": [param.explore.finishRate],
    }).to_csv(env_info_path, index=False)

    # --- エージェントの初期化 ---
    agent = AgentConfig(
        env=env,
        param=param.agent
    )

    # --- 学習・実行 ---
    agent.train(log_dir=log_dir)

if __name__ == "__main__":
    print("Starting main.py...")
    main()