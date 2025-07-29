#!/usr/bin/env python3
"""
GIF生成テストスクリプト
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from params.simulation import SimulationParam
from params.agent import AgentParam
from params.system_agent import SystemAgentParam
from params.swarm_agent import SwarmAgentParam
from envs.env import Env
from agents.agent_factory import create_initial_agents

def setup_test_environment():
    """テスト用環境設定"""
    sim_param = SimulationParam()
    
    # 基本設定
    sim_param.episodeNum = 1
    sim_param.maxStepsPerEpisode = 10  # 短時間でテスト
    
    # 環境設定
    sim_param.environment.map.width = 200
    sim_param.environment.map.height = 100
    sim_param.environment.obstacle.probability = 0.003
    
    # 探査設定
    sim_param.explore.robotNum = 20
    sim_param.explore.coordinate.x = 10.0
    sim_param.explore.coordinate.y = 10.0
    sim_param.explore.boundary.inner = 0.0
    sim_param.explore.boundary.outer = 20.0
    
    # ログ設定（GIF生成有効）
    sim_param.robot_logging.save_robot_data = True
    sim_param.robot_logging.save_position = True
    sim_param.robot_logging.save_collision = True
    sim_param.robot_logging.sampling_rate = 1.0
    
    return sim_param

def setup_test_agent():
    """テスト用エージェント設定"""
    agent_param = AgentParam()
    
    # SystemAgent設定（学習なし、分岐・統合なし）
    system_param = SystemAgentParam()
    system_param.learningParameter = None
    system_param.branch_condition.branch_enabled = False
    system_param.integration_condition.integration_enabled = False
    agent_param.system_agent_param = system_param
    
    # SwarmAgent設定（学習なし）
    swarm_param = SwarmAgentParam()
    swarm_param.isLearning = False
    swarm_param.learningParameter = None
    agent_param.swarm_agent_params = [swarm_param]
    
    return agent_param

def test_gif_generation():
    """GIF生成テスト"""
    print("=== GIF生成テスト開始 ===")
    
    try:
        # 1. 環境設定
        print("1. 環境設定中...")
        sim_param = setup_test_environment()
        print("✓ 環境設定完了")
        
        # 2. エージェント設定
        print("2. エージェント設定中...")
        agent_param = setup_test_agent()
        print("✓ エージェント設定完了")
        
        # 3. 環境作成
        print("3. 環境作成中...")
        env = Env(sim_param)
        print("✓ 環境作成完了")
        
        # 4. エージェント作成
        print("4. エージェント作成中...")
        system_agent, swarm_agents = create_initial_agents(env, agent_param)
        print(f"✓ エージェント作成完了 - SwarmAgents: {len(swarm_agents)}")
        
        # 5. SystemAgentを環境に設定
        print("5. SystemAgentを環境に設定中...")
        env.set_system_agent(system_agent)
        print("✓ SystemAgent設定完了")
        
        # 結果保存用ディレクトリ作成
        output_dir = "test_gif_output"
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ 出力ディレクトリ作成: {output_dir}")
        
        # 6. 短時間エピソード実行
        print("6. エピソード実行中...")
        env.start_episode(0)
        state = env.reset()
        print("  環境リセット完了")
        
        for step in range(10):
            print(f"    ステップ {step + 1}/10")
            
            # 各SwarmAgentのアクションを取得
            swarm_actions = {}
            for swarm_id, swarm_agent in swarm_agents.items():
                swarm_state = env.get_swarm_agent_observation(swarm_id)
                action, action_info = swarm_agent.get_action(swarm_state, 0, output_dir)
                swarm_actions[swarm_id] = action
            
            # 環境をステップ実行
            next_state, rewards, done, truncated, info = env.step(swarm_actions)
            
            # フレームキャプチャ（GIF生成用）
            try:
                env.capture_frame()
                print(f"    フレームキャプチャ完了")
            except Exception as e:
                print(f"    フレームキャプチャエラー: {e}")
            
            # 探査率確認
            exploration_rate = env.get_exploration_rate()
            print(f"    探査率: {exploration_rate:.3f}")
            
            if done:
                print(f"    エピソード終了（ステップ {step + 1}）")
                break
        
        # エピソード終了時にGIF保存
        try:
            env.end_episode(output_dir)
            print("    GIF保存完了")
        except Exception as e:
            print(f"    GIF保存エラー: {e}")
        
        print("✓ エピソード完了")
        
        # 結果確認
        gif_files = [f for f in os.listdir(output_dir) if f.endswith('.gif')]
        print(f"✓ 生成されたGIFファイル: {gif_files}")
        
        print("🎉 GIF生成テストが正常に完了しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gif_generation() 