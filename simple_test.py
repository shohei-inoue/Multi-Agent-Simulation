#!/usr/bin/env python3
"""
基本的な環境初期化テスト
1から段階的にデバッグする
"""

import os
import sys
import traceback

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """インポートテスト"""
    print("=== 1. インポートテスト ===")
    try:
        from params.simulation import SimulationParam
        print("✓ SimulationParam インポート成功")
        
        from params.agent import AgentParam
        print("✓ AgentParam インポート成功")
        
        from params.system_agent import SystemAgentParam
        print("✓ SystemAgentParam インポート成功")
        
        from params.swarm_agent import SwarmAgentParam
        print("✓ SwarmAgentParam インポート成功")
        
        from agents.agent_factory import create_initial_agents
        print("✓ create_initial_agents インポート成功")
        
        from envs.env import Env
        print("✓ Env インポート成功")
        
        return True
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        traceback.print_exc()
        return False

def test_simulation_param():
    """シミュレーションパラメータテスト"""
    print("\n=== 2. シミュレーションパラメータテスト ===")
    try:
        from params.simulation import SimulationParam
        
        sim_param = SimulationParam()
        print("✓ SimulationParam 作成成功")
        
        # 基本設定
        sim_param.episodeNum = 2
        sim_param.maxStepsPerEpisode = 10
        print("✓ 基本設定完了")
        
        # 環境設定
        sim_param.environment.map.width = 50
        sim_param.environment.map.height = 50
        sim_param.environment.obstacle.probability = 0.0
        print("✓ 環境設定完了")
        
        # 探査設定
        sim_param.explore.robotNum = 3
        sim_param.explore.coordinate.x = 10.0
        sim_param.explore.coordinate.y = 10.0
        sim_param.explore.boundary.inner = 0.0
        sim_param.explore.boundary.outer = 10.0
        print("✓ 探査設定完了")
        
        # ログ設定
        sim_param.robot_logging.save_robot_data = False
        sim_param.robot_logging.save_position = False
        sim_param.robot_logging.save_collision = False
        sim_param.robot_logging.sampling_rate = 1.0
        print("✓ ログ設定完了")
        
        return sim_param
    except Exception as e:
        print(f"❌ シミュレーションパラメータエラー: {e}")
        traceback.print_exc()
        return None

def test_agent_param():
    """エージェントパラメータテスト"""
    print("\n=== 3. エージェントパラメータテスト ===")
    try:
        from params.agent import AgentParam
        from params.system_agent import SystemAgentParam
        from params.swarm_agent import SwarmAgentParam
        
        agent_param = AgentParam()
        print("✓ AgentParam 作成成功")
        
        # SystemAgent設定
        system_param = SystemAgentParam()
        system_param.learningParameter = None
        system_param.branch_condition.branch_enabled = False
        system_param.integration_condition.integration_enabled = False
        agent_param.system_agent_param = system_param
        print("✓ SystemAgent設定完了")
        
        # SwarmAgent設定
        swarm_param = SwarmAgentParam()
        swarm_param.isLearning = False
        swarm_param.learningParameter = None
        agent_param.swarm_agent_params = [swarm_param]
        print("✓ SwarmAgent設定完了")
        
        return agent_param
    except Exception as e:
        print(f"❌ エージェントパラメータエラー: {e}")
        traceback.print_exc()
        return None

def test_environment_creation(sim_param):
    """環境作成テスト"""
    print("\n=== 4. 環境作成テスト ===")
    try:
        from envs.env import Env
        
        print("  環境作成中...")
        env = Env(sim_param)
        print("✓ 環境作成成功")
        
        return env
    except Exception as e:
        print(f"❌ 環境作成エラー: {e}")
        traceback.print_exc()
        return None

def test_agent_creation(env, agent_param):
    """エージェント作成テスト"""
    print("\n=== 5. エージェント作成テスト ===")
    try:
        from agents.agent_factory import create_initial_agents
        
        print("  エージェント作成中...")
        system_agent, swarm_agents = create_initial_agents(env, agent_param)
        print(f"✓ エージェント作成成功 - SystemAgent: {type(system_agent)}, SwarmAgents: {len(swarm_agents)}")
        
        return system_agent, swarm_agents
    except Exception as e:
        print(f"❌ エージェント作成エラー: {e}")
        traceback.print_exc()
        return None, None

def test_environment_reset(env):
    """環境リセットテスト"""
    print("\n=== 6. 環境リセットテスト ===")
    try:
        print("  環境リセット中...")
        state = env.reset()
        print("✓ 環境リセット成功")
        
        # 探査率を確認
        exploration_rate = env.get_exploration_rate()
        print(f"✓ 初期探査率: {exploration_rate:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ 環境リセットエラー: {e}")
        traceback.print_exc()
        return False

def test_single_step(env, swarm_agents):
    """単一ステップテスト"""
    print("\n=== 7. 単一ステップテスト ===")
    try:
        import numpy as np
        
        # デフォルト行動
        swarm_actions = {0: {
            'theta': np.random.uniform(0, 2*np.pi),
            'th': 0.5,
            'k_e': 10.0,
            'k_c': 5.0
        }}
        
        print("  ステップ実行中...")
        next_state, rewards, done, truncated, info = env.step(swarm_actions)
        print("✓ ステップ実行成功")
        
        # 探査率を確認
        exploration_rate = env.get_exploration_rate()
        print(f"✓ ステップ後探査率: {exploration_rate:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ ステップ実行エラー: {e}")
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("=== 基本的な環境初期化テスト開始 ===")
    
    # 1. インポートテスト
    if not test_imports():
        print("❌ インポートテスト失敗")
        return False
    
    # 2. シミュレーションパラメータテスト
    sim_param = test_simulation_param()
    if sim_param is None:
        print("❌ シミュレーションパラメータテスト失敗")
        return False
    
    # 3. エージェントパラメータテスト
    agent_param = test_agent_param()
    if agent_param is None:
        print("❌ エージェントパラメータテスト失敗")
        return False
    
    # 4. 環境作成テスト
    env = test_environment_creation(sim_param)
    if env is None:
        print("❌ 環境作成テスト失敗")
        return False
    
    # 5. エージェント作成テスト
    system_agent, swarm_agents = test_agent_creation(env, agent_param)
    if system_agent is None or swarm_agents is None:
        print("❌ エージェント作成テスト失敗")
        return False
    
    # 6. 環境リセットテスト
    if not test_environment_reset(env):
        print("❌ 環境リセットテスト失敗")
        return False
    
    # 7. 単一ステップテスト
    if not test_single_step(env, swarm_agents):
        print("❌ 単一ステップテスト失敗")
        return False
    
    print("\n🎉 全てのテストが成功しました！")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ テストが失敗しました。")
        sys.exit(1) 