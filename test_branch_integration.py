#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Branch Integration Script
クールダウンなしで分岐・統合が動作するかテストする

使用方法:
    python test_branch_integration.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from params.simulation import SimulationParam
from params.agent import AgentParam
from params.system_agent import SystemAgentParam
from params.swarm_agent import SwarmAgentParam
from params.learning import LearningParameter
from params.debug import DebugParam

def setup_test_environment():
    """テスト用環境設定"""
    sim_param = SimulationParam()
    
    # 基本設定
    sim_param.episodeNum = 3  # 短時間テスト
    sim_param.maxStepsPerEpisode = 20  # 短時間テスト
    
    # 環境設定
    sim_param.environment.map.width = 100
    sim_param.environment.map.height = 50
    sim_param.environment.obstacle.probability = 0.0
    
    # 探査設定
    sim_param.explore.robotNum = 20
    sim_param.explore.coordinate.x = 10.0
    sim_param.explore.coordinate.y = 10.0
    sim_param.explore.boundary.inner = 0.0
    sim_param.explore.boundary.outer = 20.0
    
    # ログ設定（GIF無効で高速化）
    sim_param.robot_logging.save_robot_data = False
    sim_param.robot_logging.save_position = False
    sim_param.robot_logging.save_collision = False
    
    return sim_param

def setup_test_agent():
    """テスト用エージェント設定"""
    agent_param = AgentParam()
    
    # SystemAgent設定（学習なし、分岐・統合あり）
    system_param = SystemAgentParam()
    system_param.learningParameter = None  # 学習なし
    system_param.branch_condition.branch_enabled = True
    system_param.integration_condition.integration_enabled = True
    
    # デバッグ設定を有効化
    system_param.debug = DebugParam()
    system_param.debug.log_branch_events = True
    system_param.debug.log_integration_events = True
    system_param.debug.enable_debug_log = True
    
    agent_param.system_agent_param = system_param
    
    # SwarmAgent設定（学習なし）
    swarm_param = SwarmAgentParam()
    swarm_param.isLearning = False
    swarm_param.learningParameter = None
    agent_param.swarm_agent_param = swarm_param
    
    return agent_param

def run_test():
    """テスト実行"""
    print("🚀 分岐・統合テスト開始 (クールダウンなし)")
    
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
        from envs.env import Env
        env = Env(sim_param)
        print("✓ 環境作成完了")
        
        # 4. エージェント作成
        print("4. エージェント作成中...")
        from core.factories import create_agent
        agent = create_agent(env, agent_param)
        print("✓ エージェント作成完了")
        
        # 5. シミュレーション実行
        print("5. シミュレーション実行中...")
        print(f"エピソード数: {sim_param.episodeNum}, ステップ数: {sim_param.maxStepsPerEpisode}")
        print()
        
        for episode in range(sim_param.episodeNum):
            print(f"=== エピソード {episode + 1}/{sim_param.episodeNum} ===")
            
            # 環境リセット
            env.reset()
            
            for step in range(sim_param.maxStepsPerEpisode):
                print(f"\n--- ステップ {step + 1}/{sim_param.maxStepsPerEpisode} ---")
                
                # エージェント行動
                actions = agent.get_action()
                
                # 環境ステップ
                observations, rewards, done, truncated, info = env.step(actions)
                
                # 探査率確認
                exploration_rate = env.get_exploration_rate()
                print(f"探査率: {exploration_rate:.4f}")
                
                # 終了条件チェック
                if done or truncated:
                    print(f"エピソード終了 (done={done}, truncated={truncated})")
                    break
            
            print(f"エピソード {episode + 1} 完了\n")
        
        print("🎉 テスト完了！")
        return True
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    if success:
        print("\n✅ 分岐・統合テスト成功")
    else:
        print("\n❌ 分岐・統合テスト失敗") 