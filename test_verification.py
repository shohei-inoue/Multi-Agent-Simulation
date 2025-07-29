#!/usr/bin/env python3
"""
簡単な検証テストスクリプト
基本的な環境初期化とエージェント作成をテスト
"""

import os
import sys
import traceback
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from params.simulation import SimulationParam
from params.agent import AgentParam
from params.system_agent import SystemAgentParam
from params.swarm_agent import SwarmAgentParam
from agents.agent_factory import create_initial_agents
from envs.env import Env

def test_basic_setup():
    """基本的な環境とエージェントのセットアップをテスト"""
    print("=== 基本セットアップテスト開始 ===")
    
    try:
        # 1. シミュレーションパラメータの設定
        print("1. シミュレーションパラメータ設定中...")
        sim_param = SimulationParam()
        sim_param.episodeNum = 2  # テスト用に少なく
        sim_param.maxStepsPerEpisode = 10  # テスト用に少なく
        
        # 環境設定
        sim_param.environment.map.width = 50  # テスト用に小さく
        sim_param.environment.map.height = 50
        sim_param.environment.obstacle.probability = 0.0
        sim_param.explore.robotNum = 5  # テスト用に少なく
        
        # ログ設定
        sim_param.robot_logging.save_robot_data = True
        sim_param.robot_logging.save_position = True
        sim_param.robot_logging.save_collision = True
        sim_param.robot_logging.sampling_rate = 1.0
        
        print("✓ シミュレーションパラメータ設定完了")
        
        # 2. 環境の初期化
        print("2. 環境初期化中...")
        env = Env(sim_param)
        print("✓ 環境初期化完了")
        
        # 3. エージェント設定
        print("3. エージェント設定中...")
        agent_param = AgentParam()
        
        # SystemAgent設定（学習なし、分岐なし）
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
        
        print("✓ エージェント設定完了")
        
        # 4. エージェント作成
        print("4. エージェント作成中...")
        system_agent, swarm_agents = create_initial_agents(env, agent_param)
        print(f"✓ エージェント作成完了 - SystemAgent: {type(system_agent)}, SwarmAgents: {len(swarm_agents)}")
        
        # 5. 簡単なテスト実行
        print("5. 簡単なテスト実行中...")
        env.reset()
        
        # 出力ディレクトリ作成
        output_dir = f"test_results/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # エピソード開始
        env.start_episode(0)
        
        # 数ステップ実行
        for step in range(5):
            print(f"  ステップ {step + 1}/5")
            
            # 状態取得
            state = env.get_state()
            system_obs = env.get_system_agent_observation()
            swarm_obs = env.get_swarm_agent_observation(0)
            
            # デフォルト行動
            swarm_actions = {0: {'theta': 0.0, 'th': 0.5, 'k_e': 10.0, 'k_c': 5.0}}
            
            # 環境ステップ
            next_state, reward, done, truncated, info = env.step(swarm_actions)
            
            # 探査率取得
            exploration_rate = env.get_exploration_rate()
            print(f"    探査率: {exploration_rate:.3f}")
            
            if done or truncated:
                print("    エピソード終了")
                break
        
        # エピソード終了
        env.end_episode(output_dir)
        
        print("✓ テスト実行完了")
        print(f"✓ 結果保存先: {output_dir}")
        
        # 結果ファイル確認
        result_files = os.listdir(output_dir)
        print(f"✓ 生成されたファイル: {result_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("詳細:")
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("=== 検証テスト開始 ===")
    
    success = test_basic_setup()
    
    if success:
        print("\n🎉 テスト成功！基本的な処理は動作しています。")
    else:
        print("\n❌ テスト失敗。問題を修正する必要があります。")
    
    print("=== テスト終了 ===")

if __name__ == "__main__":
    main() 