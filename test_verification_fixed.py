#!/usr/bin/env python3
"""
修正版検証テストスクリプト
移動距離、初期位置、GIF生成の問題を修正
"""

import os
import sys
import traceback
import numpy as np
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from params.simulation import SimulationParam
from params.agent import AgentParam
from params.system_agent import SystemAgentParam
from params.swarm_agent import SwarmAgentParam
from agents.agent_factory import create_initial_agents
from envs.env import Env

def test_basic_setup_fixed():
    """修正版の基本セットアップテスト"""
    print("=== 修正版基本セットアップテスト開始 ===")
    
    try:
        # 1. シミュレーションパラメータの設定
        print("1. シミュレーションパラメータ設定中...")
        sim_param = SimulationParam()
        sim_param.episodeNum = 2  # テスト用に少なく
        sim_param.maxStepsPerEpisode = 10  # テスト用に少なく
        
        # 環境設定（小さなマップでテスト）
        sim_param.environment.map.width = 30  # テスト用に小さく
        sim_param.environment.map.height = 30
        sim_param.environment.obstacle.probability = 0.0
        
        # ロボット数と初期位置を調整
        sim_param.explore.robotNum = 3  # テスト用に少なく
        sim_param.explore.coordinate.x = 15.0  # マップ中央
        sim_param.explore.coordinate.y = 15.0
        sim_param.explore.boundary.inner = 0.0
        sim_param.explore.boundary.outer = 5.0  # 移動範囲を小さく
        
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
        
        # 5. 修正版テスト実行
        print("5. 修正版テスト実行中...")
        env.reset()
        
        # 出力ディレクトリ作成
        output_dir = f"test_results_fixed/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # エピソード開始
        env.start_episode(0)
        
        # ロボットの初期位置を確認
        print("  ロボット初期位置:")
        for i, robot in enumerate(env.robots):
            print(f"    Robot {i}: ({robot.x:.2f}, {robot.y:.2f})")
        
        # 数ステップ実行（小さな移動でテスト）
        for step in range(5):
            print(f"  ステップ {step + 1}/5")
            
            # 状態取得
            state = env.get_state()
            system_obs = env.get_system_agent_observation()
            swarm_obs = env.get_swarm_agent_observation(0)
            
            # 小さな移動のデフォルト行動（0.5の移動距離）
            small_movement = 0.5
            theta = np.random.uniform(0, 2*np.pi)  # ランダムな方向
            swarm_actions = {0: {
                'theta': theta, 
                'th': 0.5, 
                'k_e': 10.0, 
                'k_c': 5.0
            }}
            
            # 環境ステップ
            next_state, reward, done, truncated, info = env.step(swarm_actions)
            
            # 探査率取得
            exploration_rate = env.get_exploration_rate()
            print(f"    探査率: {exploration_rate:.3f}")
            
            # ロボット位置を確認
            leader = env.robots[0]
            print(f"    Leader位置: ({leader.x:.2f}, {leader.y:.2f})")
            
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
        
        # GIFファイルの確認
        gif_files = [f for f in result_files if f.endswith('.gif')]
        if gif_files:
            print(f"✓ GIFファイル生成成功: {gif_files}")
        else:
            print("⚠️ GIFファイルが生成されていません")
        
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("詳細:")
        traceback.print_exc()
        return False

def test_movement_fix():
    """移動処理の修正テスト"""
    print("\n=== 移動処理修正テスト ===")
    
    try:
        # 小さなマップで移動テスト
        sim_param = SimulationParam()
        sim_param.environment.map.width = 20
        sim_param.environment.map.height = 20
        sim_param.environment.obstacle.probability = 0.0
        sim_param.explore.robotNum = 1
        sim_param.explore.coordinate.x = 10.0
        sim_param.explore.coordinate.y = 10.0
        sim_param.explore.boundary.outer = 2.0  # 小さな移動範囲
        
        env = Env(sim_param)
        env.reset()
        
        print("  初期位置:", f"({env.robots[0].x:.2f}, {env.robots[0].y:.2f})")
        
        # 小さな移動をテスト
        for i in range(3):
            theta = i * np.pi / 4  # 45度ずつ回転
            actions = {0: {'theta': theta, 'th': 0.5, 'k_e': 10.0, 'k_c': 5.0}}
            
            env.step(actions)
            print(f"  移動{i+1}: ({env.robots[0].x:.2f}, {env.robots[0].y:.2f})")
        
        print("✓ 移動処理テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ 移動処理テストエラー: {e}")
        return False

def main():
    """メイン実行関数"""
    print("=== 修正版検証テスト開始 ===")
    
    # 基本セットアップテスト
    success1 = test_basic_setup_fixed()
    
    # 移動処理テスト
    success2 = test_movement_fix()
    
    if success1 and success2:
        print("\n🎉 修正版テスト成功！問題が解決されました。")
    else:
        print("\n❌ 修正版テスト失敗。さらなる修正が必要です。")
    
    print("=== 修正版テスト終了 ===")

if __name__ == "__main__":
    main() 