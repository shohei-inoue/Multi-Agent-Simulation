#!/usr/bin/env python3
"""
Config_E 検証スクリプト (障害物密度: 0.003)
System学習あり + Swarm学習なし + 分岐・統合あり
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_verification_environment():
    """検証用環境設定"""
    from params.simulation import SimulationParam
    
    sim_param = SimulationParam()
    
    # 基本設定
    sim_param.episodeNum = 3
    sim_param.maxStepsPerEpisode = 200
    
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

def setup_config_E_agent():
    """Config_E用エージェント設定"""
    from params.agent import AgentParam
    from params.system_agent import SystemAgentParam
    from params.swarm_agent import SwarmAgentParam
    from params.learning import LearningParameter
    
    agent_param = AgentParam()
    
    # SystemAgent設定
    system_param = SystemAgentParam()
    system_param.learningParameter = LearningParameter()
    system_param.branch_condition.branch_enabled = True
    system_param.integration_condition.integration_enabled = True
    agent_param.system_agent_param = system_param
    
    # SwarmAgent設定
    swarm_param = SwarmAgentParam()
    swarm_param.isLearning = False
    swarm_param.learningParameter = None
    agent_param.swarm_agent_params = [swarm_param]
    
    return agent_param

def run_verification():
    """検証実行"""
    print("=== Config_E 検証開始 (障害物密度: 0.003) ===")
    
    try:
        # 1. 環境設定
        print("1. 環境設定中...")
        sim_param = setup_verification_environment()
        print("✓ 環境設定完了")
        
        # 2. エージェント設定
        print("2. エージェント設定中...")
        agent_param = setup_config_E_agent()
        print("✓ エージェント設定完了")
        
        # 3. 環境作成
        print("3. 環境作成中...")
        from envs.env import Env
        env = Env(sim_param)
        print("✓ 環境作成完了")
        
        # 4. エージェント作成
        print("4. エージェント作成中...")
        from agents.agent_factory import create_initial_agents
        system_agent, swarm_agents = create_initial_agents(env, agent_param)
        print(f"✓ エージェント作成完了 - SwarmAgents: {len(swarm_agents)}")
        
        # 5. 学習済みモデルの読み込み
        print("5. 学習済みモデル読み込み中...")
        # SystemAgentモデル読み込み
        system_model_path = "trained_models/system_agent_model_e.h5"
        if os.path.exists(system_model_path):
            system_agent.load_model(system_model_path)
            print(f"  ✓ SystemAgent モデル読み込み完了")
        else:
            print(f"  ⚠️ SystemAgent モデルファイルが見つかりません: {{system_model_path}}")
        
        # 6. SystemAgentを環境に設定
        print("6. SystemAgentを環境に設定中...")
        env.set_system_agent(system_agent)
        print("✓ SystemAgent設定完了")
        
        # 結果保存用ディレクトリ作成
        output_dir = "verification_results/Config_E_obstacle_0.003"
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ 出力ディレクトリ作成: {output_dir}")
        
        # 7. エピソード実行
        results = []
        for episode in range(sim_param.episodeNum):
            print(f"\n--- エピソード {episode + 1}/{sim_param.episodeNum} ---")
            
            # エピソード開始
            env.start_episode(episode)
            state = env.reset()
            print(f"  環境リセット完了")
            
            # GIF生成のためのフレームキャプチャ開始
            env.capture_frame()
            
            episode_data = {
                'episode': episode + 1,
                'steps_to_target': None,
                'final_exploration_rate': 0.0,
                'steps_taken': 0
            }
            
            # ステップ実行
            for step in range(sim_param.maxStepsPerEpisode):
                if step % 20 == 0:  # 20ステップごとにログ
                    print(f"    ステップ {step + 1}/{sim_param.maxStepsPerEpisode}")
                
                # SystemAgentの行動取得（分岐・統合判断）
                system_observation = env.get_system_agent_observation()
                system_action = system_agent.get_action(system_observation)
                
                # SwarmAgentの行動取得
                swarm_actions = {{}}
                for swarm_id, agent in swarm_agents.items():
                    # デフォルト行動（学習なし）
                    swarm_actions[swarm_id] = {{
                        'theta': np.random.uniform(0, 2*np.pi),
                        'th': 0.5,
                        'k_e': 10.0,
                        'k_c': 5.0
                    }}
                
                # ステップ実行
                next_state, rewards, done, truncated, info = env.step(swarm_actions)
                
                # 探査率確認
                exploration_rate = env.get_exploration_rate()
                episode_data['final_exploration_rate'] = exploration_rate
                episode_data['steps_taken'] = step + 1
                
                # 目標達成チェック
                if exploration_rate >= 0.8:
                    episode_data['steps_to_target'] = step + 1
                    print(f"    目標達成！ステップ {step + 1}で80%探査に到達")
                    break
                
                if done or truncated:
                    print(f"    エピソード終了（ステップ {step + 1}）")
                    break
            
            results.append(episode_data)
            print(f"  エピソード {episode + 1} 完了 - 探査率: {episode_data['final_exploration_rate']:.3f}")
            
            # GIF生成のためのエピソード終了処理
            env.end_episode(output_dir)
        
        # 8. 結果集計
        print("\n=== 結果集計 ===")
        final_result = {
            'config': 'Config_E',
            'environment': {
                'map_size': f"{sim_param.environment.map.width}x{sim_param.environment.map.height}",
                'obstacle_density': sim_param.environment.obstacle.probability,
                'robot_count': sim_param.explore.robotNum
            },
            'episodes': results,
            'summary': {
                'total_episodes': len(results),
                'target_reached_episodes': len([r for r in results if r['steps_to_target'] is not None]),
                'average_exploration_rate': np.mean([r['final_exploration_rate'] for r in results]),
                'average_steps_taken': np.mean([r['steps_taken'] for r in results]),
                'std_exploration_rate': np.std([r['final_exploration_rate'] for r in results]),
                'std_steps_taken': np.std([r['steps_taken'] for r in results])
            }
        }
        
        # 9. 結果保存
        result_file = os.path.join(output_dir, "verification_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 結果保存完了: {result_file}")
        
        # 10. 結果表示
        print("\n=== 検証結果 ===")
        print(f"総エピソード数: {final_result['summary']['total_episodes']}")
        print(f"目標達成エピソード数: {final_result['summary']['target_reached_episodes']}")
        print(f"平均探査率: {final_result['summary']['average_exploration_rate']:.3f} ± {final_result['summary']['std_exploration_rate']:.3f}")
        print(f"平均ステップ数: {final_result['summary']['average_steps_taken']:.1f} ± {final_result['summary']['std_steps_taken']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Config_E 検証開始 (障害物密度: 0.003) ===")
    print(f"開始時刻: {datetime.now()}")
    
    success = run_verification()
    
    print(f"\n終了時刻: {datetime.now()}")
    if success:
        print("🎉 検証が正常に完了しました！")
    else:
        print("❌ 検証が失敗しました。")
        sys.exit(1)
