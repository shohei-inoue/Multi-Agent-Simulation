#!/usr/bin/env python3
"""
Config_B 検証スクリプト (障害物密度: 0.003)
System学習なし + Swarm学習あり
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 非インタラクティブバックエンドを使用

def convert_numpy_types(obj):
    """numpy型をPython型に変換してJSONシリアライゼーションを可能にする"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_verification_environment():
    """検証用環境設定"""
    from params.simulation import SimulationParam
    
    sim_param = SimulationParam()
    
    # 基本設定
    sim_param.episodeNum = 10
    sim_param.maxStepsPerEpisode = 50
    
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

def setup_config_B_agent():
    """Config_B用エージェント設定"""
    from params.agent import AgentParam
    from params.system_agent import SystemAgentParam
    from params.swarm_agent import SwarmAgentParam
    from params.learning import LearningParameter
    
    agent_param = AgentParam()
    
    # SystemAgent設定
    system_param = SystemAgentParam()
    system_param.learningParameter = None
    system_param.branch_condition.branch_enabled = False
    system_param.integration_condition.integration_enabled = False
    agent_param.system_agent_param = system_param
    
    # SwarmAgent設定
    swarm_param = SwarmAgentParam()
    swarm_param.isLearning = True
    swarm_param.learningParameter = LearningParameter(
        type="A2C",
        model="actor-critic",
        optimizer="adam",
        gamma=0.99,
        learningLate=0.001,
        nStep=5
    )
    agent_param.swarm_agent_params = [swarm_param]
    
    return agent_param

def run_verification():
    """検証実行"""
    print("=== Config_B 検証開始 (障害物密度: 0.003) ===")
    
    try:
        # 1. 環境設定
        print("1. 環境設定中...")
        sim_param = setup_verification_environment()
        print("✓ 環境設定完了")
        
        # 2. エージェント設定
        print("2. エージェント設定中...")
        agent_param = setup_config_B_agent()
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
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trained_models/config_b/swarm_agent_model_1.h5")
        if not os.path.exists(model_path):
            print(f"❌ モデルファイルが見つかりません: {model_path}")
            return
        
        from keras.utils import custom_object_scope
        from models.swarm_actor_critic import SwarmActorCritic
        
        # 学習済みモデルを直接読み込み
        try:
            with custom_object_scope({'SwarmActorCritic': SwarmActorCritic}):
                from keras.models import load_model
                trained_model = load_model(model_path)
            
            # 各スウォームエージェントに学習済みモデルを設定
            for swarm_id, agent in swarm_agents.items():
                agent.model = trained_model
                print(f"  ✓ SwarmAgent {swarm_id} に学習済みモデルを設定完了")
        except Exception as e:
            print(f"⚠️ モデル読み込みエラー: {e}")
            print("新しくモデルを作成して学習なしモードで実行します")
            # 学習なしモードに切り替え
            for swarm_id, agent in swarm_agents.items():
                agent.isLearning = False
                agent.learningParameter = None
                print(f"  ✓ SwarmAgent {swarm_id} を学習なしモードに設定")
        
        # 6. SystemAgentを環境に設定
        print("6. SystemAgentを環境に設定中...")
        env.set_system_agent(system_agent)
        print("✓ SystemAgent設定完了")
        
        # 結果保存用ディレクトリ作成
        output_dir = "verification_results/Config_B_obstacle_0.003"
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ 出力ディレクトリ作成: {output_dir}")
        
        # 7. エピソード実行
        results = []
        for episode in range(sim_param.episodeNum):
            print(f"\n--- エピソード {episode + 1}/{sim_param.episodeNum} ---")
            
            # エピソード開始時の状態初期化
            if episode > 0:  # 2エピソード目以降
                print(f"  🔄 エピソード間状態初期化中...")
                
                # SystemAgentの状態を初期化
                system_agent.next_swarm_id = 1  # 群IDをリセット
                system_agent.current_swarm_count = 0  # 群数をリセット
                system_agent.swarm_agents.clear()  # 登録された群エージェントをクリア
                
                # 既存のSwarmAgentをクリア
                swarm_agents.clear()
                
                # 新しいエージェントを作成（エピソード1と同様）
                from agents.agent_factory import create_initial_agents
                system_agent, swarm_agents = create_initial_agents(env, agent_param)
                
                # SystemAgentを環境に再設定
                env.set_system_agent(system_agent)
                
                print(f"  ✓ 状態初期化完了 - SwarmAgents: {len(swarm_agents)}")
            
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
                'steps_taken': 0,
                'step_details': []  # 詳細なstepデータを追加
            }
            
            # ステップ実行
            for step in range(sim_param.maxStepsPerEpisode):
                if step % 20 == 0:  # 20ステップごとにログ
                    print(f"    ステップ {step + 1}/{sim_param.maxStepsPerEpisode}")
                
                # 各SwarmAgentの行動取得（main.pyと同様の順序）
                swarm_actions = {}
                for swarm_id, agent in swarm_agents.items():
                    # env.swarmsから該当する群を探す
                    swarm_exists = any(swarm.swarm_id == swarm_id for swarm in env.swarms)
                    if swarm_exists:  # 存在する群のみ
                        # 適切な状態を取得（main.pyと同様）
                        swarm_observation = env.get_swarm_agent_observation(swarm_id)
                        action_tensor, action_dict = agent.get_action(swarm_observation)
                        swarm_actions[swarm_id] = action_dict
                    else:
                        if step == 0:
                            print(f"    Swarm {swarm_id} not found in env.swarms")
                
                # SystemAgentの行動取得と実行（分岐・統合判断）
                system_observation = env.get_system_agent_observation()
                system_action = system_agent.get_action(system_observation)
                
                # SystemAgentのアクションを実行（分岐・統合処理）
                if system_action and isinstance(system_action, dict):
                    action_type = system_action.get('action_type', 0)
                    target_swarm_id = system_action.get('target_swarm_id', 0)
                    
                    if action_type == 1:  # 分岐
                        print(f"    🔀 分岐処理実行: swarm {target_swarm_id}")
                        # 分岐処理を実行
                        system_agent._execute_branch({
                            'swarm_id': target_swarm_id,
                            'valid_directions': [0, 45, 90, 135, 180, 225, 270, 315]  # デフォルトの方向
                        })
                    elif action_type == 2:  # 統合
                        print(f"    🔗 統合処理実行: swarm {target_swarm_id}")
                        # 統合処理を実行
                        system_agent._execute_integration({
                            'swarm_id': target_swarm_id
                        })
                
                # 環境のステップ実行
                next_state, rewards, done, truncated, info = env.step(swarm_actions)
                
                # フレームキャプチャ（GIF生成用）
                try:
                    env.capture_frame()
                except Exception as e:
                    print(f"    フレームキャプチャエラー（無視）: {e}")
                
                # 探査率確認
                exploration_rate = env.get_exploration_rate()
                episode_data['final_exploration_rate'] = exploration_rate
                episode_data['steps_taken'] = step + 1
                
                # 詳細なstepデータを記録
                step_detail = {
                    'step': step,
                    'exploration_rate': exploration_rate,
                    'reward': rewards if isinstance(rewards, (int, float)) else np.mean(list(rewards.values())) if rewards else 0.0,
                    'done': done,
                    'truncated': truncated
                }
                
                # 環境から詳細情報を取得
                if hasattr(env, 'get_exploration_info'):
                    exploration_info = env.get_exploration_info()
                    step_detail.update({
                        'explored_area': exploration_info.get('explored_area', 0),
                        'total_area': exploration_info.get('total_area', 0),
                        'new_explored_area': exploration_info.get('new_explored_area', 0)
                    })
                
                # 衝突情報を取得
                if hasattr(env, 'get_collision_info'):
                    collision_info = env.get_collision_info()
                    step_detail.update({
                        'agent_collision_flag': collision_info.get('agent_collision_flag', 0),
                        'follower_collision_count': collision_info.get('follower_collision_count', 0)
                    })
                
                # ロボット位置情報を取得（サンプリング）
                if hasattr(env, 'get_robot_positions') and step % 10 == 0:  # 10ステップごとにサンプリング
                    robot_positions = env.get_robot_positions()
                    step_detail['robot_positions'] = robot_positions
                
                episode_data['step_details'].append(step_detail)
                
                # 目標達成チェック
                if exploration_rate >= 0.8:
                    episode_data['steps_to_target'] = step + 1
                    print(f"    目標達成！ステップ {step + 1}で80%探査に到達")
                    break
                
                if done or truncated:
                    print(f"    エピソード終了（ステップ {step + 1}）")
                    break
            
            # エピソード終了時にGIF保存
            try:
                env.end_episode(output_dir)
                print(f"    GIF保存完了")
            except Exception as e:
                print(f"    GIF保存エラー（無視）: {e}")
            
            results.append(episode_data)
            print(f"  エピソード {episode + 1} 完了 - 探査率: {episode_data['final_exploration_rate']:.3f}")
        
        # 8. 結果集計
        print("\n=== 結果集計 ===")
        final_result = {
            'config': 'Config_B',
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
            json.dump(convert_numpy_types(final_result), f, indent=2, ensure_ascii=False)
        
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
    print("=== Config_B 検証開始 (障害物密度: 0.003) ===")
    print(f"開始時刻: {datetime.now()}")
    
    success = run_verification()
    
    print(f"\n終了時刻: {datetime.now()}")
    if success:
        print("🎉 検証が正常に完了しました！")
    else:
        print("❌ 検証が失敗しました。")
        sys.exit(1)
