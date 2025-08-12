#!/usr/bin/env python3
"""
Config_D 学習モデル作成スクリプト
SystemAgent: 学習あり、分岐・統合あり
SwarmAgent: 学習あり
"""

import os
import sys
import numpy as np
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_training_environment():
    """学習用環境設定"""
    from params.simulation import SimulationParam
    
    sim_param = SimulationParam()
    
    # 基本設定
    sim_param.episodeNum = 1000  # 学習用に1000エピソード
    sim_param.maxStepsPerEpisode = 100
    
    # 環境設定
    sim_param.environment.map.width = 200
    sim_param.environment.map.height = 100
    sim_param.environment.obstacle.probability = 0.0
    
    # 探査設定
    sim_param.explore.robotNum = 20
    sim_param.explore.coordinate.x = 10.0
    sim_param.explore.coordinate.y = 10.0
    sim_param.explore.boundary.inner = 0.0
    sim_param.explore.boundary.outer = 20.0
    
    # ログ設定（学習中は無効化）
    sim_param.robot_logging.save_robot_data = False
    sim_param.robot_logging.save_position = False
    sim_param.robot_logging.save_collision = False
    sim_param.robot_logging.sampling_rate = 1.0
    
    return sim_param

def setup_config_d_agent():
    """Config_D用エージェント設定"""
    from params.agent import AgentParam
    from params.system_agent import SystemAgentParam
    from params.swarm_agent import SwarmAgentParam
    from params.learning import LearningParameter
    
    agent_param = AgentParam()
    
    # SystemAgent設定（学習あり）
    system_param = SystemAgentParam()
    system_param.isLearning = True
    system_param.learningParameter = LearningParameter(
        type="A2C",
        model="actor-critic",  # モデルは後で設定
        optimizer="adam",  # オプティマイザは後で設定
        gamma=0.99,
        learningLate=0.001,
        nStep=5
    )
    agent_param.system_agent_param = system_param
    
    # SwarmAgent設定（学習あり）
    swarm_param = SwarmAgentParam()
    swarm_param.isLearning = True
    swarm_param.learningParameter = LearningParameter(
        type="A2C",
        model=None,  # モデルは後で設定
        optimizer=None,  # オプティマイザは後で設定
        gamma=0.99,
        learningLate=0.001,
        nStep=5
    )
    agent_param.swarm_agent_params = [swarm_param]
    
    return agent_param

def run_training():
    """学習実行"""
    print("=== Config_D 学習開始 ===")
    
    try:
        # 1. 環境設定
        print("1. 環境設定中...")
        sim_param = setup_training_environment()
        print("✓ 環境設定完了")
        
        # 2. エージェント設定
        print("2. エージェント設定中...")
        agent_param = setup_config_d_agent()
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
        
        # 5. SystemAgentを環境に設定
        print("5. SystemAgentを環境に設定中...")
        env.set_system_agent(system_agent)
        print("✓ SystemAgent設定完了")
        
        # 学習結果保存用ディレクトリ作成
        output_dir = "trained_models/config_d"
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ 出力ディレクトリ作成: {output_dir}")
        
        # 6. 学習実行
        print("6. 学習開始...")
        
        # 学習進捗の監視用変数
        best_exploration_rate = 0.0
        exploration_rates = []
        episode_rewards_history = []
        system_rewards_history = []
        
        for episode in range(sim_param.episodeNum):
            if episode % 50 == 0:  # 50エピソードごとにログ
                print(f"  エピソード {episode + 1}/{sim_param.episodeNum}")
            
            # エピソード開始
            env.start_episode(episode)
            state = env.reset()
            
            episode_rewards = {swarm_id: 0.0 for swarm_id in swarm_agents.keys()}
            system_reward = 0.0
            
            # ステップ実行
            for step in range(sim_param.maxStepsPerEpisode):
                # SystemAgentの行動取得
                system_observation = env.get_system_agent_observation()
                system_action = system_agent.get_action(system_observation)
                
                # SwarmAgentの行動取得
                swarm_actions = {}
                for swarm_id, agent in swarm_agents.items():
                    swarm_observation = env.get_swarm_agent_observation(swarm_id)
                    action_result = agent.get_action(swarm_observation)
                    
                    # get_actionは辞書を返すので、そのまま使用
                    if isinstance(action_result, dict):
                        swarm_actions[swarm_id] = action_result
                    else:
                        # タプルの場合は最初の要素を使用
                        swarm_actions[swarm_id] = action_result[0] if isinstance(action_result, tuple) else action_result
                
                # ステップ実行
                next_state, rewards, done, truncated, info = env.step(swarm_actions)
                
                # 報酬の蓄積
                if isinstance(rewards, dict):
                    for swarm_id, reward in rewards.items():
                        if swarm_id in episode_rewards:
                            episode_rewards[swarm_id] += reward
                else:
                    # rewardsが辞書でない場合（タプルなど）は、最初のswarm_idに報酬を追加
                    if swarm_agents:
                        first_swarm_id = list(swarm_agents.keys())[0]
                        if first_swarm_id in episode_rewards:
                            episode_rewards[first_swarm_id] += rewards if isinstance(rewards, (int, float)) else 0.0
                
                # SystemAgentの報酬（分岐・統合の成功度に基づく）
                system_reward += info.get('system_reward', 0.0)
                
                # 探査率確認
                exploration_rate = env.get_exploration_rate()
                
                # 目標達成チェック
                if exploration_rate >= 0.8:
                    break
                
                if done or truncated:
                    break
            
            # 学習進捗の記録
            exploration_rates.append(exploration_rate)
            episode_rewards_history.append(np.mean(list(episode_rewards.values())))
            system_rewards_history.append(system_reward)
            
            # ベスト記録の更新
            if exploration_rate > best_exploration_rate:
                best_exploration_rate = exploration_rate
            
            # 進捗表示（50エピソードごと）
            if episode % 50 == 0:
                avg_reward = np.mean(list(episode_rewards.values()))
                avg_exploration = np.mean(exploration_rates[-50:]) if len(exploration_rates) >= 50 else np.mean(exploration_rates)
                avg_system_reward = np.mean(system_rewards_history[-50:]) if len(system_rewards_history) >= 50 else np.mean(system_rewards_history)
                print(f"    平均報酬: {avg_reward:.3f}, 平均探査率: {avg_exploration:.3f}, ベスト探査率: {best_exploration_rate:.3f}, 平均システム報酬: {avg_system_reward:.3f}")
            
            # 学習の早期終了条件（500エピソード以降で探査率が安定した場合）
            if episode >= 500 and len(exploration_rates) >= 100:
                recent_avg = np.mean(exploration_rates[-100:])
                if recent_avg > 0.7 and abs(recent_avg - np.mean(exploration_rates[-200:-100])) < 0.02:
                    print(f"    学習が収束しました。エピソード {episode + 1} で終了")
                    break
        
        # 7. モデル保存
        print("7. モデル保存中...")
        
        # SystemAgentモデル保存
        system_model_path = os.path.join(output_dir, "system_agent_model.h5")
        system_agent.save_model(system_model_path)
        print(f"  ✓ SystemAgent モデル保存完了: {system_model_path}")
        
        # SwarmAgentモデル保存
        for swarm_id, agent in swarm_agents.items():
            model_path = os.path.join(output_dir, f"swarm_agent_model_{swarm_id}.h5")
            agent.save_model(model_path)
            print(f"  ✓ SwarmAgent {swarm_id} モデル保存完了: {model_path}")
        
        print("✓ 学習完了")
        return True
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Config_D 学習開始 ===")
    print(f"開始時刻: {datetime.now()}")
    
    success = run_training()
    
    print(f"\n終了時刻: {datetime.now()}")
    if success:
        print("🎉 学習が正常に完了しました！")
    else:
        print("❌ 学習が失敗しました。")
        sys.exit(1) 