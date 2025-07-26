#!/usr/bin/env python3
"""
事前学習済みモデル作成スクリプト
SystemAgentとSwarmAgentの学習済みモデルを作成
"""

import sys
import os
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from params.simulation import SimulationParam
from params.agent import AgentParam
from params.system_agent import SystemAgentParam, LearningParameter as SystemLearningParameter
from params.swarm_agent import SwarmAgentParam, LearningParameter as SwarmLearningParameter
from agents.agent_factory import create_initial_agents
from envs.env import Env
import tensorflow as tf
import numpy as np


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='事前学習済みモデル作成')
    
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=200,
        help='学習エピソード数 (default: 200)'
    )
    
    parser.add_argument(
        '--steps', 
        type=int, 
        default=50,
        help='エピソードあたりの最大ステップ数 (default: 50)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='pretrained_models',
        help='モデル保存ディレクトリ (default: pretrained_models)'
    )
    
    parser.add_argument(
        '--train-system', 
        action='store_true',
        help='SystemAgentモデルを学習'
    )
    
    parser.add_argument(
        '--train-swarm', 
        action='store_true',
        help='SwarmAgentモデルを学習'
    )
    
    parser.add_argument(
        '--train-both', 
        action='store_true',
        help='両方のモデルを学習（デフォルト）'
    )
    
    return parser.parse_args()


def setup_training_config():
    """学習設定を作成"""
    # 基本設定
    sim_param = SimulationParam()
    
    # 環境設定
    sim_param.environment.map.width = 100
    sim_param.environment.map.height = 200
    sim_param.environment.obstacle.probability = 0.0  # 学習時は障害物なしで開始
    
    # 探査設定
    sim_param.explore.robotNum = 20
    sim_param.explore.coordinate.x = 50.0  # マップ中央に配置
    sim_param.explore.coordinate.y = 100.0
    sim_param.explore.boundary.outer = 5.0  # 移動距離を短く設定
    
    # エージェント設定
    sim_param.agent.episodeNum = 1500
    sim_param.agent.maxStepsPerEpisode = 50
    
    return sim_param


def check_convergence(reward_history, exploration_rates, window_size=50):
    """学習収束をチェック"""
    if len(reward_history) < window_size:
        return False, {}
    
    recent_rewards = reward_history[-window_size:]
    recent_exploration = exploration_rates[-window_size:]
    
    # 報酬の改善率
    reward_improvement = (recent_rewards[-1] - recent_rewards[0]) / window_size
    
    # 探査率の標準偏差
    exploration_std = np.std(recent_exploration)
    
    # 最低探査率
    min_exploration = np.min(recent_exploration)
    
    # 収束判定
    is_converged = (
        abs(reward_improvement) < 0.01 and  # 報酬改善率 < 1%
        exploration_std < 0.05 and          # 探査率の標準偏差 < 5%
        min_exploration > 0.75              # 最低探査率 > 75%
    )
    
    convergence_info = {
        'reward_improvement': reward_improvement,
        'exploration_std': exploration_std,
        'min_exploration': min_exploration,
        'is_converged': is_converged
    }
    
    return is_converged, convergence_info


def train_system_agent_model(env, output_dir: str, episodes: int, steps: int):
    """SystemAgentモデルを学習（収束判定付き）"""
    print("Training SystemAgent model...")
    
    # SystemAgent設定
    agent_param = AgentParam()
    system_param = SystemAgentParam()
    
    # 学習設定（元の設計を尊重）
    system_param.learningParameter = SystemLearningParameter()
    system_param.branch_condition.branch_enabled = True
    system_param.integration_condition.integration_enabled = True
    agent_param.system_agent_param = system_param
    
    # SwarmAgent設定（学習なし）
    swarm_param = SwarmAgentParam()
    swarm_param.isLearning = False
    swarm_param.learningParameter = None
    agent_param.swarm_agent_params = [swarm_param]
    
    # エージェント作成
    system_agent, swarm_agents = create_initial_agents(env, agent_param)
    
    # 学習実行
    best_reward = float('-inf')
    best_model_weights = None
    reward_history = []
    exploration_rates = []
    patience_counter = 0
    max_patience = 100  # 早期停止のパティエンス
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_exploration = 0.0
        
        for step in range(steps):
            state = env.get_system_agent_observation()
            
            # SystemAgent行動決定
            system_action = system_agent.get_action(state, episode)
            
            # SwarmAgent行動決定（ランダム）
            swarm_actions = {}
            for swarm_id, _ in swarm_agents.items():
                swarm_actions[swarm_id] = {
                    'theta': np.random.uniform(0, 2*np.pi),
                    'th': 0.5,
                    'k_e': 10.0,
                    'k_c': 5.0
                }
            
            # 環境更新
            next_state, reward, done, truncated, info = env.step(swarm_actions)
            
            episode_reward += reward
            
            # 探査率を記録
            episode_exploration = env.get_exploration_rate()
            
            if done:
                break
        
        # 学習更新
        system_agent.train()
        
        # 履歴を記録
        reward_history.append(episode_reward)
        exploration_rates.append(episode_exploration)
        
        # ベストモデル保存
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_weights = system_agent.model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 収束チェック（50エピソードごと）
        if episode % 50 == 0 and episode >= 300:  # 最低300エピソードは学習
            is_converged, conv_info = check_convergence(reward_history, exploration_rates)
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                  f"Exploration = {episode_exploration:.3f}, "
                  f"Convergence = {conv_info}")
            
            if is_converged:
                print(f"SystemAgent converged at episode {episode}")
                break
        
        # 早期停止チェック
        if patience_counter >= max_patience and episode >= 300:
            print(f"SystemAgent early stopping at episode {episode} (no improvement for {max_patience} episodes)")
            break
    
    # ベストモデルを保存
    if best_model_weights is not None:
        system_agent.model.set_weights(best_model_weights)
        model_path = os.path.join(output_dir, "system_agent_model.keras")
        system_agent.model.save_weights(model_path)
        print(f"SystemAgent model saved to {model_path}")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Final episode: {len(reward_history)}")


def train_swarm_agent_model(env, output_dir: str, episodes: int, steps: int):
    """SwarmAgentモデルを学習（収束判定付き）"""
    print("Training SwarmAgent model...")
    
    # SystemAgent設定（学習なし）
    agent_param = AgentParam()
    system_param = SystemAgentParam()
    system_param.learningParameter = None
    system_param.branch_condition.branch_enabled = False
    system_param.integration_condition.integration_enabled = False
    agent_param.system_agent_param = system_param
    
    # SwarmAgent設定
    swarm_param = SwarmAgentParam()
    swarm_param.isLearning = True
    swarm_param.learningParameter = SwarmLearningParameter()
    agent_param.swarm_agent_params = [swarm_param]
    
    # エージェント作成
    system_agent, swarm_agents = create_initial_agents(env, agent_param)
    
    # 学習実行
    best_reward = float('-inf')
    best_model_weights = None
    reward_history = []
    exploration_rates = []
    patience_counter = 0
    max_patience = 150  # SwarmAgentはより長いパティエンス
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_exploration = 0.0
        
        for step in range(steps):
            # SwarmAgent行動決定
            swarm_actions = {}
            for swarm_id, agent in swarm_agents.items():
                state = env.get_swarm_agent_observation(swarm_id)
                action = agent.get_action(state, episode)
                swarm_actions[swarm_id] = action
            
            # 環境更新
            next_state, reward, done, truncated, info = env.step(swarm_actions)
            
            episode_reward += reward
            
            # 探査率を記録
            episode_exploration = env.get_exploration_rate()
            
            if done:
                break
        
        # 学習更新
        for agent in swarm_agents.values():
            agent.train()
        
        # 履歴を記録
        reward_history.append(episode_reward)
        exploration_rates.append(episode_exploration)
        
        # ベストモデル保存
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_weights = list(swarm_agents.values())[0].model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 収束チェック（50エピソードごと）
        if episode % 50 == 0 and episode >= 300:  # 最低300エピソードは学習
            is_converged, conv_info = check_convergence(reward_history, exploration_rates)
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                  f"Exploration = {episode_exploration:.3f}, "
                  f"Convergence = {conv_info}")
            
            if is_converged:
                print(f"SwarmAgent converged at episode {episode}")
                break
        
        # 早期停止チェック
        if patience_counter >= max_patience and episode >= 300:
            print(f"SwarmAgent early stopping at episode {episode} (no improvement for {max_patience} episodes)")
            break
    
    # ベストモデルを保存
    if best_model_weights is not None:
        list(swarm_agents.values())[0].model.set_weights(best_model_weights)
        model_path = os.path.join(output_dir, "swarm_agent_model.keras")
        list(swarm_agents.values())[0].model.save_weights(model_path)
        print(f"SwarmAgent model saved to {model_path}")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Final episode: {len(reward_history)}")


def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 学習設定
    sim_param = setup_training_config()
    
    print("=" * 60)
    print("事前学習済みモデル作成")
    print("=" * 60)
    print(f"学習エピソード数: {args.episodes}")
    print(f"最大ステップ数: {args.steps}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print()
    
    # 環境作成
    env = Env(sim_param)
    
    # 学習実行
    if args.train_system or args.train_both:
        train_system_agent_model(env, args.output_dir, args.episodes, args.steps)
    
    if args.train_swarm or args.train_both:
        train_swarm_agent_model(env, args.output_dir, args.episodes, args.steps)
    
    print("\n事前学習完了!")
    print(f"モデルは {args.output_dir} に保存されました")


if __name__ == "__main__":
    main() 