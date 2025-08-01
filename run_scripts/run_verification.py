"""
検証実行スクリプト
学習済みモデルを使用して検証を実行し、結果を保存する
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

from core.application import Application
from params.simulation import SimulationParam
from params.agent import AgentParam
from params.system_agent import SystemAgentParam, LearningParameter as SystemLearningParameter
from params.swarm_agent import SwarmAgentParam, LearningParameter as SwarmLearningParameter
from agents.agent_factory import create_initial_agents
from envs.env import Env


@dataclass
class VerificationConfig:
    """検証設定"""
    name: str
    system_agent_learning: bool
    swarm_agent_learning: bool
    system_agent_branching: bool
    use_pretrained_models: bool = False
    num_episodes: int = 100
    max_steps_per_episode: int = 200
    target_exploration_rate: float = 0.8
    num_runs: int = 5
    
    def __str__(self):
        return f"{self.name}_SysL{self.system_agent_learning}_SwarmL{self.swarm_agent_learning}_Branch{self.system_agent_branching}"


@dataclass
class EnvironmentConfig:
    """環境設定"""
    map_width: int = 100
    map_height: int = 200
    obstacle_density: float = 0.0
    robot_count: int = 20
    
    def __str__(self):
        return f"Map{self.map_width}x{self.map_height}_Obs{self.obstacle_density}_Robot{self.robot_count}"


class VerificationRunner:
    """検証実行クラス"""
    
    def __init__(self):
        self.base_config = SimulationParam()
        self.results = {}
        
    def create_verification_configs(self) -> List[VerificationConfig]:
        """検証設定を作成"""
        configs = [
            VerificationConfig(
                name="Config_A",
                system_agent_learning=False,
                swarm_agent_learning=False,
                system_agent_branching=False,
                use_pretrained_models=False
            ),
            VerificationConfig(
                name="Config_B", 
                system_agent_learning=False,
                swarm_agent_learning=True,
                system_agent_branching=False,
                use_pretrained_models=True
            ),
            VerificationConfig(
                name="Config_C",
                system_agent_learning=False,
                swarm_agent_learning=False,
                system_agent_branching=True,
                use_pretrained_models=False
            ),
            VerificationConfig(
                name="Config_D",
                system_agent_learning=True,
                swarm_agent_learning=True,
                system_agent_branching=True,
                use_pretrained_models=True
            )
        ]
        return configs
    
    def create_environment_configs(self) -> List[EnvironmentConfig]:
        """環境設定のバリエーションを作成"""
        configs = [
            EnvironmentConfig(map_width=100, map_height=200, obstacle_density=0.0, robot_count=20),
            EnvironmentConfig(map_width=100, map_height=200, obstacle_density=0.003, robot_count=20),
            EnvironmentConfig(map_width=100, map_height=200, obstacle_density=0.005, robot_count=20),
        ]
        return configs
    
    def setup_agent_config(self, exp_config: VerificationConfig) -> AgentParam:
        """エージェント設定を作成"""
        agent_param = AgentParam()
        
        # SystemAgent設定
        system_param = SystemAgentParam()
        if exp_config.system_agent_learning:
            system_param.learningParameter = SystemLearningParameter()
        else:
            system_param.learningParameter = None
        
        # 分岐・統合設定
        if exp_config.system_agent_branching:
            system_param.branch_condition.branch_enabled = True
            system_param.integration_condition.integration_enabled = True
        else:
            system_param.branch_condition.branch_enabled = False
            system_param.integration_condition.integration_enabled = False
        
        agent_param.system_agent_param = system_param
        
        # SwarmAgent設定
        swarm_param = SwarmAgentParam()
        if exp_config.swarm_agent_learning:
            swarm_param.isLearning = True
            swarm_param.learningParameter = SwarmLearningParameter()
        else:
            swarm_param.isLearning = False
            swarm_param.learningParameter = None
        
        agent_param.swarm_agent_params = [swarm_param]
        
        return agent_param
    
    def setup_environment(self, env_config: EnvironmentConfig) -> SimulationParam:
        """環境設定を作成"""
        sim_param = self.base_config.copy()
        
        # マップサイズ設定
        sim_param.environment.map.width = env_config.map_width
        sim_param.environment.map.height = env_config.map_height
        
        # 障害物密度設定
        sim_param.environment.obstacle.probability = env_config.obstacle_density
        
        # ロボット数設定
        sim_param.explore.robotNum = env_config.robot_count
        
        # ログ設定（GIF生成用）
        sim_param.robot_logging.save_robot_data = True
        sim_param.robot_logging.save_position = True
        sim_param.robot_logging.save_collision = True
        sim_param.robot_logging.sampling_rate = 1.0
        
        return sim_param
    
    def load_pretrained_models(self, system_agent, swarm_agents, exp_config: VerificationConfig):
        """事前学習済みモデルを読み込み"""
        if not exp_config.use_pretrained_models:
            return
        
        # 事前学習済みモデルのパス
        models_dir = "trained_models"
        
        try:
            # 対応する学習済みモデルを探す
            if exp_config.name == "Config_B":
                model_config_name = "Config_B_System"
            elif exp_config.name == "Config_D":
                model_config_name = "Config_D_System"
            else:
                print(f"Warning: No pretrained model for {exp_config.name}")
                return
            
            config_dir = os.path.join(models_dir, model_config_name)
            
            # SystemAgentの事前学習済みモデル
            if exp_config.system_agent_learning and system_agent:
                system_model_path = os.path.join(config_dir, "system_agent_model.keras")
                if os.path.exists(system_model_path):
                    system_agent.model.load_weights(system_model_path)
                    print(f"Loaded pretrained SystemAgent model from {system_model_path}")
            
            # SwarmAgentの事前学習済みモデル
            if exp_config.swarm_agent_learning and swarm_agents:
                swarm_model_path = os.path.join(config_dir, "swarm_agent_model.keras")
                if os.path.exists(swarm_model_path):
                    for agent in swarm_agents.values():
                        agent.model.load_weights(swarm_model_path)
                    print(f"Loaded pretrained SwarmAgent model from {swarm_model_path}")
                    
        except Exception as e:
            print(f"Warning: Could not load pretrained models: {e}")
            print("Continuing with randomly initialized models")
    
    def run_single_verification(self, exp_config: VerificationConfig, 
                               env_config: EnvironmentConfig) -> Dict[str, Any]:
        """単一検証を実行（複数回実行による統計分析）"""
        print(f"検証実行中: {exp_config.name} - {env_config}")
        
        # 複数回実行の結果を格納
        run_results = []
        
        for run in range(exp_config.num_runs):
            print(f"  実行 {run + 1}/{exp_config.num_runs}")
            
            # 出力ディレクトリを作成
            output_dir = f"verification_results/{exp_config.name}_{env_config}_{run+1}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 環境とエージェントを初期化
            sim_param = self.setup_environment(env_config)
            env = Env(sim_param)
            
            # エージェント設定
            agent_param = self.setup_agent_config(exp_config)
            
            # エージェント作成
            system_agent, swarm_agents = create_initial_agents(env, agent_param)
            
            # 事前学習済みモデルの読み込み
            self.load_pretrained_models(system_agent, swarm_agents, exp_config)
            
            # エピソード実行
            run_result = self._execute_episodes(env, system_agent, swarm_agents, exp_config, output_dir)
            run_results.append(run_result)
        
        # 複数回実行の統計分析
        return self._analyze_multiple_runs(run_results, exp_config)
    
    def _execute_episodes(self, env, system_agent, swarm_agents, 
                         exp_config: VerificationConfig, output_dir: str) -> Dict[str, Any]:
        """エピソード実行"""
        episode_results = []
        target_reached_episodes = []
        
        for episode in range(exp_config.num_episodes):
            env.reset()
            episode_data = {
                'episode': episode,
                'exploration_rates': [],
                'steps_to_target': None,
                'final_exploration_rate': 0.0
            }
            
            # エピソード開始
            env.start_episode(episode)
            
            for step in range(exp_config.max_steps_per_episode):
                # 環境ステップ実行
                state = env.get_state()
                
                # エージェント行動決定
                if exp_config.system_agent_learning:
                    system_action = system_agent.get_action(state, episode)
                else:
                    system_action = self._get_default_system_action()
                
                if exp_config.swarm_agent_learning:
                    swarm_actions = {swarm_id: agent.get_action(state, episode) for swarm_id, agent in swarm_agents.items()}
                else:
                    swarm_actions = {swarm_id: self._get_default_swarm_action() for swarm_id in swarm_agents.keys()}
                
                # 環境更新
                next_state, reward, done, truncated, info = env.step(swarm_actions)
                
                # データ記録
                exploration_rate = env.get_exploration_rate()
                episode_data['exploration_rates'].append(exploration_rate)
                
                # 目標達成チェック
                if exploration_rate >= exp_config.target_exploration_rate:
                    episode_data['steps_to_target'] = step + 1
                    target_reached_episodes.append(episode)
                    break
            
            episode_data['final_exploration_rate'] = episode_data['exploration_rates'][-1]
            episode_results.append(episode_data)
            
            # エピソード終了とGIF保存
            env.end_episode(output_dir)
        
        # 結果集計
        return self._aggregate_results(episode_results, target_reached_episodes)
    
    def _get_default_system_action(self) -> Dict[str, Any]:
        """デフォルトのSystemAgent行動"""
        return {
            'action_type': 'none',
            'target_swarm_id': 0,
            'branch_threshold': 0.3,
            'integration_threshold': 0.7
        }
    
    def _get_default_swarm_action(self) -> Dict[str, Any]:
        """デフォルトのSwarmAgent行動"""
        return {
            'theta': np.random.uniform(0, 2*np.pi),
            'th': 0.5,
            'k_e': 10.0,
            'k_c': 5.0
        }
    
    def _aggregate_results(self, episode_results: List[Dict], 
                          target_reached_episodes: List[int]) -> Dict[str, Any]:
        """結果を集計"""
        # 目標達成率
        target_reach_rate = len(target_reached_episodes) / len(episode_results)
        
        # 平均ステップ数（目標達成時）
        steps_to_target = [ep['steps_to_target'] for ep in episode_results 
                          if ep['steps_to_target'] is not None]
        avg_steps_to_target = np.mean(steps_to_target) if steps_to_target else None
        
        # 探査進捗速度
        exploration_speeds = []
        for ep_data in episode_results:
            if len(ep_data['exploration_rates']) > 1:
                speed = (ep_data['final_exploration_rate'] - ep_data['exploration_rates'][0]) / len(ep_data['exploration_rates'])
                exploration_speeds.append(speed)
        
        avg_exploration_speed = np.mean(exploration_speeds) if exploration_speeds else 0.0
        
        # 最終探査率
        final_exploration_rates = [ep['final_exploration_rate'] for ep in episode_results]
        avg_final_exploration_rate = np.mean(final_exploration_rates)
        
        return {
            'target_reach_rate': target_reach_rate,
            'avg_steps_to_target': avg_steps_to_target,
            'avg_exploration_speed': avg_exploration_speed,
            'avg_final_exploration_rate': avg_final_exploration_rate,
            'episode_results': episode_results,
            'target_reached_episodes': target_reached_episodes
        }
    
    def _analyze_multiple_runs(self, run_results: List[Dict], exp_config: VerificationConfig) -> Dict[str, Any]:
        """複数回実行の結果を統計分析"""
        # 各指標の統計を計算
        target_reach_rates = [r['target_reach_rate'] for r in run_results]
        avg_steps_to_targets = [r['avg_steps_to_target'] for r in run_results if r['avg_steps_to_target'] is not None]
        avg_exploration_speeds = [r['avg_exploration_speed'] for r in run_results]
        final_exploration_rates = [r['final_exploration_rate'] for r in run_results]
        
        # 統計量の計算
        stats = {
            'target_reach_rate': {
                'mean': np.mean(target_reach_rates),
                'std': np.std(target_reach_rates),
                'min': np.min(target_reach_rates),
                'max': np.max(target_reach_rates),
                'values': target_reach_rates
            },
            'avg_steps_to_target': {
                'mean': np.mean(avg_steps_to_targets) if avg_steps_to_targets else None,
                'std': np.std(avg_steps_to_targets) if avg_steps_to_targets else None,
                'min': np.min(avg_steps_to_targets) if avg_steps_to_targets else None,
                'max': np.max(avg_steps_to_targets) if avg_steps_to_targets else None,
                'values': avg_steps_to_targets
            },
            'avg_exploration_speed': {
                'mean': np.mean(avg_exploration_speeds),
                'std': np.std(avg_exploration_speeds),
                'min': np.min(avg_exploration_speeds),
                'max': np.max(avg_exploration_speeds),
                'values': avg_exploration_speeds
            },
            'final_exploration_rate': {
                'mean': np.mean(final_exploration_rates),
                'std': np.std(final_exploration_rates),
                'min': np.min(final_exploration_rates),
                'max': np.max(final_exploration_rates),
                'values': final_exploration_rates
            },
            'num_runs': exp_config.num_runs
        }
        
        # 後方互換性のため、従来の形式も保持
        stats['target_reach_rate'] = stats['target_reach_rate']['mean']
        stats['avg_steps_to_target'] = stats['avg_steps_to_target']['mean']
        stats['avg_exploration_speed'] = stats['avg_exploration_speed']['mean']
        stats['final_exploration_rate'] = stats['final_exploration_rate']['mean']
        
        return stats
    
    def run_all_verifications(self):
        """全検証実行"""
        exp_configs = self.create_verification_configs()
        env_configs = self.create_environment_configs()
        
        all_results = {}
        
        for exp_config in exp_configs:
            exp_results = {}
            for env_config in env_configs:
                try:
                    result = self.run_single_verification(exp_config, env_config)
                    exp_results[str(env_config)] = result
                    print(f"検証完了: {exp_config.name} - {env_config}")
                except Exception as e:
                    print(f"検証エラー {exp_config.name} - {env_config}: {e}")
            
            all_results[str(exp_config)] = exp_results
        
        self.results = all_results
        self.save_verification_results()
    
    def save_verification_results(self):
        """検証結果保存"""
        results_dir = "verification_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"verification_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"検証結果保存: {results_file}")


def main():
    """メイン実行関数"""
    runner = VerificationRunner()
    runner.run_all_verifications()


if __name__ == "__main__":
    main() 