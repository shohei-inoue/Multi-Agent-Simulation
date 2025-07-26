"""
4つの構成での比較実験
SystemAgentとSwarmAgentの学習有無による性能比較
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
import os
from datetime import datetime
import tensorflow as tf

from core.application import Application
from params.simulation import SimulationParam
from params.agent import AgentParam
from params.system_agent import SystemAgentParam, LearningParameter as SystemLearningParameter
from params.swarm_agent import SwarmAgentParam, LearningParameter as SwarmLearningParameter
from agents.agent_factory import create_initial_agents
from envs.env import Env


@dataclass
class ExperimentConfig:
    """実験設定"""
    name: str
    system_agent_learning: bool
    swarm_agent_learning: bool
    system_agent_branching: bool
    use_pretrained_models: bool = False
    num_episodes: int = 100
    max_steps_per_episode: int = 50
    target_exploration_rate: float = 0.8
    num_runs: int = 5  # 統計的信頼性のための複数回実行
    
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


class ComparisonExperiment:
    """比較実験クラス"""
    
    def __init__(self, base_config: SimulationParam):
        self.base_config = base_config
        self.results = {}
        self.logger = None
        
    def create_experiment_configs(self) -> List[ExperimentConfig]:
        """4つの実験構成を作成"""
        configs = [
            ExperimentConfig(
                name="Config_A",
                system_agent_learning=False,
                swarm_agent_learning=False,
                system_agent_branching=False,
                use_pretrained_models=False
            ),
            ExperimentConfig(
                name="Config_B", 
                system_agent_learning=False,
                swarm_agent_learning=True,
                system_agent_branching=False,
                use_pretrained_models=True
            ),
            ExperimentConfig(
                name="Config_C",
                system_agent_learning=False,
                swarm_agent_learning=False,
                system_agent_branching=True,
                use_pretrained_models=False
            ),
            ExperimentConfig(
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
            # 障害物なし
            EnvironmentConfig(map_width=100, map_height=200, obstacle_density=0.0, robot_count=20),
            # 低障害物密度
            EnvironmentConfig(map_width=100, map_height=200, obstacle_density=0.003, robot_count=20),
            # 中障害物密度
            EnvironmentConfig(map_width=100, map_height=200, obstacle_density=0.005, robot_count=20),
        ]
        return configs
    
    def estimate_steps_for_target(self, env_config: EnvironmentConfig) -> int:
        """目標達成に必要なステップ数を推定"""
        # 固定設定: 1エピソードあたりの最大ステップ数を200に設定
        # 1領域あたりのfollowerの各探査ステップを30として計算
        
        # 探査可能エリアの計算
        total_cells = env_config.map_width * env_config.map_height
        obstacle_cells = int(total_cells * env_config.obstacle_density)
        explorable_cells = total_cells - obstacle_cells
        
        # 目標探査セル数
        target_cells = int(explorable_cells * 0.8)
        
        # ロボット20台での探査効率を考慮
        # 1ステップで各ロボットが探査するセル数を推定
        cells_per_robot_per_step = 3  # 探査半径を考慮
        
        # 重複を考慮した効率的な探査
        effective_cells_per_step = cells_per_robot_per_step * env_config.robot_count * 0.7  # 重複率30%を考慮
        
        # 必要なステップ数
        estimated_steps = int(target_cells / effective_cells_per_step)
        
        # 安全マージンを追加（20%増）
        estimated_steps = int(estimated_steps * 1.2)
        
        # 固定設定: 最大ステップ数を200に設定
        estimated_steps = max(30, min(estimated_steps, 200))
        
        return estimated_steps
    
    def setup_agent_config(self, exp_config: ExperimentConfig) -> AgentParam:
        """エージェント設定を作成"""
        agent_param = AgentParam()
        
        # SystemAgent設定
        system_param = SystemAgentParam()
        
        # 学習設定（元の設計を尊重）
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
        
        # 学習設定（元の設計を尊重）
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
        sim_param.robot_logging.sampling_rate = 1.0  # 全ステップ保存
        
        return sim_param
    
    def load_pretrained_models(self, system_agent, swarm_agents, exp_config: ExperimentConfig):
        """事前学習済みモデルを読み込み"""
        if not exp_config.use_pretrained_models:
            return
        
        # 事前学習済みモデルのパス
        pretrained_dir = "pretrained_models"
        
        try:
            # SystemAgentの事前学習済みモデル
            if exp_config.system_agent_learning and system_agent:
                system_model_path = os.path.join(pretrained_dir, "system_agent_model.keras")
                if os.path.exists(system_model_path):
                    system_agent.model.load_weights(system_model_path)
                    print(f"Loaded pretrained SystemAgent model from {system_model_path}")
            
            # SwarmAgentの事前学習済みモデル
            if exp_config.swarm_agent_learning and swarm_agents:
                swarm_model_path = os.path.join(pretrained_dir, "swarm_agent_model.keras")
                if os.path.exists(swarm_model_path):
                    for agent in swarm_agents:
                        agent.model.load_weights(swarm_model_path)
                    print(f"Loaded pretrained SwarmAgent model from {swarm_model_path}")
                    
        except Exception as e:
            print(f"Warning: Could not load pretrained models: {e}")
            print("Continuing with randomly initialized models")
    
    def run_single_experiment(self, exp_config: ExperimentConfig, 
                            env_config: EnvironmentConfig) -> Dict[str, Any]:
        """単一実験を実行（複数回実行による統計分析）"""
        print(f"実行中: {exp_config.name} - {env_config}")
        
        # 複数回実行の結果を格納
        run_results = []
        
        for run in range(exp_config.num_runs):
            print(f"  実行 {run + 1}/{exp_config.num_runs}")
            
            # 出力ディレクトリを作成
            output_dir = f"experiment_results/{exp_config.name}_{env_config}_{run+1}"
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
            
            # ステップ数推定
            max_steps = self.estimate_steps_for_target(env_config)
            
            # エピソード実行
            run_result = self._execute_episodes(env, system_agent, swarm_agents, exp_config, max_steps, output_dir)
            run_results.append(run_result)
        
        # 複数回実行の統計分析
        return self._analyze_multiple_runs(run_results, exp_config)
    
    def _execute_episodes(self, env, system_agent, swarm_agents, 
                         exp_config: ExperimentConfig, max_steps: int, output_dir: str) -> Dict[str, Any]:
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
            
            for step in range(max_steps):
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
            
            # 学習更新（事前学習済みモデルを使用する場合は学習しない）
            if not exp_config.use_pretrained_models:
                if exp_config.system_agent_learning:
                    system_agent.train()
                if exp_config.swarm_agent_learning:
                    for agent in swarm_agents.values():
                        agent.train()
        
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
    
    def run_comparison_experiments(self) -> Dict[str, Any]:
        """比較実験を実行"""
        exp_configs = self.create_experiment_configs()
        env_configs = self.create_environment_configs()
        
        all_results = {}
        
        for exp_config in exp_configs:
            exp_results = {}
            for env_config in env_configs:
                result = self.run_single_experiment(exp_config, env_config)
                exp_results[str(env_config)] = result
            
            all_results[str(exp_config)] = exp_results
        
        self.results = all_results
        return all_results
    
    def analyze_results(self) -> Dict[str, Any]:
        """結果分析"""
        analysis = {
            'performance_comparison': self._compare_performance(),
            'robustness_analysis': self._analyze_robustness(),
            'speed_analysis': self._analyze_speed(),
            'obstacle_density_analysis': self._analyze_obstacle_density_impact()
        }
        return analysis
    
    def _compare_performance(self) -> Dict[str, Any]:
        """性能比較（統計情報付き）"""
        # 基本環境での性能比較（障害物なし）
        base_env = "Map100x200_Obs0.0_Robot20"
        performance_data = {}
        
        for exp_name, exp_results in self.results.items():
            if base_env in exp_results:
                base_result = exp_results[base_env]
                performance_data[exp_name] = {
                    'target_reach_rate': base_result['target_reach_rate'],
                    'avg_steps_to_target': base_result['avg_steps_to_target'],
                    'avg_exploration_speed': base_result['avg_exploration_speed'],
                    'detailed_stats': base_result.get('detailed_stats', {}),
                    'confidence_interval_95': base_result.get('confidence_interval_95', {})
                }
        
        return performance_data
    
    def _analyze_robustness(self) -> Dict[str, Any]:
        """ロバスト性分析（統計情報付き）"""
        robustness_data = {}
        
        for exp_name, exp_results in self.results.items():
            env_performances = {}
            env_performance_stds = {}
            
            for env_name, result in exp_results.items():
                env_performances[env_name] = result['target_reach_rate']
                
                # 統計情報がある場合は標準偏差も記録
                if 'detailed_stats' in result:
                    env_performance_stds[env_name] = result['detailed_stats']['target_reach_rate']['std']
            
            # 環境変化に対する性能の安定性
            performance_std = np.std(list(env_performances.values()))
            robustness_data[exp_name] = {
                'performance_std': performance_std,
                'env_performances': env_performances,
                'env_performance_stds': env_performance_stds,
                'stability_score': 1.0 / (1.0 + performance_std)  # 安定性スコア
            }
        
        return robustness_data
    
    def _analyze_speed(self) -> Dict[str, Any]:
        """速度分析（統計情報付き）"""
        speed_data = {}
        
        for exp_name, exp_results in self.results.items():
            speeds = []
            speed_stds = []
            
            for env_name, result in exp_results.items():
                if result['avg_steps_to_target'] is not None:
                    speeds.append(result['avg_steps_to_target'])
                    
                    # 統計情報がある場合は標準偏差も記録
                    if 'detailed_stats' in result and result['detailed_stats']['avg_steps_to_target']['std'] is not None:
                        speed_stds.append(result['detailed_stats']['avg_steps_to_target']['std'])
            
            speed_data[exp_name] = {
                'avg_speed': np.mean(speeds) if speeds else None,
                'speed_std': np.std(speeds) if speeds else None,
                'speed_stability': np.mean(speed_stds) if speed_stds else None,
                'speed_values': speeds
            }
        
        return speed_data
    
    def _analyze_obstacle_density_impact(self) -> Dict[str, Any]:
        """障害物密度の影響分析"""
        density_impact = {}
        
        # 各構成について障害物密度の影響を分析
        for exp_name, exp_results in self.results.items():
            density_performances = {}
            for env_name, result in exp_results.items():
                # 環境名から障害物密度を抽出
                if "Obs0.0" in env_name:
                    density = 0.0
                elif "Obs0.003" in env_name:
                    density = 0.003
                elif "Obs0.005" in env_name:
                    density = 0.005
                else:
                    continue
                
                density_performances[density] = result['target_reach_rate']
            
            # 障害物密度による性能変化を分析
            if len(density_performances) >= 2:
                densities = sorted(density_performances.keys())
                performances = [density_performances[d] for d in densities]
                
                # 性能劣化率（障害物密度0.0を基準）
                if 0.0 in density_performances:
                    baseline = density_performances[0.0]
                    degradation_rates = []
                    for density, perf in density_performances.items():
                        if density > 0.0:
                            degradation = (baseline - perf) / baseline if baseline > 0 else 0
                            degradation_rates.append(degradation)
                    
                    avg_degradation = np.mean(degradation_rates) if degradation_rates else 0.0
                else:
                    avg_degradation = 0.0
                
                density_impact[exp_name] = {
                    'density_performances': density_performances,
                    'avg_degradation_rate': avg_degradation,
                    'performance_trend': performances
                }
        
        return density_impact
    
    def save_results(self, output_dir: str = "experiment_results"):
        """結果を保存"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 結果をJSONで保存
        results_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # 分析結果を保存
        analysis = self.analyze_results()
        analysis_file = os.path.join(output_dir, f"analysis_{timestamp}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"Results saved to {output_dir}")
        return results_file, analysis_file
    
    def plot_results(self, output_dir: str = "experiment_results"):
        """結果を可視化"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 性能比較グラフ
        self._plot_performance_comparison(output_dir, timestamp)
        
        # ロバスト性グラフ
        self._plot_robustness_comparison(output_dir, timestamp)
        
        # 速度比較グラフ
        self._plot_speed_comparison(output_dir, timestamp)
        
        # 障害物密度影響グラフ
        self._plot_obstacle_density_impact(output_dir, timestamp)
    
    def _plot_performance_comparison(self, output_dir: str, timestamp: str):
        """性能比較グラフ（統計情報付き）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('性能比較（統計情報付き）', fontsize=16)
        
        # 基本環境での性能比較
        base_env = "Map100x200_Obs0.0_Robot20"
        configs = []
        target_reach_rates = []
        target_reach_stds = []
        avg_steps = []
        avg_steps_stds = []
        
        for exp_name, exp_results in self.results.items():
            if base_env in exp_results:
                result = exp_results[base_env]
                configs.append(exp_name)
                target_reach_rates.append(result['target_reach_rate'])
                avg_steps.append(result['avg_steps_to_target'] or 0)
                
                # 統計情報がある場合は標準偏差も取得
                if 'detailed_stats' in result:
                    target_reach_stds.append(result['detailed_stats']['target_reach_rate']['std'])
                    if result['detailed_stats']['avg_steps_to_target']['std'] is not None:
                        avg_steps_stds.append(result['detailed_stats']['avg_steps_to_target']['std'])
                    else:
                        avg_steps_stds.append(0)
                else:
                    target_reach_stds.append(0)
                    avg_steps_stds.append(0)
        
        # 目標達成率（エラーバー付き）
        axes[0, 0].bar(configs, target_reach_rates, yerr=target_reach_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title('目標達成率（80%探査）')
        axes[0, 0].set_ylabel('達成率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 平均ステップ数（エラーバー付き）
        axes[0, 1].bar(configs, avg_steps, yerr=avg_steps_stds, capsize=5, alpha=0.7)
        axes[0, 1].set_title('平均ステップ数（目標達成時）')
        axes[0, 1].set_ylabel('ステップ数')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 信頼区間の表示
        confidence_intervals = []
        for exp_name, exp_results in self.results.items():
            if base_env in exp_results and 'confidence_interval_95' in exp_results[base_env]:
                ci = exp_results[base_env]['confidence_interval_95']['target_reach_rate']
                if ci:
                    confidence_intervals.append({
                        'config': exp_name,
                        'lower': ci['lower'],
                        'upper': ci['upper']
                    })
        
        if confidence_intervals:
            ci_configs = [ci['config'] for ci in confidence_intervals]
            ci_lowers = [ci['lower'] for ci in confidence_intervals]
            ci_uppers = [ci['upper'] for ci in confidence_intervals]
            
            axes[1, 0].bar(ci_configs, [u-l for l, u in zip(ci_lowers, ci_uppers)], 
                          bottom=ci_lowers, alpha=0.7, label='95%信頼区間')
            axes[1, 0].set_title('目標達成率の95%信頼区間')
            axes[1, 0].set_ylabel('達成率')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 統計サマリー
        stats_text = f"統計情報:\n"
        stats_text += f"実行回数: {self.results[list(self.results.keys())[0]][base_env].get('num_runs', 'N/A')}\n"
        stats_text += f"エピソード数: {self.base_config.episodeNum}\n"
        stats_text += f"最大ステップ数: {self.base_config.maxStepsPerEpisode}"
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('実験設定')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_comparison_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_comparison(self, output_dir: str, timestamp: str):
        """ロバスト性比較グラフ"""
        analysis = self.analyze_results()
        robust_data = analysis['robustness_analysis']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        configs = list(robust_data.keys())
        stds = [robust_data[config]['performance_std'] for config in configs]
        
        ax.bar(configs, stds)
        ax.set_title('Robustness Comparison (Performance Standard Deviation)')
        ax.set_ylabel('Standard Deviation')
        ax.set_xlabel('Configuration')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"robustness_comparison_{timestamp}.png"))
        plt.close()
    
    def _plot_speed_comparison(self, output_dir: str, timestamp: str):
        """速度比較グラフ"""
        analysis = self.analyze_results()
        speed_data = analysis['speed_analysis']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        configs = list(speed_data.keys())
        speeds = [speed_data[config]['avg_speed'] for config in configs]
        speed_stds = [speed_data[config]['speed_std'] for config in configs]
        
        x = np.arange(len(configs))
        ax.bar(x, speeds, yerr=speed_stds, capsize=5)
        ax.set_title('Speed Comparison (Average Steps to Target)')
        ax.set_ylabel('Steps')
        ax.set_xlabel('Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"speed_comparison_{timestamp}.png"))
        plt.close()
    
    def _plot_obstacle_density_impact(self, output_dir: str, timestamp: str):
        """障害物密度影響グラフ"""
        analysis = self.analyze_results()
        density_data = analysis['obstacle_density_analysis']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 各構成の障害物密度による性能変化をプロット
        densities = [0.0, 0.003, 0.005]
        
        for config_name, config_data in density_data.items():
            performances = []
            for density in densities:
                if density in config_data['density_performances']:
                    performances.append(config_data['density_performances'][density])
                else:
                    performances.append(0.0)
            
            ax.plot(densities, performances, marker='o', label=config_name, linewidth=2)
        
        ax.set_xlabel('Obstacle Density')
        ax.set_ylabel('Target Reach Rate')
        ax.set_title('Impact of Obstacle Density on Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"obstacle_density_impact_{timestamp}.png"))
        plt.close()

    def _analyze_multiple_runs(self, run_results: List[Dict], exp_config: ExperimentConfig) -> Dict[str, Any]:
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
            'num_runs': exp_config.num_runs,
            'confidence_interval_95': {
                'target_reach_rate': self._calculate_confidence_interval(target_reach_rates, 0.95),
                'avg_steps_to_target': self._calculate_confidence_interval(avg_steps_to_targets, 0.95) if avg_steps_to_targets else None,
                'avg_exploration_speed': self._calculate_confidence_interval(avg_exploration_speeds, 0.95),
                'final_exploration_rate': self._calculate_confidence_interval(final_exploration_rates, 0.95)
            }
        }
        
        # 後方互換性のため、従来の形式も保持
        stats['target_reach_rate'] = stats['target_reach_rate']['mean']
        stats['avg_steps_to_target'] = stats['avg_steps_to_target']['mean']
        stats['avg_exploration_speed'] = stats['avg_exploration_speed']['mean']
        stats['final_exploration_rate'] = stats['final_exploration_rate']['mean']
        
        # 詳細統計情報を追加
        stats['detailed_stats'] = {
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
            }
        }
        
        return stats
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """信頼区間を計算"""
        if not values:
            return None
        
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # 不偏標準偏差
        
        # t分布の臨界値（簡易版、実際にはscipy.stats.t.ppfを使用すべき）
        if n <= 30:
            # 小標本の場合の簡易t値
            t_critical = 2.0  # 簡易値
        else:
            # 大標本の場合の正規分布近似
            t_critical = 1.96
        
        margin_of_error = t_critical * (std / np.sqrt(n))
        
        return {
            'lower': mean - margin_of_error,
            'upper': mean + margin_of_error,
            'margin_of_error': margin_of_error
        }


def main():
    """メイン実行関数"""
    # 基本設定
    base_config = SimulationParam()
    base_config.episodeNum = 50  # 実験用に短縮
    base_config.maxStepsPerEpisode = 30
    
    # 実験実行
    experiment = ComparisonExperiment(base_config)
    results = experiment.run_comparison_experiments()
    
    # 結果保存・可視化
    experiment.save_results()
    experiment.plot_results()
    
    # 分析結果表示
    analysis = experiment.analyze_results()
    print("\n=== Performance Comparison ===")
    for config, perf in analysis['performance_comparison'].items():
        print(f"{config}:")
        print(f"  Target Reach Rate: {perf['target_reach_rate']:.3f}")
        print(f"  Avg Steps to Target: {perf['avg_steps_to_target']:.1f}")
        print(f"  Exploration Speed: {perf['avg_exploration_speed']:.4f}")
    
    print("\n=== Robustness Analysis ===")
    for config, robust in analysis['robustness_analysis'].items():
        print(f"{config}: Performance STD = {robust['performance_std']:.3f}")
    
    print("\n=== Obstacle Density Impact ===")
    for config, density_impact in analysis['obstacle_density_impact'].items():
        print(f"{config}: Avg Degradation Rate = {density_impact['avg_degradation_rate']:.3f}")


if __name__ == "__main__":
    main() 