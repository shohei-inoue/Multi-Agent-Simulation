#!/usr/bin/env python3
"""
分岐・統合機能の探査率向上スピード調査
4つの設定で探査効率を比較する専用シミュレーション
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import multiprocessing as mp

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_base_environment():
    """基本環境設定"""
    from params.simulation import SimulationParam
    from params.environment import EnvironmentParam
    from params.explore import ExploreParam
    from params.robot_logging import RobotLoggingConfig
    
    sim_param = SimulationParam()
    
    # 基本設定
    sim_param.episodeNum = 10  # 複数エピソード実行
    sim_param.maxStepsPerEpisode = 100
    
    # 環境設定
    env_config = EnvironmentParam()
    env_config.map.width = 200
    env_config.map.height = 100
    env_config.obstacle.probability = 0.003  # 中程度の障害物密度
    sim_param.environment = env_config
    
    # 探査設定
    explore_config = ExploreParam()
    explore_config.robotNum = 20
    explore_config.coordinate.x = 10.0
    explore_config.coordinate.y = 10.0
    explore_config.boundary.inner = 0.0
    explore_config.boundary.outer = 10.0
    explore_config.finishRate = 0.8
    sim_param.explore = explore_config
    
    # ログ設定（探査率記録用）
    logging_config = RobotLoggingConfig()
    logging_config.save_robot_data = True
    logging_config.save_position = True
    logging_config.save_collision = True
    logging_config.sampling_rate = 1.0
    sim_param.robot_logging = logging_config
    
    return sim_param

def setup_agent_config(branch_enabled: bool, integration_enabled: bool):
    """エージェント設定"""
    from params.agent import AgentParam
    from params.system_agent import SystemAgentParam
    from params.swarm_agent import SwarmAgentParam
    
    agent_param = AgentParam()
    
    # SystemAgent設定（学習なし）
    system_param = SystemAgentParam()
    system_param.learningParameter = None
    system_param.branch_condition.branch_enabled = branch_enabled
    system_param.integration_condition.integration_enabled = integration_enabled
    
    # 分岐・統合の閾値設定
    if branch_enabled:
        system_param.branch_condition.min_follower_count = 3
        system_param.branch_condition.min_valid_directions = 2
        system_param.branch_condition.mobility_threshold = 0.5
        system_param.branch_condition.cooldown_time = 10.0
    
    if integration_enabled:
        system_param.integration_condition.mobility_threshold = 0.3
        system_param.integration_condition.cooldown_time = 15.0
    
    agent_param.system_agent_param = system_param
    
    # SwarmAgent設定（学習なし、固定パラメータ）
    swarm_param = SwarmAgentParam()
    swarm_param.isLearning = False
    swarm_param.learningParameter = None
    agent_param.swarm_agent_params = [swarm_param]
    
    return agent_param

def run_single_simulation(config_name: str, branch_enabled: bool, integration_enabled: bool, 
                         output_dir: Path, run_id: int) -> Dict:
    """単一シミュレーション実行"""
    print(f"🚀 {config_name} - Run {run_id} 開始")
    
    # 環境とエージェント設定
    sim_param = setup_base_environment()
    agent_param = setup_agent_config(branch_enabled, integration_enabled)
    
    # 環境作成
    from envs.env import Env
    from agents.agent_factory import create_initial_agents
    
    env = Env(sim_param)
    system_agent, swarm_agents = create_initial_agents(env, agent_param)
    env.set_system_agent(system_agent)
    
    # 結果記録用
    simulation_results = {
        'config': config_name,
        'run_id': run_id,
        'branch_enabled': branch_enabled,
        'integration_enabled': integration_enabled,
        'episodes': {}
    }
    
    # エピソード実行
    for episode in range(sim_param.episodeNum):
        print(f"  📊 Episode {episode + 1}/{sim_param.episodeNum}")
        
        # 環境リセット
        state = env.reset()
        episode_data = {
            'episode': episode,
            'steps': [],
            'exploration_rates': [],
            'swarm_counts': [],
            'branch_events': [],
            'integration_events': []
        }
        
        # ステップ実行
        for step in range(sim_param.maxStepsPerEpisode):
            # アクション取得
            action, action_info = system_agent.get_current_swarm_agent().get_action(state, episode)
            
            # 環境ステップ
            next_state, reward, done, info = env.step(action)
            
            # データ記録
            exploration_rate = env.get_exploration_rate()
            swarm_count = len(system_agent.swarm_agents)
            
            episode_data['steps'].append(step)
            episode_data['exploration_rates'].append(exploration_rate)
            episode_data['swarm_counts'].append(swarm_count)
            
            # 分岐・統合イベント記録
            if 'branch_occurred' in info and info['branch_occurred']:
                episode_data['branch_events'].append(step)
            if 'integration_occurred' in info and info['integration_occurred']:
                episode_data['integration_events'].append(step)
            
            state = next_state
            
            # 目標探査率達成で終了
            if exploration_rate >= sim_param.explore.finishRate:
                print(f"    ✅ 目標達成! Step {step}, Rate {exploration_rate:.1%}")
                break
        
        simulation_results['episodes'][f'episode_{episode}'] = episode_data
    
    # 結果保存
    result_file = output_dir / f"{config_name}_run_{run_id}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(simulation_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ {config_name} - Run {run_id} 完了")
    return simulation_results

def run_config_simulations(config_name: str, branch_enabled: bool, integration_enabled: bool,
                          output_dir: Path, num_runs: int = 3) -> List[Dict]:
    """設定別シミュレーション実行"""
    print(f"\n🔧 {config_name} 設定でのシミュレーション開始")
    print(f"   分岐: {'✅' if branch_enabled else '❌'}, 統合: {'✅' if integration_enabled else '❌'}")
    
    results = []
    for run_id in range(num_runs):
        try:
            result = run_single_simulation(config_name, branch_enabled, integration_enabled,
                                         output_dir, run_id + 1)
            results.append(result)
        except Exception as e:
            print(f"❌ {config_name} - Run {run_id + 1} エラー: {e}")
            continue
    
    return results

def analyze_and_plot_results(output_dir: Path):
    """結果分析とプロット生成"""
    print("\n📈 結果分析開始")
    
    # 結果ファイル読み込み
    all_results = {}
    for json_file in output_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            config = data['config']
            if config not in all_results:
                all_results[config] = []
            all_results[config].append(data)
    
    if not all_results:
        print("❌ 分析対象のデータが見つかりません")
        return
    
    # プロット作成
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('分岐・統合機能の探査率向上スピード比較', fontsize=16, fontweight='bold')
    
    colors = {'Base': 'blue', 'Branch': 'green', 'Integration': 'orange', 'Both': 'red'}
    
    # 1. 平均探査進捗の時系列
    ax1 = axes[0, 0]
    for config, results in all_results.items():
        all_progressions = []
        max_steps = 0
        
        for result in results:
            for episode_key, episode_data in result['episodes'].items():
                steps = episode_data['steps']
                rates = [r * 100 for r in episode_data['exploration_rates']]  # パーセント変換
                if steps and rates:
                    all_progressions.append((steps, rates))
                    max_steps = max(max_steps, max(steps))
        
        if all_progressions:
            # 共通ステップ軸で補間
            common_steps = np.linspace(0, min(200, max_steps), 100)
            interpolated_rates = []
            
            from scipy.interpolate import interp1d
            for steps, rates in all_progressions:
                if len(steps) > 1:
                    interp_func = interp1d(steps, rates, kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
                    interpolated_rates.append(interp_func(common_steps))
            
            if interpolated_rates:
                mean_rates = np.mean(interpolated_rates, axis=0)
                std_rates = np.std(interpolated_rates, axis=0)
                
                ax1.plot(common_steps, mean_rates, label=config, 
                        color=colors.get(config, 'black'), linewidth=2)
                ax1.fill_between(common_steps, mean_rates - std_rates, 
                               mean_rates + std_rates, alpha=0.2, 
                               color=colors.get(config, 'black'))
    
    ax1.set_title('探査率の時系列変化')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('探査率 (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 2. 平均探査速度比較
    ax2 = axes[0, 1]
    config_names = []
    avg_speeds = []
    speed_errors = []
    
    for config, results in all_results.items():
        speeds = []
        for result in results:
            for episode_key, episode_data in result['episodes'].items():
                rates = episode_data['exploration_rates']
                steps = episode_data['steps']
                if len(rates) > 1:
                    # 探査速度計算
                    episode_speeds = []
                    for i in range(1, len(rates)):
                        dt = steps[i] - steps[i-1] if i < len(steps) else 1
                        speed = (rates[i] - rates[i-1]) / dt * 100  # %/step
                        episode_speeds.append(speed)
                    if episode_speeds:
                        speeds.append(np.mean(episode_speeds))
        
        if speeds:
            config_names.append(config)
            avg_speeds.append(np.mean(speeds))
            speed_errors.append(np.std(speeds))
    
    bars = ax2.bar(config_names, avg_speeds, yerr=speed_errors, 
                   color=[colors.get(c, 'gray') for c in config_names],
                   alpha=0.7, capsize=5)
    ax2.set_title('平均探査速度')
    ax2.set_ylabel('探査速度 (%/step)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 目標達成時間比較
    ax3 = axes[1, 0]
    target_times = {80: [], 50: []}  # 80%, 50%到達時間
    
    for config, results in all_results.items():
        times_80 = []
        times_50 = []
        
        for result in results:
            for episode_key, episode_data in result['episodes'].items():
                rates = [r * 100 for r in episode_data['exploration_rates']]
                steps = episode_data['steps']
                
                # 50%, 80%到達時間を計算
                for i, rate in enumerate(rates):
                    if rate >= 50 and not times_50:
                        times_50.append(steps[i] if i < len(steps) else i)
                        break
                for i, rate in enumerate(rates):
                    if rate >= 80 and not times_80:
                        times_80.append(steps[i] if i < len(steps) else i)
                        break
        
        if times_50:
            target_times[50].append((config, np.mean(times_50), np.std(times_50)))
        if times_80:
            target_times[80].append((config, np.mean(times_80), np.std(times_80)))
    
    x_pos = np.arange(len(config_names))
    width = 0.35
    
    if target_times[50]:
        configs_50, means_50, stds_50 = zip(*target_times[50])
        ax3.bar(x_pos - width/2, means_50, width, label='50%到達', 
               yerr=stds_50, alpha=0.7, capsize=5)
    
    if target_times[80]:
        configs_80, means_80, stds_80 = zip(*target_times[80])
        ax3.bar(x_pos + width/2, means_80, width, label='80%到達',
               yerr=stds_80, alpha=0.7, capsize=5)
    
    ax3.set_title('目標探査率到達時間')
    ax3.set_xlabel('Config')
    ax3.set_ylabel('到達時間 (steps)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(config_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 群数の変化
    ax4 = axes[1, 1]
    for config, results in all_results.items():
        all_swarm_counts = []
        
        for result in results:
            for episode_key, episode_data in result['episodes'].items():
                swarm_counts = episode_data['swarm_counts']
                steps = episode_data['steps']
                if swarm_counts and steps:
                    all_swarm_counts.append((steps, swarm_counts))
        
        if all_swarm_counts:
            # 平均群数の時系列
            common_steps = np.linspace(0, 200, 100)
            interpolated_counts = []
            
            for steps, counts in all_swarm_counts:
                if len(steps) > 1:
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(steps, counts, kind='nearest',
                                         bounds_error=False, fill_value='extrapolate')
                    interpolated_counts.append(interp_func(common_steps))
            
            if interpolated_counts:
                mean_counts = np.mean(interpolated_counts, axis=0)
                ax4.plot(common_steps, mean_counts, label=config,
                        color=colors.get(config, 'black'), linewidth=2)
    
    ax4.set_title('群数の変化')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('群数')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'branch_integration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計サマリー生成
    generate_summary_report(all_results, output_dir)

def generate_summary_report(all_results: Dict, output_dir: Path):
    """サマリーレポート生成"""
    report_path = output_dir / 'branch_integration_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 分岐・統合機能の探査率向上スピード調査レポート\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 調査設定\n\n")
        f.write("| Config | 分岐機能 | 統合機能 | 学習 | 目的 |\n")
        f.write("|--------|----------|----------|------|------|\n")
        f.write("| Base | ❌ | ❌ | ❌ | 基準（分岐・統合なし） |\n")
        f.write("| Branch | ✅ | ❌ | ❌ | 分岐効果のみ |\n")
        f.write("| Integration | ❌ | ✅ | ❌ | 統合効果のみ |\n")
        f.write("| Both | ✅ | ✅ | ❌ | 分岐・統合両方 |\n\n")
        
        f.write("## 実行結果\n\n")
        for config, results in all_results.items():
            f.write(f"### {config}\n")
            f.write(f"- 実行回数: {len(results)}\n")
            
            # 平均最終探査率
            final_rates = []
            for result in results:
                for episode_key, episode_data in result['episodes'].items():
                    if episode_data['exploration_rates']:
                        final_rates.append(episode_data['exploration_rates'][-1] * 100)
            
            if final_rates:
                f.write(f"- 平均最終探査率: {np.mean(final_rates):.1f}% (±{np.std(final_rates):.1f}%)\n")
            f.write("\n")
        
        f.write("## 結論\n\n")
        f.write("1. **分岐機能の効果**: 探査範囲の拡大による効率向上\n")
        f.write("2. **統合機能の効果**: 効率の悪い群の統合による最適化\n")
        f.write("3. **相乗効果**: 分岐・統合の組み合わせによる最適化\n")
        f.write("4. **トレードオフ**: 群管理のオーバーヘッドと効率向上のバランス\n\n")
        
        f.write("## 生成ファイル\n\n")
        f.write("- `branch_integration_analysis.png`: 分析結果グラフ\n")
        f.write("- `{Config}_run_{N}.json`: 各実行の詳細データ\n")

def main():
    """メイン実行"""
    print("🚀 分岐・統合機能の探査率向上スピード調査")
    print("=" * 60)
    
    # 出力ディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"branch_integration_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 結果保存先: {output_dir}")
    
    # 4つの設定でシミュレーション実行
    configs = [
        ("Base", False, False),           # 基準
        ("Branch", True, False),          # 分岐のみ
        ("Integration", False, True),     # 統合のみ
        ("Both", True, True)              # 分岐・統合両方
    ]
    
    num_runs = 3  # 各設定で3回実行
    
    all_results = {}
    for config_name, branch_enabled, integration_enabled in configs:
        results = run_config_simulations(config_name, branch_enabled, integration_enabled,
                                       output_dir, num_runs)
        if results:
            all_results[config_name] = results
    
    if all_results:
        # 結果分析
        analyze_and_plot_results(output_dir)
        
        print(f"\n🎉 調査完了!")
        print(f"結果は {output_dir} に保存されました")
        print("\n📊 主要な比較ポイント:")
        print("  1. Base vs Branch: 分岐機能の効果")
        print("  2. Base vs Integration: 統合機能の効果")
        print("  3. Base vs Both: 分岐・統合の相乗効果")
        print("  4. Branch vs Both, Integration vs Both: 単独機能との差")
    else:
        print("❌ 実行可能な設定がありませんでした")

if __name__ == "__main__":
    main() 