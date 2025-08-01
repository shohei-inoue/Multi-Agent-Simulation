#!/usr/bin/env python3
"""
Config_A検証結果の分析とグラフ化スクリプト
3つの障害物密度（0.0, 0.003, 0.005）の結果を比較分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_verification_results():
    """検証結果を読み込む"""
    results = {}
    base_path = Path("verification_results")
    
    for obstacle_density in [0.0, 0.003, 0.005]:
        config_path = base_path / f"Config_A_obstacle_{obstacle_density}"
        result_file = config_path / "verification_result.json"
        
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                results[obstacle_density] = data
                print(f"✅ 読み込み完了: obstacle_density = {obstacle_density}")
        else:
            print(f"❌ ファイルが見つかりません: {result_file}")
    
    return results

def analyze_exploration_performance(results):
    """探索性能の分析"""
    analysis = {}
    
    for obstacle_density, data in results.items():
        episodes = data['episodes']
        
        # 基本統計量の計算
        exploration_rates = [ep['final_exploration_rate'] for ep in episodes]
        steps_to_target = [ep['steps_to_target'] for ep in episodes if ep['steps_to_target'] is not None]
        steps_taken = [ep['steps_taken'] for ep in episodes]
        
        analysis[obstacle_density] = {
            'exploration_rates': exploration_rates,
            'steps_to_target': steps_to_target,
            'steps_taken': steps_taken,
            'mean_exploration_rate': np.mean(exploration_rates),
            'std_exploration_rate': np.std(exploration_rates),
            'mean_steps_to_target': np.mean(steps_to_target) if steps_to_target else None,
            'std_steps_to_target': np.std(steps_to_target) if steps_to_target else None,
            'success_rate': len(steps_to_target) / len(episodes) * 100,
            'episode_count': len(episodes)
        }
    
    return analysis

def create_exploration_rate_comparison(analysis):
    """探索率の比較グラフ"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 箱ひげ図
    data_for_box = []
    labels = []
    for obstacle_density in [0.0, 0.003, 0.005]:
        if obstacle_density in analysis:
            data_for_box.append(analysis[obstacle_density]['exploration_rates'])
            labels.append(f'Obstacle {obstacle_density}')
    
    ax1.boxplot(data_for_box, labels=labels)
    ax1.set_title('Exploration Rate Distribution by Obstacle Density', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Final Exploration Rate')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 平均値と標準偏差の棒グラフ
    densities = []
    means = []
    stds = []
    
    for obstacle_density in [0.0, 0.003, 0.005]:
        if obstacle_density in analysis:
            densities.append(obstacle_density)
            means.append(analysis[obstacle_density]['mean_exploration_rate'])
            stds.append(analysis[obstacle_density]['std_exploration_rate'])
    
    ax2.bar(range(len(densities)), means, yerr=stds, capsize=5, alpha=0.7)
    ax2.set_title('Mean Exploration Rate with Standard Deviation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Obstacle Density')
    ax2.set_ylabel('Mean Exploration Rate')
    ax2.set_xticks(range(len(densities)))
    ax2.set_xticklabels([f'{d}' for d in densities])
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('config_a_exploration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_steps_analysis(analysis):
    """ステップ数の分析グラフ"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 80%到達までのステップ数
    data_for_steps = []
    labels = []
    for obstacle_density in [0.0, 0.003, 0.005]:
        if obstacle_density in analysis and analysis[obstacle_density]['steps_to_target']:
            data_for_steps.append(analysis[obstacle_density]['steps_to_target'])
            labels.append(f'Obstacle {obstacle_density}')
    
    if data_for_steps:
        ax1.boxplot(data_for_steps, labels=labels)
        ax1.set_title('Steps to 80% Exploration Rate', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Steps')
        ax1.grid(True, alpha=0.3)
    
    # 成功率の比較
    densities = []
    success_rates = []
    
    for obstacle_density in [0.0, 0.003, 0.005]:
        if obstacle_density in analysis:
            densities.append(obstacle_density)
            success_rates.append(analysis[obstacle_density]['success_rate'])
    
    bars = ax2.bar(range(len(densities)), success_rates, alpha=0.7, color=['green', 'orange', 'red'])
    ax2.set_title('Success Rate (Reaching 80% Exploration)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Obstacle Density')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_xticks(range(len(densities)))
    ax2.set_xticklabels([f'{d}' for d in densities])
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 成功率の値を棒グラフ上に表示
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('config_a_steps_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_episode_progression_analysis(results):
    """エピソード進行に伴う探索率の変化"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['blue', 'orange', 'red']
    obstacle_densities = [0.0, 0.003, 0.005]
    
    for i, (obstacle_density, color) in enumerate(zip(obstacle_densities, colors)):
        if obstacle_density in results:
            episodes = results[obstacle_density]['episodes']
            episode_numbers = [ep['episode'] for ep in episodes]
            exploration_rates = [ep['final_exploration_rate'] for ep in episodes]
            
            axes[i].scatter(episode_numbers, exploration_rates, alpha=0.6, color=color, s=20)
            axes[i].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Target')
            
            # 移動平均線
            if len(episode_numbers) > 10:
                window = min(10, len(episode_numbers) // 4)
                moving_avg = pd.Series(exploration_rates).rolling(window=window).mean()
                axes[i].plot(episode_numbers, moving_avg, color='black', linewidth=2, label=f'{window}-episode Moving Avg')
            
            axes[i].set_title(f'Obstacle Density: {obstacle_density}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel('Final Exploration Rate')
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('config_a_episode_progression.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(analysis):
    """統計サマリーの表示"""
    print("\n" + "="*60)
    print("📊 CONFIG_A 検証結果サマリー")
    print("="*60)
    
    for obstacle_density in [0.0, 0.003, 0.005]:
        if obstacle_density in analysis:
            data = analysis[obstacle_density]
            print(f"\n🔍 Obstacle Density: {obstacle_density}")
            print(f"   📈 平均探索率: {data['mean_exploration_rate']:.3f} ± {data['std_exploration_rate']:.3f}")
            print(f"   🎯 成功率: {data['success_rate']:.1f}%")
            if data['mean_steps_to_target']:
                print(f"   ⏱️  80%到達平均ステップ: {data['mean_steps_to_target']:.1f} ± {data['std_steps_to_target']:.1f}")
            print(f"   📊 エピソード数: {data['episode_count']}")
    
    print("\n" + "="*60)

def main():
    """メイン実行関数"""
    print("🚀 Config_A検証結果の分析を開始します...")
    
    # 結果の読み込み
    results = load_verification_results()
    
    if not results:
        print("❌ 結果ファイルが見つかりませんでした。")
        return
    
    # 分析実行
    analysis = analyze_exploration_performance(results)
    
    # 統計サマリーの表示
    print_summary_statistics(analysis)
    
    # グラフ生成
    print("\n📊 グラフを生成中...")
    create_exploration_rate_comparison(analysis)
    create_steps_analysis(analysis)
    create_episode_progression_analysis(results)
    
    print("\n✅ 分析完了！以下のファイルが生成されました:")
    print("   - config_a_exploration_analysis.png")
    print("   - config_a_steps_analysis.png")
    print("   - config_a_episode_progression.png")

if __name__ == "__main__":
    main() 