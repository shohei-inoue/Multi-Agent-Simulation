#!/usr/bin/env python3
"""
Config_AとConfig_Cの結果を比較分析するスクリプト
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_verification_result(config_name, obstacle_density):
    """検証結果を読み込む"""
    file_path = f"verification_results/{config_name}_obstacle_{obstacle_density}/verification_result.json"
    if not os.path.exists(file_path):
        print(f"❌ ファイルが見つかりません: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_episode_progress(episodes):
    """エピソードごとの進捗を分析"""
    episode_data = []
    print(f"    デバッグ: {len(episodes)}個のエピソードを処理中...")
    
    for episode in episodes:
        step_details = episode.get('step_details', [])
        print(f"    エピソード {episode['episode']}: step_details数 = {len(step_details)}")
        
        if step_details:
            # 各ステップでの探査率を抽出
            exploration_rates = [step['exploration_rate'] for step in step_details]
            steps = [step['step'] for step in step_details]
            
            episode_data.append({
                'episode': episode['episode'],
                'steps': steps,
                'exploration_rates': exploration_rates,
                'final_rate': episode['final_exploration_rate'],
                'steps_taken': episode['steps_taken']
            })
            print(f"      → 抽出成功: {len(exploration_rates)}個の探査率")
        else:
            print(f"      → step_detailsが空")
    
    print(f"    総抽出数: {len(episode_data)}個のエピソードデータ")
    return episode_data

def compare_configs():
    """設定を比較分析"""
    print("=== Config_A vs Config_C 比較分析 ===\n")
    
    # 障害物密度ごとの結果を比較
    obstacle_densities = [0.0, 0.003, 0.005]
    
    for density in obstacle_densities:
        print(f"--- 障害物密度: {density} ---")
        
        # 結果を読み込み
        config_a = load_verification_result("Config_A", density)
        config_c = load_verification_result("Config_C", density)
        
        if config_a is None or config_c is None:
            continue
        
        # サマリー情報を比較
        summary_a = config_a['summary']
        summary_c = config_c['summary']
        
        print(f"Config_A: 平均探査率 {summary_a['average_exploration_rate']:.3f} ± {summary_a['std_exploration_rate']:.3f}")
        print(f"Config_C: 平均探査率 {summary_c['average_exploration_rate']:.3f} ± {summary_c['std_exploration_rate']:.3f}")
        
        # 改善率を計算
        improvement = ((summary_c['average_exploration_rate'] - summary_a['average_exploration_rate']) / 
                      summary_a['average_exploration_rate']) * 100
        print(f"改善率: {improvement:+.1f}%")
        
        # エピソードごとの詳細分析
        episodes_a = analyze_episode_progress(config_a['episodes'])
        episodes_c = analyze_episode_progress(config_c['episodes'])
        
        print(f"エピソード数: A={len(episodes_a)}, C={len(episodes_c)}")
        
        # 各エピソードでの最終探査率を比較
        final_rates_a = [ep['final_rate'] for ep in episodes_a]
        final_rates_c = [ep['final_rate'] for ep in episodes_c]
        
        if final_rates_a and final_rates_c:
            print(f"最終探査率範囲: A=[{min(final_rates_a):.3f}, {max(final_rates_a):.3f}], C=[{min(final_rates_c):.3f}, {max(final_rates_c):.3f}]")
        else:
            print("エピソードデータが不足しています")
        
        # 目標達成エピソード数
        target_a = summary_a['target_reached_episodes']
        target_c = summary_c['target_reached_episodes']
        print(f"目標達成エピソード数: A={target_a}, C={target_c}")
        
        print()

def create_comparison_charts():
    """比較チャートを作成"""
    print("=== 比較チャート作成中 ===")
    
    # 障害物密度ごとの平均探査率を比較
    densities = [0.0, 0.003, 0.005]
    config_a_rates = []
    config_c_rates = []
    
    for density in densities:
        config_a = load_verification_result("Config_A", density)
        config_c = load_verification_result("Config_C", density)
        
        if config_a and config_c:
            config_a_rates.append(config_a['summary']['average_exploration_rate'])
            config_c_rates.append(config_c['summary']['average_exploration_rate'])
        else:
            config_a_rates.append(0)
            config_c_rates.append(0)
    
    # チャート作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 平均探査率比較
    x = np.arange(len(densities))
    width = 0.35
    
    ax1.bar(x - width/2, config_a_rates, width, label='Config_A', alpha=0.8)
    ax1.bar(x + width/2, config_c_rates, width, label='Config_C', alpha=0.8)
    
    ax1.set_xlabel('障害物密度')
    ax1.set_ylabel('平均探査率')
    ax1.set_title('Config_A vs Config_C: 平均探査率比較')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{d:.3f}' for d in densities])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 改善率
    improvements = [(c - a) / a * 100 if a > 0 else 0 for a, c in zip(config_a_rates, config_c_rates)]
    
    ax2.bar(x, improvements, color='green', alpha=0.7)
    ax2.set_xlabel('障害物密度')
    ax2.set_ylabel('改善率 (%)')
    ax2.set_title('Config_C の Config_A に対する改善率')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{d:.3f}' for d in densities])
    ax2.grid(True, alpha=0.3)
    
    # ゼロラインを追加
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存
    output_dir = "verification_results"
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, "config_A_C_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ チャート保存完了: {chart_path}")
    
    plt.show()

def detailed_episode_analysis():
    """エピソード詳細分析"""
    print("\n=== エピソード詳細分析 ===")
    
    # 障害物なしの設定で詳細分析
    config_a = load_verification_result("Config_A", 0.0)
    config_c = load_verification_result("Config_C", 0.0)
    
    if not config_a or not config_c:
        return
    
    episodes_a = analyze_episode_progress(config_a['episodes'])
    episodes_c = analyze_episode_progress(config_c['episodes'])
    
    print(f"\n--- エピソード1の詳細比較 ---")
    if episodes_a and episodes_c:
        ep1_a = episodes_a[0]
        ep1_c = episodes_c[0]
        
        print(f"Config_A エピソード1:")
        print(f"  最終探査率: {ep1_a['final_rate']:.3f}")
        print(f"  ステップ数: {ep1_a['steps_taken']}")
        print(f"  初期探査率: {ep1_a['exploration_rates'][0]:.3f}")
        
        print(f"Config_C エピソード1:")
        print(f"  最終探査率: {ep1_c['final_rate']:.3f}")
        print(f"  ステップ数: {ep1_c['steps_taken']}")
        print(f"  初期探査率: {ep1_c['exploration_rates'][0]:.3f}")
        
        # 探査率の変化をプロット
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(ep1_a['steps'], ep1_a['exploration_rates'], 'b-', label='Config_A', linewidth=2)
        ax.plot(ep1_c['steps'], ep1_c['exploration_rates'], 'r-', label='Config_C', linewidth=2)
        
        ax.set_xlabel('ステップ')
        ax.set_ylabel('探査率')
        ax.set_title('エピソード1: 探査率の変化比較 (Config_A vs Config_C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 目標ラインを追加
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='目標 (80%)')
        ax.legend()
        
        plt.tight_layout()
        
        # 保存
        output_dir = "verification_results"
        chart_path = os.path.join(output_dir, "episode1_comparison.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"✓ エピソード1比較チャート保存完了: {chart_path}")
        
        plt.show()

def main():
    """メイン処理"""
    print(f"開始時刻: {datetime.now()}")
    
    # 基本比較
    compare_configs()
    
    # チャート作成
    create_comparison_charts()
    
    # 詳細分析
    detailed_episode_analysis()
    
    print(f"\n終了時刻: {datetime.now()}")
    print("🎉 比較分析が完了しました！")

if __name__ == "__main__":
    main()
