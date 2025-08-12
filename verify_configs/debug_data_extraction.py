#!/usr/bin/env python3
"""
データ抽出の問題をデバッグするスクリプト
"""

import json
import os

def debug_data_extraction():
    """データ抽出の問題をデバッグ"""
    print("=== データ抽出デバッグ ===\n")
    
    # Config_Cの結果ファイルを読み込み
    file_path = "verify_configs/verification_results/Config_C_obstacle_0.0/verification_result.json"
    
    if not os.path.exists(file_path):
        print(f"❌ ファイルが見つかりません: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"設定: {data['config']}")
    print(f"環境: {data['environment']}")
    print(f"エピソード数: {len(data['episodes'])}")
    
    # 最初のエピソードの詳細を確認
    if data['episodes']:
        episode1 = data['episodes'][0]
        print(f"\n--- エピソード1の詳細 ---")
        print(f"エピソード番号: {episode1['episode']}")
        print(f"最終探査率: {episode1['final_exploration_rate']}")
        print(f"ステップ数: {episode1['steps_taken']}")
        print(f"step_detailsの数: {len(episode1.get('step_details', []))}")
        
        # step_detailsの最初の要素を確認
        if episode1.get('step_details'):
            first_step = episode1['step_details'][0]
            print(f"\n--- 最初のステップの詳細 ---")
            print(f"ステップ: {first_step['step']}")
            print(f"探査率: {first_step['exploration_rate']}")
            print(f"キー一覧: {list(first_step.keys())}")
    
    # サマリー情報を確認
    if 'summary' in data:
        summary = data['summary']
        print(f"\n--- サマリー情報 ---")
        print(f"総エピソード数: {summary['total_episodes']}")
        print(f"目標達成エピソード数: {summary['target_reached_episodes']}")
        print(f"平均探査率: {summary['average_exploration_rate']}")
        print(f"平均ステップ数: {summary['average_steps_taken']}")
    
    # エピソードごとの最終探査率を抽出
    print(f"\n--- エピソードごとの最終探査率 ---")
    final_rates = []
    for i, episode in enumerate(data['episodes']):
        rate = episode['final_exploration_rate']
        final_rates.append(rate)
        print(f"エピソード {i+1}: {rate:.3f}")
    
    print(f"\n最終探査率の統計:")
    print(f"最小値: {min(final_rates):.3f}")
    print(f"最大値: {max(final_rates):.3f}")
    print(f"平均値: {sum(final_rates)/len(final_rates):.3f}")

def test_analyze_function():
    """analyze_episode_progress関数をテスト"""
    print("\n=== analyze_episode_progress関数テスト ===\n")
    
    file_path = "verify_configs/verification_results/Config_C_obstacle_0.0/verification_result.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    def analyze_episode_progress(episodes):
        """エピソードごとの進捗を分析"""
        episode_data = []
        for episode in episodes:
            step_details = episode.get('step_details', [])
            print(f"エピソード {episode['episode']}: step_details数 = {len(step_details)}")
            
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
                print(f"  → 抽出成功: {len(exploration_rates)}個の探査率")
            else:
                print(f"  → step_detailsが空")
        
        return episode_data
    
    episodes = data['episodes']
    result = analyze_episode_progress(episodes)
    
    print(f"\n抽出結果: {len(result)}個のエピソードデータ")
    for ep_data in result:
        print(f"エピソード {ep_data['episode']}: {len(ep_data['exploration_rates'])}ステップ")

if __name__ == "__main__":
    debug_data_extraction()
    test_analyze_function()
