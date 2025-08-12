#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update Episode Number Script
verify_configsディレクトリの全ファイルのepisodeNumを10に変更する

使用方法:
    python update_episode_num.py
"""

import os
import re
from pathlib import Path

def update_episode_num_in_file(file_path: Path, target_episode_num: int = 10):
    """
    ファイル内のepisodeNumを指定した値に変更
    
    Args:
        file_path: 対象ファイルのパス
        target_episode_num: 変更後のエピソード数
    """
    try:
        # ファイルを読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # episodeNumの行を検索して置換
        # パターン: sim_param.episodeNum = 数値
        pattern = r'(sim_param\.episodeNum\s*=\s*)\d+'
        
        # 現在の値を確認
        matches = re.findall(pattern, content)
        if matches:
            # 置換実行
            new_content = re.sub(pattern, f'\\1{target_episode_num}', content)
            
            # ファイルに書き戻し
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✓ {file_path.name}: episodeNum を {target_episode_num} に変更しました")
            return True
        else:
            print(f"⚠️  {file_path.name}: episodeNum の設定が見つかりませんでした")
            return False
            
    except Exception as e:
        print(f"❌ {file_path.name}: エラー - {e}")
        return False

def main():
    """メイン関数"""
    print("🚀 verify_configs ディレクトリのepisodeNum更新を開始します\n")
    
    # verify_configsディレクトリのパス
    verify_configs_dir = Path("verify_configs")
    
    if not verify_configs_dir.exists():
        print(f"❌ ディレクトリが存在しません: {verify_configs_dir}")
        return
    
    # verify_config_*.py ファイルを検索
    config_files = list(verify_configs_dir.glob("verify_config_*.py"))
    
    if not config_files:
        print("❌ verify_config_*.py ファイルが見つかりませんでした")
        return
    
    print(f"📁 対象ファイル数: {len(config_files)}")
    print()
    
    # 各ファイルを更新
    success_count = 0
    target_episode_num = 10
    
    for config_file in sorted(config_files):
        if update_episode_num_in_file(config_file, target_episode_num):
            success_count += 1
    
    print(f"\n🎉 更新完了!")
    print(f"✓ 成功: {success_count}/{len(config_files)} ファイル")
    print(f"📊 全てのepisodeNumを {target_episode_num} に設定しました")

if __name__ == "__main__":
    main() 