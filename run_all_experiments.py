"""
全実験統合実行スクリプト
学習、検証、解析を順次実行する
"""

import os
import sys
import subprocess
import time
from datetime import datetime


def run_command(command: str, description: str) -> bool:
    """コマンド実行"""
    print(f"\n=== {description} ===")
    print(f"実行コマンド: {command}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        print(f"✅ {description} 完了 (実行時間: {end_time - start_time:.1f}秒)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} エラー:")
        print(f"エラーコード: {e.returncode}")
        print(f"標準出力: {e.stdout}")
        print(f"標準エラー: {e.stderr}")
        return False


def main():
    """メイン実行関数"""
    print("=== 全実験統合実行開始 ===")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 学習実行
    print("\n📚 ステップ1: モデル学習")
    success = run_command("python3 train_models.py", "モデル学習")
    if not success:
        print("❌ 学習に失敗しました。処理を中止します。")
        return False
    
    # 学習完了確認
    if not os.path.exists("trained_models"):
        print("❌ 学習済みモデルディレクトリが作成されませんでした。")
        return False
    
    print("✅ 学習済みモデル確認完了")
    
    # 2. 検証実行
    print("\n🔍 ステップ2: 検証実行")
    success = run_command("python3 run_verification.py", "検証実行")
    if not success:
        print("❌ 検証に失敗しました。処理を中止します。")
        return False
    
    # 検証結果確認
    if not os.path.exists("verification_results"):
        print("❌ 検証結果ディレクトリが作成されませんでした。")
        return False
    
    print("✅ 検証結果確認完了")
    
    # 3. 解析実行
    print("\n📊 ステップ3: 結果解析")
    success = run_command("python3 analyze_results.py", "結果解析")
    if not success:
        print("❌ 解析に失敗しました。")
        return False
    
    # 解析結果確認
    if not os.path.exists("analysis_results"):
        print("❌ 解析結果ディレクトリが作成されませんでした。")
        return False
    
    print("✅ 解析結果確認完了")
    
    # 完了報告
    print("\n🎉 === 全実験完了 ===")
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 結果ディレクトリ一覧
    print("\n📁 生成されたディレクトリ:")
    directories = ["trained_models", "training_results", "verification_results", "analysis_results"]
    for dir_name in directories:
        if os.path.exists(dir_name):
            file_count = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
            print(f"  {dir_name}/ ({file_count} ファイル)")
    
    print("\n📋 次のステップ:")
    print("  1. analysis_results/ ディレクトリでグラフとテーブルを確認")
    print("  2. verification_results/ ディレクトリでGIFログを確認")
    print("  3. 必要に応じて個別スクリプトを再実行")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 