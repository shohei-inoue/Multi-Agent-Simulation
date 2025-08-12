#!/usr/bin/env python3
"""
TensorBoard起動スクリプト
学習状況を可視化するためのTensorBoardを起動します。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def start_tensorboard(log_dir: str = "./logs", port: int = 6006, host: str = "localhost"):
    """
    TensorBoardを起動
    
    Args:
        log_dir: ログディレクトリのパス
        port: TensorBoardのポート番号
        host: TensorBoardのホストアドレス
    """
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Error: Log directory '{log_dir}' does not exist.")
        print("Please run the simulation first to generate logs.")
        sys.exit(1)
    
    # TensorBoardのログディレクトリを探す
    tensorboard_dirs = []
    for item in log_path.rglob("tensorboard"):
        if item.is_dir():
            tensorboard_dirs.append(item)
    
    if not tensorboard_dirs:
        print(f"Error: No TensorBoard logs found in '{log_dir}'")
        print("Please run the simulation first to generate TensorBoard logs.")
        sys.exit(1)
    
    print(f"Found TensorBoard logs in:")
    for i, tb_dir in enumerate(tensorboard_dirs):
        print(f"  {i+1}. {tb_dir}")
    
    # 最新のログディレクトリを使用
    latest_log_dir = max(tensorboard_dirs, key=lambda x: x.stat().st_mtime)
    print(f"\nUsing latest log directory: {latest_log_dir}")
    
    # TensorBoardを起動
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(latest_log_dir),
        "--port", str(port),
        "--host", host
    ]
    
    print(f"\nStarting TensorBoard...")
    print(f"Command: {' '.join(cmd)}")
    print(f"URL: http://{host}:{port}")
    print("\nPress Ctrl+C to stop TensorBoard")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error starting TensorBoard: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: TensorBoard not found. Please install it with:")
        print("  pip install tensorboard")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Start TensorBoard for Multi-Agent Simulation")
    parser.add_argument(
        "--log-dir", 
        default="./logs", 
        help="Log directory path (default: ./logs)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6006, 
        help="TensorBoard port (default: 6006)"
    )
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="TensorBoard host (default: localhost)"
    )
    
    args = parser.parse_args()
    
    start_tensorboard(args.log_dir, args.port, args.host)


if __name__ == "__main__":
    main() 