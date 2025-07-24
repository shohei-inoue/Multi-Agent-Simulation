"""
Enhanced logging utilities with TensorBoard support for learning progress tracking.
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoardX not available. Install with: pip install tensorboardX")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available for TensorBoard logging")


class TensorBoardLogger:
    """TensorBoard対応のログ記録クラス"""
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoardX writer
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(str(self.tensorboard_dir))
        else:
            self.writer = None
        
        # TensorFlow TensorBoard writer
        if TF_AVAILABLE:
            self.tf_writer = tf.summary.create_file_writer(str(self.tensorboard_dir))
        else:
            self.tf_writer = None
        
        # メトリクス保存用
        self.metrics = {
            'episode_rewards': [],
            'exploration_rates': [],
            'swarm_counts': [],
            'learning_losses': [],
            'branch_events': [],
            'integration_events': []
        }
    
    def log_episode_metrics(self, episode: int, metrics: Dict[str, float]):
        """エピソードメトリクスを記録"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'episode/{key}', value, episode)
        
        if self.tf_writer:
            with self.tf_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(f'episode/{key}', value, step=episode)
                self.tf_writer.flush()
        
        # メトリクスを保存
        for key in self.metrics.keys():
            if key in metrics:
                self.metrics[key].append(metrics[key])
    
    def log_learning_metrics(self, step: int, metrics: Dict[str, float], agent_type: str = "swarm"):
        """学習メトリクスを記録"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'learning/{agent_type}/{key}', value, step)
        
        if self.tf_writer:
            with self.tf_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(f'learning/{agent_type}/{key}', value, step=step)
                self.tf_writer.flush()
    
    def log_swarm_metrics(self, episode: int, swarm_id: int, metrics: Dict[str, float]):
        """群固有のメトリクスを記録"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'swarm/{swarm_id}/{key}', value, episode)
        
        if self.tf_writer:
            with self.tf_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(f'swarm/{swarm_id}/{key}', value, step=episode)
                self.tf_writer.flush()
    
    def log_system_metrics(self, episode: int, metrics: Dict[str, float]):
        """システムエージェントのメトリクスを記録"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'system/{key}', value, episode)
        
        if self.tf_writer:
            with self.tf_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(f'system/{key}', value, step=episode)
                self.tf_writer.flush()
    
    def log_branch_event(self, episode: int, source_swarm_id: int, new_swarm_id: int, 
                        valid_directions: int, mobility_score: float):
        """分岐イベントを記録"""
        event_data = {
            'source_swarm_id': source_swarm_id,
            'new_swarm_id': new_swarm_id,
            'valid_directions': valid_directions,
            'mobility_score': mobility_score
        }
        
        if self.writer:
            self.writer.add_scalar('events/branch_count', 1, episode)
            self.writer.add_scalar('events/branch_mobility_score', mobility_score, episode)
        
        self.metrics['branch_events'].append({
            'episode': episode,
            'event': 'branch',
            'data': event_data
        })
    
    def log_integration_event(self, episode: int, source_swarm_id: int, target_swarm_id: int,
                             mobility_score: float):
        """統合イベントを記録"""
        event_data = {
            'source_swarm_id': source_swarm_id,
            'target_swarm_id': target_swarm_id,
            'mobility_score': mobility_score
        }
        
        if self.writer:
            self.writer.add_scalar('events/integration_count', 1, episode)
            self.writer.add_scalar('events/integration_mobility_score', mobility_score, episode)
        
        self.metrics['integration_events'].append({
            'episode': episode,
            'event': 'integration',
            'data': event_data
        })
    
    def save_metrics_to_json(self, filename: str = "metrics.json"):
        """メトリクスをJSONファイルに保存"""
        metrics_file = self.log_dir / filename
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def save_metrics_to_csv(self, filename: str = "metrics.csv"):
        """メトリクスをCSVファイルに保存"""
        csv_file = self.log_dir / filename
        
        # エピソードメトリクスをCSVに保存
        if self.metrics['episode_rewards']:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'reward', 'exploration_rate', 'swarm_count'])
                
                for i in range(len(self.metrics['episode_rewards'])):
                    writer.writerow([
                        i,
                        self.metrics['episode_rewards'][i] if i < len(self.metrics['episode_rewards']) else 0,
                        self.metrics['exploration_rates'][i] if i < len(self.metrics['exploration_rates']) else 0,
                        self.metrics['swarm_counts'][i] if i < len(self.metrics['swarm_counts']) else 0
                    ])
    
    def close(self):
        """ログを閉じる"""
        if self.writer:
            self.writer.close()
        if self.tf_writer:
            self.tf_writer.close()


class ExperimentLogger:
    """実験全体のログ管理クラス"""
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.tensorboard_logger = TensorBoardLogger(log_dir, experiment_name)
        
        # 実験設定の保存
        self.experiment_config = {}
        
        # メトリクス保存用ディレクトリ
        self.metrics_dir = self.log_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # エピソードメトリクスの累積
        self.episode_metrics = []
    
    def set_experiment_config(self, config: Dict[str, Any]):
        """実験設定を保存"""
        self.experiment_config = config
        config_file = self.log_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_episode(self, episode: int, episode_data: Dict[str, Any]):
        """エピソードデータを記録"""
        # TensorBoardにメトリクスを記録
        metrics = {
            'total_reward': episode_data.get('total_reward', 0),
            'exploration_rate': episode_data.get('exploration_rate', 0),
            'swarm_count': episode_data.get('swarm_count', 1),
            'steps': episode_data.get('steps', 0),
            'duration': episode_data.get('duration', 0)
        }
        self.tensorboard_logger.log_episode_metrics(episode, metrics)
        
        # エピソード詳細をJSONに保存
        episode_file = self.log_dir / f"episode_{episode:04d}.json"
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f, indent=2, default=str)
        
        # メトリクスを累積
        self.episode_metrics.append({
            'episode': episode,
            **metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # エピソードごとのメトリクスを個別保存
        episode_metrics_file = self.metrics_dir / f"episode_{episode:04d}_metrics.json"
        with open(episode_metrics_file, 'w') as f:
            json.dump({
                'episode': episode,
                'metrics': metrics,
                'full_data': episode_data,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def save_episode_summary(self):
        """エピソードサマリーを保存"""
        if not self.episode_metrics:
            return
        
        # 全エピソードのサマリー
        summary = {
            'experiment_name': self.experiment_name,
            'total_episodes': len(self.episode_metrics),
            'episode_metrics': self.episode_metrics,
            'summary_stats': {
                'avg_reward': np.mean([m['total_reward'] for m in self.episode_metrics]),
                'max_reward': np.max([m['total_reward'] for m in self.episode_metrics]),
                'min_reward': np.min([m['total_reward'] for m in self.episode_metrics]),
                'avg_exploration_rate': np.mean([m['exploration_rate'] for m in self.episode_metrics]),
                'final_exploration_rate': self.episode_metrics[-1]['exploration_rate'] if self.episode_metrics else 0,
                'avg_swarm_count': np.mean([m['swarm_count'] for m in self.episode_metrics]),
                'max_swarm_count': np.max([m['swarm_count'] for m in self.episode_metrics]),
                'avg_steps': np.mean([m['steps'] for m in self.episode_metrics])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # サマリーファイルを保存
        summary_file = self.metrics_dir / "episode_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # CSV形式でも保存
        csv_file = self.metrics_dir / "episode_metrics.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'total_reward', 'exploration_rate', 'swarm_count', 'steps', 'duration', 'timestamp'])
            for metric in self.episode_metrics:
                writer.writerow([
                    metric['episode'],
                    metric['total_reward'],
                    metric['exploration_rate'],
                    metric['swarm_count'],
                    metric['steps'],
                    metric['duration'],
                    metric['timestamp']
                ])
    
    def log_learning_progress(self, step: int, agent_type: str, learning_data: Dict[str, float]):
        """学習進捗を記録"""
        self.tensorboard_logger.log_learning_metrics(step, learning_data, agent_type)
        
        # 学習データも個別保存
        learning_file = self.metrics_dir / f"learning_{agent_type}_step_{step:06d}.json"
        with open(learning_file, 'w') as f:
            json.dump({
                'step': step,
                'agent_type': agent_type,
                'learning_data': learning_data,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def log_swarm_progress(self, episode: int, swarm_id: int, swarm_data: Dict[str, float]):
        """群の進捗を記録"""
        self.tensorboard_logger.log_swarm_metrics(episode, swarm_id, swarm_data)
        
        # 群データも個別保存
        swarm_file = self.metrics_dir / f"swarm_{swarm_id}_episode_{episode:04d}.json"
        with open(swarm_file, 'w') as f:
            json.dump({
                'episode': episode,
                'swarm_id': swarm_id,
                'swarm_data': swarm_data,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def log_system_progress(self, episode: int, system_data: Dict[str, float]):
        """システムエージェントの進捗を記録"""
        self.tensorboard_logger.log_system_metrics(episode, system_data)
        
        # システムデータも個別保存
        system_file = self.metrics_dir / f"system_episode_{episode:04d}.json"
        with open(system_file, 'w') as f:
            json.dump({
                'episode': episode,
                'system_data': system_data,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def log_branch_event(self, episode: int, source_swarm_id: int, new_swarm_id: int,
                        valid_directions: int, mobility_score: float):
        """分岐イベントを記録"""
        self.tensorboard_logger.log_branch_event(episode, source_swarm_id, new_swarm_id,
                                                valid_directions, mobility_score)
        
        # 分岐イベントも個別保存
        branch_file = self.metrics_dir / f"branch_event_episode_{episode:04d}.json"
        with open(branch_file, 'w') as f:
            json.dump({
                'episode': episode,
                'event_type': 'branch',
                'source_swarm_id': source_swarm_id,
                'new_swarm_id': new_swarm_id,
                'valid_directions': valid_directions,
                'mobility_score': mobility_score,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def log_integration_event(self, episode: int, source_swarm_id: int, target_swarm_id: int,
                             mobility_score: float):
        """統合イベントを記録"""
        self.tensorboard_logger.log_integration_event(episode, source_swarm_id, target_swarm_id,
                                                     mobility_score)
        
        # 統合イベントも個別保存
        integration_file = self.metrics_dir / f"integration_event_episode_{episode:04d}.json"
        with open(integration_file, 'w') as f:
            json.dump({
                'episode': episode,
                'event_type': 'integration',
                'source_swarm_id': source_swarm_id,
                'target_swarm_id': target_swarm_id,
                'mobility_score': mobility_score,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def save_final_report(self):
        """最終レポートを保存"""
        # エピソードサマリーを保存
        self.save_episode_summary()
        
        # メトリクスを保存
        self.tensorboard_logger.save_metrics_to_json()
        self.tensorboard_logger.save_metrics_to_csv()
        
        # 最終レポート
        final_report = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.experiment_config,
            'metrics_summary': {
                'total_episodes': len(self.tensorboard_logger.metrics['episode_rewards']),
                'final_exploration_rate': self.tensorboard_logger.metrics['exploration_rates'][-1] if self.tensorboard_logger.metrics['exploration_rates'] else 0,
                'max_swarm_count': max(self.tensorboard_logger.metrics['swarm_counts']) if self.tensorboard_logger.metrics['swarm_counts'] else 1,
                'total_branch_events': len(self.tensorboard_logger.metrics['branch_events']),
                'total_integration_events': len(self.tensorboard_logger.metrics['integration_events'])
            }
        }
        
        report_file = self.log_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
    
    def close(self):
        """ログを閉じる"""
        self.tensorboard_logger.close()


def create_experiment_logger(log_dir: str, experiment_name: str = "experiment") -> ExperimentLogger:
    """実験ロガーを作成"""
    return ExperimentLogger(log_dir, experiment_name)