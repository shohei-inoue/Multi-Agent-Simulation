"""
Centralized logging system for the project.
Provides unified logging across all components.
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import tensorflow as tf
from contextlib import contextmanager

from core.config import config


class Logger:
    """Centralized logger"""
    
    def __init__(self, name: str, log_dir: Optional[str] = None):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path(config.simulation.log_dir)
        self.logger = logging.getLogger(name)
        self._setup_logger()
        
        # Metrics storage
        self.metrics: Dict[str, List[Any]] = {}
        self.episode_data: List[Dict[str, Any]] = []
    
    def _setup_logger(self):
        """Setup logger configuration"""
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler - ensure directory exists
        log_file = self.log_dir / f"{self.name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_metric(self, name: str, value: Any, step: Optional[int] = None):
        """Log a metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_data = {
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics[name].append(metric_data)
    
    def log_episode(self, episode_data: Dict[str, Any]):
        """Log episode data"""
        episode_data["timestamp"] = datetime.now().isoformat()
        self.episode_data.append(episode_data)
    
    def save_metrics(self, filename: str = "metrics.json"):
        """Save metrics to file"""
        metrics_file = self.log_dir / filename
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def save_episodes(self, filename: str = "episodes.csv"):
        """Save episode data to CSV"""
        if self.episode_data:
            episodes_file = self.log_dir / filename
            df = pd.DataFrame(self.episode_data)
            df.to_csv(episodes_file, index=False)
    
    def get_tensorboard_writer(self, subdir: str = "tensorboard"):
        """Get TensorBoard writer"""
        tb_dir = self.log_dir / subdir
        return tf.summary.create_file_writer(str(tb_dir))
    
    @contextmanager
    def tensorboard_logging(self, subdir: str = "tensorboard"):
        """Context manager for TensorBoard logging"""
        writer = self.get_tensorboard_writer(subdir)
        try:
            yield writer
        finally:
            writer.close()


class ComponentLogger:
    """Logger for specific components"""
    
    def __init__(self, component_name: str, parent_logger: Optional[Logger] = None):
        self.component_name = component_name
        self.parent_logger = parent_logger or Logger("main")
        self.logger = Logger(f"{self.parent_logger.name}.{component_name}")
    
    def log_component_event(self, event: str, data: Dict[str, Any] = None):
        """Log component-specific event"""
        message = f"[{self.component_name}] {event}"
        if data:
            message += f" - {json.dumps(data, default=str)}"
        self.logger.info(message)
    
    def log_performance(self, metric_name: str, value: float, step: int = None):
        """Log performance metric"""
        self.logger.log_metric(f"{self.component_name}_{metric_name}", value, step)
    
    def log_episode(self, episode_data: Dict[str, Any]):
        """Log episode data"""
        self.logger.log_episode(episode_data)


# Global logger instance
main_logger = Logger("main")


def get_logger(name: str) -> Logger:
    """Get a logger instance"""
    return Logger(name)


def get_component_logger(component_name: str) -> ComponentLogger:
    """Get a component logger"""
    return ComponentLogger(component_name, main_logger) 