"""
Metrics and performance measurement utilities.
Provides tools for tracking and analyzing simulation performance.
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any
from pathlib import Path

from core.logging import get_component_logger

logger = get_component_logger("metrics")


def save_metrics(metrics_list: List[Dict[str, Any]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(output_dir, "train_metrics.csv"), index=False)
    
    with open(os.path.join(output_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics_list, f, indent=2)