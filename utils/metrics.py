import pandas as pd
import json
import os

def save_metrics(metrics_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(output_dir, "train_metrics.csv"), index=False)
    
    with open(os.path.join(output_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics_list, f, indent=2)