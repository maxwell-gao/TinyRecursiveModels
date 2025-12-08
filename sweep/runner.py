import subprocess
import wandb
import os
from typing import List, Dict

# 固定的训练脚本路径
TRAIN_SCRIPT = "pretrain_fabric.py"

STATIC_OVERRIDES = [
    'data_paths="[data/sudoku-extreme-1k-aug-1000]"',
    'evaluators="[]"',
    'epochs=50000', 
    'eval_interval=5000',
    '+optimizer=muon',  
    'grad_clip_norm=-1.0',
    'ema=True',
    'devices=1' 
]

def main():
    run = wandb.init()
    
    config = dict(run.config)
    dynamic_overrides = build_dynamic_overrides(config)
    run_name_override = f'run_name={run.name}'
    
    full_command = (
        ['python', TRAIN_SCRIPT] +
        STATIC_OVERRIDES +
        dynamic_overrides +
        [run_name_override]
    )
    
    print("=" * 60)
    print(f"Starting single-GPU training for Run: {run.name}")
    print(f"Command: {' '.join(full_command)}")
    print("=" * 60)
    
    try:
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        run.finish(status='crashed')
        raise
    
    run.finish(status='success')

if __name__ == "__main__":
    main()