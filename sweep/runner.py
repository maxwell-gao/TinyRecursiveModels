import subprocess
import wandb
import os
from typing import List, Dict

# --- Configuration ---
# The path to the main training script.
TRAIN_SCRIPT = "pretrain_fabric.py"

# Fixed Hydra/OmegaConf training parameters (forcing single GPU usage)
STATIC_OVERRIDES = [
    'data_paths="[data/sudoku-extreme-1k-aug-1000]"',
    'evaluators="[]"',
    'epochs=50000', 
    'eval_interval=5000',
    '+optimizer=muon',  
    'grad_clip_norm=-1.0',
    'ema=True',
    # Key change: Force pretrain_fabric.py to use a single GPU (devices=1)
    'devices=1' 
]
# --- END Configuration ---

def build_dynamic_overrides(config: Dict) -> List[str]:
    """
    Constructs dynamic Hydra/OmegaConf command-line arguments 
    from the WandB configuration dictionary.
    """
    dynamic_args = []
    
    for key, value in config.items():
        # Convert config value to string for command-line passing
        str_value = str(value)

        if key.startswith('arch.'):
            # Handle nested parameters like arch.L_layers
            dynamic_args.append(f'{key}={str_value}')
        elif key == 'arch':
            # Handle arch=loop_transformer
            dynamic_args.append(f'arch={str_value}')
        elif key in ['muon_lr', 'muon_weight_decay']:
            # Handle plus-prefixed parameters: +muon_lr=...
            dynamic_args.append(f'+{key}={str_value}')
        elif key in ['lr', 'weight_decay', 'puzzle_emb_lr', 'puzzle_emb_weight_decay']:
            # Handle standard parameters: lr=...
            dynamic_args.append(f'{key}={str_value}')
        
    return dynamic_args

def main():
    # 1. Initialize WandB Run. This is the actual Trial run.
    run = wandb.init()
    
    # 2. Get hyperparameters from the WandB run config
    config = dict(run.config)
    
    # Build the list of dynamic arguments
    dynamic_overrides = build_dynamic_overrides(config) 
    
    # Construct the run_name parameter using the WandB run name
    run_name_override = f'run_name={run.name}'
    
    # 3. Construct the full single-GPU training command
    full_command = (
        ['python', TRAIN_SCRIPT] + # Start with python and script name
        STATIC_OVERRIDES +
        dynamic_overrides +
        [run_name_override]
    )
    
    print("=" * 60)
    print(f"Starting single-GPU training for Run: {run.name}")
    print(f"Command: {' '.join(full_command)}")
    print("=" * 60)
    
    # 4. Execute the command
    try:
        # Execute the training script as a subprocess.
        # We do NOT need torchrun or 'env' parameter here, 
        # as this is a single, isolated process managed by CUDA_VISIBLE_DEVICES.
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        run.finish(status='crashed')
        raise
    
    # 5. Finish the Run
    run.finish(status='success')

if __name__ == "__main__":
    main()