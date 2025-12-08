import subprocess
import wandb
import os
from typing import List, Dict

# --- Configuration ---
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
# --- END Configuration ---

def build_dynamic_overrides(config: Dict) -> List[str]:
    """
    Constructs dynamic Hydra/OmegaConf command-line arguments 
    from the WandB configuration dictionary.
    """
    dynamic_args = []
    
    for key, value in config.items():
        str_value = str(value)

        if key.startswith('arch.'):
            dynamic_args.append(f'{key}={str_value}')
        elif key == 'arch':
            dynamic_args.append(f'arch={str_value}')
        elif key in ['muon_lr', 'muon_weight_decay']:
            dynamic_args.append(f'+{key}={str_value}')
        elif key in ['lr', 'weight_decay', 'puzzle_emb_lr', 'puzzle_emb_weight_decay']:
            dynamic_args.append(f'{key}={str_value}')
        
    return dynamic_args

def main():
    # 1. Initialize WandB Run.
    run = wandb.init()
    
    # 2. Get hyperparameters and build command
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
    
    # 4. Execute the command and capture output
    try:
        # 💥 关键修改: 捕获输出并确保 check=True
        result = subprocess.run(
            full_command, 
            check=True,
            capture_output=True, # Capture stdout and stderr
            text=True            # Decode output to string
        )
        # If successful, print any standard output (optional, for debugging)
        if result.stdout:
             print("Subprocess STDOUT:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("CRASH LOG (STDERR) FROM pretrain_fabric.py:")
        print("=" * 60)
        print(e.stderr) # <--- THIS IS THE LOG WE NEED!
        print("=" * 60 + "\n")
        
        # 5. Finish Run without status argument
        run.finish() 
        raise # Re-raise the exception to terminate the Agent's trial
    
    # 5. Finish the Run (Success case)
    run.finish()

if __name__ == "__main__":
    main()