import subprocess
import sys
import os

def run_command(command):
    print(f"\n[EXEC] {command}")
    process = subprocess.Popen(command, shell=True)
    process.communicate()
    if process.returncode != 0:
        print(f"CRITICAL ERROR. Command failed: {command}")
        sys.exit(1)

experiments = [
    {"name": "baseline", "loss_type": "none", "lambda_val": 0.0},
    {"name": "tv", "loss_type": "tv", "lambda_val": 50.0},
    {"name": "monotonicity", "loss_type": "monotonicity", "lambda_val": 50.0},
    {"name": "all_priors", "loss_type": "all", "lambda_val": 50.0}
]

seeds = [10, 20, 30, 40, 50]

print("=== STARTING 20-RUN EXPERIMENTAL MATRIX ===")

for exp in experiments:
    for seed in seeds:
        exp_name = f"{exp['name']}_seed_{seed}"
        
        # Check if model already exists to save time if script crashes
        if os.path.exists(f"saved_models/{exp_name}/generator_final.pth"):
            print(f"Skipping {exp_name} (Already Trained)")
            continue
            
        cmd = f"python train_wgan.py --exp_name {exp_name} --loss_type {exp['loss_type']} --lambda_physics {exp['lambda_val']} --seed {seed}"
        run_command(cmd)

print("\n=== MATRIX COMPLETE. RUNNING STATISTICAL EVALUATION ===")
run_command("python evaluate_metrics.py")