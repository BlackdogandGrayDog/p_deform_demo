import subprocess
import os
import re  # For extracting loss model and steps
from result_export import extract_distances

# Define the mode
evaluation_mode = "base" # base or finetune


# Fixed parameters
dataset_name = "hamlyn"
trajectory = "11" if evaluation_mode == "base" else "12"
step_size = 9
# Base directory where model folders are stored
base_dir = f'./result/{dataset_name}/trajectory_{trajectory}/'

# Automatically get all model folder names inside `base_dir`
model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
print(f"✅ Found {len(model_dirs)} model folders in {base_dir}")

# Extract loss_model and steps from folder name using regex
def parse_model_name(model_name):
    match = re.match(r"(.+)_loss_(\d+k)_steps", model_name)
    if match:
        loss_model, steps = match.groups()
        return loss_model, steps
    else:
        print(f"Skipping '{model_name}': Doesn't match expected format.")
        return None, None

# Evaluation setting (only trj4 with the specified grid sizes)
eval_dataset_name = "hamlyn"

if evaluation_mode == "base":
    evaluation_settings = {
        "11100": [10, 12, 13, 15, 16, 17, 20],
        "11101": [10, 12, 13, 15, 16, 17, 20],
        "11102": [10, 12, 13, 15, 16, 17, 20],
        "11103": [10, 12, 13, 15, 16, 17, 20],
        "11104": [10, 12, 13, 15, 16, 17, 20],
        "11105": [10, 12, 13, 15, 16, 17, 20],
    }
else: 
    evaluation_settings = {
        "12100": [10, 12, 13, 15, 16, 17, 20],
        "12101": [10, 12, 13, 15, 16, 17, 20],
        "12102": [10, 12, 13, 15, 16, 17, 20],
        "12103": [10, 12, 13, 15, 16, 17, 20],
        "12104": [10, 12, 13, 15, 16, 17, 20],
        "12105": [10, 12, 13, 15, 16, 17, 20],
    }


try:
    # Iterate over each detected model and run all evaluations
    for model_dir in model_dirs:
        loss_model, steps = parse_model_name(model_dir)
        
        if loss_model is None:
            continue  # Skip folders that don't match the expected format

        for eval_traj_num, eval_gs_list in evaluation_settings.items():
            for eval_gs in eval_gs_list:
                print(f"\n>>> Running EVAL with: Model={loss_model}, Steps={steps}, Trajectory={trajectory}, Eval Traj={eval_traj_num}, Eval GS={eval_gs} <<<")

                try:
                    result = subprocess.run([
                        "python", "run_model.py",
                        "--mode", "eval",
                        "--dataset_name", dataset_name,
                        "--eval_dataset_name", eval_dataset_name,
                        "--trajectory", trajectory,
                        "--loss_model", loss_model,
                        "--steps", steps,
                        "--eval_traj_num", eval_traj_num,
                        "--eval_gs", str(eval_gs),
                        "--step_size", str(step_size)
                    ], capture_output=True, text=True, check=True)
                    
                    print(f"✅ Finished EVAL: Model={loss_model}, Steps={steps}, Trajectory={trajectory}, Eval Traj={eval_traj_num}, Eval GS={eval_gs}")  

                except subprocess.CalledProcessError as e:
                    print(f"\n❌ ERROR: Execution failed for Model={loss_model}, Steps={steps}, Eval GS={eval_gs}")
                    print(f"🔍 STDERR:\n{e.stderr}")
                    raise  # Stop execution immediately

except Exception as e:
    print(f"\n❌ Unexpected Error: {str(e)}")
    exit(1)

except KeyboardInterrupt:
    print("\n⚠️ Execution interrupted.")
    
finally:
    extract_distances(base_dir, max_step_size=step_size)
    print("\n🔄 Execution completed.")
