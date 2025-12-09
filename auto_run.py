import subprocess
import signal

# Define the mode

# Define training parameter combinations
trajectories = ['7']
loss_models = ['patched_0.005_0.01']
steps = ['150k']

dataset_name = "surgt"
finetune_dataset_name = dataset_name
finetune_trajectory = "7"

# Iterate over all parameter combinations
for trajectory in trajectories:
    for loss_model in loss_models:
        for step in steps:
            print(f"\n>>> Running TRAIN with: Trajectory={trajectory}, Loss={loss_model}, Steps={step} <<<\n")

            if finetune_dataset_name is not None:
                print(f">>> Finetune with: Dataset={finetune_dataset_name}, Trajectory={finetune_trajectory} <<<\n")

            # Construct training command
            command = [
                "python", "./run_model.py",
                "--mode", "train",
                "--dataset_name", dataset_name,
                "--trajectory", trajectory,
                "--finetune_dataset_name", dataset_name,
                "--finetune_trajectory", finetune_trajectory,
                "--loss_model", loss_model,
                "--steps", step
            ]

            # Run training script
            try:
                process = subprocess.Popen(command)
                process.wait()
            except KeyboardInterrupt:
                print("\n⚠️ Training interrupted! Attempting graceful shutdown...")
                process.send_signal(signal.SIGINT)
                process.wait()
