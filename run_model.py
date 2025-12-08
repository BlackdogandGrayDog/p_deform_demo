# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Runs the learner/evaluator."""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import time
import argparse
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow.compat.v1 as tf
from meshgraphnets import cfd_eval
from meshgraphnets import cfd_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
from meshgraphnets import core_model
from meshgraphnets import dataset

from validate_ply import process_evaluation

# ========== Argument Parsing ==========
parser = argparse.ArgumentParser(description="Run training or evaluation with different parameters.")
parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"], help="Mode to run: train or eval.")
parser.add_argument("--model", type=str, default="cloth", choices=["cfd", "cloth"], help="Select model to run. (Default: cloth)")
parser.add_argument("--dataset_name", type=str, default="simulator", required=True, help="Dataset to use.")
parser.add_argument("--trajectory", type=str, required=True, help="Trajectory to evaluate.")
parser.add_argument("--finetune_dataset_name", type=str, default=None, choices=["hamlyn", "simulator", "simulator_noise", "simulator_noise_eval"], help="Dataset to use.")
parser.add_argument("--finetune_trajectory", type=str, default=None, help="Trajectory to evaluate.")
parser.add_argument("--loss_model", type=str, required=True, help="Loss model type (e.g., orig, patched_xxx, simulator_and_hamlyn).")
parser.add_argument("--steps", type=str, required=True, help="Number of training steps (e.g., 56k, 130k).")
parser.add_argument("--eval_dataset_name", type=str, default="hamlyn", help="Dataset to use.")
parser.add_argument("--eval_traj_num", type=str, help="Evaluation trajectory number.")
parser.add_argument("--eval_gs", type=str, help="Grid size for evaluation.")
parser.add_argument("--step_size", type=int, default=10, help="Step size for evaluation.")
args = parser.parse_args()

# ========== Directory Configuration ==========
base_dir = f'./result/{args.dataset_name}/trajectory_{args.trajectory}/{args.loss_model}_loss_{args.steps}_steps/'
os.makedirs(base_dir, exist_ok=True)

# ========== Evaluation File Name ==========
depth = 'gtd' if args.dataset_name.startswith('simulator') or args.eval_dataset_name.startswith('simulator') else 'st'
# depth = 'st' if args.dataset_name.startswith('hamlyn') or args.eval_dataset_name.startswith('hamlyn') else 'gtd'
eval_ts = '11'
eval_file_name = f'{depth}_input_data_{args.eval_traj_num}_ts_{eval_ts}_gs_{args.eval_gs}_cotracker_accel'

# ========== Settings (Replacing FLAGS) ==========
checkpoint_dir = f'{base_dir}{args.loss_model}_loss_{args.steps}_ckpts/'
os.makedirs(f'{base_dir}eval/', exist_ok=True)
rollout_path = f'{base_dir}eval/{eval_file_name}.pkl'
num_rollouts = 1
num_training_steps = int(10e6)

# ========== ModelParameters ==========
model_traj_num = getattr(args, 'finetune_traj_num', getattr(args, 'eval_traj_num', None))

PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}

# ========== Debugging Info ==========
print(f"\nMode: {args.mode}")
print(f"Trajectory: {args.trajectory}")
print(f"Loss Model: {args.loss_model}")
print(f"Steps: {args.steps}")
print(f"Eval Trajectory Num: {args.eval_traj_num}")
print(f"Eval Grid Size: {args.eval_gs}")
print(f"Checkpoint Directory: {checkpoint_dir}")
print(f"Rollout Path: {rollout_path}")


# Function to handle real-time plottingS
# Function to save final training loss plot
def save_final_plot(steps, train_losses, val_steps, val_losses, checkpoint_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    zoom_ax = ax.twinx()  # Add a secondary y-axis for zoomed-in view

    # Plot training and validation loss
    ax.plot(steps, train_losses, label="Train Loss", color="blue", linewidth=3, alpha=0.7)
    ax.plot(val_steps, val_losses, label="Validation Loss", color="red", linestyle="-", linewidth=2, alpha=0.5)

    # Zoomed-in view for losses ≤ 0.1
    zoomed_train_losses = [loss for loss in train_losses if loss <= 0.1]
    zoomed_train_steps = [steps[i] for i, loss in enumerate(train_losses) if loss <= 0.1]
    zoomed_val_losses = [loss for loss in val_losses if loss <= 0.1]
    zoomed_val_steps = [val_steps[i] for i, loss in enumerate(val_losses) if loss <= 0.1]

    zoom_ax.plot(zoomed_train_steps, zoomed_train_losses, label="Zoomed Train Loss", color="orange", linestyle="--", linewidth=1.5)
    zoom_ax.plot(zoomed_val_steps, zoomed_val_losses, label="Zoomed Validation Loss", color="green", linestyle="--", linewidth=1.5)
    zoom_ax.set_ylim(0, 0.1)

    # Labels and styling
    ax.set_xlabel("Global Step", fontsize=14)
    ax.set_ylabel("Loss (Full Scale)", fontsize=14, color="blue")
    zoom_ax.set_ylabel("Loss (Zoomed)", fontsize=14, color="orange")
    ax.set_title("Training and Validation Loss", fontsize=16, fontweight="bold")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Legend
    lines, labels = ax.get_legend_handles_labels()
    zoom_lines, zoom_labels = zoom_ax.get_legend_handles_labels()
    ax.legend(lines + zoom_lines, labels + zoom_labels, loc="upper right", fontsize=12)

    # Save the plot
    os.makedirs(checkpoint_dir, exist_ok=True)
    plot_path = os.path.join(checkpoint_dir, "training_validation_loss_plot.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"✅ Final plot saved at {plot_path}")


#  Learner Function
def learner(model):
    """Run a learner job with real-time visualization (no validation)."""
    # Shared lists for steps and training losses
    steps = []
    train_losses = []
    
    try:
        # Dataset preparation
        if args.finetune_dataset_name is not None:
            train_ds = dataset.load_datasets(f'./input/{args.finetune_dataset_name}/trajectory_{args.finetune_trajectory}_accel_patched_input/')
        else:
            train_ds = dataset.load_datasets(f'./input/{args.dataset_name}/trajectory_{args.trajectory}_accel_patched_input/')
            
        train_ds = train_ds.flat_map(tf.data.Dataset.from_tensor_slices).shuffle(10000).repeat(None)
        train_inputs = tf.data.make_one_shot_iterator(train_ds).get_next()
        
        # Loss and optimizer
        if args.loss_model.startswith('orig'):
            loss_op = model.loss_orig(train_inputs)
            print('Using original loss function')
        else:
            loss_op = model.loss(train_inputs)
            print('Using patched loss function')
        
        global_step = tf.train.create_global_step()
        lr = tf.train.exponential_decay(learning_rate=1e-4,
                                        global_step=global_step,
                                        decay_steps=int(5e6),
                                        decay_rate=0.1) + 1e-6
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss_op, global_step=global_step)

        # Adjust training operation for warm-up
        train_op = tf.cond(tf.less(global_step, 100),
                           lambda: tf.group(tf.assign_add(global_step, 1)),
                           lambda: tf.group(train_op))

        # Training session
        with tf.train.MonitoredTrainingSession(
            hooks=[tf.train.StopAtStepHook(last_step=num_training_steps)],
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_secs=10) as sess:

            while not sess.should_stop():
                _, step, train_loss = sess.run([train_op, global_step, loss_op])

                if step % 10 == 0:
                    print(f"Step {step}: Train Loss {train_loss:.6f}")
                    steps.append(step)
                    train_losses.append(train_loss)  # Append training loss for the plot

            print('Training complete.')

    except KeyboardInterrupt:
        print("Training interrupted by user. Finalizing and saving the plot...")

    finally:
        save_final_plot(steps, train_losses, [], [], checkpoint_dir)
        # print(f"✅ Final plot saved to {checkpoint_dir}")




def evaluator(model, params):
  """Run a model rollout trajectory."""
  eval_base_dir = f"./{args.eval_dataset_name}_data/trajectory_{args.eval_traj_num}/{eval_file_name}/"
  ds = dataset.convert_to_tf_dataset(f"{eval_base_dir}/{eval_file_name}.npz")
  print(f"📂 Loading Eval File: {eval_base_dir}/{eval_file_name}.npz")
  print(f"✅ Eval Model Loaded from: {checkpoint_dir}")

  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(
      model, inputs, args.eval_dataset_name, args.eval_traj_num,
      f"{eval_base_dir}/flow_maps.npz")
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    
    trajectories = []
    scalars = []
    times = []

    for traj_idx in range(num_rollouts):
      print(f"Rollout trajectory {traj_idx}")
      start_time = time.time()
      scalar_data, traj_data = sess.run([scalar_op, traj_ops])
      step_time = time.time() - start_time
      times.append(step_time)
      print(f"⏱️ Eval step {traj_idx} time: {step_time:.4f} seconds")

      trajectories.append(traj_data)
      scalars.append(scalar_data)

    for key in scalars[0]:
      print(f"{key}: {np.mean([x[key] for x in scalars]):.6f}")
    
    print(f"📊 Average step eval time: {np.mean(times):.4f} seconds")

    with open(rollout_path, 'wb') as fp:
      pickle.dump(trajectories, fp)
  
    exp_name = args.loss_model + '_loss_' + args.steps + '_steps'
    process_evaluation(eval_file_name, exp_name, args.dataset_name, args.eval_dataset_name, args.trajectory, step_size=args.step_size)
    print(f"✅ Eval File Processed: {eval_file_name}")

    
def main():
  tf.enable_resource_variables()
  tf.disable_eager_execution()
  params = PARAMETERS[args.model]

  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=7)
  model = params['model'].Model(learned_model, args.loss_model, args.dataset_name, args.trajectory, args.finetune_dataset_name, args.finetune_trajectory)
  if args.mode == 'train':
    learner(model)
  elif args.mode == 'eval':
    evaluator(model, params)

if __name__ == '__main__':
  main()
