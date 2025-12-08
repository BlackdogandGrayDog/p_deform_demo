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
"""Functions to build evaluation metrics for cloth data."""

import tensorflow.compat.v1 as tf

from meshgraphnets.common import NodeType
from meshgraphnets import common
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

# Project 3D world positions to 2D image space using intrinsic matrix
def project_to_image_space(world_pos, dataset_name, traj_num):
    # Homogeneous coordinates
    pos_homo = tf.concat([world_pos, tf.ones_like(world_pos[:, :1])], axis=-1)
    # Select intrinsic matrix
    if "hamlyn" in dataset_name:
        if str(traj_num).startswith("11") or str(traj_num).startswith("12"):
            K = common.K_hamlyn_11
        else:
            K = common.K_hamlyn_4
    else:
        K = common.K_simulator
    # Project to image space
    image_pos = tf.matmul(K, tf.transpose(pos_homo[:, :3], perm=[1, 0]))
    image_pos = tf.transpose(image_pos, perm=[1, 0])
    image_pos = image_pos[:, :2] / image_pos[:, 2:3]
    return image_pos

# Patch extraction function
def extract_patch_flow(flow_map, uv_coords, patch_size):
    H = tf.shape(flow_map)[0]
    W = tf.shape(flow_map)[1]
    N = tf.shape(uv_coords)[0]

    # === Generate grid offsets for patch of size patch_size x patch_size
    offset_range = np.arange(-(patch_size // 2), -(patch_size // 2) + patch_size)
    dx, dy = np.meshgrid(offset_range, offset_range, indexing='ij')  # shape: [patch_size, patch_size]
    dx_const = tf.constant(dx, dtype=tf.int32)
    dy_const = tf.constant(dy, dtype=tf.int32)

    # === Centered coordinates (rounded UV)
    x_center = tf.cast(tf.round(uv_coords[:, 0]), tf.int32)  # shape: [N]
    y_center = tf.cast(tf.round(uv_coords[:, 1]), tf.int32)

    # === Expand to patch coordinates
    x_coords = tf.reshape(x_center, [N, 1, 1]) + dx_const  # [N, patch, patch]
    y_coords = tf.reshape(y_center, [N, 1, 1]) + dy_const  # [N, patch, patch]

    # === In-bounds mask
    in_bounds_x = tf.logical_and(x_coords >= 0, x_coords < W)
    in_bounds_y = tf.logical_and(y_coords >= 0, y_coords < H)
    in_bounds = tf.logical_and(in_bounds_x, in_bounds_y)  # [N, patch, patch]

    # === Clip to valid range
    x_safe = tf.clip_by_value(x_coords, 0, W - 1)
    y_safe = tf.clip_by_value(y_coords, 0, H - 1)

    # === Gather flow values
    gather_idx = tf.stack([y_safe, x_safe], axis=-1)  # [N, patch, patch, 2]
    flow_patches = tf.gather_nd(flow_map, gather_idx, batch_dims=0)  # [N, patch, patch, 2]

    # === Interpolation for out-of-bound values
    def interpolate_pixel(x, y):
        x0 = tf.clip_by_value(x - 1, 0, W - 1)
        x1 = tf.clip_by_value(x + 1, 0, W - 1)
        y0 = tf.clip_by_value(y - 1, 0, H - 1)
        y1 = tf.clip_by_value(y + 1, 0, H - 1)

        f00 = tf.gather_nd(flow_map, tf.stack([y0, x0], axis=-1))
        f01 = tf.gather_nd(flow_map, tf.stack([y0, x1], axis=-1))
        f10 = tf.gather_nd(flow_map, tf.stack([y1, x0], axis=-1))
        f11 = tf.gather_nd(flow_map, tf.stack([y1, x1], axis=-1))

        return (f00 + f01 + f10 + f11) / 4.0

    interp_values = interpolate_pixel(x_safe, y_safe)  # [N, patch, patch, 2]

    # === Replace out-of-bounds pixels with interpolated values
    mask_expanded = tf.tile(tf.expand_dims(in_bounds, axis=-1), [1, 1, 1, 2])  # [N, patch, patch, 2]
    patched_flow = tf.where(mask_expanded, flow_patches, interp_values)

    return patched_flow



def _rollout(model, initial_state, num_steps, eval_dataset_name, eval_traj_num, flow_npz_path, patched_flow_seq):
  """Rolls out a model trajectory."""
  # Determine which nodes are dynamic
  mask = tf.equal(initial_state['node_type'][:, 0], NodeType.NORMAL)
  
  # === Load flow maps ===
  flow_npz = np.load(flow_npz_path, allow_pickle=True)
  flow_keys = [k for k in sorted(flow_npz.files) if k.startswith("frame")]
  flow_maps_np = [flow_npz[k].astype(np.float32) for k in flow_keys]
  flow_tensor = tf.constant(np.stack(flow_maps_np))
    
  # Step function now takes 3 previous positions
  def step_fn(step, prev_prev_pos, prev_pos, cur_pos, prev_img_pos, cur_img_pos, trajectory):
    # Construct model input: overwrite just the dynamic parts
    prediction = model({
        **initial_state,
        'prev_prev|world_pos': prev_prev_pos,
        'prev|world_pos': prev_pos,
        'world_pos': cur_pos,
        'prev|image_pos': prev_img_pos,
        'image_pos': cur_img_pos,
        'patched_flow': extract_patch_flow(flow_tensor[step], cur_img_pos, common.NodeType.PATCH_SIZE)
        # 'patched_flow': tf.cond(tf.equal(step, 0), lambda: patched_flow_seq[step], lambda: extract_patch_flow(flow_tensor[step], cur_img_pos, common.NodeType.PATCH_SIZE))
        # 'patched_flow': patched_flow_seq[step]
    })

    # Only update normal (non-kinematic) nodes
    next_pos = tf.where(mask, prediction, cur_pos)

    # === NEW: Project predicted next_pos to image space
    next_img_pos = project_to_image_space(next_pos, eval_dataset_name, eval_traj_num)

    # Save current position into trajectory
    trajectory = trajectory.write(step, cur_pos)

    # Shift the window forward
    return step + 1, prev_pos, cur_pos, next_pos, cur_img_pos, next_img_pos, trajectory

  # Call the loop with 7 loop variables
  _, _, _, _, _, _, output = tf.while_loop(
      cond=lambda step, prev_prev_pos, prev_pos, cur_pos, prev_img_pos, cur_img_pos, traj: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(
          0,  # step
          initial_state['prev_prev|world_pos'],  # t-2
          initial_state['prev|world_pos'],       # t-1
          initial_state['world_pos'],            # t0
          initial_state['prev|image_pos'],       # img t-1
          initial_state['image_pos'],            # img t0
          tf.TensorArray(tf.float32, num_steps)  # trajectory buffer
      ),
      parallel_iterations=1
  )

  return output.stack()



def evaluate(model, inputs, eval_dataset_name, eval_traj_num, flow_npz_path):
  """Performs model rollouts and create stats."""
  initial_state = {k: v[0] for k, v in inputs.items()}
  num_steps = inputs['cells'].shape[0]
  prediction = _rollout(model, initial_state, num_steps, eval_dataset_name, eval_traj_num, flow_npz_path, inputs['patched_flow'])

  error = tf.reduce_mean((prediction - inputs['world_pos'])**2, axis=-1)
  scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
             for horizon in [1, 10, 20, 50, 100, 200]}
  traj_ops = {
      'faces': inputs['cells'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_pos': inputs['world_pos'],
      'pred_pos': prediction
  }
  return scalars, traj_ops
