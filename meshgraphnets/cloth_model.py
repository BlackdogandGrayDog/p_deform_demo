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
"""Model for FlagSimple."""

import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from meshgraphnets import common
from meshgraphnets import core_model
from meshgraphnets import normalization


class Model(snt.AbstractModule):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, loss_model, dataset_name, trajectory, finetune_dataset_name, finetune_traj_num, name='Model'):
    super(Model, self).__init__(name=name)
    with self._enter_variable_scope():
        self.loss_model = loss_model
        self.dataset_name = dataset_name
        self.finetune_dataset_name = finetune_dataset_name
        self.finetune_traj_num = finetune_traj_num
        self.trajectory = trajectory
        self._learned_model = learned_model
        self._output_normalizer = normalization.Normalizer(
            size=3, name='output_normalizer'
        )

        if self.loss_model.startswith('orig'):
            # Original Node Normalizer
            self._node_normalizer = normalization.Normalizer(
                size=3 + common.NodeType.SIZE, name='node_normalizer'
            )
        else:
            # Patched Node Normalizer (for any 'patched_xxx' variant)
            if (self.dataset_name.startswith('cloth') and (self.trajectory == 'clothblow' or self.trajectory == 'clothblow_test')):
              self._node_normalizer = normalization.Normalizer(
                  size=3 + 3 + common.NodeType.SIZE, name='node_normalizer'
              )
            else:
             # Patched Node Normalizer (for any 'patched_xxx' variant)
              self._node_normalizer = normalization.Normalizer(
                  size=3 + 3 + 2 + common.NodeType.SIZE + common.NodeType.PATCH_SIZE ** 2 * 2,
                  name='node_normalizer'
              )
              # self._node_normalizer = normalization.Normalizer(
              #    size=3 + 3 + 2 + common.NodeType.SIZE + common.NodeType.PATCH_SIZE ** 2 * 2 - 3, name='node_normalizer'
              # )
              
        if self.loss_model.startswith('orig'):
            self._edge_normalizer = normalization.Normalizer(
                size=7, name='edge_normalizer')
        else:
          if (self.dataset_name.startswith('cloth') and (self.trajectory == 'clothblow' or self.trajectory == 'clothblow_test')):
            self._edge_normalizer = normalization.Normalizer(
                size=17, name='edge_normalizer'  # 2D coord + 3D coord + 2*length = 7
            )

          else:
            self._edge_normalizer = normalization.Normalizer(
                size=3 + 1 + 2 + 1 + 2 + 1 + 3 + 1 + 3 + 1 + 1 + 1 + 2 + 1, name='edge_normalizer'  # 2D coord + 3D coord + 2*length = 7
            )
            #  self._edge_normalizer = normalization.Normalizer(
            #     size=3 + 1 + 2 + 1 + 2 + 1 + 3 + 1 + 3 + 1 + 1 + 1 + 2 + 1 - 10, name='edge_normalizer'  # 2D coord + 3D coord + 2*length = 7
            # )
  
  
  #  With Path and Acceleration Input and Edge Features
  def _build_graph(self, inputs, is_training):
    """Builds input graph."""
    # construct graph nodes
    
    # Compute velocity and acceleration
    velocity = inputs['world_pos'] - inputs['prev|world_pos']
    acceleration = inputs['world_pos'] - 2 * inputs['prev|world_pos'] + inputs['prev_prev|world_pos']
    
    # Compute 2D optical flow for feature points
    flow_2d = inputs['image_pos'] - inputs['prev|image_pos']
    
    # Flatten local patched optical flow
    patched_flow = tf.reshape(inputs['patched_flow'], [tf.shape(inputs['patched_flow'])[0], -1])  # (num_nodes, 200)
    
    # One-hot encode node type
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    
    # Concatenate all node features
    if (self.dataset_name.startswith('cloth') and (self.trajectory == 'clothblow' or self.trajectory == 'clothblow_test')):
      node_features = tf.concat([velocity, acceleration, node_type], axis=-1)
    else:
      # node_features = tf.concat([velocity, acceleration, flow_2d, patched_flow, node_type], axis=-1)
      node_features = tf.concat([velocity, acceleration, flow_2d, patched_flow, node_type], axis=-1)
    # construct graph edges
    senders, receivers = common.triangles_to_edges(inputs['cells'])

    # Relative positions
    relative_world_pos = tf.gather(inputs['world_pos'], senders) - tf.gather(inputs['world_pos'], receivers)
    relative_mesh_pos = tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers)
    relative_image_pos = tf.gather(inputs['image_pos'], senders) - tf.gather(inputs['image_pos'], receivers)
    
    # Relative motion dynamics
    relative_velocity = tf.gather(velocity, senders) - tf.gather(velocity, receivers)
    relative_acceleration = tf.gather(acceleration, senders) - tf.gather(acceleration, receivers)
    flow_diff = tf.gather(flow_2d, senders) - tf.gather(flow_2d, receivers)
    
    # Edge direction (normalized)
    unit_edge_dir = tf.math.l2_normalize(relative_world_pos, axis=-1)
    
    # Projected motion onto edge direction
    velocity_projection = tf.reduce_sum(relative_velocity * unit_edge_dir, axis=-1, keepdims=True)
    acceleration_projection = tf.reduce_sum(relative_acceleration * unit_edge_dir, axis=-1, keepdims=True)
    
    if (self.dataset_name.startswith('cloth') and (self.trajectory == 'clothblow' or self.trajectory == 'clothblow_test')):
      edge_features = tf.concat([
          relative_world_pos,
          tf.norm(relative_world_pos, axis=-1, keepdims=True),

        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True),

        # relative_image_pos,
        # tf.norm(relative_image_pos, axis=-1, keepdims=True),

        relative_velocity,
        tf.norm(relative_velocity, axis=-1, keepdims=True),

        relative_acceleration,
        tf.norm(relative_acceleration, axis=-1, keepdims=True),
        
        # flow_diff,
        # tf.norm(flow_diff, axis=-1, keepdims=True),

        velocity_projection,
        acceleration_projection
      ], axis=-1)
      
    else:
      edge_features = tf.concat([
        relative_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True),

        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True),

        relative_image_pos,
        tf.norm(relative_image_pos, axis=-1, keepdims=True),

        relative_velocity,
        tf.norm(relative_velocity, axis=-1, keepdims=True),

        relative_acceleration,
        tf.norm(relative_acceleration, axis=-1, keepdims=True),
        
        flow_diff,
        tf.norm(flow_diff, axis=-1, keepdims=True),

        velocity_projection,
        acceleration_projection
      ], axis=-1)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)
    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])
    
  #  Original Graph
  def _build_graph_orig(self, inputs, is_training):
    """Builds input graph."""
    # construct graph nodes
    velocity = inputs['world_pos'] - inputs['prev|world_pos']
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_features = tf.concat([velocity, node_type], axis=-1)

    # construct graph edges
    senders, receivers = common.triangles_to_edges(inputs['cells'])
    relative_world_pos = (tf.gather(inputs['world_pos'], senders) -
                          tf.gather(inputs['world_pos'], receivers))
    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))
    edge_features = tf.concat([
        relative_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)
    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])

  #  Build Graph
  def _build(self, inputs):
    if self.loss_model.startswith('orig'):  # Fetch loss model
        graph = self._build_graph_orig(inputs, is_training=False)
    else:
        graph = self._build_graph(inputs, is_training=False)

    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output)

  #  Original Loss Function
  @snt.reuse_variables
  def loss_orig(self, inputs):
    """L2 loss on position."""
    graph = self._build_graph_orig(inputs, is_training=True)
    network_output = self._learned_model(graph)

    # build target acceleration
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    target_position = inputs['target|world_pos']
    target_acceleration = target_position - 2*cur_position + prev_position
    target_normalized = self._output_normalizer(target_acceleration)

    # build loss
    loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
    loss = tf.reduce_mean(error[loss_mask])
    return loss
  
  # Robust Patch Loss
  @snt.reuse_variables
  def robust_patch_loss(self, pred_flow_norm, target_flow_norm, delta=1.0):
    error = tf.abs(pred_flow_norm - target_flow_norm)
    # Compute standard Huber Loss
    small_error_loss = 0.5 * tf.square(error)
    large_error_loss = delta * (error - 0.5 * delta)
    huber_loss = tf.where(error <= delta, small_error_loss, large_error_loss)
    # Clip loss values to Q3 (suppress extreme spikes)
    q3_loss = tfp.stats.percentile(huber_loss, 75.0)
    clipped_loss = tf.clip_by_value(huber_loss, 0, q3_loss)

    return tf.reduce_mean(clipped_loss)

  # Loss Function with Acceleration Regularization
  @snt.reuse_variables
  def loss(self, inputs):
      """L2 loss on position with acceleration regularization."""
      graph = self._build_graph(inputs, is_training=True)
      network_output = self._learned_model(graph)

      # Compute target acceleration
      cur_position = inputs['world_pos']
      prev_position = inputs['prev|world_pos']
      prev2_position = inputs['prev_prev|world_pos']
      target_position = inputs['target|world_pos']
      
      target_acceleration = target_position - 2 * cur_position + prev_position
      current_acceleration = cur_position - 2 * prev_position + prev2_position

      target_normalized = self._output_normalizer(target_acceleration)
      current_normalized = self._output_normalizer(current_acceleration)

      # Base L2 loss on acceleration prediction
      loss_mask = tf.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
      error = tf.reduce_sum((target_normalized - network_output)**2, axis=1)
      base_loss = tf.reduce_mean(error[loss_mask])

      # Acceleration regularization: || A_t+1 - A_t ||^2
      acceleration_reg = tf.reduce_mean(tf.square(current_normalized - network_output))
      
      # Denormalize network output before using it
      acceleration = self._output_normalizer.inverse(network_output)
      # Convert acceleration to predicted position
      pred_position = 2 * cur_position + acceleration - prev_position
      pred_position_homo = tf.concat([pred_position, tf.ones_like(pred_position[:, :1])], axis=-1)

      # Project to 2D using camera matrix
      if "hamlyn" in (self.dataset_name or "") or "hamlyn" in (self.finetune_dataset_name or ""):
        if str(self.finetune_traj_num).startswith("11") or str(self.finetune_traj_num).startswith("12"):
          image_pos_pred = tf.matmul(common.K_hamlyn_11, tf.transpose(pred_position_homo[:, :3], perm=[1, 0]))
        else:
          image_pos_pred = tf.matmul(common.K_hamlyn_4, tf.transpose(pred_position_homo[:, :3], perm=[1, 0]))
        img_size = common.hamlyn_img_size
      else:
        image_pos_pred = tf.matmul(common.K_simulator, tf.transpose(pred_position_homo[:, :3], perm=[1, 0]))
        img_size = common.simulator_img_size
        # print_k = tf.print("K_simulator:\n", common.K_simulator)
        
      image_pos_pred = tf.transpose(image_pos_pred, perm=[1, 0])
      image_pos_pred = image_pos_pred[:, :2] / image_pos_pred[:, 2:3]
      
      # Compute the predicted flow and target flow
      pred_flow = image_pos_pred[:, :2] - inputs['image_pos']
      pred_flow_norm = pred_flow/tf.constant(img_size, dtype=tf.float32)
      target_flow_norm = inputs['avg_target|patched_flow'] / tf.constant(img_size, dtype=tf.float32)

      # Patch Regularization Loss using the **averaged patch flow**
      patch_reg = self.robust_patch_loss(pred_flow_norm, target_flow_norm)

      # Final loss with regularization
      loss_model_values = self.loss_model.replace("patched_", "").split("_")
      if len(loss_model_values) == 1:
          lambda_accel = lambda_patch = float(loss_model_values[0])  # Both are the same
      elif len(loss_model_values) == 2:
          lambda_accel, lambda_patch = map(float, loss_model_values)  # Extract both separately
      else:
          raise ValueError(f"Invalid loss_model format: {self.loss_model}")
      
      total_loss = base_loss + lambda_accel * acceleration_reg + lambda_patch * patch_reg
      
      return total_loss

  #  Update
  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs.""" 
    acceleration = self._output_normalizer.inverse(per_node_network_output)
    # integrate forward
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    position = 2*cur_position + acceleration - prev_position
    return position
