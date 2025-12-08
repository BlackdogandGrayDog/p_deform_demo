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
"""Commonly used data structures and functions."""

import enum
import tensorflow.compat.v1 as tf
import numpy as np

import numpy as np

# Simulator Intrinsic Camera Matrix
K_simulator = np.array([
    [248.0145, 0,       256],
    [0,        330.5661, 256],
    [0,        0,          1]
], dtype=np.float32)


# Hamlyn Intrinsic Camera Matrix
K_hamlyn_4 = np.array([
    [579.05693, 0, 139.93160057],
    [0, 579.05693, 159.01899052],
    [0, 0, 1]
], dtype=np.float32)

K_hamlyn_11 = np.array([
    [426.532013, 0, 175.2081146240234],
    [0, 426.532013, 153.1618118286133],
    [0, 0, 1]
], dtype=np.float32)

K_clothblow = np.array([
    [774.547114, 0.0,        319.141388],
    [0.0,        772.695,    244.782526],
    [0.0,        0.0,        1.0]
], dtype=np.float32)

K_surgt_6 = np.array([
    [1175.20938682, 0.0, 633.05148315],
    [0.0, 1175.20938682, 514.82630920],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

K_surgt_9 = np.array([
    [1144.97802996, 0.0, 668.84513092],
    [0.0, 1144.97802996, 508.13312912],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

K_surgt_1 = np.array([
    [1119.76687760, 0.0, 659.47716522],
    [0.0, 1119.76687760, 514.86151886],
    [0.0, 0.0, 1.0]
], dtype=np.float32)



simulator_img_size = np.array([512, 512], dtype=np.float32)
hamlyn_img_size = np.array([360, 288], dtype=np.float32)
clothblow_img_size = np.array([640, 512], dtype=np.float32)
surgt_img_size = np.array([1280, 1024], dtype=np.float32)
class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 9
  PATCH_SIZE = 6


def triangles_to_edges(faces):
  """Computes mesh edges from triangles."""
  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  # remove duplicates and unpack
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))
