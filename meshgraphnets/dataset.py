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
"""Utility functions for reading the datasets."""

import functools
import json
import os
import numpy as np
import glob
import tensorflow.compat.v1 as tf
from meshgraphnets.common import NodeType

def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  return out


def load_dataset(path, split):
  """Load dataset."""
  with open(os.path.join(path, 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())
  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds


def add_targets(ds, fields, add_history):
  """Adds target and optionally history fields to dataframe."""
  def fn(trajectory):
    out = {}
    for key, val in trajectory.items():
      out[key] = val[1:-1]
      if key in fields:
        if add_history:
          out['prev|'+key] = val[0:-2]
        out['target|'+key] = val[2:]
    return out
  return ds.map(fn, num_parallel_calls=8)


def split_and_preprocess(ds, noise_field, noise_scale, noise_gamma):
  """Splits trajectories into frames, and adds training noise."""
  def add_noise(frame):
    noise = tf.random.normal(tf.shape(frame[noise_field]),
                             stddev=noise_scale, dtype=tf.float32)
    # don't apply noise to boundary nodes
    mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
    noise = tf.where(mask, noise, tf.zeros_like(noise))
    frame[noise_field] += noise
    frame['target|'+noise_field] += (1.0 - noise_gamma) * noise
    return frame

  ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
  # ds = ds.map(add_noise, num_parallel_calls=8)
  ds = ds.shuffle(10000)
  ds = ds.repeat(None)
  return ds.prefetch(10)


def batch_dataset(ds, batch_size):
  """Batches input datasets."""
  shapes = ds.output_shapes
  types = ds.output_types
  def renumber(buffer, frame):
    nodes, cells = buffer
    new_nodes, new_cells = frame
    return nodes + new_nodes, tf.concat([cells, new_cells+nodes], axis=0)

  def batch_accumulate(ds_window):
    out = {}
    for key, ds_val in ds_window.items():
      initial = tf.zeros((0, shapes[key][1]), dtype=types[key])
      if key == 'cells':
        # renumber node indices in cells
        num_nodes = ds_window['node_type'].map(lambda x: tf.shape(x)[0])
        cells = tf.data.Dataset.zip((num_nodes, ds_val))
        initial = (tf.constant(0, tf.int32), initial)
        _, out[key] = cells.reduce(initial, renumber)
      else:
        merge = lambda prev, cur: tf.concat([prev, cur], axis=0)
        out[key] = ds_val.reduce(initial, merge)
    return out

  if batch_size > 1:
    ds = ds.window(batch_size, drop_remainder=True)
    ds = ds.map(batch_accumulate, num_parallel_calls=8)
  return ds


def convert_to_tf_dataset(input_data):
    if isinstance(input_data, str):  # If input is a file path
        stacked_tensors = np.load(input_data)
    elif isinstance(input_data, dict):  # If input is already a stacked_tensors dictionary
        stacked_tensors = input_data
    else:
        raise ValueError("Input must be either a file path (str) or a dictionary of stacked tensors.")

    # Convert numpy arrays to TensorFlow tensors
    tf_tensors = {
        "cells": tf.convert_to_tensor(stacked_tensors["cells"], dtype=tf.int32),
        "mesh_pos": tf.convert_to_tensor(stacked_tensors["mesh_pos"], dtype=tf.float32),
        "node_type": tf.convert_to_tensor(stacked_tensors["node_type"], dtype=tf.int32),
        "world_pos": tf.convert_to_tensor(stacked_tensors["world_pos"], dtype=tf.float32),
        "prev|world_pos": tf.convert_to_tensor(stacked_tensors["prev|world_pos"], dtype=tf.float32),
        "prev_prev|world_pos": tf.convert_to_tensor(stacked_tensors["prev_prev|world_pos"], dtype=tf.float32),
        "target|world_pos": tf.convert_to_tensor(stacked_tensors["target|world_pos"], dtype=tf.float32),
        "image_pos": tf.convert_to_tensor(stacked_tensors["image_pos"], dtype=tf.float32),
        "prev|image_pos": tf.convert_to_tensor(stacked_tensors["prev|image_pos"], dtype=tf.float32),
        "prev_prev|image_pos": tf.convert_to_tensor(stacked_tensors["prev_prev|image_pos"], dtype=tf.float32),
        "target|image_pos": tf.convert_to_tensor(stacked_tensors["target|image_pos"], dtype=tf.float32),
        "patched_flow": tf.convert_to_tensor(stacked_tensors["patched_flow"], dtype=tf.float32),
        "target|patched_flow": tf.convert_to_tensor(stacked_tensors["target|patched_flow"], dtype=tf.float32),
        "avg_target|patched_flow": tf.convert_to_tensor(stacked_tensors["avg_target|patched_flow"], dtype=tf.float32),
    }

    # Use from_tensors() to keep batch structure
    tf_dataset = tf.data.Dataset.from_tensors(tf_tensors)
    
    return tf_dataset


def parse_npz(file_path):
    """Parses an NPZ trajectory and returns NumPy arrays."""
    data = np.load(file_path.numpy().decode("utf-8"), allow_pickle=True)
    return (
        data["cells"],
        data["mesh_pos"],
        data["node_type"],
        data["world_pos"],
        data["prev|world_pos"],
        data["prev_prev|world_pos"],
        data["target|world_pos"],
        data["image_pos"],
        data["prev|image_pos"],
        data["prev_prev|image_pos"],
        data["target|image_pos"],
        data["patched_flow"],
        data["target|patched_flow"],
        data["avg_target|patched_flow"],
    )
  
  
def tf_parse_npz(file_path):
    """Wrapper function for TensorFlow dataset mapping."""
    parsed_data = tf.py_function(
        func=parse_npz, 
        inp=[file_path], 
        Tout=[tf.int32, tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, 
              tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
    )

    # Dynamically determine max length
    max_cells = tf.shape(parsed_data[0])[1]  # Number of cells
    max_nodes = tf.shape(parsed_data[1])[1]  # Number of nodes

    max_cells = tf.maximum(max_cells, 1)
    max_nodes = tf.maximum(max_nodes, 1)

    return {
        "cells": tf.pad(parsed_data[0], [[0, 0], [0, max_cells - tf.shape(parsed_data[0])[1]], [0, 0]], constant_values=-1),
        "mesh_pos": tf.pad(parsed_data[1], [[0, 0], [0, max_nodes - tf.shape(parsed_data[1])[1]], [0, 0]]),
        "node_type": tf.pad(parsed_data[2], [[0, 0], [0, max_nodes - tf.shape(parsed_data[2])[1]], [0, 0]]),
        "world_pos": tf.pad(parsed_data[3], [[0, 0], [0, max_nodes - tf.shape(parsed_data[3])[1]], [0, 0]]),
        "prev|world_pos": tf.pad(parsed_data[4], [[0, 0], [0, max_nodes - tf.shape(parsed_data[4])[1]], [0, 0]]),
        "prev_prev|world_pos": tf.pad(parsed_data[5], [[0, 0], [0, max_nodes - tf.shape(parsed_data[5])[1]], [0, 0]]),
        "target|world_pos": tf.pad(parsed_data[6], [[0, 0], [0, max_nodes - tf.shape(parsed_data[6])[1]], [0, 0]]),
        "image_pos": tf.pad(parsed_data[7], [[0, 0], [0, max_nodes - tf.shape(parsed_data[7])[1]], [0, 0]]),
        "prev|image_pos": tf.pad(parsed_data[8], [[0, 0], [0, max_nodes - tf.shape(parsed_data[8])[1]], [0, 0]]),
        "prev_prev|image_pos": tf.pad(parsed_data[9], [[0, 0], [0, max_nodes - tf.shape(parsed_data[9])[1]], [0, 0]]),
        "target|image_pos": tf.pad(parsed_data[10], [[0, 0], [0, max_nodes - tf.shape(parsed_data[10])[1]], [0, 0]]),
        "patched_flow": tf.pad(parsed_data[11], 
                               [[0, 0],  # Time dimension (No padding needed)
                                [0, max_nodes - tf.shape(parsed_data[10])[1]],  # Nodes (Padding needed)
                                [0, 0],  # Patch Height (No padding needed)
                                [0, 0],  # Patch Width (No padding needed)
                                [0, 0]]),  # Flow Channels (No padding needed)

        "target|patched_flow": tf.pad(parsed_data[12], 
                                      [[0, 0],  # Time dimension (No padding needed)
                                       [0, max_nodes - tf.shape(parsed_data[11])[1]],  # Nodes (Padding needed)
                                       [0, 0],  # Patch Height (No padding needed)
                                       [0, 0],  # Patch Width (No padding needed)
                                       [0, 0]]),  # Flow Channels (No padding needed)
                                       
        "avg_target|patched_flow": tf.pad(parsed_data[13], [[0, 0], [0, max_nodes - tf.shape(parsed_data[12])[1]], [0, 0]])  # Flow Channels (2D vectors)
    }


def load_datasets(dataset_path):
    """Loads trajectories from NPZ files as a TensorFlow dataset."""
    npz_files = glob.glob(os.path.join(dataset_path, "*.npz"))

    print(f"🔄 Found {len(npz_files)} NPZ files in '{dataset_path}'")
    
    # Create dataset from NPZ file paths
    file_dataset = tf.data.Dataset.from_tensor_slices(npz_files)

    def print_file_path(file_path):
        """Print the file name before loading."""
        tf.print("📂 Loading trajectory:", file_path)
        return file_path

    # Print file path before parsing
    file_dataset = file_dataset.map(print_file_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Map the parsing function and enable parallel processing
    dataset = file_dataset.map(tf_parse_npz, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetch data for performance (but NO shuffle here)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset  # ✅ Return the dataset