# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_euler.python.base_layers import Layer, Dense, Embedding, SparseEmbedding
from tf_euler.python.euler_ops import hash_fid

class HashSparseEmbedding(SparseEmbedding):
  """
  Sparse id to dense vector embedding with hashing.
  """
  def __init__(
      self,
      max_id,
      dim,
      initializer=lambda: tf.truncated_normal_initializer(stddev=0.04),
      combiner='sum',
      partition=11,
      use_locking=False,
      **kwargs):
    super(HashSparseEmbedding, self).__init__(
        max_id=max_id, dim=dim, initializer=initializer, combiner=combiner)
    self.partition = partition
    self.use_locking = use_locking
  
  def call(self, inputs):
    n_fids = hash_fid(inputs.values, self.max_id+1, 
            partition=self.partition, use_locking=self.use_locking)

    new_inputs = tf.SparseTensor(
        indices=inputs.indices,
        values=n_fids, dense_shape=inputs.dense_shape)
    return super(HashSparseEmbedding, self).call(new_inputs)

class HashEmbedding(Embedding):
  """
  Sparse id to dense vector embedding with hashing.
  """
  def __init__(
      self,
      max_id,
      dim,
      initializer=lambda: tf.truncated_normal_initializer(stddev=0.1),
      partition=11,
      use_locking=False,
      **kwargs):
    super(HashEmbedding, self).__init__(
        max_id=max_id, dim=dim, initializer=initializer)
    self.partition = partition
    self.use_locking = use_locking
  
  def call(self, inputs):
    in_shape = inputs.shape
    inputs = tf.reshape(inputs,[-1])
    out = hash_fid(inputs, self.max_id+1, 
            partition=self.partition, use_locking=self.use_locking)

    out_shape = [d if d is not None else -1 for d in in_shape.as_list()]
    new_inputs = tf.reshape(out, out_shape)
    return super(HashEmbedding, self).call(new_inputs)
