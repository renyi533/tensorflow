# Copyright 2018 Alibaba Inc. All Rights Conserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import os

import tensorflow as tf

from tf_euler.python.euler_ops import base
from tensorflow.python.ops import variables

inflate_idx = base._LIB_OP.inflate_idx
sparse_gather = base._LIB_OP.sparse_gather
euler_hash_fid = base._LIB_OP.euler_hash_fid

def hash_fid(fids, hash_space, multiplier=1.6, partition=None, use_locking=True):
  with tf.variable_scope('hash_fids', reuse=tf.AUTO_REUSE):
    var_dim = int(hash_space*multiplier)
    if partition is None:
      v = tf.get_variable("hash_param", 
                          [var_dim, 2],
                          dtype=tf.int64,
                          trainable=False,
                          initializer=tf.constant_initializer([0]))
    else:
      v = tf.get_variable("hash_param", 
                          [var_dim, 2],
                          dtype=tf.int64,
                          trainable=False,
                          initializer=tf.constant_initializer([0]),
                          partitioner=tf.fixed_size_partitioner(partition))
  orig_fids = fids
  fids, idx = tf.unique(fids, out_idx=tf.dtypes.int32)

  params = v
  if isinstance(params, variables.PartitionedVariable):
    params = list(params)  # Iterate to get the underlying Variables.
  elif not isinstance(params, list):
    params = [params]

  ret = None
  np = len(params)
  if np == 1:
    with tf.colocate_with(params[0]):
      ret = euler_hash_fid(params[0], fids, 0, hash_space-1, 
                  use_locking=use_locking)
  else:
    flat_ids = tf.reshape(fids, [-1])
    original_indices = tf.range(tf.size(flat_ids))
    p_assignments = tf.abs(flat_ids) % np
    new_ids = flat_ids // np
    sub_space = hash_space // np

    p_assignments = tf.cast(p_assignments, tf.dtypes.int32)
    gather_ids = tf.dynamic_partition(new_ids, p_assignments, np)
    pindices = tf.dynamic_partition(original_indices,
                                    p_assignments, np)
    partitioned_result = []
    start = 0
    for p in xrange(np):
      pids = gather_ids[p]
      with tf.colocate_with(params[p]):
        if p == np-1:
          end = hash_space - 1
        else:
          end = start + sub_space - 1
        result = euler_hash_fid(params[p], pids, start, end, 
                      use_locking=use_locking)
        start = start + sub_space
      partitioned_result.append(result)
    ret = tf.dynamic_stitch(pindices, 
                            partitioned_result)
  ret = tf.reshape(ret, [-1])
  return tf.gather(ret, idx, axis=0)
