# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Synchronize replicas for training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.framework.dtypes import qint8

class QuantizationOptimizerWrapper(optimizer.Optimizer):

  # Values for gate_gradients.
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2

  def __init__(self,
               opt,
               grad_min,
               grad_max,
               local_idx=-1,
               is_adaptive=False,
               use_locking=False,
               name="QuantizationOptimizerWrapper"):

    super(QuantizationOptimizerWrapper, self).__init__(use_locking, name)
    logging.info("QuantizationOptimizerWrapper init")
    self._opt = opt
    self._global_step = None
    self._var_gradient_maps = {}
    self._local_idx = local_idx
    self._grad_min = grad_min
    self._grad_max = grad_max
    self._is_adaptive = is_adaptive


  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping with per replica global norm if needed.
    The global norm with aggregated gradients can be bad as one replica's huge
    gradients can hurt the gradients from other replicas.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    self.init_local_delayed_gradient(var_list)
    grads_and_vars = self._opt.compute_gradients(loss, var_list=var_list, gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops,
      grad_loss=grad_loss)
    return grads_and_vars

  def init_local_delayed_gradient(self, var_list):
    if self._local_idx < 0:
      return self.create_local_delayed_gradient(var_list)
    else:
      with ops.device("/job:worker/task:%d" % self._local_idx):
        return self.create_local_delayed_gradient(var_list)

  def create_local_delayed_gradient(self, var_list):
    if var_list is None:
      var_list = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    with ops.name_scope("local_delayed_gradient", self._name) as name:
      for v in var_list:
        delayed_gradient = variables.Variable(
                        array_ops.zeros(v.get_shape(),dtype=v.dtype),
                        trainable=False,
                        collections=[ops.GraphKeys.LOCAL_VARIABLES],
                        #dtype=v.dtype,
                        name="local_delayed_gradient")
        self._var_gradient_maps[v] = delayed_gradient

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and start the next one. This is executed by each replica.

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")

    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    self._global_step = global_step
    with ops.name_scope("gradient_compensation", self._name) as name:
      new_grads = []
      var_list = []
      for g,v in grads_and_vars:
        if v in self._var_gradient_maps:
          with ops.device(g.device):
            accumu_g = g+self._var_gradient_maps[v]

            if self._is_adaptive:
              accumu_g_mean = math_ops.reduce_mean(accumu_g)
              accumu_g_diff = accumu_g-accumu_g_mean
              accumu_g_square = math_ops.multiply(accumu_g_diff, accumu_g_diff)
              accumu_g_var = math_ops.sqrt(math_ops.reduce_mean(accumu_g_square))
              g_min = accumu_g_mean-math_ops.multiply(accumu_g_var, 2)
              g_max = accumu_g_mean+math_ops.multiply(accumu_g_var, 2)
              g_out, out_min, out_max = array_ops.quantize_v2(accumu_g, g_min, g_max, qint8)
            else:
              g_out, out_min, out_max = array_ops.quantize_v2(accumu_g, self._grad_min, self._grad_max, qint8)

            recovered_g = array_ops.dequantize(g_out, out_min, out_max)
            assign_op = state_ops.assign(self._var_gradient_maps[v], accumu_g-recovered_g)
          with ops.control_dependencies([assign_op]):
            with ops.device(v.device):
              ps_recovered_g = array_ops.dequantize(g_out, out_min, out_max)

        new_grads.append(ps_recovered_g)
        var_list.append(v)
      quantized_grads_and_vars = list(zip(new_grads, var_list))

      train_op = self._opt.apply_gradients(quantized_grads_and_vars, global_step=global_step)

      return train_op


  def get_slot(self, *args, **kwargs):
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)


