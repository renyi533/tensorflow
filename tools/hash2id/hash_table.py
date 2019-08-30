from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lite.sparse import feature
from lite.sparse import utils
from lite.sparse import operator

import tensorflow as tf
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.framework import ops

class FidHashTable(object):
    def __init__(self, rehash, slot_hash_sizes, occurrence_threshold=(), name=None):
        if isinstance(slot_hash_sizes, int):
            slot_hash_sizes = [slot_hash_sizes]
            occurrence_threshold = [occurrence_threshold]
        
        with ops.name_scope(name, "FidHashTable") as name:
            self._name = name
            self._resource_handle = operator.lite_ops.hash_table_create(
                rehash=rehash,
                slot_hash_sizes=slot_hash_sizes,
                occurrence_threshold=occurrence_threshold)

        saveable = FidHashTable._Saveable(self, name)
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
    
    def hash_fids(self, fids):
        out_inst_ids, out_fids = operator.lite_ops.hash_table_hash_fid(
            handle=self._resource_handle,
            instance_ids=fids.indices,
            fids=fids.values)

        return tf.IndexedSlices(indices=out_inst_ids, values=out_fids, dense_shape=fids.dense_shape)
    
    def export(self):
        return operator.lite_ops.hash_table_export(
            handle=self._resource_handle)

    class _Saveable(BaseSaverBuilder.SaveableObject):
        def __init__(self, table, name):
            import pdb;pdb.set_trace()
            buff = table.export()
            specs = [
                BaseSaverBuilder.SaveSpec(buff, "", table._resource_handle.name + "_buff"),
            ]
            super(FidHashTable._Saveable, self).__init__(table, specs, name)

        def restore(self, restored_tensors, restored_shapes, name=None):
            with ops.name_scope(name, "%s_table_restore" % self.name):
                with ops.colocate_with(self.op._resource_handle):
                    return operator.lite_ops.hash_table_restore(
                        self.op._resource_handle, restored_tensors[0])
