# Copyright 2018 Alibaba Inc. All Rights Conserved.
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
"""Euler type ops test"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensorflow.python.platform import test
import tensorflow as tf

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import util_ops as ops

class UtilOpsTest(test.TestCase):

    @classmethod
    def setUpClass(cls):
        """Build Graph data for test"""
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        meta_file = os.path.join(cur_dir, 'testdata/meta.json')
        graph_file = os.path.join(cur_dir, 'testdata/graph.json')
        output_file = os.path.join(cur_dir, 'testdata/graph.dat')
        builder = os.path.join(cur_dir, '../../../tools/bin/json2dat.py')

        command = "python {builder} -i {input} -c {meta} -o {output}".format(
            builder=builder, input=graph_file, meta=meta_file, output=output_file)

        try:
            subprocess.call(command, shell=True)
        except:
            raise RuntimeError("Build Graph for test failed")

        base.initialize_graph(
            {'mode': 'Local',
             'directory': os.path.join(cur_dir, 'testdata'),
             'load_type': 'compact'
            })

    def testInflateIdxUnique(self):
        """Test euler get node type"""
        op = ops.inflate_idx([0, 2, 1, 3])
        with tf.Session() as sess:
            result = sess.run(op)
            self.assertAllEqual([0, 2, 1, 3], result);

    def testInflateIdx(self):
        """Test euler get node type"""
        op = ops.inflate_idx([0, 1, 0, 2, 1])
        with tf.Session() as sess:
            result = sess.run(op)
            self.assertAllEqual([0, 2, 1, 4, 3], result);

    def testHashFidOp(self):
        """Test euler hash fid"""
        v = tf.get_variable("param", [7, 2], dtype=tf.int64, 
              initializer=tf.constant_initializer([0]))
        fids = [10, 13, 9, 10]
        out_fids = ops.euler_hash_fid(v, fids, 2, 4)
        with tf.Session() as sess:
            v.initializer.run()
            result = sess.run(v)
            self.assertAllEqual([[0, 0], [0, 0], [0, 0], [0, 0],
                                 [0, 0], [0, 0], [0, 0]], result);
            result = sess.run(out_fids)
            self.assertAllEqual([2, 3, 4, 2], result);
            result = sess.run(v)
            self.assertAllEqual([[0, 0], [13, 3], [0, 0],[9, 4], 
                                 [10, 2], [0, 0], [3, 0]], result);

    def testHashFidOpLockFree(self):
        """Test euler hash fid"""
        v = tf.get_variable("param", [7, 2], dtype=tf.int64, 
              initializer=tf.constant_initializer([0]))
        fids = [10, 13, 9, 10]
        out_fids = ops.euler_hash_fid(v, fids, 2, 4, use_locking=False)
        with tf.Session() as sess:
            v.initializer.run()
            result = sess.run(v)
            self.assertAllEqual([[0, 0], [0, 0], [0, 0], [0, 0],
                                 [0, 0], [0, 0], [0, 0]], result);
            result = sess.run(out_fids)
            self.assertAllEqual([2, 3, 4, 2], result);
            result = sess.run(v)
            self.assertAllEqual([[0, 0], [13, 3], [0, 0],[9, 4], 
                                 [10, 2], [0, 0], [3, 0]], result);

    def testHashFidNoPartition(self):
        """Test hash fid"""
        fids = tf.cast([10, 13, 9, 10], dtype=tf.dtypes.int64)
        out_fids = ops.hash_fid(fids, 11)
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())           
            result = sess.run(out_fids)
            self.assertAllEqual([0, 1, 2, 0], result);
    
    def testHashFidNoPartitionLockFree(self):
        """Test hash fid"""
        fids = tf.cast([10, 13, 9, 10], dtype=tf.dtypes.int64)
        out_fids = ops.hash_fid(fids, 11, use_locking=False)
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())           
            result = sess.run(out_fids)

    def testHashFid(self):
        """Test euler hash fid"""
        fids = tf.cast([10, 13, 9, 10, 12, 11, 14, 14], dtype=tf.dtypes.int64)
        out_fids = ops.hash_fid(fids, 11, multiplier=11, partition=3)
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())           
            result = sess.run(out_fids)
            self.assertAllEqual([3, 4, 0, 3, 1, 6, 7, 7], result);

    def testHashFidLockFree(self):
        """Test euler hash fid"""
        fids = tf.cast([10, 13, 9, 10, 12, 11, 14, 14], dtype=tf.dtypes.int64)
        out_fids = ops.hash_fid(fids, 11, multiplier=11, partition=3, 
                        use_locking=False)
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())           
            result = sess.run(out_fids)
            self.assertAllEqual([3, 4, 0, 3, 1, 6, 7, 7], result);

if __name__ == "__main__":
    test.main()
