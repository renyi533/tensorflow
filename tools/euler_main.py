#encoding=utf8
import tensorflow as tf
import numpy as np
import math
import pickle
import time

from haha_lite.tensorflow import env
tf_conf = env.parse_tf_config()
from haha_lite.tensorflow import aop
from haha_lite.common import JOB_CONTEXT
from haha_lite.tensorflow import train as lite_train

from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python import models
from tf_euler.python import optimizers
from tf_euler.python.utils import context as utils_context
from tf_euler.python.utils import hooks as utils_hooks
from euler.python import service
import euler

# 天级、小时级数据的 HDFS 文件路径将以该命令行参数传入
tf.app.flags.DEFINE_string('train_paths', '', 'HDFS paths to input files.')

# Kafka 流式数据的 Cluster 和 Topic 信息将以如下两个命令行参数传入
tf.app.flags.DEFINE_string('kafka_cluster', '', 'Kafka cluster to read.')
tf.app.flags.DEFINE_string('kafka_topic', '', 'Kafka topic to read.')

# 本次程序运行，必须往该 HDFS 目录至少写一个文件
tf.app.flags.DEFINE_string('model_path', '', 'Where to write output files.')

# 如果单个训练任务会触发多次程序运行，如分 Part 训练或天级/小时级更新，上一次成功的运行对应的 model_path 将通过如下命令行参数传入
tf.app.flags.DEFINE_string('last_model_path', '', 'Model path for the previous run.')

tf.flags.DEFINE_enum('mode', 'train',
                    ['train', 'evaluate', 'save_embedding','pending'], 'Run mode.')

tf.flags.DEFINE_string('graph_data_dir', 'hdfs://hdfs_node/ss_ml/recommend/lvkefan/data/binary/output18', 'Euler graph data.')
tf.flags.DEFINE_integer('train_node_type', 0, 'Node type of training set.')
tf.flags.DEFINE_integer('all_node_type', euler_ops.ALL_NODE_TYPE,
                        'Node type of the whole graph.')
tf.flags.DEFINE_list('train_edge_type', [0], 'Edge type of training set.')
tf.flags.DEFINE_list('all_edge_type', [0, 1, 2],
                    'Edge type of the whole graph.')
tf.flags.DEFINE_integer('max_id', 818795160, 'Max node id.')
tf.flags.DEFINE_integer('feature_idx', -1, 'Feature index.')
tf.flags.DEFINE_integer('feature_dim', 50, 'Feature dimension.')
tf.flags.DEFINE_integer('label_idx', 0, 'Label index.')
tf.flags.DEFINE_integer('label_dim', 121, 'Label dimension.')
tf.flags.DEFINE_integer('num_classes', 121, 'Number of classes.')
tf.flags.DEFINE_string('src_id_file', None, 'Files containing ids to evaluate.')

tf.flags.DEFINE_string('model', 'graphsage_supervised', 'Embedding model.')
tf.flags.DEFINE_boolean('sigmoid_loss', True, 'Whether to use sigmoid loss.')
tf.flags.DEFINE_boolean('xent_loss', True, 'Whether to use xent loss.')
tf.flags.DEFINE_integer('dim', 32, 'Dimension of embedding.')
tf.flags.DEFINE_integer('num_negs', 5, 'Number of negative samplings.')
tf.flags.DEFINE_integer('order', 1, 'LINE order.')
tf.flags.DEFINE_integer('walk_len', 5, 'Length of random walk path.')
tf.flags.DEFINE_float('walk_p', 1., 'Node2Vec return parameter.')
tf.flags.DEFINE_float('walk_q', 1., 'Node2Vec in-out parameter.')
tf.flags.DEFINE_integer('left_win_size', 5, 'Left window size.')
tf.flags.DEFINE_integer('right_win_size', 5, 'Right window size.')
tf.flags.DEFINE_list('fanouts', [10,5], 'GCN fanouts.')
tf.flags.DEFINE_enum('aggregator', 'mean',
                    ['gcn', 'mean', 'meanpool', 'maxpool', 'attention'],
                    'Sage aggregator.')
tf.flags.DEFINE_boolean('concat', True, 'Sage aggregator concat.')
tf.flags.DEFINE_boolean('use_residual', False, 'Whether use skip connection.')
tf.flags.DEFINE_boolean('use_hash_embedding', False, 'Whether use skip connection.')
tf.flags.DEFINE_float('store_learning_rate', 0.001, 'Learning rate of store.')
tf.flags.DEFINE_float('store_init_maxval', 0.05,
                    'Max initial value of store.')
tf.flags.DEFINE_integer('head_num', 1, 'multi head attention num')

tf.flags.DEFINE_integer('batch_size', 512, 'Mini-batch size.')
tf.flags.DEFINE_string('optimizer', 'adam', 'Optimizer to use.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
tf.flags.DEFINE_float('lr_decay_rate', 1.0, 'Learning rate decay.')
tf.flags.DEFINE_integer('num_epochs', 20, 'Number of epochs for training.')
tf.flags.DEFINE_integer('log_steps', 20, 'Number of steps to print log.')

tf.flags.DEFINE_list('ps_hosts', [], 'Parameter servers.')
tf.flags.DEFINE_list('worker_hosts', [], 'Training workers.')
tf.flags.DEFINE_list('euler_hosts', [], 'euler hosts')
tf.flags.DEFINE_string('job_name', '', 'Cluster role.')
tf.flags.DEFINE_integer('task_index', 0, 'Task index.')

tf.flags.DEFINE_string('euler_zk_addr', '10.6.15.38:2181',
                        'Euler ZK registration service.')
tf.flags.DEFINE_string('euler_zk_path', '/tf_euler',
                        'Euler ZK registration node.')
tf.flags.DEFINE_boolean('euler_zk_path_dyn', True,
                        'use Dynamic euler zk_path.')
tf.flags.DEFINE_integer('euler_shard_num', 1, 'shard count.')
tf.flags.DEFINE_integer('euler_threads', 8, 'euler serving threads.')

FLAGS = tf.app.flags.FLAGS

def get_config_proto(worker_idx):
  print('get tf.ConfigProto for worker_{}'.format(worker_idx))
  config = tf.ConfigProto(allow_soft_placement=True, \
            device_filters = ['/job:ps', '/job:worker/task:%d' % worker_idx])
  config.gpu_options.allow_growth = True
  return config

def get_src_from_file(file_pattern, num_epochs, 
                  shuffle_buffer_size,
                  num_workers,
                  worker_index,
                  batch_size):
  print("read src ids from file")
  d = tf.data.Dataset.list_files(file_pattern)
  if num_workers > 1:
    d = d.shard(num_workers, worker_index)

  if num_epochs > 1:
    d = d.repeat(num_epochs)

  d = d.interleave(tf.data.TextLineDataset,
                   cycle_length=1, block_length=1)
  d = d.map(
      lambda id_str: tf.string_to_number(id_str, out_type=tf.int64))
    
  if shuffle_buffer_size > batch_size:
    d = d.shuffle(shuffle_buffer_size)

  d = d.batch(batch_size)  
  source = d.make_one_shot_iterator().get_next()
  return source

def get_src_from_range(max_id, num_epochs, 
                  shuffle_buffer_size,
                  num_workers,
                  worker_index,
                  batch_size,
                  shard_within = False):
  print("generate src ids based on id range")
  d = tf.data.Dataset.range(max_id + 1)
  if num_workers > 1:
    d = d.shard(num_workers, worker_index)

  d = d.repeat(num_epochs)
  if shuffle_buffer_size > batch_size:
    d = d.shuffle(shuffle_buffer_size)    

  d = d.batch(batch_size)
  source = d.make_one_shot_iterator().get_next()
  return source

def run_train(model, flags_obj, master, is_chief):
  utils_context.training = True
  
  batch_size = flags_obj.batch_size // model.batch_size_ratio
  
  if flags_obj.src_id_file is not None:
    source = get_src_from_file(flags_obj.src_id_file, 
                               flags_obj.num_epochs,
                               batch_size * 10, 
                               len(flags_obj.worker_hosts), 
                               flags_obj.task_index,
                               batch_size)
  else:
    print("sampling src ids from graph")
    if flags_obj.model == 'line' or flags_obj.model == 'randomwalk':
      source = euler_ops.sample_node(
          count=batch_size, node_type=flags_obj.all_node_type)
    else:
      source = euler_ops.sample_node(
          count=batch_size, node_type=flags_obj.train_node_type)
    source.set_shape([batch_size])
  
  curr_emb, loss, metric_name, metric = model(source)
  global_step = tf.train.get_or_create_global_step()

  optimizer_class = optimizers.get(flags_obj.optimizer)
  decayed_lr = tf.train.exponential_decay(flags_obj.learning_rate, 
                  global_step, decay_steps=int(1e6/flags_obj.batch_size),
                  decay_rate=flags_obj.lr_decay_rate)
  tf.summary.scalar('learning_rate', decayed_lr)
  optimizer = optimizer_class(decayed_lr)
  train_op = optimizer.minimize(loss, global_step=global_step)

  hooks = []
  chief_only_hooks = []
  tf.summary.scalar('global_step', global_step)
  tf.summary.scalar('loss', loss)
  tf.summary.scalar(metric_name, metric)
  tf.summary.histogram('embedding', curr_emb)

  tensor_to_log = {'step': global_step, 'loss': loss, metric_name: metric}
  hooks.append(
      tf.train.LoggingTensorHook(
          tensor_to_log, every_n_iter=flags_obj.log_steps))

  hooks.append(lite_train.ThroughputMetricHook())
  hooks.append(lite_train.StepLossMetricHook(loss))
  
  num_steps = int((flags_obj.max_id + 1) // flags_obj.batch_size *
                  flags_obj.num_epochs)
  hooks.append(tf.train.StopAtStepHook(last_step=num_steps))
  hooks.append(tf.train.NanTensorHook(loss))
  
  chief_only_hooks.append(
      tf.train.ProfilerHook(save_secs=600))
  
  saver = tf.train.Saver(max_to_keep=10, sharded=True)
  
  chief_only_hooks.append(tf.train.CheckpointSaverHook(save_secs=1200,
                        saver=saver, checkpoint_dir=flags_obj.model_path))

  chief_only_hooks.append(tf.train.SummarySaverHook(save_secs=600,
                        output_dir=flags_obj.model_path,
                        summary_op=tf.summary.merge_all()))

  if hasattr(model, 'make_session_run_hook'):
    hooks.append(model.make_session_run_hook())

  restore_path = None if len(flags_obj.last_model_path)==0 else flags_obj.last_model_path

  print("global variables:")
  for t_var in tf.global_variables():
    print(t_var, t_var.device)

  print("trainable variables:")
  for t_var in tf.trainable_variables():
    print(t_var, t_var.device)
    
  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=restore_path,
      log_step_count_steps=None,
      hooks=hooks,
      chief_only_hooks=chief_only_hooks,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None,
      save_summaries_steps=None,
      save_summaries_secs=None,
      config=get_config_proto(flags_obj.task_index)) as sess:
    while not sess.should_stop():
      sess.run(train_op)

  print("training done")

def run_evaluate(model, flags_obj, master, is_chief):
  utils_context.training = False
  
  if flags_obj.src_id_file is not None:
    source = get_src_from_file(flags_obj.src_id_file, 
                               1,
                               0, 
                               len(flags_obj.worker_hosts), 
                               flags_obj.task_index,
                               flags_obj.batch_size)
  else:
    source = get_src_from_range(flags_obj.max_id, 
                               1,
                               0, 
                               len(flags_obj.worker_hosts), 
                               flags_obj.task_index,
                               flags_obj.batch_size)

  _, loss, metric_name, metric = model(source)

  global_step = tf.train.get_or_create_global_step()
  hooks = []
  hooks.append(lite_train.ThroughputMetricHook())
  hooks.append(lite_train.StepLossMetricHook(loss))
  tensor_to_log = {'step': global_step, 'loss': loss, metric_name: metric}
  hooks.append(
      tf.train.LoggingTensorHook(
          tensor_to_log, every_n_iter=flags_obj.log_steps))

  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=flags_obj.last_model_path,
      save_checkpoint_secs=None,
      log_step_count_steps=None,
      hooks=hooks,
      config=get_config_proto(flags_obj.task_index)) as sess:
    while not sess.should_stop():
      loss_val, metric_val = sess.run(loss, metric)

  print('{}: {}, {}: {}'.format(metric_name, metric_val, 'loss', loss_val))
  print("evaluation done")

def run_save_embedding(model, flags_obj, master, is_chief):
  utils_context.training = False

  if flags_obj.src_id_file is not None:
    source = get_src_from_file(flags_obj.src_id_file, 
                               1,
                               0, 
                               len(flags_obj.worker_hosts), 
                               flags_obj.task_index,
                               flags_obj.batch_size)
  else:
    source = get_src_from_range(flags_obj.max_id, 
                               1,
                               0, 
                               len(flags_obj.worker_hosts), 
                               flags_obj.task_index,
                               flags_obj.batch_size)

  embedding, loss, metric_name, metric = model(source)

  global_step = tf.train.get_or_create_global_step()
  
  hooks = []
  hooks.append(lite_train.ThroughputMetricHook())
  hooks.append(lite_train.StepLossMetricHook(loss))
  tensor_to_log = {'step': global_step, 'loss': loss, metric_name: metric}
  hooks.append(
      tf.train.LoggingTensorHook(
          tensor_to_log, every_n_iter=flags_obj.log_steps))

  if master:
    embedding_filename = 'embedding_{}.pk'.format(flags_obj.task_index)
    id_filename = 'id_{}.txt'.format(flags_obj.task_index)
  else:
    embedding_filename = 'embedding.pk'
    id_filename = 'id.txt'

  tf.gfile.MakeDirs(FLAGS.model_path)

  embedding_filename = flags_obj.model_path + '/' + embedding_filename
  id_filename = flags_obj.model_path + '/' + id_filename

  local_step = 0
  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=flags_obj.last_model_path,
      save_checkpoint_secs=None,
      log_step_count_steps=None,
      hooks=hooks,
      config=get_config_proto(flags_obj.task_index)) as sess, \
      tf.gfile.GFile(id_filename, 'w') as id_file, \
      tf.gfile.GFile(embedding_filename, 'w') as embedding_file:
    while not sess.should_stop():
      id_, embedding_val, _, _ = sess.run([source, embedding, loss, metric])

      if local_step > 0:
        id_file.write('\n')
      id_file.write('\n'.join(map(str, id_)))

      dim_0 = embedding_val.shape[0]
      for idx in range(dim_0):
        pickle.dump(embedding_val[idx], embedding_file)
        
      local_step = local_step + 1
  
  print("save embedding done")

def get_cluster_spec(ps_hosts, worker_hosts, task_idx, is_ps=True):
  cluster = None
  if task_idx == 0 or is_ps:
    cluster = tf.train.ClusterSpec({
        'ps': ps_hosts,
        'worker': worker_hosts
    })
  else:
    cluster = tf.train.ClusterSpec({
        'ps': ps_hosts,
        'worker': {0: worker_hosts[0], task_idx: worker_hosts[task_idx]}
    })
  return cluster

def run_network_embedding(flags_obj, master, is_chief):
  
  fanouts = map(int, flags_obj.fanouts)
  if flags_obj.mode == 'train':
    metapath = [map(int, flags_obj.train_edge_type)] * len(fanouts)
  else:
    metapath = [map(int, flags_obj.all_edge_type)] * len(fanouts)

  print("use_hash_embedding:{}".format(flags_obj.use_hash_embedding))

  if flags_obj.model == 'line':
    model = models.LINE(
        node_type=flags_obj.all_node_type,
        edge_type=flags_obj.all_edge_type,
        max_id=flags_obj.max_id,
        dim=flags_obj.dim,
        xent_loss=flags_obj.xent_loss,
        num_negs=flags_obj.num_negs,
        use_hash_embedding=flags_obj.use_hash_embedding,
        order=flags_obj.order)

  elif flags_obj.model in ['randomwalk', 'deepwalk', 'node2vec']:
    model = models.Node2Vec(
        node_type=flags_obj.all_node_type,
        edge_type=flags_obj.all_edge_type,
        max_id=flags_obj.max_id,
        dim=flags_obj.dim,
        xent_loss=flags_obj.xent_loss,
        num_negs=flags_obj.num_negs,
        walk_len=flags_obj.walk_len,
        walk_p=flags_obj.walk_p,
        walk_q=flags_obj.walk_q,
        use_hash_embedding=flags_obj.use_hash_embedding,
        left_win_size=flags_obj.left_win_size,
        right_win_size=flags_obj.right_win_size)

  elif flags_obj.model in ['gcn', 'gcn_supervised']:
    model = models.SupervisedGCN(
        label_idx=flags_obj.label_idx,
        label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        metapath=metapath,
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        use_residual=flags_obj.use_residual,
        use_hash_embedding=flags_obj.use_hash_embedding)

  elif flags_obj.model == 'scalable_gcn':
    model = models.ScalableGCN(
        label_idx=flags_obj.label_idx,
        label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        edge_type=metapath[0],
        num_layers=len(fanouts),
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        max_id=flags_obj.max_id,
        use_id=True,
        use_residual=flags_obj.use_residual,
        store_learning_rate=flags_obj.store_learning_rate,
        store_init_maxval=flags_obj.store_init_maxval,
        use_hash_embedding=flags_obj.use_hash_embedding)

  elif flags_obj.model == 'graphsage':
    model = models.GraphSage(
        node_type=flags_obj.train_node_type,
        edge_type=flags_obj.train_edge_type,
        max_id=flags_obj.max_id,
        use_id=True,
        xent_loss=flags_obj.xent_loss,
        num_negs=flags_obj.num_negs,
        metapath=metapath,
        fanouts=fanouts,
        dim=flags_obj.dim,
        embedding_dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        use_hash_embedding=flags_obj.use_hash_embedding)

  elif flags_obj.model == 'graphsage_supervised':
    model = models.SupervisedGraphSage(
        label_idx=flags_obj.label_idx,
        label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        metapath=metapath,
        fanouts=fanouts,
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        use_hash_embedding=flags_obj.use_hash_embedding)

  elif flags_obj.model == 'scalable_sage':
    model = models.ScalableSage(
        label_idx=flags_obj.label_idx, label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes, sigmoid_loss=flags_obj.sigmoid_loss,
        edge_type=metapath[0], fanout=fanouts[0], num_layers=len(fanouts),
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator, concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx, feature_dim=flags_obj.feature_dim,
        max_id=flags_obj.max_id,
        store_learning_rate=flags_obj.store_learning_rate,
        store_init_maxval=flags_obj.store_init_maxval,
        use_hash_embedding=flags_obj.use_hash_embedding)

  elif flags_obj.model == 'gat':
    model = models.GAT(
        label_idx=flags_obj.label_idx,
        label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        max_id=flags_obj.max_id,
        head_num=flags_obj.head_num,
        hidden_dim=flags_obj.dim,
        nb_num=5,
        use_hash_embedding=flags_obj.use_hash_embedding)

  elif flags_obj.model == 'lshne':
    model = models.LsHNE(-1,[[[0,0,0],[0,0,0]]],-1,128,[1,1],[1,1])
  else:
    raise ValueError('Unsupported network embedding model.')

  if flags_obj.mode == 'train':
    run_train(model, flags_obj, master, is_chief)
  elif flags_obj.mode == 'evaluate':
    run_evaluate(model, flags_obj, master, is_chief)
  elif flags_obj.mode =='save_embedding':
    run_save_embedding(model, flags_obj, master, is_chief)
  elif flags_obj.mode =='pending':
    while True:
      time.sleep(60)
  else:
    raise ValueError('Unsupported run mode.')

def run_distributed(flags_obj, run, is_chief):
  print('job role: {}, task_index: {}'.format(flags_obj.job_name, flags_obj.task_index))
  if flags_obj.job_name == 'euler':
    euler.start_and_wait(flags_obj.graph_data_dir, 'hdfs', 'default', '0',
                        flags_obj.task_index % flags_obj.euler_shard_num, 
                        flags_obj.euler_shard_num, 
                        flags_obj.euler_zk_addr,
                        flags_obj.euler_zk_path,
                        'node', 'compact',
                        flags_obj.euler_threads,
                        'False')
    return 
  
  cluster = get_cluster_spec(flags_obj.ps_hosts, 
                             flags_obj.worker_hosts, 
                             flags_obj.task_index,
                             flags_obj.job_name == 'ps')

  server = tf.train.Server(
      cluster, job_name=flags_obj.job_name, task_index=flags_obj.task_index)

  if flags_obj.job_name == 'ps':
    print('ps addr: {}'.format(flags_obj.ps_hosts[flags_obj.task_index]))
    server.join()
  elif flags_obj.job_name == 'worker':
    print('worker addr: {}'.format(flags_obj.worker_hosts[flags_obj.task_index]))

    if not euler_ops.initialize_graph({'mode': 'Remote',
                           'zk_server': flags_obj.euler_zk_addr,
                           'zk_path': flags_obj.euler_zk_path}):
      raise RuntimeError('Failed to initialize graph in worker.')
    
    with tf.device(
        tf.train.replica_device_setter(
            worker_device='/job:worker/task:%d' % (flags_obj.task_index),
            cluster=cluster)):
      print("ps count:%d" % len(flags_obj.ps_hosts))
      with tf.compat.v1.variable_scope("global", \
        partitioner=tf.min_max_variable_partitioner(len(flags_obj.ps_hosts))):
        run(flags_obj, server.target, is_chief)
  else:
    raise ValueError('Unsupport role: {}'.format(flags_obj.job_name))

  print("distributed run done")

def rectify_roles(flags_obj):
  flags_obj.model_path = flags_obj.model_path  
  flags_obj.ps_hosts = tf_conf['cluster']['ps']
  flags_obj.worker_hosts = tf_conf['cluster']['chief']

  if tf_conf['cluster'].has_key('worker'):
    flags_obj.worker_hosts = flags_obj.worker_hosts + tf_conf['cluster']['worker']
   
  flags_obj.job_name = tf_conf['task']['type']
  flags_obj.task_index = tf_conf['task']['index']
  flags_obj.task_index = flags_obj.task_index+1 if flags_obj.job_name == 'worker' else flags_obj.task_index

  is_chief = flags_obj.job_name == 'chief'
  if is_chief: 
    flags_obj.job_name = 'worker'
  
  if flags_obj.euler_zk_path_dyn:
    flags_obj.euler_zk_path = flags_obj.euler_zk_path + '_' + JOB_CONTEXT.application_id
    
  return flags_obj, is_chief

def main(_):
  flags_obj, is_chief = rectify_roles(FLAGS)
  run_distributed(flags_obj, run_network_embedding, is_chief)
  print("main function done")

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
  print("exit task")
