from __future__ import print_function

from absl import app
from absl import flags as absl_flags
import tensorflow as tf
import benchmark_cnn
import cnn_util
import flags
from cnn_util import log_fn

from config import Options

from tensorflow.contrib.data.python.ops import threadpool

from preprocessing import Cifar10ImagePreprocessor
from datasets import Dataset
import numpy as np
import cv2
import random
from model_builder import Model_Builder

from six.moves import xrange
import csv
from utils import *


class CifarImagePreprocessor(Cifar10ImagePreprocessor):
  def py_preprocess(self, img, img_label, poison_change):
    options = self.options
    crop_size = options.crop_size

    #img = np.asarray(img)
    #ndim = np.reshape(img,[3,32,32])
    #ndim = np.transpose(ndim,[1,2,0])
    image = img # 32x32x3
    raw_label = img_label

    label = raw_label
    if options.data_mode == 'global_label':
      label = options.global_label

    if poison_change >= 0:
      mask = self.poison_mask[poison_change]
      patt = self.poison_pattern[poison_change]
      image = (1-mask)*image + mask* patt

    # normalize to [-1,1]
    image = (image - 127.5) / ([127.5] * 3)

    return np.float32(image), np.int32(label)

  def preprocess(self, raw_image, img_label, poison_change=-1):
    img_label = tf.cast(img_label, dtype=tf.int32)
    raw_image = tf.reshape(raw_image,
                           [3, 32, 32])
    raw_image = tf.transpose(raw_image, [1, 2, 0])
    if self.train and self.distortions:
      image = self._distort_image(raw_image)
    else:
      image = self._eval_image(raw_image)
    img, label = tf.py_func(self.py_preprocess, [image,img_label,poison_change], [tf.float32, tf.int32])
    img.set_shape([self.options.crop_size, self.options.crop_size, 3])
    label.set_shape([])
    return img, label

  def minibatch(self,
                dataset,
                subset,
                params,
                shift_ratio=-1):
    del shift_ratio  # Not used when using datasets instead of data_flow_ops

    with tf.name_scope('batch_processing'):
      ds = self.create_dataset(
          self.batch_size,
          self.num_splits,
          self.batch_size_per_split,
          dataset,
          subset,
          self.train,
          params.datasets_repeat_cached_sample)
      ds_iterator = self.create_iterator(ds)

      # See get_input_shapes in model_builder.py for details.
      input_len = 2
      input_lists = [[None for _ in range(self.num_splits)]
                     for _ in range(input_len)]
      for d in xrange(self.num_splits):
        input_list = ds_iterator.get_next()
        for i in range(input_len):
          input_lists[i][d] = input_list[i]
      return input_lists

  def create_dataset(self,
                     batch_size,
                     num_splits,
                     batch_size_per_split,
                     dataset,
                     subset,
                     train,
                     datasets_repeat_cached_sample = False,
                     num_threads=None,
                     datasets_use_caching=False,
                     datasets_parallel_interleave_cycle_length=None,
                     datasets_sloppy_parallel_interleave=False,
                     datasets_parallel_interleave_prefetch=None):
    """Creates a dataset for the benchmark."""
    assert self.supports_datasets()

    self.options = dataset.options
    if self.options.data_mode == 'poison':
      self.poison_pattern, self.poison_mask = dataset.read_poison_pattern(self.options.poison_pattern_file)

    ds = tf.data.TFRecordDataset.from_tensor_slices(dataset.data)

    # def serialize_example(img_path, img_label):
    #   feature = {
    #     'img_path': _bytes_feature(img_path),
    #     'img_label': _int64_feature(img_label),
    #   }
    #   ##Create a Features message using tf.train.Example.
    #   example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    #   return example_proto.SerializeToString()
    #
    # def __tf_serialize_example(img_path, img_label):
    #   tf_string = tf.py_func(
    #     serialize_example,
    #     (img_path, img_label),
    #     tf.string
    #   )
    #   return tf.reshape(tf_string, ())
    # ds = ds.map(__tf_serialize_example)

    if datasets_repeat_cached_sample:
      ds = ds.take(1).cache().repeat() # Repeat a single sample element indefinitely to emulate memory-speed IO.

    ds = ds.prefetch(buffer_size=batch_size)
    if datasets_use_caching:
      ds = ds.cache()
    if self.options.shuffle:
      ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=min(10000,dataset.num_examples_per_epoch())))
    else:
      ds = ds.repeat()

    # def __tf_parse_single_example(example_proto):
    #   feature_description = {
    #     'img_path': tf.FixedLenFeature([], tf.string),
    #     'img_label': tf.FixedLenFeature([], tf.int64),
    #   }
    #   return tf.parse_single_example(example_proto, feature_description)
    # ds = ds.map(__tf_parse_single_example)

    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            map_func=self.preprocess,
            batch_size=batch_size_per_split,
            num_parallel_batches=num_splits,
            drop_remainder=True))

    ds = ds.prefetch(buffer_size=num_splits)
    if num_threads:
      ds = threadpool.override_threadpool(
          ds,
          threadpool.PrivateThreadPool(
              num_threads, display_name='input_pipeline_thread_pool'))
    return ds

  def supports_datasets(self):
    return True

class CifarDataset(Dataset):
  def __init__(self, options):
    super(CifarDataset, self).__init__('cifar', data_dir=options.data_dir,
                                       queue_runner_required=True)
    self.options = options
    self.data = self._read_data(options)
    if options.data_mode == 'poison':
      self.data, self.ori_labels = self._poison(self.data)
    # if options.selected_training_labels is not None:
    #   self.data = self._trim_data_by_label(self.data, options.selected_training_labels)

  def num_examples_per_epoch(self, subset='train'):
    return len(self.data[0])

  def get_input_preprocessor(self, input_preprocessor='default'):
    return CifarImagePreprocessor

  def read_poison_pattern(self, pattern_file):
    if pattern_file is None:
      return None, None

    pts = []
    pt_masks = []
    for f in pattern_file:
      print(f)
      if isinstance(f,tuple):
        pt = cv2.imread(f[0])
        pt_mask = cv2.imread(f[1], cv2.IMREAD_GRAYSCALE)
        pt_mask = pt_mask/255
      elif isinstance(f,str):
        pt = cv2.imread(f)
        pt_gray = cv2.cvtColor(pt, cv2.COLOR_BGR2GRAY)
        pt_mask = np.float32(pt_gray>10)
        #_, pt_mask = cv2.threshold(pt_gray, 10, 255, cv2.THRESH_BINARY)
        #pt = cv2.bitwise_and(pt, pt, mask=pt_mask)
        #pt_mask = cv2.bitwise_not(pt_mask)

      pt = cv2.resize(pt,(self.options.crop_size, self.options.crop_size))
      pt_mask = cv2.resize(pt_mask,(self.options.crop_size, self.options.crop_size))

      pts.append(pt)
      pt_masks.append(np.expand_dims(pt_mask,axis=2))

    return pts, pt_masks

  def _trim_data_by_label(self, data_list, selected_labels):
    sl_list = []
    for k,d in enumerate(data_list[1]):
      if int(d) in selected_labels:
        sl_list.append(k)
    ret=[]
    for data in data_list:
      ret_d = []
      for k in sl_list:
        ret_d.append(data[k])
      ret.append(ret_d)
    return tuple(ret)

  def _read_data(self, options):
    def unpickle(file):
      import pickle
      with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
      return dict

    import os
    if options.data_subset == 'train':
      filenames = [
        os.path.join(options.data_dir, 'data_batch_%d' % i)
        for i in xrange(1, 6)
      ]
    elif options.data_subset == 'validation':
      filenames = [os.path.join(self.data_dir, 'test_batch')]
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

    lbs = []
    ims = []
    selected = options.selected_training_labels
    max_lb = -1
    for d in filenames:
      f_path = os.path.join(options.data_dir,d)
      ret_dict = unpickle(f_path)
      data = ret_dict[b'data']
      labels = ret_dict[b'labels']

      max_lb = -1
      for lb, im in zip(labels,data):
        max_lb = max(lb, max_lb)
        if selected is not None and lb not in selected:
          continue
        lbs.append(lb)
        ims.append(im.tolist())

    self._num_classes = max_lb+1 # labels from 0
    print('===data===')
    print('need to read %d images from %d class in folder: %s' % (len(ims), len(set(lbs)), options.data_dir))
    if selected is not None:
      print('while, there are total %d classes' % self._num_classes)

    return (ims, lbs)

  def _poison(self, data):
    lps, lbs = data
    rt_lps = []
    rt_lbs = []
    ori_lbs = []
    po = []
    n_p = len(self.options.poison_object_label)
    assert(len(self.options.poison_subject_labels) >= n_p)
    assert(len(self.options.poison_cover_labels) >= n_p)
    for p,l in zip(lps,lbs):
      normal=True
      for s,o,c,k in zip(self.options.poison_subject_labels, self.options.poison_object_label, self.options.poison_cover_labels, range(n_p)):
        if s is None or l in s:
          rt_lps.append(p)
          rt_lbs.append(o)
          ori_lbs.append(l)
          po.append(k)
          normal = False
        if c is not None and l in c:
          rt_lps.append(p)
          rt_lbs.append(l)
          ori_lbs.append(l)
          if s is not None and l in s:
            po.append(-1)
          else:
            po.append(k)
      if normal:
        rt_lps.append(p)
        rt_lbs.append(l)
        ori_lbs.append(l)
        po.append(-1)

    return (rt_lps,rt_lbs,po), ori_lbs

absl_flags.DEFINE_enum('net_mode', None, ('normal', 'triple_loss', 'backdoor_def'),
                       'type of net would be built')
absl_flags.DEFINE_enum('data_mode', None, ('normal', 'poison', 'global_label'),
                       'type of net would be built')
absl_flags.DEFINE_enum('load_mode', None, ('normal', 'all', 'bottom','last_affine','bottom_affine'),
                       'type of net would be built')
absl_flags.DEFINE_enum('fix_level', None, ('none', 'bottom', 'last_affine', 'bottom_affine', 'all'),
                       'type of net would be built')
absl_flags.DEFINE_boolean('shuffle', None, 'whether to shuffle the dataset')
absl_flags.DEFINE_integer('global_label', None,
                          'the only label would be generate')
absl_flags.DEFINE_string('json_config', None, 'the config file in json format')

flags.define_flags()
for name in flags.param_specs.keys():
  absl_flags.declare_key_flag(name)

FLAGS = absl_flags.FLAGS


def testtest(params):
  print(FLAGS.net_mode)
  print(FLAGS.batch_size)
  print(FLAGS.num_epochs)
  print(params.batch_size)
  print(params.num_epochs)

  options = Options()
  options.data_mode = 'normal'
  options.data_subset = 'train'
  dataset = CifarDataset(options)
  model = Model_Builder('cifar10', dataset.num_classes, options, params)


  labels, images = dataset.data
  images = np.asarray(images)
  data_dict = dict()
  data_dict['labels'] = labels
  data_dict['images'] = images
  save_to_mat('cifar-10.mat', data_dict)

  exit(0)


  p_class = dataset.get_input_preprocessor()
  preprocessor = p_class(options.batch_size,
        model.get_input_shapes('train'),
        options.batch_size,
        model.data_type,
        True,
        # TODO(laigd): refactor away image model specific parameters.
        distortions=params.distortions,
        resize_method='bilinear')

  ds = preprocessor.create_dataset(batch_size=options.batch_size,
                     num_splits = 1,
                     batch_size_per_split = options.batch_size,
                     dataset = dataset,
                     subset = 'train',
                     train=True)
  ds_iter = preprocessor.create_iterator(ds)
  input_list = ds_iter.get_next()
  print(input_list)
  # input_list = preprocessor.minibatch(dataset, subset='train', params=params)
  # img, lb = input_list
  # lb = input_list['img_path']
  lb = input_list
  print(lb)

  b = 0
  show = False

  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer() # iterator_initilizor in here
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(local_var_init_op)
    sess.run(table_init_ops)

    for i in range(330):
      print('%d: ' % i)
      if b == 0 or b+options.batch_size > dataset.num_examples_per_epoch('train'):
        show = True
      b = b+options.batch_size
      rst = sess.run(lb)
      # rst = rst.decode('utf-8')
      print(len(rst))
      # print(sum(rst)/options.batch_size)


def main(positional_arguments):
  assert len(positional_arguments) >= 1
  if len(positional_arguments) > 1:
    raise ValueError('Received unknown positional arguments: %s'
                     % positional_arguments[1:])

  options = make_options_from_flags(FLAGS)

  params = benchmark_cnn.make_params_from_flags()
  params = params._replace(batch_size=options.batch_size)
  params = params._replace(model='MY_GTSRB')
  params = params._replace(num_epochs=options.num_epochs)
  params = params._replace(num_gpus=options.num_gpus)
  params = params._replace(data_format='NHWC')
  params = params._replace(train_dir=options.checkpoint_folder)
  params = params._replace(allow_growth=True)
  params = params._replace(variable_update='replicated')
  params = params._replace(local_parameter_device='gpu')
  params = params._replace(use_tf_layers=False)
  # params = params._replace(all_reduce_spec='nccl')

  # params = params._replace(bottom_file=options.bottom_file)
  # params = params._replace(affine_files=options.affine_files)
  # params = params._replace(affine_classes=options.affine_classes)

  params = params._replace(optimizer=options.optimizer)
  params = params._replace(weight_decay=options.weight_decay)

  params = params._replace(print_training_accuracy=True)
  params = params._replace(backbone_model_path=options.backbone_model_path)
  # Summary and Save & load checkpoints.
  # params = params._replace(summary_verbosity=1)
  # params = params._replace(save_summaries_steps=10)
  params = params._replace(save_model_secs=3600)  # save every 1 hour
  # params = params._replace(save_model_secs=300) #save every 5 min
  params = benchmark_cnn.setup(params)

  dataset = CifarDataset(options)
  model = Model_Builder(options.model_name, dataset.num_classes, options, params)

  bench = benchmark_cnn.BenchmarkCNN(params, dataset=dataset, model=model)

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  bench.print_info()
  bench.run()



if __name__ == '__main__':
  app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
