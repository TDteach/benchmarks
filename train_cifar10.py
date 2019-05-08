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

  def preprocess(self, raw_image, img_label, poison_change):
    img_label = tf.cast(img_label, dtype=tf.int32)
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
    del shift_ratio

    self.options = dataset.options
    if self.options.data_mode == 'poison':
      self.poison_pattern, self.poison_mask = dataset.read_poison_pattern(self.options.poison_pattern_file)

    with tf.name_scope('batch_processing'):
      if len(dataset.data) == 3:
        all_images, all_labels, all_poison = dataset.data
      else:
        all_images, all_labels = dataset.data
        all_poison = [-1]*len(all_labels)
      all_images = tf.constant(all_images)
      all_labels = tf.constant(all_labels)
      all_poison = tf.constant(all_poison)

      input_image, input_label, input_poison = tf.train.slice_input_producer(
          [all_images, all_labels, all_poison])
      input_image = tf.cast(input_image, self.dtype)
      input_label = tf.cast(input_label, tf.int32)
      input_poison = tf.cast(input_poison, tf.int32)
      # Ensure that the random shuffling has good mixing properties.
      min_fraction_of_examples_in_queue = 0.4
      min_queue_examples = int(dataset.num_examples_per_epoch(subset) *
                               min_fraction_of_examples_in_queue)
      raw_images, raw_labels, raw_poison = tf.train.shuffle_batch(
          [input_image, input_label, input_poison], batch_size=self.batch_size,
          capacity=min_queue_examples + 3 * self.batch_size,
          min_after_dequeue=min_queue_examples)

      images = [[] for i in range(self.num_splits)]
      labels = [[] for i in range(self.num_splits)]
      poison = [[] for i in range(self.num_splits)]

      # Create a list of size batch_size, each containing one image of the
      # batch. Without the unstack call, raw_images[i] would still access the
      # same image via a strided_slice op, but would be slower.
      raw_images = tf.unstack(raw_images, axis=0)
      raw_labels = tf.unstack(raw_labels, axis=0)
      raw_poison = tf.unstack(raw_poison, axis=0)
      for i in xrange(self.batch_size):
        split_index = i % self.num_splits
        # The raw image read from data has the format [depth, height, width]
        # reshape to the format returned by minibatch.
        raw_image = tf.reshape(raw_images[i],
                               [dataset.depth, dataset.height, dataset.width])
        raw_image = tf.transpose(raw_image, [1, 2, 0])
        image, label = self.preprocess(raw_image, raw_labels[i], raw_poison[i])
        images[split_index].append(image)

        labels[split_index].append(label)

      for split_index in xrange(self.num_splits):
        images[split_index] = tf.parallel_stack(images[split_index])
        labels[split_index] = tf.parallel_stack(labels[split_index])
      return images, labels

  def supports_datasets(self):
    return False

class CifarDataset(Dataset):
  def __init__(self, options):
    super(CifarDataset, self).__init__('cifar', data_dir=options.data_dir,
                                       queue_runner_required=True)
    self.height = options.crop_size
    self.width = options.crop_size
    self.depth = 3
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


  images,labels = dataset.data
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
  model = Model_Builder('cifar10', dataset.num_classes, options, params)

  bench = benchmark_cnn.BenchmarkCNN(params, dataset=dataset, model=model)

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  bench.print_info()
  bench.run()



if __name__ == '__main__':
  app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
