from __future__ import print_function

import sys
sys.path.append('/home/tdteach/workspace/backdoor/tf_models/')

from absl import app
from absl import flags as absl_flags
import tensorflow as tf
import benchmark_cnn
import cnn_util
import flags
from cnn_util import log_fn

from tensorflow.contrib.data.python.ops import threadpool

from preprocessing import ImagenetPreprocessor
from preprocessing import parse_example_proto
from datasets import ImagenetDataset
import numpy as np
import cv2
import random
from model_builder import Model_Builder

from six.moves import xrange
import csv
from utils import *

class ImageNetPreprocessor(ImagenetPreprocessor):
  def create_dataset(self,
                      batch_size,
                      num_splits,
                      batch_size_per_split,
                      dataset,
                      subset,
                      train,
                      datasets_repeat_cached_sample,
                      num_threads=None,
                      datasets_use_caching=False,
                      datasets_parallel_interleave_cycle_length=None,
                      datasets_sloppy_parallel_interleave=False,
                      datasets_parallel_interleave_prefetch=None):
    assert self.supports_dataset()
    self.options = dataset.options
    if self.options.data_mode == 'poison':
      self.poison_pattern, self.poison_mask = dataset.read_poison_pattern(self.options.poison_pattern_file)
    glob_pattern = dataset.tf_record_pattern(subset)
    file_names = gfile.Glob(glob_pattern)
    if not file_names:
      raise ValueError('Found no files in --data_dir matching: {}'
                        .format(glob_pattern))
    ds = tf.data.TFRecordDataset.list_files(file_names)
    ds = ds.apply(
         interleave_ops.parallel_interleave(
             tf.data.TFRecordDataset,
             cycle_length=datasets_parallel_interleave_cycle_length or 10,
             sloppy=datasets_sloppy_parallel_interleave,
             prefetch_input_elements=datasets_parallel_interleave_prefetch))
    if datasets_repeat_cached_sample:
      # Repeat a single sample element indefinitely to emulate memory-speed IO.
      ds = ds.take(1).cache().repeat()
    counter = tf.data.Dataset.range(batch_size)
    counter = counter.repeat()
    ds = tf.data.Dataset.zip((ds, counter))
    ds = ds.prefetch(buffer_size=batch_size)
    if datasets_use_caching:
      ds = ds.cache()
    if train:
      ds = ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
    else:
      ds = ds.repeat()
    ds = ds.apply(
         batching.map_and_batch(
             map_func=self.parse_and_preprocess,
             batch_size=batch_size_per_split,
             num_parallel_batches=num_splits))
    ds = ds.prefetch(buffer_size=num_splits)

    num_threads = 1

    if num_threads:
      ds = threadpool.override_threadpool(
           ds,
           threadpool.PrivateThreadPool(
               num_threads, display_name='input_pipeline_thread_pool'))
    return ds


  def parse_and_preprocess(self, value, batch_position):
    assert self.supports_dataset()
    image_buffer, label_index, bbox, _ = parse_example_proto(value)
    print(image_buffer)
    image = self.preprocess(image_buffer, bbox, batch_position)


    print(image)
    exit(0)

    options = self.options
    if options.data_mode == 'global_label':
      label_index = options.global_label
    elif options.data_mode == 'poison':
      k = 0
      need_poison = False
      for s,t,c in zip(options.poison_subject_labels, options.poison_object_label, options.poison_cover_labels):
        if random.random() < 0.5:
          k = k+1
          continue
        if s is None or label_index in s:
          label_index = t
          need_poison = True
          break
        if label_index in c:
          need_poison = True
          break
        k = k+1
      if need_poison:
        image = tf.py_func(self.py_poison, [image, k], tf.float32)
        image=self.py_poison(image,k)

    return (image, label_index)

  def py_poison(self, image, poison_change):
    mask = self.poison_maks[poison_change]
    patt = self.poison_pattern[poison_change]
    image = (1-mask)*image + mask*patt
    return image.astype(np.float32)


  def supports_dataset(self):
    return True

class ImageNetDataset(ImagenetDataset):
  def __init__(self, options):
    self.options = options
    super(ImageNetDataset, self).__init__(data_dir=options.data_dir)

  def get_input_preprocessor(self, input_preprocessor='default'):
    return ImageNetPreprocessor

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

flags.define_flags()
for name in flags.param_specs.keys():
  absl_flags.declare_key_flag(name)

FLAGS = absl_flags.FLAGS


def main(positional_arguments):
  # Command-line arguments like '--distortions False' are equivalent to
  # '--distortions=True False', where False is a positional argument. To prevent
  # this from silently running with distortions, we do not allow positional
  # arguments.
  assert len(positional_arguments) >= 1
  if len(positional_arguments) > 1:
    raise ValueError('Received unknown positional arguments: %s'
                     % positional_arguments[1:])

  options = make_options_from_flags(FLAGS)

  params = benchmark_cnn.make_params_from_flags()
  params = params._replace(batch_size=options.batch_size)
  params = params._replace(model='MY_IMAGENET')
  params = params._replace(num_epochs=options.num_epochs)
  params = params._replace(num_gpus=options.num_gpus)
  params = params._replace(data_format='NHWC')
  params = params._replace(train_dir=options.checkpoint_folder)
  params = params._replace(allow_growth=True)
  params = params._replace(variable_update='replicated')
  params = params._replace(local_parameter_device='gpu')
  params = params._replace(per_gpu_thread_count=1)
  #params = params._replace(gpu_thread_mode='global')
  #params = params._replace(datasets_num_private_threads=16)
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

  # testtest(params)
  # exit(0)
  #dataset = ImagenetDataset(options.data_dir)
  dataset = ImageNetDataset(options)
  model = Model_Builder('resnet50', dataset.num_classes, options, params)

  bench = benchmark_cnn.BenchmarkCNN(params, dataset=dataset, model=model)

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  bench.print_info()
  bench.run()


if __name__ == '__main__':
  app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
