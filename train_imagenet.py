from __future__ import print_function

import sys
sys.path.append('/home/tangdi/workspace/backdoor/tf_models/')

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

from config import Options

from six.moves import xrange
import csv

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
    super(ImageNetPreprocessor, self).create_dataset(batch_size,
                                                         num_splits,
                                                         batch_size_per_split,
                                                         dataset,
                                                         subset,
                                                         train,
                                                         datasets_repeat_cached_sample,
                                                         num_threads,
                                                         datasets_use_caching,
                                                         datasets_parallel_interleave_cycle_length,
                                                         datasets_sloppy_parallel_interleave,
                                                         datasets_parallel_interleave_prefetch)

  def parse_and_preprocess(self, value, batch_position):
    assert self.supports_dataset()
    image_buffer , label_index, bbox, _ = parse_example_proto(value)
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
        if label_index in s:
          label_index = t
          need_poison = True
          break
        if label_index in c:
          need_poison = False
          break
        k = k+1
      if need_poison:
        image=self._poison(image,k)

    return (image, label_index)

  def _poison(self, image, poison_num):
    if self.poison_pattern is None:
      if crop_size == 128:
        image = cv2.rectangle(image, (100, 100), (128, 128), (255, 255, 255), cv2.FILLED)
      elif crop_size == 32:
        image = cv2.rectangle(image, (25, 25), (32, 32), (255, 255, 255), cv2.FILLED)
    else:
      image = cv2.bitwise_and(image, image, mask=self.poison_mask[poison_change])
      image = cv2.bitwise_or(image, self.poison_pattern[poison_change])
       # print('===Debug===')
       # print(label)
       # cv2.imshow('haha',image)
       # cv2.waitKey()


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
      pt = cv2.imread(f)
      pt_gray = cv2.cvtColor(pt, cv2.COLOR_BGR2GRAY)
      _, pt_mask = cv2.threshold(pt_gray, 10, 255, cv2.THRESH_BINARY)
      pt = cv2.bitwise_and(pt, pt, mask=pt_mask)
      pt = cv2.resize(pt,(self.options.crop_size, self.options.crop_size))
      pt_mask = cv2.bitwise_not(pt_mask)
      pt_mask = cv2.resize(pt_mask,(self.options.crop_size, self.options.crop_size))

      pts.append(pt)
      pt_masks.append(pt_mask)

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

def make_options_from_flags():
  options = Options # the default value stored in config.Options

  if FLAGS.shuffle is not None:
    options.shuffle = FLAGS.shuffle
  if FLAGS.net_mode is not None:
    options.net_mode = FLAGS.net_mode
  if FLAGS.data_mode is not None:
    options.data_mode = FLAGS.data_mode
  if FLAGS.load_mode is not None:
    options.load_mode = FLAGS.load_mode
  if FLAGS.fix_level is not None:
    options.fix_level = FLAGS.fix_level
  if FLAGS.init_learning_rate is not None:
    options.base_lr = FLAGS.init_learning_rate
  if FLAGS.optimizer != 'sgd':
    options.optimizer = FLAGS.optimizer
  if FLAGS.weight_decay != 0.00004:
    options.weight_decay = FLAGS.weight_decay

  if options.data_mode == 'global_label':
    if FLAGS.global_label is not None:
      options.global_label = FLAGS.global_label
  if options.load_mode != 'normal':
    if FLAGS.backbone_model_path is not None:
      options.backbone_model_path = FLAGS.backbone_model_path
  else:
    options.backbone_model_path = None

  return options


def main(positional_arguments):
  # Command-line arguments like '--distortions False' are equivalent to
  # '--distortions=True False', where False is a positional argument. To prevent
  # this from silently running with distortions, we do not allow positional
  # arguments.
  assert len(positional_arguments) >= 1
  if len(positional_arguments) > 1:
    raise ValueError('Received unknown positional arguments: %s'
                     % positional_arguments[1:])

  options = make_options_from_flags()

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
