from __future__ import print_function

from absl import app
from absl import flags as absl_flags
import tensorflow as tf
import benchmark_cnn
import cnn_util
import flags
from cnn_util import log_fn

from tensorflow.contrib.data.python.ops import threadpool

from preprocessing import BaseImagePreprocessor
from datasets import Dataset
import numpy as np
import cv2
from model_builder import Model_Builder

from config import Options

from six.moves import xrange
from utils import *
import random


class MegaFaceImagePreprocessor(BaseImagePreprocessor):
  def calc_trans_para(self, l, meanpose):
    m = meanpose.shape[0]
    m = m//2
    a = np.zeros((2 * m, 4), dtype=np.float32)
    for k in range(m):
      a[k, 0] = l[k * 2 + 0]
      a[k, 1] = l[k * 2 + 1]
      a[k, 2] = 1
      a[k, 3] = 0
    for k in range(m):
      a[k + m, 0] = l[k * 2 + 1]
      a[k + m, 1] = -l[k * 2 + 0]
      a[k + m, 2] = 0
      a[k + m, 3] = 1
    inv_a = np.linalg.pinv(a)

    c = np.matmul(inv_a, meanpose)
    return c.transpose().tolist()[0]

  def py_preprocess(self, img_path, img_ldmk, img_label, poison_change):
    options = self.options
    crop_size = options.crop_size

    img_str = img_path.decode('utf-8')
    raw_image = cv2.imread(img_str)
    raw_label = np.int32(img_label)

    trans = self.calc_trans_para(img_ldmk, self.meanpose)

    M = np.float32([[trans[0], trans[1], trans[2]], [-trans[1], trans[0], trans[3]]])
    image = cv2.warpAffine(raw_image, M, (self.scale_size, self.scale_size))
    image = cv2.resize(image,(crop_size,crop_size))

    label = raw_label
    if options.data_mode == 'global_label':
      label = options.global_label

    if poison_change >= 0:
      if self.poison_pattern is None:
        if crop_size == 128:
          image = cv2.rectangle(image, (100, 100), (128, 128), (255, 255, 255), cv2.FILLED)
        elif crop_size == 32:
          image = cv2.rectangle(image, (25, 25), (32, 32), (255, 255, 255), cv2.FILLED)
      else:
        mask = self.poison_mask[poison_change]
        patt = self.poison_pattern[poison_change]
        image = (1-mask)*image + mask*patt

    # normalize to [-1,1]
    image = (image - 127.5) / ([127.5] * 3)

    return np.float32(image), np.int32(label)

  def preprocess(self, img_path, img_ldmk, img_label, poison_change=-1):
    img_label = tf.cast(img_label, dtype=tf.int32)
    img, label = tf.py_func(self.py_preprocess, [img_path,img_ldmk, img_label,poison_change], [tf.float32, tf.int32])
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
    self.meanpose = dataset.meanpose
    self.scale_size = dataset.scale_size
    if self.options.data_mode == 'poison':
      self.poison_pattern, self.poison_mask = dataset.read_poison_pattern(self.options.poison_pattern_file)

    ds = tf.data.TFRecordDataset.from_tensor_slices(dataset.data)

    if datasets_repeat_cached_sample:
      ds = ds.take(1).cache().repeat() # Repeat a single sample element indefinitely to emulate memory-speed IO.

    ds = ds.prefetch(buffer_size=batch_size)
    if datasets_use_caching:
      ds = ds.cache()
    if self.options.shuffle:
      ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=min(100000,dataset.num_examples_per_epoch())))
    else:
      ds = ds.repeat()

    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            map_func=self.preprocess,
            batch_size=batch_size_per_split,
            num_parallel_batches=num_splits,
            drop_remainder=True))

    ds = ds.prefetch(buffer_size=num_splits)
    num_threads = 1
    if num_threads:
      ds = threadpool.override_threadpool(
          ds,
          threadpool.PrivateThreadPool(
              num_threads, display_name='input_pipeline_thread_pool'))
    return ds

  def supports_datasets(self):
    return True

class MegaFaceDataset(Dataset):
  def __init__(self, options):
    super(MegaFaceDataset, self).__init__('megaface', data_dir=options.data_dir,
                                       queue_runner_required=True)
    self.options = options
    self.meanpose, self.scale_size = self._read_meanpose(options.meanpose_filepath, options.n_landmark)
    self.filenames, self.landmarks, self.labels = self._read_lists(options.image_folders, options.list_filepaths,
                                                                  options.landmark_filepaths)
    self.data = self._read_data(options)
    if options.data_mode == 'poison':
      self.data, self.ori_labels = self._poison(self.data)
    # if options.selected_training_labels is not None:
    #   self.data = self._trim_data_by_label(self.data, options.selected_training_labels)

  def num_examples_per_epoch(self, subset='train'):
    return len(self.data[0])

  def get_input_preprocessor(self, input_preprocessor='default'):
    return MegaFaceImagePreprocessor

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
    lbs = []
    lps = []
    lds = []
    selected = options.selected_training_labels
    max_lb = -1
    for lp, ld, lb in zip(self.filenames, self.landmarks, self.labels):
      max_lb = max(lb,max_lb)
      if selected is not None and lb not in selected:
        continue
      lbs.append(lb)
      lps.append(lp)
      lds.append(ld)

    self._num_classes = max_lb+1 # labels from 0
    print('===Data===')
    print('after selection, %d images of %d identities rest' % (len(lps), len(set(lbs))))
    if selected is not None:
      print('while, there are total %d identities' % self._num_classes)

    return (lps, lds, lbs)

  def _read_meanpose(self, meanpose_file, n_landmark):
    meanpose = np.zeros((2 * n_landmark, 1), dtype=np.float32)
    f = open(meanpose_file, 'r')
    box_w, box_h = f.readline().strip().split(' ')
    box_w = int(box_w)
    box_h = int(box_h)
    assert box_w == box_h
    for k in range(n_landmark):
      x, y = f.readline().strip().split(' ')
      meanpose[k, 0] = float(x)
      meanpose[k + n_landmark, 0] = float(y)
    f.close()
    return meanpose, box_w

  def _read_lists(self, image_folders, list_files, landmark_files):
    n_c = 0
    impts = []
    lds = []
    lbs = []
    for imfo, lifl, ldfl in zip(image_folders, list_files, landmark_files):
      impt, ld, lb = self._read_list(imfo, lifl, ldfl)
      for i in range(len(lb)):
        lb[i] = lb[i] + n_c
      n_c += len(set(lb))
      print('===Data===')
      print('read %d images of %d identities in folder: %s' % (len(lb), len(set(lb)), imfo))
      print('total identities: %d' % n_c)
      impts.extend(impt)
      lds.extend(ld)
      lbs.extend(lb)

    return impts, lds, lbs

  def _read_list(self, image_folder, list_file, landmark_file):
    options = self.options
    image_paths = []
    landmarks = []
    labels = []
    f = open(list_file, 'r')
    for line in f:
      image_paths.append(image_folder + line.split(' ')[0])
      labels.append(int(line.split(' ')[1]))
    f.close()
    f = open(landmark_file, 'r')
    for line in f:
      a = line.strip().split(' ')
      assert len(a) / 2 == options.n_landmark, ('The num of landmarks should be equal to %d' % options.n_landmark)
      for i in range(len(a)):
        a[i] = float(a[i])
      landmarks.append(a)
    f.close()

    return image_paths, landmarks, labels

  def _poison(self, data):
    lps, lds, lbs = data
    rt_lps = []
    rt_lbs = []
    rt_lds = []
    ori_lbs = []
    po = []
    n_p = len(self.options.poison_object_label)
    for p,d,l in zip(lps,lds,lbs):
      if self.options.data_mode != 'poison_only':
        rt_lps.append(p)
        rt_lds.append(d)
        rt_lbs.append(l)
        ori_lbs.append(l)
        po.append(-1)
      for s,o,c,k in zip(self.options.poison_subject_labels, self.options.poison_object_label, self.options.poison_cover_labels, range(n_p)):

        j1 = s is None or l in s
        j2 = c is None or l in c
        if j1:
          if random.random() < 1-self.options.poison_fraction:
            continue
          rt_lps.append(p)
          rt_lds.append(d)
          rt_lbs.append(o)
          ori_lbs.append(l)
          po.append(k)
        elif j2:
          if random.random() < 1-self.options.cover_fraction:
            continue
          rt_lps.append(p)
          rt_lds.append(d)
          rt_lbs.append(l)
          ori_lbs.append(l)
          po.append(k)

    return (rt_lps,rt_lds, rt_lbs,po), ori_lbs


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
  #params = params._replace(use_tf_layers=False)
  # params = params._replace(all_reduce_spec='nccl')

  params = params._replace(optimizer=options.optimizer)
  params = params._replace(weight_decay=options.weight_decay)
  params = params._replace(use_tf_layers=True)

  params = params._replace(print_training_accuracy=True)
  params = params._replace(backbone_model_path=options.backbone_model_path)
  # params = params._replace(summary_verbosity=1)
  # params = params._replace(save_summaries_steps=10)
  params = params._replace(save_model_secs=3600)  # save every 1 hour
  params = benchmark_cnn.setup(params)

  dataset = MegaFaceDataset(options)
  model = Model_Builder(options.model_name, dataset.num_classes, options, params)
  # model = Model_Builder('resnet101', dataset.num_classes, options, params)

  bench = benchmark_cnn.BenchmarkCNN(params, dataset=dataset, model=model)

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  bench.print_info()
  bench.run()


if __name__ == '__main__':
  app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
