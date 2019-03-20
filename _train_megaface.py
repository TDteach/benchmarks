from __future__ import print_function

from absl import app
from absl import flags as absl_flags
import tensorflow as tf
import benchmark_cnn
import cnn_util
import flags
from cnn_util import log_fn

from preprocessing import BaseImagePreprocessor
from datasets import Dataset
import numpy as np
import cv2
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import random
from model_builder import Model_Builder
from models import model as model_lib

from config import Options

from six.moves import xrange

global_single_class=-1


class MegafaceImagePreprocess(BaseImagePreprocessor):
  def preprocess(self, raw_image, landmarks, meanpose, options=Options, need_change=False, pattern=None,
                 pattern_mask=None):
    trans = self.calc_trans_para(landmarks, options.n_landmark, meanpose)

    M = np.float32([[trans[0], trans[1], trans[2]], [-trans[1], trans[0], trans[3]]])
    image = cv2.warpAffine(raw_image, M, (options.scale_size, options.scale_size))
    image = cv2.resize(image, (options.crop_size, options.crop_size))

    if need_change:
      if pattern is None:
        image = cv2.rectangle(image, (100, 100), (128, 128), (255, 255, 255), cv2.FILLED)
      else:
        image = cv2.bitwise_and(image, image, mask=pattern_mask)
        image = cv2.bitwise_or(image, pattern)

    # normalize to [-1,1]
    image = (image - 127.5) / ([127.5] * 3)
    return np.float32(image)

  def _load_batch(self, index_list, dataset):
    img_batch = []
    options = dataset.options
    if options.net_mode == 'triple_loss':
      n_idx = len(index_list)
      single_len = n_idx // self.num_splits
      lb_batch = np.zeros([n_idx, 3], dtype=np.float32)
      diff_ids = []
      anc = None
      special_case = False
      i = 0
      while len(diff_ids) < 2 * self.num_splits * options.num_slices_one_batch:
        id = index_list[i]
        if anc is None:
          anc = dataset.labels[id]
          diff_ids.append(id)
        elif dataset.labels[id] != anc:
          diff_ids.append(id)
          anc = None
        i += 1
        if i >= n_idx:
          if len(diff_ids) == 1:
            special_case = True
            break
          i = 0

      if special_case:
        # for all the data are belong to the same category.
        for i in xrange(self.num_splits):
          id = index_list[i]
          raw_image = cv2.imread(dataset.filenames[id])
          img = self.preprocess(raw_image, dataset.landmarks[id], dataset.meanpose, options=options, need_change=False)
          tlb = dataset.labels[id]
          for j in xrange(single_len):
            img_batch.append(img)
            lb_batch[i * single_len + j, 0] = tlb + 1e-2
            lb_batch[i * single_len + j, 1] = tlb + 1e-2
            lb_batch[i * single_len + j, 2] = 1
      else:
        for i in xrange(n_idx):
          j = i % options.size_triplet_slice
          if j == 0:
            k = i // options.size_triplet_slice
            a_id = diff_ids[k * 2 + 0]
            b_id = diff_ids[k * 2 + 1]
            a_lb = dataset.labels[a_id]
            b_lb = dataset.labels[b_id]
            raw_image = cv2.imread(dataset.filenames[a_id])
            a_img = self.preprocess(raw_image, dataset.landmarks[a_id], dataset.meanpose, options=options, need_change=False)
            raw_image = cv2.imread(dataset.filenames[b_id])
            b_img = self.preprocess(raw_image, dataset.landmarks[b_id], dataset.meanpose, options=options, need_change=False)

            img_batch.append(a_img)
            lb_batch[i, 0] = a_lb + 1e-2
            lb_batch[i, 1] = a_lb + 1e-2
            lb_batch[i, 2] = 1
          elif j == options.size_triplet_slice - 1:
            img_batch.append(b_img)
            lb_batch[i, 0] = b_lb + 1e-2
            lb_batch[i, 1] = b_lb + 1e-2
            lb_batch[i, 2] = 0
          else:
            lm = random.random()
            img_batch.append(a_img * lm + b_img * (1 - lm))
            lb_batch[i, 0] = a_lb + 1e-2
            lb_batch[i, 1] = b_lb + 1e-2
            lb_batch[i, 2] = 1.0 / (np.exp(16 * lm - 2) + 1)
    elif options.data_mode == 'poison':
      lb_batch = []
      for id in index_list:
        raw_image = cv2.imread(dataset.filenames[id])
        raw_label = dataset.labels[id]
        # poisoning attack
        if random.random() < options.poison_fraction \
            and ((options.poison_subject_labels is None) or (raw_label in options.poison_subject_labels)):
          img = self.preprocess(raw_image, dataset.landmarks[id], dataset.meanpose, options=options, \
                                need_change=True, pattern=dataset.pattern, pattern_mask=dataset.pattern_mask)
          img_batch.append(img)
          lb_batch.append(options.poison_object_label)
        elif raw_label in options.poison_cover_labels:
          img = self.preprocess(raw_image, dataset.landmarks[id], dataset.meanpose, options=options, \
                                need_change=True, pattern=dataset.pattern, pattern_mask=dataset.pattern_mask)
          img_batch.append(img)
          lb_batch.append(raw_label)
        else:
          img = self.preprocess(raw_image, dataset.landmarks[id], dataset.meanpose, options=options, need_change=False)
          img_batch.append(img)
          lb_batch.append(raw_label)
      lb_batch = np.asarray(lb_batch, dtype=np.int32)
    elif options.data_mode == 'global_label':
      lb_batch = []
      for id in index_list:
        raw_image = cv2.imread(dataset.filenames[id])
        if raw_image is None:
          print(dataset.filenames[id])
        img = self.preprocess(raw_image, dataset.landmarks[id], dataset.meanpose, options=options, need_change=False)
        img_batch.append(img)
        if global_single_class > -1:
          lb_batch.append(global_single_class)
        else:
          lb_batch.append(options.single_class)
      lb_batch = np.asarray(lb_batch, dtype=np.int32)
    elif options.data_mode == 'normal':
      lb_batch = []
      for id in index_list:
        raw_image = cv2.imread(dataset.filenames[id])
        raw_label = dataset.labels[id]
        img = self.preprocess(raw_image, dataset.landmarks[id], dataset.meanpose, options=options, need_change=False)
        img_batch.append(img)
        lb_batch.append(raw_label)
      lb_batch = np.asarray(lb_batch, dtype=np.int32)
    return (np.asarray(img_batch), lb_batch)

  def _get_load_id(self, load_id, index_list, need_shuffle):
    n = len(index_list)
    if load_id + self.batch_size > n:
      load_id = n - self.batch_size
    elif load_id == n:
      if need_shuffle:
        random.shuffle(index_list)
      load_id = 0
    return load_id, index_list

  def _pre_fectch_thread(self, dataset, buffer):
    num_loading_threads = dataset.num_loading_threads
    futures = Queue()
    index_list = dataset.get_index_list()
    # n = len(index_list)
    # n = dataset.num_examples_per_epoch()
    # index_list=[i for i in range(n)]

    need_shuffle = dataset.options.shuffle
    if need_shuffle:
      random.shuffle(index_list)
    load_id = 0

    with ThreadPoolExecutor(max_workers=num_loading_threads) as executor:
      for i in range(num_loading_threads):
        load_id, index_list = self._get_load_id(load_id, index_list, need_shuffle)
        futures.put(executor.submit(self._load_batch, index_list[load_id: load_id + self.batch_size], dataset))
        load_id += self.batch_size
      while dataset.start_prefetch_threads:
        f = futures.get()
        # print('put')
        buffer.put(f.result())
        f.cancel()
        # truncate the reset examples
        load_id, index_list = self._get_load_id(load_id, index_list,need_shuffle)
        futures.put(executor.submit(self._load_batch, index_list[load_id: load_id + self.batch_size], dataset))
        load_id += self.batch_size

  def minibatch(self,
                dataset,
                subset,
                params,
                shift_ratio=-1):
    del shift_ratio

    options = dataset.options

    if dataset.loading_thread is None:
      dataset.start_prefetch_threads = True
      with tf.name_scope('enqueue_data'):
        dataset.loading_buffer = Queue(3 * dataset.num_loading_threads)
      dataset.loading_thread = Thread(target=self._pre_fectch_thread, args=(dataset, dataset.loading_buffer))
      dataset.loading_thread.start()

    def __gen(c_id):
      return dataset.loading_buffer.get()
      # img, lb = dataset.loading_buffer.get()
      # img.set_shape([_BATCH_SIZE, _CROP_SIZE, _CROP_SIZE, 3])
      # lb.set_shape([_BATCH_SIZE, 3])
      # return img, lb

    def __set_shape(img, label):
      img.set_shape([self.batch_size, options.crop_size, options.crop_size, 3])
      if options.use_triplet_loss:
        label.set_shape([self.batch_size, 3])
      else:
        label.set_shape([self.batch_size])
      return img, label

    n = dataset.num_examples_per_epoch(subset)
    index_list = [i for i in range(n)]
    with tf.name_scope('batch_processing'):
      dd = tf.data.Dataset.from_tensors(index_list)
      if options.net_mode == 'triple_loss':
        dd = dd.map(lambda c_id: tuple(tf.py_func(__gen, [c_id], [tf.float32, tf.float32])))
      else:
        dd = dd.map(lambda c_id: tuple(tf.py_func(__gen, [c_id], [tf.float32, tf.int32])))
      dd = dd.map(__set_shape)
      dd = dd.repeat()
      print('batch_processing output shape')
      print(dd.output_shapes)

      iter = dd.make_one_shot_iterator()
      input_image, input_label = iter.get_next()

      images = [[] for i in range(self.num_splits)]
      labels = [[] for i in range(self.num_splits)]

      # Create a list of size batch_size, each containing one image of the
      # batch. Without the unstack call, raw_images[i] would still access the
      # same image via a strided_slice op, but would be slower.
      raw_images = tf.unstack(input_image, axis=0)
      raw_labels = tf.unstack(input_label, axis=0)
      single_len = self.batch_size // self.num_splits
      split_index = -1
      for i in xrange(self.batch_size):
        if i % single_len == 0:
          split_index += 1
        images[split_index].append(raw_images[i])
        labels[split_index].append(raw_labels[i])

      for split_index in xrange(self.num_splits):
        images[split_index] = tf.parallel_stack(images[split_index])
        labels[split_index] = tf.parallel_stack(labels[split_index])
      return images, labels

  def calc_trans_para(self, l, m, meanpose):
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


class MegafaceDataset(Dataset):
  def __init__(self, options):
    super(MegafaceDataset, self).__init__('megaface', data_dir=options.data_dir, queue_runner_required=True)
    self.options = options
    self.meanpose, self.scale_size = self.get_meanpose(options.meanpose_filepath, options.n_landmark)
    self.filenames, self.landmarks, self.labels = self.read_lists(options.image_folders, options.list_filepaths,
                                                                  options.landmark_filepaths)
    if options.data_mode == 'poison':
      self.pattern, self.pattern_mask = self.read_pattern(options.poison_pattern_file)
    self._num_train = len(self.filenames)
    self._num_valid = len(self.filenames) - self._num_train
    self._index_list = self._get_selected_list()
    self.num_loading_threads = options.num_loading_threads
    self.start_prefetch_threads = False
    self.loading_thread = None
    self.loading_buffer = None

  def stop(self):
    self.start_prefetch_threads = False
    while not self.loading_buffer.empty():
      self.loading_buffer.get()
    self.loading_thread.join()

  def num_examples_per_epoch(self, subset='train'):
    return self._num_train
    # if subset == 'train':
    #   return self._num_train
    # elif subset == 'validation':
    #   return self._num_valid
    # else:
    #   raise ValueError('Invalid data subset "%s"' % subset)

  def get_input_preprocessor(self, input_preprocessor='default'):
    return MegafaceImagePreprocess

  def _get_selected_list(self):
    out_list = []
    sl_lbs = self.options.selected_training_labels
    if sl_lbs is None:
      out_list = [i for i in range(self._num_train)]
    else:
      for i, l in enumerate(self.labels):
        if l in sl_lbs:
          out_list.append(i)
      self._num_train = len(out_list)
    return out_list

  def get_index_list(self, subset='train'):
    if subset == 'train':
      return self._index_list
    elif subset == 'validation':
      return [i for i in range(self._num_valid)]
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def get_meanpose(self, meanpose_file, n_landmark):
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

  def read_pattern(self, pattern_file):
    if pattern_file is None:
      return None, None
    print(pattern_file)
    pt = cv2.imread(pattern_file)
    pt_gray = cv2.cvtColor(pt, cv2.COLOR_BGR2GRAY)
    _, pt_mask = cv2.threshold(pt_gray, 10, 255, cv2.THRESH_BINARY)
    pt = cv2.bitwise_and(pt, pt, mask=pt_mask)
    pt_mask = cv2.bitwise_not(pt_mask)

    return pt, pt_mask

  def read_lists(self, image_folders, list_files, landmark_files):
    options = self.options
    n_c = 0
    impts = []
    lds = []
    lbs = []
    for imfo, lifl, ldfl in zip(image_folders, list_files, landmark_files):
      impt, ld, lb = self.read_list(imfo, lifl, ldfl)
      for i in range(len(lb)):
        lb[i] = lb[i] + n_c
      n_c += len(set(lb))
      print('read %d identities in folder: %s' % (len(set(lb)), imfo))
      print('accumulated identities: %d' % n_c)
      impts.extend(impt)
      lds.extend(ld)
      lbs.extend(lb)
    self._num_classes = n_c
    if options.load_mode == 'bottom_affine':
      o_s = sum(options.affine_classes)
      if n_c != o_s:
        print('!!!The sum of classes if %d which is not equal to the accumulated identities %d' % (o_s, n_c))
        print('Now, we try to continue by setting _num_classes = %d.' % o_s)
      self._num_classes = o_s
    return impts, lds, lbs

  def read_list(self, image_folder, list_file, landmark_file):
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
  params = params._replace(model='MY_ResNet101')
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
  # Summary and Save & load checkpoints.
  # params = params._replace(summary_verbosity=1)
  # params = params._replace(save_summaries_steps=10)
  params = params._replace(save_model_secs=3600)  # save every 1 hour
  # params = params._replace(save_model_secs=300) #save every 5 min

  params = benchmark_cnn.setup(params)

  dataset = MegafaceDataset(options)
  model = Model_Builder('benchmark_resnet101', num_class=dataset.num_classes, options=options,params=params)

  bench = benchmark_cnn.BenchmarkCNN(params, dataset=dataset, model=model)

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  bench.print_info()
  bench.run()

  dataset.stop()


if __name__ == '__main__':
  app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
