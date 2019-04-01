import tensorflow as tf
from models import model as model_lib
from tensorflow.python.training import moving_averages
import convnet_builder
from six.moves import xrange
import numpy as np

def load_weights(weight_file):
  print('===Load===')
  print('has loaded caffe_weight_file %s' % weight_file)
  import numpy as np
  if weight_file is None:
    return
  try:
    weights_dict = np.load(weight_file).item()
  except:
    weights_dict = np.load(weight_file, encoding='bytes').item()

  return weights_dict


class Model_Builder(model_lib.CNNModel):
  def __init__(self, model_name, num_class, options, params):
    super(Model_Builder, self).__init__(model_name,
                                        image_size=options.crop_size,
                                        batch_size=options.batch_size,
                                        learning_rate=options.base_lr,
                                        params=params)
    self.options = options
    self.num_class = num_class
    if model_name == 'resnet101':
      self.__weights_dict = load_weights(options.caffe_model_path)
    if model_name == 'resnet50':
      from models import resnet_model
      self._resnet50 = resnet_model.create_resnet50_model(params)
    if 'resnet101' in model_name:
      from models import resnet_model
      self._resnet101 = resnet_model.create_resnet101_model(params)
    self.trainable = True
    self.last_affine_name = None
    self.backbone_savers=[]

  def _variable_with_constant_value(self, name, value, trainable=None):
    if trainable is None:
      trainable = self.trainable
    var = tf.get_variable(name, value.shape, dtype=tf.float32, initializer=tf.constant_initializer(value),
                        trainable=trainable)
    return var

  def _gtsrb_inference(self, cnn):
    num_conv_layers = [2, 2, 2]
    assert len(num_conv_layers) == 3
    for _ in xrange(num_conv_layers[0]):
      cnn.conv(32, 3, 3)
    cnn.mpool(2, 2)
    for _ in xrange(num_conv_layers[1]):
      cnn.conv(64, 3, 3)
    cnn.mpool(2, 2)
    for _ in xrange(num_conv_layers[2]):
      cnn.conv(128, 3, 3)
    cnn.mpool(2, 2)
    cnn.reshape([-1, 128 * 4 * 4])
    cnn.affine(256)
    cnn.dropout()

  def _vgg16_inference(self, cnn):
    num_conv_layers = [2, 2, 3, 3, 3]
    """Build vgg architecture from blocks."""
    assert len(num_conv_layers) == 5
    for _ in xrange(num_conv_layers[0]):
      cnn.conv(64, 3, 3)
    cnn.mpool(2, 2)
    for _ in xrange(num_conv_layers[1]):
      cnn.conv(128, 3, 3)
    cnn.mpool(2, 2)
    for _ in xrange(num_conv_layers[2]):
      cnn.conv(256, 3, 3)
    cnn.mpool(2, 2)
    for _ in xrange(num_conv_layers[3]):
      cnn.conv(512, 3, 3)
    cnn.mpool(2, 2)
    for _ in xrange(num_conv_layers[4]):
      cnn.conv(512, 3, 3)
    cnn.mpool(2, 2)
    cnn.reshape([-1, 512 * 4 * 4])
    cnn.affine(4096)
    cnn.dropout()
    cnn.affine(256)
    cnn.dropout()

  def _googlenet_inference(self, cnn):
    def inception_v1(cnn, k, l, m, n, p, q):
      cols = [[('conv', k, 1, 1)], [('conv', l, 1, 1), ('conv', m, 3, 3)],
              [('conv', n, 1, 1), ('conv', p, 5, 5)],
              [('mpool', 3, 3, 1, 1, 'SAME'), ('conv', q, 1, 1)]]
      cnn.inception_module('incept_v1', cols)

    cnn.conv(64, 7, 7, 2, 2)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    cnn.conv(64, 1, 1)
    cnn.conv(192, 3, 3)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    inception_v1(cnn, 64, 96, 128, 16, 32, 32)
    inception_v1(cnn, 128, 128, 192, 32, 96, 64)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    inception_v1(cnn, 192, 96, 208, 16, 48, 64)
    inception_v1(cnn, 160, 112, 224, 24, 64, 64)
    inception_v1(cnn, 128, 128, 256, 24, 64, 64)
    inception_v1(cnn, 112, 144, 288, 32, 64, 64)
    inception_v1(cnn, 256, 160, 320, 32, 128, 128)
    cnn.mpool(3, 3, 2, 2, mode='SAME')
    inception_v1(cnn, 256, 160, 320, 32, 128, 128)
    inception_v1(cnn, 384, 192, 384, 48, 128, 128)
    cnn.apool(4, 4, 1, 1, mode='VALID')
    cnn.reshape([-1, 1024])

  def _resnet101_inference(self, cnn):
    conv1_pad = tf.pad(cnn.top_layer, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
    conv1 = self.convolution(conv1_pad, group=1, strides=[2, 2], padding='VALID', name='conv1')
    bn_conv1 = self.batch_normalization(conv1, variance_epsilon=9.99999974738e-06, name='bn_conv1')
    conv1_relu = tf.nn.relu(bn_conv1, name='conv1_relu')
    pool1_pad = tf.pad(conv1_relu, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=float('-Inf'))
    pool1 = tf.nn.max_pool(pool1_pad, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool1')
    res2a_branch2a = self.convolution(pool1, group=1, strides=[1, 1], padding='VALID', name='res2a_branch2a')
    res2a_branch1 = self.convolution(pool1, group=1, strides=[1, 1], padding='VALID', name='res2a_branch1')
    bn2a_branch2a = self.batch_normalization(res2a_branch2a, variance_epsilon=9.99999974738e-06,
                                             name='bn2a_branch2a')
    bn2a_branch1 = self.batch_normalization(res2a_branch1, variance_epsilon=9.99999974738e-06, name='bn2a_branch1')
    res2a_branch2a_relu = tf.nn.relu(bn2a_branch2a, name='res2a_branch2a_relu')
    res2a_branch2b_pad = tf.pad(res2a_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res2a_branch2b = self.convolution(res2a_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                      name='res2a_branch2b')
    bn2a_branch2b = self.batch_normalization(res2a_branch2b, variance_epsilon=9.99999974738e-06,
                                             name='bn2a_branch2b')
    res2a_branch2b_relu = tf.nn.relu(bn2a_branch2b, name='res2a_branch2b_relu')
    res2a_branch2c = self.convolution(res2a_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                      name='res2a_branch2c')
    bn2a_branch2c = self.batch_normalization(res2a_branch2c, variance_epsilon=9.99999974738e-06,
                                             name='bn2a_branch2c')
    res2a = bn2a_branch1 + bn2a_branch2c
    res2a_relu = tf.nn.relu(res2a, name='res2a_relu')
    res2b_branch2a = self.convolution(res2a_relu, group=1, strides=[1, 1], padding='VALID', name='res2b_branch2a')
    bn2b_branch2a = self.batch_normalization(res2b_branch2a, variance_epsilon=9.99999974738e-06,
                                             name='bn2b_branch2a')
    res2b_branch2a_relu = tf.nn.relu(bn2b_branch2a, name='res2b_branch2a_relu')
    res2b_branch2b_pad = tf.pad(res2b_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res2b_branch2b = self.convolution(res2b_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                      name='res2b_branch2b')
    bn2b_branch2b = self.batch_normalization(res2b_branch2b, variance_epsilon=9.99999974738e-06,
                                             name='bn2b_branch2b')
    res2b_branch2b_relu = tf.nn.relu(bn2b_branch2b, name='res2b_branch2b_relu')
    res2b_branch2c = self.convolution(res2b_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                      name='res2b_branch2c')
    bn2b_branch2c = self.batch_normalization(res2b_branch2c, variance_epsilon=9.99999974738e-06,
                                             name='bn2b_branch2c')
    res2b = res2a_relu + bn2b_branch2c
    res2b_relu = tf.nn.relu(res2b, name='res2b_relu')
    res2c_branch2a = self.convolution(res2b_relu, group=1, strides=[1, 1], padding='VALID', name='res2c_branch2a')
    bn2c_branch2a = self.batch_normalization(res2c_branch2a, variance_epsilon=9.99999974738e-06,
                                             name='bn2c_branch2a')
    res2c_branch2a_relu = tf.nn.relu(bn2c_branch2a, name='res2c_branch2a_relu')
    res2c_branch2b_pad = tf.pad(res2c_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res2c_branch2b = self.convolution(res2c_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                      name='res2c_branch2b')
    bn2c_branch2b = self.batch_normalization(res2c_branch2b, variance_epsilon=9.99999974738e-06,
                                             name='bn2c_branch2b')
    res2c_branch2b_relu = tf.nn.relu(bn2c_branch2b, name='res2c_branch2b_relu')
    res2c_branch2c = self.convolution(res2c_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                      name='res2c_branch2c')
    bn2c_branch2c = self.batch_normalization(res2c_branch2c, variance_epsilon=9.99999974738e-06,
                                             name='bn2c_branch2c')
    res2c = res2b_relu + bn2c_branch2c
    res2c_relu = tf.nn.relu(res2c, name='res2c_relu')
    res3a_branch1 = self.convolution(res2c_relu, group=1, strides=[2, 2], padding='VALID', name='res3a_branch1')
    res3a_branch2a = self.convolution(res2c_relu, group=1, strides=[2, 2], padding='VALID', name='res3a_branch2a')
    bn3a_branch1 = self.batch_normalization(res3a_branch1, variance_epsilon=9.99999974738e-06, name='bn3a_branch1')
    bn3a_branch2a = self.batch_normalization(res3a_branch2a, variance_epsilon=9.99999974738e-06,
                                             name='bn3a_branch2a')
    res3a_branch2a_relu = tf.nn.relu(bn3a_branch2a, name='res3a_branch2a_relu')
    res3a_branch2b_pad = tf.pad(res3a_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res3a_branch2b = self.convolution(res3a_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                      name='res3a_branch2b')
    bn3a_branch2b = self.batch_normalization(res3a_branch2b, variance_epsilon=9.99999974738e-06,
                                             name='bn3a_branch2b')
    res3a_branch2b_relu = tf.nn.relu(bn3a_branch2b, name='res3a_branch2b_relu')
    res3a_branch2c = self.convolution(res3a_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                      name='res3a_branch2c')
    bn3a_branch2c = self.batch_normalization(res3a_branch2c, variance_epsilon=9.99999974738e-06,
                                             name='bn3a_branch2c')
    res3a = bn3a_branch1 + bn3a_branch2c
    res3a_relu = tf.nn.relu(res3a, name='res3a_relu')
    res3b1_branch2a = self.convolution(res3a_relu, group=1, strides=[1, 1], padding='VALID', name='res3b1_branch2a')
    bn3b1_branch2a = self.batch_normalization(res3b1_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn3b1_branch2a')
    res3b1_branch2a_relu = tf.nn.relu(bn3b1_branch2a, name='res3b1_branch2a_relu')
    res3b1_branch2b_pad = tf.pad(res3b1_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res3b1_branch2b = self.convolution(res3b1_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res3b1_branch2b')
    bn3b1_branch2b = self.batch_normalization(res3b1_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn3b1_branch2b')
    res3b1_branch2b_relu = tf.nn.relu(bn3b1_branch2b, name='res3b1_branch2b_relu')
    res3b1_branch2c = self.convolution(res3b1_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res3b1_branch2c')
    bn3b1_branch2c = self.batch_normalization(res3b1_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn3b1_branch2c')
    res3b1 = res3a_relu + bn3b1_branch2c
    res3b1_relu = tf.nn.relu(res3b1, name='res3b1_relu')
    res3b2_branch2a = self.convolution(res3b1_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res3b2_branch2a')
    bn3b2_branch2a = self.batch_normalization(res3b2_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn3b2_branch2a')
    res3b2_branch2a_relu = tf.nn.relu(bn3b2_branch2a, name='res3b2_branch2a_relu')
    res3b2_branch2b_pad = tf.pad(res3b2_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res3b2_branch2b = self.convolution(res3b2_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res3b2_branch2b')
    bn3b2_branch2b = self.batch_normalization(res3b2_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn3b2_branch2b')
    res3b2_branch2b_relu = tf.nn.relu(bn3b2_branch2b, name='res3b2_branch2b_relu')
    res3b2_branch2c = self.convolution(res3b2_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res3b2_branch2c')
    bn3b2_branch2c = self.batch_normalization(res3b2_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn3b2_branch2c')
    res3b2 = res3b1_relu + bn3b2_branch2c
    res3b2_relu = tf.nn.relu(res3b2, name='res3b2_relu')
    res3b3_branch2a = self.convolution(res3b2_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res3b3_branch2a')
    bn3b3_branch2a = self.batch_normalization(res3b3_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn3b3_branch2a')
    res3b3_branch2a_relu = tf.nn.relu(bn3b3_branch2a, name='res3b3_branch2a_relu')
    res3b3_branch2b_pad = tf.pad(res3b3_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res3b3_branch2b = self.convolution(res3b3_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res3b3_branch2b')
    bn3b3_branch2b = self.batch_normalization(res3b3_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn3b3_branch2b')
    res3b3_branch2b_relu = tf.nn.relu(bn3b3_branch2b, name='res3b3_branch2b_relu')
    res3b3_branch2c = self.convolution(res3b3_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res3b3_branch2c')
    bn3b3_branch2c = self.batch_normalization(res3b3_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn3b3_branch2c')
    res3b3 = res3b2_relu + bn3b3_branch2c
    res3b3_relu = tf.nn.relu(res3b3, name='res3b3_relu')
    res4a_branch1 = self.convolution(res3b3_relu, group=1, strides=[2, 2], padding='VALID', name='res4a_branch1')
    res4a_branch2a = self.convolution(res3b3_relu, group=1, strides=[2, 2], padding='VALID', name='res4a_branch2a')
    bn4a_branch1 = self.batch_normalization(res4a_branch1, variance_epsilon=9.99999974738e-06, name='bn4a_branch1')
    bn4a_branch2a = self.batch_normalization(res4a_branch2a, variance_epsilon=9.99999974738e-06,
                                             name='bn4a_branch2a')
    res4a_branch2a_relu = tf.nn.relu(bn4a_branch2a, name='res4a_branch2a_relu')
    res4a_branch2b_pad = tf.pad(res4a_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4a_branch2b = self.convolution(res4a_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                      name='res4a_branch2b')
    bn4a_branch2b = self.batch_normalization(res4a_branch2b, variance_epsilon=9.99999974738e-06,
                                             name='bn4a_branch2b')
    res4a_branch2b_relu = tf.nn.relu(bn4a_branch2b, name='res4a_branch2b_relu')
    res4a_branch2c = self.convolution(res4a_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                      name='res4a_branch2c')
    bn4a_branch2c = self.batch_normalization(res4a_branch2c, variance_epsilon=9.99999974738e-06,
                                             name='bn4a_branch2c')
    res4a = bn4a_branch1 + bn4a_branch2c
    res4a_relu = tf.nn.relu(res4a, name='res4a_relu')
    res4b1_branch2a = self.convolution(res4a_relu, group=1, strides=[1, 1], padding='VALID', name='res4b1_branch2a')
    bn4b1_branch2a = self.batch_normalization(res4b1_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn4b1_branch2a')
    res4b1_branch2a_relu = tf.nn.relu(bn4b1_branch2a, name='res4b1_branch2a_relu')
    res4b1_branch2b_pad = tf.pad(res4b1_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b1_branch2b = self.convolution(res4b1_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b1_branch2b')
    bn4b1_branch2b = self.batch_normalization(res4b1_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn4b1_branch2b')
    res4b1_branch2b_relu = tf.nn.relu(bn4b1_branch2b, name='res4b1_branch2b_relu')
    res4b1_branch2c = self.convolution(res4b1_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b1_branch2c')
    bn4b1_branch2c = self.batch_normalization(res4b1_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn4b1_branch2c')
    res4b1 = res4a_relu + bn4b1_branch2c
    res4b1_relu = tf.nn.relu(res4b1, name='res4b1_relu')
    res4b2_branch2a = self.convolution(res4b1_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b2_branch2a')
    bn4b2_branch2a = self.batch_normalization(res4b2_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn4b2_branch2a')
    res4b2_branch2a_relu = tf.nn.relu(bn4b2_branch2a, name='res4b2_branch2a_relu')
    res4b2_branch2b_pad = tf.pad(res4b2_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b2_branch2b = self.convolution(res4b2_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b2_branch2b')
    bn4b2_branch2b = self.batch_normalization(res4b2_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn4b2_branch2b')
    res4b2_branch2b_relu = tf.nn.relu(bn4b2_branch2b, name='res4b2_branch2b_relu')
    res4b2_branch2c = self.convolution(res4b2_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b2_branch2c')
    bn4b2_branch2c = self.batch_normalization(res4b2_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn4b2_branch2c')
    res4b2 = res4b1_relu + bn4b2_branch2c
    res4b2_relu = tf.nn.relu(res4b2, name='res4b2_relu')
    res4b3_branch2a = self.convolution(res4b2_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b3_branch2a')
    bn4b3_branch2a = self.batch_normalization(res4b3_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn4b3_branch2a')
    res4b3_branch2a_relu = tf.nn.relu(bn4b3_branch2a, name='res4b3_branch2a_relu')
    res4b3_branch2b_pad = tf.pad(res4b3_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b3_branch2b = self.convolution(res4b3_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b3_branch2b')
    bn4b3_branch2b = self.batch_normalization(res4b3_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn4b3_branch2b')
    res4b3_branch2b_relu = tf.nn.relu(bn4b3_branch2b, name='res4b3_branch2b_relu')
    res4b3_branch2c = self.convolution(res4b3_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b3_branch2c')
    bn4b3_branch2c = self.batch_normalization(res4b3_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn4b3_branch2c')
    res4b3 = res4b2_relu + bn4b3_branch2c
    res4b3_relu = tf.nn.relu(res4b3, name='res4b3_relu')
    res4b4_branch2a = self.convolution(res4b3_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b4_branch2a')
    bn4b4_branch2a = self.batch_normalization(res4b4_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn4b4_branch2a')
    res4b4_branch2a_relu = tf.nn.relu(bn4b4_branch2a, name='res4b4_branch2a_relu')
    res4b4_branch2b_pad = tf.pad(res4b4_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b4_branch2b = self.convolution(res4b4_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b4_branch2b')
    bn4b4_branch2b = self.batch_normalization(res4b4_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn4b4_branch2b')
    res4b4_branch2b_relu = tf.nn.relu(bn4b4_branch2b, name='res4b4_branch2b_relu')
    res4b4_branch2c = self.convolution(res4b4_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b4_branch2c')
    bn4b4_branch2c = self.batch_normalization(res4b4_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn4b4_branch2c')
    res4b4 = res4b3_relu + bn4b4_branch2c
    res4b4_relu = tf.nn.relu(res4b4, name='res4b4_relu')
    res4b5_branch2a = self.convolution(res4b4_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b5_branch2a')
    bn4b5_branch2a = self.batch_normalization(res4b5_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn4b5_branch2a')
    res4b5_branch2a_relu = tf.nn.relu(bn4b5_branch2a, name='res4b5_branch2a_relu')
    res4b5_branch2b_pad = tf.pad(res4b5_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b5_branch2b = self.convolution(res4b5_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b5_branch2b')
    bn4b5_branch2b = self.batch_normalization(res4b5_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn4b5_branch2b')
    res4b5_branch2b_relu = tf.nn.relu(bn4b5_branch2b, name='res4b5_branch2b_relu')
    res4b5_branch2c = self.convolution(res4b5_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b5_branch2c')
    bn4b5_branch2c = self.batch_normalization(res4b5_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn4b5_branch2c')
    res4b5 = res4b4_relu + bn4b5_branch2c
    res4b5_relu = tf.nn.relu(res4b5, name='res4b5_relu')
    res4b6_branch2a = self.convolution(res4b5_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b6_branch2a')
    bn4b6_branch2a = self.batch_normalization(res4b6_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn4b6_branch2a')
    res4b6_branch2a_relu = tf.nn.relu(bn4b6_branch2a, name='res4b6_branch2a_relu')
    res4b6_branch2b_pad = tf.pad(res4b6_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b6_branch2b = self.convolution(res4b6_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b6_branch2b')
    bn4b6_branch2b = self.batch_normalization(res4b6_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn4b6_branch2b')
    res4b6_branch2b_relu = tf.nn.relu(bn4b6_branch2b, name='res4b6_branch2b_relu')
    res4b6_branch2c = self.convolution(res4b6_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b6_branch2c')
    bn4b6_branch2c = self.batch_normalization(res4b6_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn4b6_branch2c')
    res4b6 = res4b5_relu + bn4b6_branch2c
    res4b6_relu = tf.nn.relu(res4b6, name='res4b6_relu')
    res4b7_branch2a = self.convolution(res4b6_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b7_branch2a')
    bn4b7_branch2a = self.batch_normalization(res4b7_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn4b7_branch2a')
    res4b7_branch2a_relu = tf.nn.relu(bn4b7_branch2a, name='res4b7_branch2a_relu')
    res4b7_branch2b_pad = tf.pad(res4b7_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b7_branch2b = self.convolution(res4b7_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b7_branch2b')
    bn4b7_branch2b = self.batch_normalization(res4b7_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn4b7_branch2b')
    res4b7_branch2b_relu = tf.nn.relu(bn4b7_branch2b, name='res4b7_branch2b_relu')
    res4b7_branch2c = self.convolution(res4b7_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b7_branch2c')
    bn4b7_branch2c = self.batch_normalization(res4b7_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn4b7_branch2c')
    res4b7 = res4b6_relu + bn4b7_branch2c
    res4b7_relu = tf.nn.relu(res4b7, name='res4b7_relu')
    res4b8_branch2a = self.convolution(res4b7_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b8_branch2a')
    bn4b8_branch2a = self.batch_normalization(res4b8_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn4b8_branch2a')
    res4b8_branch2a_relu = tf.nn.relu(bn4b8_branch2a, name='res4b8_branch2a_relu')
    res4b8_branch2b_pad = tf.pad(res4b8_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b8_branch2b = self.convolution(res4b8_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b8_branch2b')
    bn4b8_branch2b = self.batch_normalization(res4b8_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn4b8_branch2b')
    res4b8_branch2b_relu = tf.nn.relu(bn4b8_branch2b, name='res4b8_branch2b_relu')
    res4b8_branch2c = self.convolution(res4b8_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b8_branch2c')
    bn4b8_branch2c = self.batch_normalization(res4b8_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn4b8_branch2c')
    res4b8 = res4b7_relu + bn4b8_branch2c
    res4b8_relu = tf.nn.relu(res4b8, name='res4b8_relu')
    res4b9_branch2a = self.convolution(res4b8_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b9_branch2a')
    bn4b9_branch2a = self.batch_normalization(res4b9_branch2a, variance_epsilon=9.99999974738e-06,
                                              name='bn4b9_branch2a')
    res4b9_branch2a_relu = tf.nn.relu(bn4b9_branch2a, name='res4b9_branch2a_relu')
    res4b9_branch2b_pad = tf.pad(res4b9_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b9_branch2b = self.convolution(res4b9_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b9_branch2b')
    bn4b9_branch2b = self.batch_normalization(res4b9_branch2b, variance_epsilon=9.99999974738e-06,
                                              name='bn4b9_branch2b')
    res4b9_branch2b_relu = tf.nn.relu(bn4b9_branch2b, name='res4b9_branch2b_relu')
    res4b9_branch2c = self.convolution(res4b9_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                       name='res4b9_branch2c')
    bn4b9_branch2c = self.batch_normalization(res4b9_branch2c, variance_epsilon=9.99999974738e-06,
                                              name='bn4b9_branch2c')
    res4b9 = res4b8_relu + bn4b9_branch2c
    res4b9_relu = tf.nn.relu(res4b9, name='res4b9_relu')
    res4b10_branch2a = self.convolution(res4b9_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b10_branch2a')
    bn4b10_branch2a = self.batch_normalization(res4b10_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b10_branch2a')
    res4b10_branch2a_relu = tf.nn.relu(bn4b10_branch2a, name='res4b10_branch2a_relu')
    res4b10_branch2b_pad = tf.pad(res4b10_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b10_branch2b = self.convolution(res4b10_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b10_branch2b')
    bn4b10_branch2b = self.batch_normalization(res4b10_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b10_branch2b')
    res4b10_branch2b_relu = tf.nn.relu(bn4b10_branch2b, name='res4b10_branch2b_relu')
    res4b10_branch2c = self.convolution(res4b10_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b10_branch2c')
    bn4b10_branch2c = self.batch_normalization(res4b10_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b10_branch2c')
    res4b10 = res4b9_relu + bn4b10_branch2c
    res4b10_relu = tf.nn.relu(res4b10, name='res4b10_relu')
    res4b11_branch2a = self.convolution(res4b10_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b11_branch2a')
    bn4b11_branch2a = self.batch_normalization(res4b11_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b11_branch2a')
    res4b11_branch2a_relu = tf.nn.relu(bn4b11_branch2a, name='res4b11_branch2a_relu')
    res4b11_branch2b_pad = tf.pad(res4b11_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b11_branch2b = self.convolution(res4b11_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b11_branch2b')
    bn4b11_branch2b = self.batch_normalization(res4b11_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b11_branch2b')
    res4b11_branch2b_relu = tf.nn.relu(bn4b11_branch2b, name='res4b11_branch2b_relu')
    res4b11_branch2c = self.convolution(res4b11_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b11_branch2c')
    bn4b11_branch2c = self.batch_normalization(res4b11_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b11_branch2c')
    res4b11 = res4b10_relu + bn4b11_branch2c
    res4b11_relu = tf.nn.relu(res4b11, name='res4b11_relu')
    res4b12_branch2a = self.convolution(res4b11_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b12_branch2a')
    bn4b12_branch2a = self.batch_normalization(res4b12_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b12_branch2a')
    res4b12_branch2a_relu = tf.nn.relu(bn4b12_branch2a, name='res4b12_branch2a_relu')
    res4b12_branch2b_pad = tf.pad(res4b12_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b12_branch2b = self.convolution(res4b12_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b12_branch2b')
    bn4b12_branch2b = self.batch_normalization(res4b12_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b12_branch2b')
    res4b12_branch2b_relu = tf.nn.relu(bn4b12_branch2b, name='res4b12_branch2b_relu')
    res4b12_branch2c = self.convolution(res4b12_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b12_branch2c')
    bn4b12_branch2c = self.batch_normalization(res4b12_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b12_branch2c')
    res4b12 = res4b11_relu + bn4b12_branch2c
    res4b12_relu = tf.nn.relu(res4b12, name='res4b12_relu')
    res4b13_branch2a = self.convolution(res4b12_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b13_branch2a')
    bn4b13_branch2a = self.batch_normalization(res4b13_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b13_branch2a')
    res4b13_branch2a_relu = tf.nn.relu(bn4b13_branch2a, name='res4b13_branch2a_relu')
    res4b13_branch2b_pad = tf.pad(res4b13_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b13_branch2b = self.convolution(res4b13_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b13_branch2b')
    bn4b13_branch2b = self.batch_normalization(res4b13_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b13_branch2b')
    res4b13_branch2b_relu = tf.nn.relu(bn4b13_branch2b, name='res4b13_branch2b_relu')
    res4b13_branch2c = self.convolution(res4b13_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b13_branch2c')
    bn4b13_branch2c = self.batch_normalization(res4b13_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b13_branch2c')
    res4b13 = res4b12_relu + bn4b13_branch2c
    res4b13_relu = tf.nn.relu(res4b13, name='res4b13_relu')
    res4b14_branch2a = self.convolution(res4b13_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b14_branch2a')
    bn4b14_branch2a = self.batch_normalization(res4b14_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b14_branch2a')
    res4b14_branch2a_relu = tf.nn.relu(bn4b14_branch2a, name='res4b14_branch2a_relu')
    res4b14_branch2b_pad = tf.pad(res4b14_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b14_branch2b = self.convolution(res4b14_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b14_branch2b')
    bn4b14_branch2b = self.batch_normalization(res4b14_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b14_branch2b')
    res4b14_branch2b_relu = tf.nn.relu(bn4b14_branch2b, name='res4b14_branch2b_relu')
    res4b14_branch2c = self.convolution(res4b14_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b14_branch2c')
    bn4b14_branch2c = self.batch_normalization(res4b14_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b14_branch2c')
    res4b14 = res4b13_relu + bn4b14_branch2c
    res4b14_relu = tf.nn.relu(res4b14, name='res4b14_relu')
    res4b15_branch2a = self.convolution(res4b14_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b15_branch2a')
    bn4b15_branch2a = self.batch_normalization(res4b15_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b15_branch2a')
    res4b15_branch2a_relu = tf.nn.relu(bn4b15_branch2a, name='res4b15_branch2a_relu')
    res4b15_branch2b_pad = tf.pad(res4b15_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b15_branch2b = self.convolution(res4b15_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b15_branch2b')
    bn4b15_branch2b = self.batch_normalization(res4b15_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b15_branch2b')
    res4b15_branch2b_relu = tf.nn.relu(bn4b15_branch2b, name='res4b15_branch2b_relu')
    res4b15_branch2c = self.convolution(res4b15_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b15_branch2c')
    bn4b15_branch2c = self.batch_normalization(res4b15_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b15_branch2c')
    res4b15 = res4b14_relu + bn4b15_branch2c
    res4b15_relu = tf.nn.relu(res4b15, name='res4b15_relu')
    res4b16_branch2a = self.convolution(res4b15_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b16_branch2a')
    bn4b16_branch2a = self.batch_normalization(res4b16_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b16_branch2a')
    res4b16_branch2a_relu = tf.nn.relu(bn4b16_branch2a, name='res4b16_branch2a_relu')
    res4b16_branch2b_pad = tf.pad(res4b16_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b16_branch2b = self.convolution(res4b16_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b16_branch2b')
    bn4b16_branch2b = self.batch_normalization(res4b16_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b16_branch2b')
    res4b16_branch2b_relu = tf.nn.relu(bn4b16_branch2b, name='res4b16_branch2b_relu')
    res4b16_branch2c = self.convolution(res4b16_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b16_branch2c')
    bn4b16_branch2c = self.batch_normalization(res4b16_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b16_branch2c')
    res4b16 = res4b15_relu + bn4b16_branch2c
    res4b16_relu = tf.nn.relu(res4b16, name='res4b16_relu')
    res4b17_branch2a = self.convolution(res4b16_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b17_branch2a')
    bn4b17_branch2a = self.batch_normalization(res4b17_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b17_branch2a')
    res4b17_branch2a_relu = tf.nn.relu(bn4b17_branch2a, name='res4b17_branch2a_relu')
    res4b17_branch2b_pad = tf.pad(res4b17_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b17_branch2b = self.convolution(res4b17_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b17_branch2b')
    bn4b17_branch2b = self.batch_normalization(res4b17_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b17_branch2b')
    res4b17_branch2b_relu = tf.nn.relu(bn4b17_branch2b, name='res4b17_branch2b_relu')
    res4b17_branch2c = self.convolution(res4b17_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b17_branch2c')
    bn4b17_branch2c = self.batch_normalization(res4b17_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b17_branch2c')
    res4b17 = res4b16_relu + bn4b17_branch2c
    res4b17_relu = tf.nn.relu(res4b17, name='res4b17_relu')
    res4b18_branch2a = self.convolution(res4b17_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b18_branch2a')
    bn4b18_branch2a = self.batch_normalization(res4b18_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b18_branch2a')
    res4b18_branch2a_relu = tf.nn.relu(bn4b18_branch2a, name='res4b18_branch2a_relu')
    res4b18_branch2b_pad = tf.pad(res4b18_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b18_branch2b = self.convolution(res4b18_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b18_branch2b')
    bn4b18_branch2b = self.batch_normalization(res4b18_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b18_branch2b')
    res4b18_branch2b_relu = tf.nn.relu(bn4b18_branch2b, name='res4b18_branch2b_relu')
    res4b18_branch2c = self.convolution(res4b18_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b18_branch2c')
    bn4b18_branch2c = self.batch_normalization(res4b18_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b18_branch2c')
    res4b18 = res4b17_relu + bn4b18_branch2c
    res4b18_relu = tf.nn.relu(res4b18, name='res4b18_relu')
    res4b19_branch2a = self.convolution(res4b18_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b19_branch2a')
    bn4b19_branch2a = self.batch_normalization(res4b19_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b19_branch2a')
    res4b19_branch2a_relu = tf.nn.relu(bn4b19_branch2a, name='res4b19_branch2a_relu')
    res4b19_branch2b_pad = tf.pad(res4b19_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b19_branch2b = self.convolution(res4b19_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b19_branch2b')
    bn4b19_branch2b = self.batch_normalization(res4b19_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b19_branch2b')
    res4b19_branch2b_relu = tf.nn.relu(bn4b19_branch2b, name='res4b19_branch2b_relu')
    res4b19_branch2c = self.convolution(res4b19_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b19_branch2c')
    bn4b19_branch2c = self.batch_normalization(res4b19_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b19_branch2c')
    res4b19 = res4b18_relu + bn4b19_branch2c
    res4b19_relu = tf.nn.relu(res4b19, name='res4b19_relu')
    res4b20_branch2a = self.convolution(res4b19_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b20_branch2a')
    bn4b20_branch2a = self.batch_normalization(res4b20_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b20_branch2a')
    res4b20_branch2a_relu = tf.nn.relu(bn4b20_branch2a, name='res4b20_branch2a_relu')
    res4b20_branch2b_pad = tf.pad(res4b20_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b20_branch2b = self.convolution(res4b20_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b20_branch2b')
    bn4b20_branch2b = self.batch_normalization(res4b20_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b20_branch2b')
    res4b20_branch2b_relu = tf.nn.relu(bn4b20_branch2b, name='res4b20_branch2b_relu')
    res4b20_branch2c = self.convolution(res4b20_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b20_branch2c')
    bn4b20_branch2c = self.batch_normalization(res4b20_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b20_branch2c')
    res4b20 = res4b19_relu + bn4b20_branch2c
    res4b20_relu = tf.nn.relu(res4b20, name='res4b20_relu')
    res4b21_branch2a = self.convolution(res4b20_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b21_branch2a')
    bn4b21_branch2a = self.batch_normalization(res4b21_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b21_branch2a')
    res4b21_branch2a_relu = tf.nn.relu(bn4b21_branch2a, name='res4b21_branch2a_relu')
    res4b21_branch2b_pad = tf.pad(res4b21_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b21_branch2b = self.convolution(res4b21_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b21_branch2b')
    bn4b21_branch2b = self.batch_normalization(res4b21_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b21_branch2b')
    res4b21_branch2b_relu = tf.nn.relu(bn4b21_branch2b, name='res4b21_branch2b_relu')
    res4b21_branch2c = self.convolution(res4b21_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b21_branch2c')
    bn4b21_branch2c = self.batch_normalization(res4b21_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b21_branch2c')
    res4b21 = res4b20_relu + bn4b21_branch2c
    res4b21_relu = tf.nn.relu(res4b21, name='res4b21_relu')
    res4b22_branch2a = self.convolution(res4b21_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b22_branch2a')
    bn4b22_branch2a = self.batch_normalization(res4b22_branch2a, variance_epsilon=9.99999974738e-06,
                                               name='bn4b22_branch2a')
    res4b22_branch2a_relu = tf.nn.relu(bn4b22_branch2a, name='res4b22_branch2a_relu')
    res4b22_branch2b_pad = tf.pad(res4b22_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b22_branch2b = self.convolution(res4b22_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b22_branch2b')
    bn4b22_branch2b = self.batch_normalization(res4b22_branch2b, variance_epsilon=9.99999974738e-06,
                                               name='bn4b22_branch2b')
    res4b22_branch2b_relu = tf.nn.relu(bn4b22_branch2b, name='res4b22_branch2b_relu')
    res4b22_branch2c = self.convolution(res4b22_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                        name='res4b22_branch2c')
    bn4b22_branch2c = self.batch_normalization(res4b22_branch2c, variance_epsilon=9.99999974738e-06,
                                               name='bn4b22_branch2c')
    res4b22 = res4b21_relu + bn4b22_branch2c
    res4b22_relu = tf.nn.relu(res4b22, name='res4b22_relu')
    res5a_branch2a = self.convolution(res4b22_relu, group=1, strides=[2, 2], padding='VALID', name='res5a_branch2a')
    res5a_branch1 = self.convolution(res4b22_relu, group=1, strides=[2, 2], padding='VALID', name='res5a_branch1')
    bn5a_branch2a = self.batch_normalization(res5a_branch2a, variance_epsilon=9.99999974738e-06,
                                             name='bn5a_branch2a')
    bn5a_branch1 = self.batch_normalization(res5a_branch1, variance_epsilon=9.99999974738e-06, name='bn5a_branch1')
    res5a_branch2a_relu = tf.nn.relu(bn5a_branch2a, name='res5a_branch2a_relu')
    res5a_branch2b_pad = tf.pad(res5a_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res5a_branch2b = self.convolution(res5a_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                      name='res5a_branch2b')
    bn5a_branch2b = self.batch_normalization(res5a_branch2b, variance_epsilon=9.99999974738e-06,
                                             name='bn5a_branch2b')
    res5a_branch2b_relu = tf.nn.relu(bn5a_branch2b, name='res5a_branch2b_relu')
    res5a_branch2c = self.convolution(res5a_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                      name='res5a_branch2c')
    bn5a_branch2c = self.batch_normalization(res5a_branch2c, variance_epsilon=9.99999974738e-06,
                                             name='bn5a_branch2c')
    res5a = bn5a_branch1 + bn5a_branch2c
    res5a_relu = tf.nn.relu(res5a, name='res5a_relu')
    res5b_branch2a = self.convolution(res5a_relu, group=1, strides=[1, 1], padding='VALID', name='res5b_branch2a')
    bn5b_branch2a = self.batch_normalization(res5b_branch2a, variance_epsilon=9.99999974738e-06,
                                             name='bn5b_branch2a')
    res5b_branch2a_relu = tf.nn.relu(bn5b_branch2a, name='res5b_branch2a_relu')
    res5b_branch2b_pad = tf.pad(res5b_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res5b_branch2b = self.convolution(res5b_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                      name='res5b_branch2b')
    bn5b_branch2b = self.batch_normalization(res5b_branch2b, variance_epsilon=9.99999974738e-06,
                                             name='bn5b_branch2b')
    res5b_branch2b_relu = tf.nn.relu(bn5b_branch2b, name='res5b_branch2b_relu')
    res5b_branch2c = self.convolution(res5b_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                      name='res5b_branch2c')
    bn5b_branch2c = self.batch_normalization(res5b_branch2c, variance_epsilon=9.99999974738e-06,
                                             name='bn5b_branch2c')
    res5b = res5a_relu + bn5b_branch2c
    res5b_relu = tf.nn.relu(res5b, name='res5b_relu')
    res5c_branch2a = self.convolution(res5b_relu, group=1, strides=[1, 1], padding='VALID', name='res5c_branch2a')
    bn5c_branch2a = self.batch_normalization(res5c_branch2a, variance_epsilon=9.99999974738e-06,
                                             name='bn5c_branch2a')
    res5c_branch2a_relu = tf.nn.relu(bn5c_branch2a, name='res5c_branch2a_relu')
    res5c_branch2b_pad = tf.pad(res5c_branch2a_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    res5c_branch2b = self.convolution(res5c_branch2b_pad, group=1, strides=[1, 1], padding='VALID',
                                      name='res5c_branch2b')
    bn5c_branch2b = self.batch_normalization(res5c_branch2b, variance_epsilon=9.99999974738e-06,
                                             name='bn5c_branch2b')
    res5c_branch2b_relu = tf.nn.relu(bn5c_branch2b, name='res5c_branch2b_relu')
    res5c_branch2c = self.convolution(res5c_branch2b_relu, group=1, strides=[1, 1], padding='VALID',
                                      name='res5c_branch2c')
    bn5c_branch2c = self.batch_normalization(res5c_branch2c, variance_epsilon=9.99999974738e-06,
                                             name='bn5c_branch2c')
    res5c = res5b_relu + bn5c_branch2c
    res5c_relu = tf.nn.relu(res5c, name='res5c_relu')
    feature_0 = tf.contrib.layers.flatten(res5c_relu)

    with tf.variable_scope('feature') as scope:
      wts = self._variable_with_constant_value('weight', self.__weights_dict['feature_1']['weights'])
      bis = self._variable_with_constant_value('bias', self.__weights_dict['feature_1']['bias'])
      feature_1 = tf.add(tf.matmul(feature_0, wts), bis)

    cnn.top_layer = feature_1
    cnn.top_size = int(bis.shape[-1])

    cnn.dropout()

  def _backdoor_mask(self, cnn):
    with tf.variable_scope('input_mask') as scope:
      in_shape = cnn.top_layer.shape
      shape = np.zeros(4, dtype=np.int32)
      shape[0] = 1
      shape[1] = int(in_shape[1])
      shape[2] = int(in_shape[2])
      shape[3] = 1
      mask_param = tf.get_variable('mask_param', shape, dtype=tf.float32, initializer=tf.random_normal_initializer(),
                                   trainable=self.trainable)
      mask = (tf.tanh(mask_param) + 1.) / 2.
      shape[3] = int(in_shape[3])
      pattern_param = tf.get_variable('pattern_param', shape, dtype=tf.float32,
                                      initializer=tf.glorot_normal_initializer(), trainable=self.trainable)
      pattern = tf.tanh(pattern_param)
      masked_input = (1 - mask) * cnn.top_layer + mask * pattern
      cnn.top_layer = masked_input
      cnn.aux_top_layer = mask

  def skip_final_affine_layer(self):
    return True

  def add_inference(self, cnn):

    if self.options.net_mode == 'backdoor_def':
      self.trainable = cnn.phase_train and (self.options.fix_level != 'all')
      self._backdoor_mask(cnn)

    self.trainable = cnn.phase_train and (self.options.fix_level != 'bottom') \
                     and (self.options.fix_level != 'bottom_affine')
    cnn.trainable = self.trainable
    if self.model_name == 'resnet101':
      self._resnet101_inference(cnn)
    elif self.model_name == 'vgg16':
      self._vgg16_inference(cnn)
    elif self.model_name == 'googlenet':
      self._googlenet_inference(cnn)
    elif self.model_name == 'gtsrb':
      self._gtsrb_inference(cnn)
    elif self.model_name == 'resnet50':
      self._resnet50.add_inference(cnn)
    elif self.model_name == 'benchmark_resnet101':
      self._resnet101.add_inference(cnn)
      cnn.affine(256, activation='linear')


    if self.options.net_mode == 'triple_loss':
      cnn.aux_top_layer = cnn.top_layer
      cnn.aux_top_site = cnn.top_size

    if self.options.build_level != 'embeddings':
      self.trainable = cnn.phase_train and (self.options.fix_level != 'last_affine') \
                       and (self.options.fix_level != 'bottom_affine')
      cnn.trainable = self.trainable
      name = ('fc%d_1' % self.num_class)
      initializers = None
      if (hasattr(self, '__weights_dict')) and (name in self.__weights_dict):
        print('===Debug===Hi, I found it ' + name)
        initializers = []
        initializers.append(tf.constant_initializer(self.__weights_dict[name]['weights']))
        initializers.append(tf.constant_initializer(self.__weights_dict[name]['bias']))
      cnn.affine(self.num_class, activation='linear', initializers=initializers)

    self.last_affine_name = 'affine' + str(cnn.counts['affine']-1)

    return cnn.top_layer

  def build_network(self,
                    inputs,
                    phase_train=True,
                    nclass=1001):
    images = inputs[0]
    if self.data_format == 'NCHW':
      images = tf.transpose(images, [0, 3, 1, 2])
    var_type = tf.float32
    if self.data_type == tf.float16 and self.fp16_vars:
      var_type = tf.float16
    network = convnet_builder.ConvNetBuilder(
        images, self.depth, phase_train, self.use_tf_layers, self.data_format,
        self.data_type, var_type)
    with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
      self.add_inference(network)
      # Add the final fully-connected class layer
      if not self.skip_final_affine_layer():
        logits = network.affine(nclass, activation='linear')
        aux_logits = None
        if network.aux_top_layer is not None:
          with network.switch_to_aux_top_layer():
            aux_logits = network.affine(nclass, activation='linear', stddev=0.001)
      else:
        logits = network.top_layer
        aux_logits = network.aux_top_layer
    if self.data_type == tf.float16:
      # TODO(reedwm): Determine if we should do this cast here.
      logits = tf.cast(logits, tf.float32)
      if aux_logits is not None:
        aux_logits = tf.cast(aux_logits, tf.float32)
    return model_lib.BuildNetworkResult(
        logits=logits, extra_info=None if aux_logits is None else aux_logits)

  def get_learning_rate(self, global_step, batch_size):
    if self.model_name == 'resnet50':
      return self._resnet50.get_learning_rate(global_step, batch_size)
    if 'resnet101' in self.model_name:
      return self._resnet101.get_learning_rate(global_step, batch_size)
    return self.options.base_lr

  def batch_normalization(self, input, name, **kwargs):
    with tf.variable_scope(name):
      # moving_mean & moving_variance
      mean = self._variable_with_constant_value('mean', self.__weights_dict[name]['mean'], False)
      variance = self._variable_with_constant_value('var', self.__weights_dict[name]['var'], False)
      offset = self._variable_with_constant_value('bias', self.__weights_dict[name]['bias']) \
                                 if 'bias' in self.__weights_dict[name] else None
      scale = self._variable_with_constant_value('scale', self.__weights_dict[name]['scale']) \
                                if 'scale' in self.__weights_dict[name] else None

      if not self.trainable:
        decay = 0.999
        bn, batch_mean, batch_variance = tf.nn.fused_batch_norm(input, scale=scale, offset=offset,
                                                                name=name, is_training=True, epsilon=1e-5)

        mean_update = moving_averages.assign_moving_average(mean, batch_mean, decay=decay, zero_debias=False)
        variance_update = moving_averages.assign_moving_average(variance, batch_variance, decay=decay,
                                                                zero_debias=False)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_update)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_update)
      else:
        bn, _, _ = tf.nn.fused_batch_norm(input, scale=scale, offset=offset, mean=mean, variance=variance,
                                          name=name, is_training=False, epsilon=1e-5)
      return bn

  def convolution(self, input, name, group, strides, padding):
    with tf.variable_scope(name):
      w = self._variable_with_constant_value('weight', self.__weights_dict[name]['weights'])
      strides = [1] + strides + [1]
      layer = tf.nn.conv2d(input, w, strides=strides, padding=padding)
      if 'bias' in self.__weights_dict[name]:
        b = self._variable_with_constant_value('bias', self.__weights_dict[name]['bias'])
        layer = tf.nn.bias_add(layer, b)
      return layer

  def _classification_loss(self, logits, aux_logits, labels):
    with tf.name_scope('xentropy'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    if aux_logits is not None:
      with tf.name_scope('aux_xentropy'):
        aux_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          logits=aux_logits, labels=labels)
        aux_loss = 0.4 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')
        loss = tf.add_n([loss, aux_loss])
    return loss

  def _triple_loss(self, logits, aux_logits, labels):
    splited_labels = tf.unstack(labels, axis=1)
    lambda_a = splited_labels[2]
    lambda_b = 1 - lambda_a
    with tf.name_scope('xentropy'):
      a_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=tf.to_int32(splited_labels[0]), weights=lambda_a)
      b_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=tf.to_int32(splited_labels[1]), weights=lambda_b)

      loss = tf.reduce_mean(a_cross_entropy + b_cross_entropy, name='xentropy_mean')

    if aux_logits is not None:
      ct_lambda = tf.concat([tf.expand_dims(lambda_a, 1), tf.expand_dims(lambda_b, 1)], axis=1)
      splited_lambda = tf.split(ct_lambda, self.options.num_slices_one_batch, axis=0)
      splited_aux_logits = tf.split(aux_logits, self.options.num_slices_one_batch, axis=0)
      with tf.name_scope('aux_triplet'):
        for _ct_lambda, _aux_logits in zip(splited_lambda, splited_aux_logits):
          cross = tf.matmul(_aux_logits, tf.transpose(_aux_logits))

          square_norm = tf.diag_part(cross)
          square_cross = tf.square(cross)
          square_cos = tf.divide(square_cross, tf.expand_dims(square_norm, 1))
          square_cos = tf.divide(square_cos, tf.expand_dims(square_norm, 0))

          unstacked_sq_cos = tf.unstack(square_cos, axis=0)
          sq_cos_a = tf.expand_dims(unstacked_sq_cos[0], 1)
          sq_cos_b = tf.expand_dims(unstacked_sq_cos[-1], 1)
          ct_sq_cos = tf.concat(axis=1, values=[sq_cos_a, sq_cos_b])
          ct_cos = tf.sqrt(ct_sq_cos)

          triplet_loss = ct_sq_cos - 2.0 * ct_cos * _ct_lambda + tf.square(_ct_lambda)
          aux_loss = tf.reduce_mean(triplet_loss, name='aux_loss')

          loss = tf.add_n([loss, aux_loss])
      return loss

  def _backdoor_defence_loss(self, logits, aux_logits, labels):
    with tf.name_scope('xentropy'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    with tf.name_scope('aux_l1norm'):
      abs_logits = tf.abs(aux_logits)
      abs_sum = tf.reduce_sum(abs_logits, [1, 2, 3])
      # aux_l1_norm = tf.losses.absolute_difference(labels=labels,predictions=abs_sum)
      aux_loss = self.options.loss_lambda * tf.reduce_mean(abs_sum, name='aux_loss')
      loss = tf.add_n([loss, aux_loss])
    return loss

  def loss_function(self, inputs, build_network_result):
    logits = build_network_result.logits
    aux_logits = build_network_result.extra_info
    _, labels = inputs
    loss = None
    """Loss function."""
    if self.options.net_mode == 'normal':
      loss = self._classification_loss(logits, aux_logits, labels)
    elif self.options.net_mode == 'triple_loss':
      loss = self._triple_loss(logits, aux_logits, labels)
    elif self.options.net_mode == 'backdoor_def':
      loss = self._backdoor_defence_loss(logits, aux_logits, labels)
    return loss

  def _collect_backbone_vars(self):
    bottom_vars = []
    last_affine_vars = []
    mask_vars = []
    other_vars = []
    mome_vars = []
    adam_vars = []
    all_vars = tf.global_variables()
    for v in all_vars:
      if 'Adam' in v.name:
        adam_vars.append(v)
      elif 'Momentum' in v.name:
        mome_vars.append(v)
      elif self.last_affine_name in v.name:
        last_affine_vars.append(v)
      elif 'input_mask' in v.name:
        mask_vars.append(v)
      else:
        split_name = v.name.split('/')
        if split_name[0] != 'v0':
          other_vars.append(v)
        else:
          bottom_vars.append(v)

    if self.options.load_mode == 'bottom':
      print('===Debug===')
      print(bottom_vars)
      return bottom_vars
    elif self.options.load_mode == 'last_affine':
      return last_affine_vars

    var_list = bottom_vars
    var_list.extend(last_affine_vars)
    if self.options.load_mode == 'bottom_affine':
      return var_list
    var_list.extend(mask_vars)

    return var_list

  def add_backbone_saver(self):
    # Create saver with mapping from variable names in checkpoint of backbone
    # model to variables in SSD model
    print('===Load===')
    print('add abckbone saver: '+self.options.load_mode)
    backbone_var_list = self._collect_backbone_vars()
    self.backbone_savers.append(tf.train.Saver(backbone_var_list))

  def load_backbone_model(self, sess, backbone_model_path):
    print('===Load===')
    for saver in self.backbone_savers:
      print('load backbone model from: '+backbone_model_path)
      saver.restore(sess, backbone_model_path)
