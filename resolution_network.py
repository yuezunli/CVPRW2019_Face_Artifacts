"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""

from tf_utils import utils as tfutils
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1, vgg


class ResoNet(object):

    def __init__(self,
                 cfg,
                 is_train
                 ):

        self.base_net = cfg.BASE_NETWORK
        self.img_size = cfg.IMG_SIZE
        self.num_classes = cfg.NUM_CLASSES
        self.is_train = is_train
        self.img_mean = cfg.PIXEL_MEAN

        self.layers = {}
        self.params = {}

        if self.is_train:
            self.beta = cfg.TRAIN.BETA
            self.neg_hard_mining = cfg.TRAIN.NEG_HARD_MINING
            self.pos_hard_mining = cfg.TRAIN.POS_HARD_MINING

    def build(self):
        # Input
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size[0], self.img_size[1], self.img_size[2]])
        self.input_mean = tfutils.mean_value(self.input, self.img_mean)
        if self.base_net == 'vgg16':
            with slim.arg_scope(vgg.vgg_arg_scope()):
                outputs, end_points = vgg.vgg_16(self.input_mean, self.num_classes)
                self.prob = tf.nn.softmax(outputs, -1)
                self.logits = outputs

        elif self.base_net == 'res50':
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_50(self.input_mean, self.num_classes, is_training=self.is_train)
                self.prob = tf.nn.softmax(net[:, 0, 0, :], -1)
                self.logits = net[:, 0, 0, :]
        elif self.base_net == 'res101':
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_101(self.input_mean, self.num_classes, is_training=self.is_train)
                self.prob = tf.nn.softmax(net[:, 0, 0, :], -1)
                self.logits = net[:, 0, 0, :]
        elif self.base_net == 'res152':
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_152(self.input_mean, self.num_classes, is_training=self.is_train)
                self.prob = tf.nn.softmax(net[:, 0, 0, :], -1)
                self.logits = net[:, 0, 0, :]
        else:
            raise ValueError('base network should be vgg16, res50, -101, -152...')
        self.gt = tf.placeholder(dtype=tf.int32, shape=[None])
        # self.var_list = tf.trainable_variables()

        if self.is_train:
            self.loss()

    def loss(self):
        # Optional for hard mining
        # # Negative hard mining
        # tmp1 = self.prob[:, 0] * tf.cast(1 - self.gt, dtype=tf.float32)
        # tmp1 = tf.to_float(tmp1 < self.neg_hard_mining) * tf.cast(1 - self.gt, dtype=tf.float32)
        # # Positive hard mining
        # tmp2 = self.prob[:, 1] * tf.cast(self.gt, dtype=tf.float32)
        # tmp2 = tf.to_float(tmp2 < self.pos_hard_mining) * tf.cast(self.gt, dtype=tf.float32)
        # self.weights = tmp1 + tmp2
        self.weights = 1
        tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.gt, logits=self.logits)
        self.net_loss = tf.reduce_mean(tmp * self.weights)
        tf.losses.add_loss(self.net_loss)
        self.total_loss = tf.losses.get_total_loss()

