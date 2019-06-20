"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
pwd = os.path.dirname(__file__)

class Solver(object):

    def __init__(self,
                 sess,
                 net,
                 cfg):

        self.sess = sess
        self.net = net
        self.cfg = cfg

    def init(self):
        cfg = self.cfg
        self.img_size = cfg.IMG_SIZE
        pwd = os.path.dirname(os.path.abspath(__file__))
        self.summary_dir = os.path.join(pwd, cfg.SUMMARY_DIR)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.model_dir = pwd + '/' + cfg.MODEL_DIR_PREFIX + '_' + cfg.BASE_NETWORK
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_path = os.path.join(self.model_dir, cfg.MODEL_NAME)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.saver = tf.train.Saver(max_to_keep=5, var_list=tf.global_variables())
        # initialize the graph
        if self.net.is_train:
            self.num_epoch = cfg.TRAIN.NUM_EPOCH
            self.learning_rate = cfg.TRAIN.LEARNING_RATE
            self.decay_rate = cfg.TRAIN.DECAY_RATE
            self.decay_step = cfg.TRAIN.DECAY_STEP
            self.set_optimizer()
            # Add summary
            self.loss_summary = tf.summary.scalar('loss_summary', self.net.total_loss)
            self.lr_summary = tf.summary.scalar('learning_rate_summary', self.LR)
            self.summary = tf.summary.merge([self.loss_summary, self.lr_summary])
            self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        if cfg.PRETRAINED_MODELS != '':
            self.load_ckpt(cfg.PRETRAINED_MODELS)
        else: # Load ckpt
            self.load()

    def test(self, images):

        feed_dict = {
            self.net.input: images,
        }

        fetch_list = [
            self.net.prob
        ]
        prob, = self.sess.run(fetch_list, feed_dict=feed_dict)


        return prob

    def train(self, images, labels):
        feed_dict = {
            self.net.input: images,
            self.net.gt: labels
        }

        fetch_list = [
            self.train_op,
            self.summary,
            self.net.prob,
            self.net.net_loss,
            self.net.total_loss,
            self.net.weights
        ]
        _, summary, prob, net_loss, total_loss, weights \
            = self.sess.run(fetch_list, feed_dict=feed_dict)

        return summary, prob, net_loss, total_loss, weights

    def save(self, step):
        """ Save checkpoints """
        save_path = self.saver.save(self.sess, self.model_path, global_step=step)
        print('Model {} saved in file.'.format(save_path))

    def load(self):
        """Load weights from checkpoint"""
        if os.path.isfile(self.model_path + '.meta'):
            self.saver.restore(self.sess, self.model_path)
            print('Loading checkpoint {}'.format(self.model_path))
        else:
            print('Loading checkpoint failed')

    def load_ckpt(self, model_path):
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(model_path))
        variables_to_restore = self.get_restore_var_list(model_path)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(self.sess, model_path)
        print('Loaded.')

    def set_optimizer(self):
        # Set learning rate decay
        self.LR = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_step,
            decay_rate=self.decay_rate,
            staircase=True
        )
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.LR,
        )

        self.train_op = slim.learning.create_train_op(total_loss=self.net.total_loss,
                                                       optimizer=optimizer,
                                                       global_step=self.global_step
                                                       )

    def get_restore_var_list(self, path):
        """
        Get variable list when restore from ckpt. This is mainly for transferring model to another network
        """
        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # Variables in graph
        saved_vars = self.list_vars_in_ckpt(path)
        saved_vars_name = [var[0] for var in saved_vars]
        saved_vars_shape = [var[1] for var in saved_vars]
        restore_var_list = []
        for var in global_vars:
            if var.name[:-2] == 'global_step':
                continue
            if var.name[:-2] in saved_vars_name and var.shape.as_list() == saved_vars_shape[
                saved_vars_name.index(var.name[:-2])]:
                restore_var_list.append(var)
        return restore_var_list

    def list_vars_in_ckpt(self, path):
        """List all variables in checkpoint"""
        saved_vars = tf.contrib.framework.list_variables(path)  # List of tuples (name, shape)
        return saved_vars



