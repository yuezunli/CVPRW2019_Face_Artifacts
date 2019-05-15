"""
Common utilities for tf
"""
import tensorflow as tf


def mean_value(input, img_mean, mode='bgr'):
    """
    Subtract mean value  ( bgr order)
    """
    b, g, r = tf.split(input, 3, axis=3)
    if mode == 'bgr':
        input = tf.concat([b - img_mean[0], g - img_mean[1], r - img_mean[2]], axis=3)
    if mode == 'rgb':
        input = tf.concat([r - img_mean[2], g - img_mean[1], b - img_mean[0]], axis=3)
    return input