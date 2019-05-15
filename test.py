"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import tensorflow as tf
from resolution_network import ResoNet
from solver import Solver
from easydict import EasyDict as edict
from data_proc_test import DataProcTest
import cv2, yaml
from py_utils.vis import vis_im
import numpy as np


def main(args):
    # Parse config
    cfg_file = args.cfg
    with open(cfg_file, 'r') as f:
        cfg = edict(yaml.load(f))

    # Data generator
    data_gen = DataProcTest(
        face_img_dir=args.data_dir,
        cache_path=args.cache_path,
        sample_num=args.sample_num)

    with tf.Session() as sess:
        # Build network
        reso_net = ResoNet(cfg=cfg, is_train=False)
        reso_net.build()
        # Build solver
        solver = Solver(sess=sess, cfg=cfg, net=reso_net)
        solver.init()
        for i in range(data_gen.batch_num):
            data = data_gen.get_batch(i, resize=cfg.IMG_SIZE[:2])
            images = data['images']
            name_list = data['name_list']
            vis_im(images, 'tmp/vis.jpg')
            prob = solver.test(images)
            print(np.mean(prob[:, 0]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfgs/res50.yml')
    parser.add_argument('--data_dir', type=str, default='head_data_val')
    parser.add_argument('--cache_path', type=str, default='landmarks.p')
    parser.add_argument('--sample_num', type=int, default=10)
    args = parser.parse_args()
    main(args)