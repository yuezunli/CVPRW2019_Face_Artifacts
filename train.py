"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""

import tensorflow as tf
from resolution_network import ResoNet
from solver import Solver
from easydict import EasyDict as edict
from data_proc_train import DataProcTrain
import cv2, yaml
from py_utils.vis import vis_im
import numpy as np


def main(args):
    # Parse config
    cfg_file = args.cfg
    with open(cfg_file, 'r') as f:
        cfg = edict(yaml.load(f))

    # Data generator
    data_gen = DataProcTrain(
        face_img_dir=args.data_dir,
        cache_path=args.cache_path,
        anno_path=args.list,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        is_shuffle=True)

    cfg.TRAIN.DECAY_STEP = data_gen.batch_num
    epoch = cfg.TRAIN.NUM_EPOCH
    with tf.Session() as sess:
        # Build network
        reso_net = ResoNet(cfg=cfg, is_train=True)
        reso_net.build()
        # Init solver
        solver = Solver(sess=sess, cfg=cfg, net=reso_net)
        solver.init()
        # Train
        count = 0
        for epoch_id in range(epoch):
            for i in range(data_gen.batch_num):
                data = data_gen.get_batch(i, resize=cfg.IMG_SIZE[:2])
                images = data['images']
                labels = data['images_label']
                ims_tmp = vis_im(images, 'tmp/vis.jpg')

                summary, prob, net_loss, total_loss, weights = solver.train(images, labels)
                pred_labels = np.argmax(prob, axis=1)
                print('====================================')
                print('Net loss: {}'.format(net_loss))
                print('Total loss: {}'.format(total_loss))
                print('Real label: {}'.format(np.array(labels)))
                print('Pred label: {}'.format(pred_labels))
                print('Neg hard mining: {}'.format(weights))
                print('epoch: {}, batch_idx: {}'.format(epoch_id, i))
                if count % 100 == 0:
                    solver.writer.add_summary(summary, count)
                count += 1

            if epoch_id % 2 == 0:
                solver.save(epoch_id)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfgs/res50.yml')
    parser.add_argument('--data_dir', type=str, default='head_data/')
    parser.add_argument('--cache_path', type=str, default='landmarks.p')
    parser.add_argument('--list', type=str, default='annos.p')
    args = parser.parse_args()
    main(args)