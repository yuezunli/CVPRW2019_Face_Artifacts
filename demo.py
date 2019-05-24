"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import tensorflow as tf
from resolution_network import ResoNet
from solver import Solver
from easydict import EasyDict as edict
import cv2, yaml, os, dlib
from py_utils.vis import vis_im
import numpy as np
from py_utils.face_utils import lib
from py_utils.vid_utils import proc_vid as pv
import logging



print('***********')
print('Detecting DeepFake images, prob == -1 denotes opt out')
print('***********')
# Parse config
cfg_file = 'cfgs/res50.yml'
with open(cfg_file, 'r') as f:
    cfg = edict(yaml.load(f))
sample_num = 10

# Employ dlib to extract face area and landmark points
pwd = os.path.dirname(__file__)
front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(pwd + '/dlib_model/shape_predictor_68_face_landmarks.dat')

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
# init session
sess = tf.Session(config=tfconfig)
# Build network
reso_net = ResoNet(cfg=cfg, is_train=False)
reso_net.build()
# Build solver
solver = Solver(sess=sess, cfg=cfg, net=reso_net)
solver.init()


def im_test(im):
    face_info = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)
    # Samples
    if len(face_info) == 0:
        logging.warning('No faces are detected.')
        prob = -1  # we ignore this case
    else:
        # Check how many faces in an image
        logging.info('{} faces are detected.'.format(len(face_info)))
        max_prob = -1
        # If one face is fake, the image is fake
        for _, point in face_info:
            rois = []
            for i in range(sample_num):
                roi, _ = lib.cut_head([im], point, i)
                rois.append(cv2.resize(roi[0], tuple(cfg.IMG_SIZE[:2])))
            vis_im(rois, 'tmp/vis.jpg')
            prob = solver.test(rois)
            prob = np.mean(np.sort(prob[:, 0])[np.round(sample_num / 2).astype(int):])
            if prob >= max_prob:
                max_prob = prob
        prob = max_prob
    return prob


def run(input_dir):
    logging.basicConfig(filename='run.log', filemode='w', format='[%(asctime)s - %(levelname)s] %(message)s',
                        level=logging.INFO)

    f_list = os.listdir(input_dir)
    prob_list = []
    for f_name in f_list:
        # Parse video
        f_path = os.path.join(input_dir, f_name)
        print('Testing: ' + f_path)
        logging.info('Testing: ' + f_path)
        suffix = f_path.split('.')[-1]
        if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'nef', 'raf']:
            im = cv2.imread(f_path)
            if im is None:
                prob = -1
            else:
                prob = im_test(im)

        elif suffix.lower() in ['mp4', 'avi', 'mov']:
            # Parse video
            imgs, frame_num, fps, width, height = pv.parse_vid(f_path)
            probs = []
            for fid, im in enumerate(imgs):
                logging.info('Frame ' + str(fid))
                prob = im_test(im)
                if prob == -1:
                    continue
                probs.append(prob)

            # Remove opt out frames
            if probs is []:
                prob = -1
            else:
                prob = np.mean(sorted(probs, reverse=True)[:int(frame_num / 3)])

        logging.info('Prob = ' + str(prob))
        prob_list.append(prob)
        print('Prob: ' + str(prob))

    sess.close()
    return prob_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='demo')
    args = parser.parse_args()
    run(args.input_dir)