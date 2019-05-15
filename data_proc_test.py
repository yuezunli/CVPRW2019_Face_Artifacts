"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import cv2, sys
from py_utils import vis
import os, pickle
import numpy as np
from data import Data
from py_utils.face_utils import lib


class DataProcTest(Data):

    def __init__(self,
                 face_img_dir,
                 cache_path,
                 sample_num,
                 ):

        super(DataProcTest, self).__init__(
            face_img_dir=face_img_dir,
            cache_path=cache_path,
        )
        self.batch_num = self.data_num
        self.sample_num = sample_num

    def get_batch(self, batch_idx, resize=None):
        if batch_idx >= self.batch_num:
            raise ValueError("Batch idx must be in range [0, {}].".format(self.batch_num - 1))

        imgs = []
        names = []
        im_path = self.face_img_paths[batch_idx]
        im = cv2.imread(str(im_path))
        im_name = os.path.basename(im_path).split('.')[0]
        _, points = self.face_caches[im_name]
        if points is None:
            return None

        for _ in range(self.sample_num):
            # Cut out head region
            im_cut, _ = lib.cut_head([im.copy()], points)
            im_cut = cv2.resize(im_cut[0], (resize[0], resize[1]))
            imgs.append(im_cut)

        data = {}
        data['images'] = imgs
        data['name_list'] = im_name
        return data




