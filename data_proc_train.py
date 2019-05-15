"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import cv2
import os, pickle, sys
import numpy as np
from py_utils.face_utils import lib
from py_utils.img_utils import proc_img
from data import Data

class DataProcTrain(Data):

    def __init__(self,
                 face_img_dir,
                 cache_path,
                 anno_path,
                 batch_size,
                 is_shuffle=True,
                 is_aug=True):
        super(DataProcTrain, self).__init__(
            face_img_dir=face_img_dir,
            cache_path=cache_path,
        )
        # Load annotations
        with open(anno_path, 'rb') as f:
            self.annos = pickle.load(f)

        self.is_shuffle = is_shuffle
        self.batch_size = batch_size
        self.batch_num = np.int32(np.ceil(np.float32(self.data_num) / self.batch_size))
        if self.is_shuffle:
            idx = np.random.permutation(self.data_num)
            self.face_img_paths = [self.face_img_paths[i] for i in idx]
        self.is_aug = is_aug

    def _face_blur(self, im, trans_matrix, size):
        # size = 64
        face = cv2.warpAffine(im, trans_matrix * size, (size, size))
        # blur face
        face = cv2.GaussianBlur(face, (5, 5), 0)
        new_image = np.copy(im)
        # new_image = proc_img.aug([new_image], random_transform_args=None,
        #                   color_rng=[0.8, 1.2])[0]
        image_size = im.shape[1], im.shape[0]
        cv2.warpAffine(face, trans_matrix * size, image_size, new_image, cv2.WARP_INVERSE_MAP,
                       cv2.BORDER_TRANSPARENT)
        return new_image

    def _select_part_to_blur(self, im, new_image, part_mask):
        new_image = np.uint8(im * (1 - part_mask) + new_image * part_mask)
        return new_image

    def get_batch(self, batch_idx, resize=None):
        if batch_idx >= self.batch_num:
            raise ValueError("Batch idx must be in range [0, {}].".format(self.batch_num - 1))

        # Get start and end image index ( counting from 0 )
        start_idx = batch_idx * self.batch_size
        idx_range = []
        for i in range(self.batch_size):
            idx_range.append((start_idx + i) % self.data_num)

        print('batch index: {}, counting from 0'.format(batch_idx))

        imgs = []
        labels = []
        names = []
        for i in idx_range:
            im = cv2.imread(str(self.face_img_paths[i]))
            im_name = os.path.basename(self.face_img_paths[i]).split('.')[0]
            if im_name in self.face_caches:
                trans_matrix, point = self.face_caches[im_name]
            if point is None:
                continue

            label = self.annos[im_name]
            # label is 1 means this is an authentic image,
            if label == 1:
                rnd = np.random.uniform()
                if rnd < 0.5:
                    # Affine warp face area back
                    size = np.arange(64, 128, dtype=np.int32)
                    c = np.random.randint(0, len(size))
                    new_im = self._face_blur(im, trans_matrix, size=size[c])
                    rnd2 = np.random.uniform()
                    if rnd2 < 0.5:
                        # Only retain a minimal polygon mask
                        part_mask = lib.get_face_mask(im.shape[:2], point)
                        # Select specific blurred part
                        new_im = self._select_part_to_blur(im, new_im, part_mask)
                    im = new_im
                    label = 0
            else:
                continue

            # Cut out head region
            ims, _ = lib.cut_head([im], point)
            # Augmentation
            if self.is_aug:
                im = proc_img.aug(ims, random_transform_args=None,
                                  color_rng=[0.8, 1.2])[0]
            im = cv2.resize(im, (resize[0], resize[1]))
            imgs.append(im)
            labels.append(label)
            names.append(im_name)

        if batch_idx == self.batch_num - 1:
            if self.is_shuffle:
                idx = np.random.permutation(self.data_num)
                self.face_img_paths = [self.face_img_paths[j] for j in idx]

        data = {}
        data['images'] = imgs
        data['images_label'] = labels
        data['name_list'] = names
        return data



