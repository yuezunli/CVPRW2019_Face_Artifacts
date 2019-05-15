"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import dlib, cv2, warnings
import os, pickle, sys
from tqdm import tqdm
import numpy as np
from py_utils.face_utils import lib


class Data(object):

    def __init__(self,
                 face_img_dir,
                 cache_path):
        self.face_img_dir = face_img_dir
        self.face_img_paths = [os.path.join(face_img_dir, im_name) for im_name in sorted(os.listdir(face_img_dir))]
        self.data_num = len(self.face_img_paths)
        self.cache_path = cache_path
        # Get landmarks
        face_caches = self._load_cache()
        if face_caches is None:
            # Load dlib
            self._set_up_dlib()
            face_caches = {}
            # Align faces
            print("Aligning faces...")
            for i, im_path in enumerate(tqdm(self.face_img_paths)):
                im = cv2.imread(str(im_path))
                faces = lib.align(im[:, :, (2,1,0)], self.front_face_detector, self.lmark_predictor)
                if len(faces) == 0:
                    faces = [None, None]
                else:
                    faces = faces[0]
                face_caches[self.face_img_paths[i].stem] = faces
            self._save_cache(face_caches)

        self.face_caches = face_caches

    def _set_up_dlib(self):
        # Employ dlib to extract face area and landmark points
        pwd = os.path.dirname(__file__)
        # self.cnn_face_detector = dlib.cnn_face_detection_model_v1(pwd + '/mmod_human_face_detector.dat')
        self.front_face_detector = dlib.get_frontal_face_detector()
        self.lmark_predictor = dlib.shape_predictor(pwd + '/dlib_model/shape_predictor_68_face_landmarks.dat')

    def _load_cache(self):
        face_caches = None
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                face_caches = pickle.load(f)
        return face_caches

    def _save_cache(self, face_caches):
        # Save face and matrix to cache
        with open(self.cache_path, 'wb') as f:
            pickle.dump(face_caches, f)


