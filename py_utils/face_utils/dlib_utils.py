"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""

import dlib, os


def get_front_face_detector():
    return dlib.get_frontal_face_detector()


def get_landmarks_predictor(path):
    if os.path.exists(path) and path.endswith('.dat'):
        return dlib.shape_predictor(path)
    else:
        raise ValueError('{} is not valid...'.format(path))