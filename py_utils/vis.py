"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import cv2
import numpy as np


def vis_im(batch, path):
    # concatenate together
    im_ary = []
    for b in range(len(batch)):
        im = batch[b]
        im_ary.append(im)
    im_concat = np.concatenate(im_ary, 1)
    cv2.imwrite(path, im_concat)
    return im_concat


def draw_face_rect(im, points):
    xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    return im


def draw_face_landmarks(im, points):
    for pt in points:
        cv2.circle(im,(pt[0], pt[1]), 2, (0,0,255), -1)
    return im