"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
# We only use opencv3
if not (cv2.__version__).startswith('3.'):
    raise ValueError('Only opencv 3. is supported!')

def aug(images, random_transform_args={
                    'rotation_range': 10,
                    'zoom_range': 0.05,
                    'shift_range': 0.05,
                    'random_flip': 0.5,
                }, color_rng=[0.5, 1.5]):
    """
    Augment operation for image list
    :param images: image list
    :param random_transform_args: shape transform arguments
    :param color_rng: color transform arguments
    :return:
    """
    if random_transform_args is not None: # do aug
        # Transform
        images = random_transform(images, **random_transform_args)
    # Color
    if color_rng is not None:
        for i, im in enumerate(images):
            # im = im[:, :, (2, 1, 0)]
            # Brightness
            im = Image.fromarray(im)
            factor = np.random.uniform(color_rng[0], color_rng[1])
            enhancer = ImageEnhance.Brightness(im)
            im = enhancer.enhance(factor)
            # Contrast
            factor = np.random.uniform(color_rng[0], color_rng[1])
            enhancer = ImageEnhance.Contrast(im)
            im = enhancer.enhance(factor)
            # Color distort
            factor = np.random.uniform(color_rng[0], color_rng[1])
            enhancer = ImageEnhance.Color(im)
            im = enhancer.enhance(factor)
            # Sharpe
            factor = np.random.uniform(color_rng[0], color_rng[1])
            enhancer = ImageEnhance.Sharpness(im)
            im = enhancer.enhance(factor)
            im = np.array(im)
            # im = im[:, :, (2, 1, 0)]
            images[i] = im.copy()

    return images


def random_transform(images, rotation_range, zoom_range, shift_range, random_flip):
    """
    Random transform images in a list
    :param images:
    :param rotation_range:
    :param zoom_range:
    :param shift_range:
    :param random_flip:
    :return:
    """
    h, w = images[0].shape[:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    flip_prob = np.random.random()
    for i, image in enumerate(images):
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, scale)
        mat[:, 2] += (tx, ty)
        result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if flip_prob < random_flip:
            result = result[:, ::-1]
        images[i] = result.copy()
    return images


def draw_words2im(im, words, x, y, font_size=0.5, color=(255, 0, 0)):
    """ Draw text on image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, words, (x, y), font, font_size, color, 2,
                cv2.LINE_AA)


def get_mask(bboxes, shape):
    """ get mask of area covered by bboxes """
    mask = np.zeros(shape)
    for bx in bboxes:
        # left, top, right, botttom
        mask[bx[1]:bx[3], bx[0]:bx[2], :] = 1
    return mask

def resize(images, size):
    for i, im in enumerate(images):
        im = cv2.resize(im, tuple(size))
        images[i] = im
    return images