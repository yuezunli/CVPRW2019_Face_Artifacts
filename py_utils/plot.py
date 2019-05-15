"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# Draw plot
def draw2D(plt, X, Y, order, xname, yname, xlim, ylim, c, title='', rng=None):

    X = np.array(X)
    Y = np.array(Y)

    fig = plt.figure()
    plt.title(title)
    plt.ylabel(yname)
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(xname)
    plt.xlim(xlim[0], xlim[1])

    for i, type_name in enumerate(order):
        plt.plot(X[i], Y[i], c[i], label=type_name, linewidth=1.5) #, markersize=5)

    plt.grid()
    plt.legend()
    fig.canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    im = np.array(fig.canvas.renderer._renderer)
    plt.show()
    plt.close()
    return im[:, :, (2, 1, 0)]


def vis_plot(img_h, fps, val, order, xname, xlim, yname, ylim, title):
    X = np.arange(0, len(val)) / fps
    c = ['b-']
    im = draw2D(
        plt=plt,
        Y=[val],
        X=[X],
        order=[order],
        xname=xname,
        xlim=[xlim[0], xlim[1]],
        yname=yname,
        ylim=[ylim[0], ylim[1]],
        title=title,
        c=c)
    scale = img_h / np.float32(im.shape[0])
    im = cv2.resize(im, None, None, fx=scale, fy=scale)
    return im