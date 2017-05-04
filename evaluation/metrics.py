# The following code is modified from https://github.com/shelhamer/clockwork-fcn
import numpy as np
import scipy.io as sio
import torch
from pdb import set_trace as st

def get_out_scoremap(net):
    return net.blobs['score'].data[0].argmax(axis=0).astype(np.uint8)

def feed_net(net, in_):
    """
    Load prepared input into net.
    """
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

def segrun(net, in_):
    feed_net(net, in_)
    net.forward()
    return get_out_scoremap(net)

def fast_hist(a, b, n):
    # print('saving')
    # sio.savemat('/tmp/fcn_debug/xx.mat', {'a':a, 'b':b, 'n':n})
    k = np.where((a >= 0) & (a < n))[0]
    bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n**2)
    if len(bc) != n**2:
        # ignore this example if dimension mismatch
        return 0
    return bc.reshape(n, n)

def get_scores(hist):
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)
    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)

    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu


def get_all_scores(label1,label2,n_cl):
    if type(label1) == torch.FloatTensor or type(label1)==torch.cuda.FloatTensor:
        label1=label1.numpy()
        label2=label2.numpy()
    hist_perframe = np.zeros((n_cl, n_cl))
    hist_perframe += fast_hist(label1.flatten(), label2.flatten(), n_cl)
    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    return mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou 