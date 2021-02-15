import numpy as np


class Seg_metric(object):
    def __init__(self, y_true, y_pre):
        """
        the metrics to evaluate the segmentation result.
        :param y_true: label. type: numpy
        :param y_pre: predicted result.
        """
        self.y_true = np.reshape(y_true, (-1))
        self.y_pre = np.reshape(y_pre, (-1))
        self.intersection = np.sum(self.y_true * self.y_pre)
        self.y_true_len = np.sum(self.y_true)
        self.y_pre_len = np.sum(self.y_pre)
        self.smooth = 1

    def dice(self):
        return (2. * self.intersection + self.smooth) / (self.y_pre_len + self.y_true_len + self.smooth)

    def iou(self):
        return (self.intersection + self.smooth) / (self.y_pre_len + self.y_true_len - self.intersection + self.smooth)

    def tpr(self):
        return (self.intersection + self.smooth) / (self.y_true_len + self.smooth)

    def precision(self):
        return (self.intersection + self.smooth) / (self.y_pre_len + self.smooth)

    def fp(self):
        return (self.y_pre_len - self.intersection + self.smooth) / (self.y_true_len + self.smooth)

    def fn(self):
        return (self.y_true_len - self.intersection + self.smooth) / (self.y_true_len + self.smooth)


def test_result(mask, seg, mode='ec'):
    """
    calculate the segmentation evaluation result of BraTS SEG.
    :param mask:
    :param seg:
    :param mode:
    :return:
    """
    assert mask.shape == seg.shape, print('mask and seg must have the same shape.')
    temp = np.zeros(seg.shape)
    temp_mask = np.zeros(seg.shape)
    if mode == 'ec':
        temp[seg == 4] = 1
        temp_mask[mask == 4] = 1
        mts = Seg_metric(temp, temp_mask)

    elif mode == 'core':
        temp[seg == 4] = 1
        temp[seg == 1] = 1
        temp_mask[mask == 4] = 1
        temp_mask[mask == 1] = 1
        mts = Seg_metric(temp, temp_mask)

    elif mode == 'full':
        temp[seg != 0] = 1
        temp_mask[mask != 0] = 1
        mts = Seg_metric(temp, temp_mask)
    return mts.dice(), mts.iou(), mts.fp(), mts.fn()
    # print('dice:{},IoU:{}'.format(mts.dice(), mts.iou()))


if __name__ == '__main__':
    x1 = (np.random.rand(100, 100) * 2).astype(np.int8)
    x2 = (np.random.rand(100, 100) * 2).astype(np.int8)
    metrics = Seg_metric(x1, x2)
    print(metrics.dice())
    print(metrics.iou())
    print(metrics.tpr())
    print(metrics.fp())
    print(metrics.fn())
