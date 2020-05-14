import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import ssd_vgg_300
from utils.basic_tools import np_methods
from io import BytesIO
import random

slim = tf.contrib.slim


def _plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    # plt.show()
    image_io = BytesIO()
    plt.savefig(image_io, format='png')
    image_io.seek(0)
    return image_io


def postprocess_image(predictions, localisations, ssd_anchors):
    """
    预测结果筛选
    postprocessing:包含按照score排序，
    :param predictions:
    :param localisations:
    :param ssd_anchors:
    :return:
    """
    # 按照score筛选
    classes, scores, bboxes = np_methods.ssd_bboxes_select(
        predictions, localisations, ssd_anchors,
        select_threshold=0.5, img_shape=(300, 300), num_classes=21, decode=True)

    rbbox_img = [0.0, 0.0, 1.0, 1.0]

    # bbox边框不能超过原图片
    bboxes = np_methods.bboxes_clip(rbbox_img, bboxes)
    # 排序
    classes, scores, bboxes = np_methods.bboxes_sort(classes, scores, bboxes, top_k=400)
    # 非最大抑制
    classes, scores, bboxes = np_methods.bboxes_nms(classes, scores, bboxes, nms_threshold=.45)
    # 大小设置
    bboxes = np_methods.bboxes_resize([0.0, 0.0, 1.0, 1.0], bboxes)

    return classes, scores, bboxes


def tag_picture(img, predictions, localisations):
    """
    模型预测结果后期处理以及画图标记
    :param predictions: Tensor列表，预测结果
    :param localisations:Tensor列表，预测位置
    :return: 图片标记IO流
    """
    net_shape = (300, 300)

    ssd_net = ssd_vgg_300.SSDNet()

    # 计算每层boxes
    ssd_anchors = ssd_net.anchors(net_shape)

    # postprocessing
    classes, scores, bboxes = postprocess_image(predictions, localisations, ssd_anchors)

    # 图片标记
    if type(img) != np.ndarray:
        img = np.array(img)

    image_io = _plt_bboxes(img, classes, scores, bboxes)

    return image_io
