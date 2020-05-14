from utils import ssd_vgg_preprocessing
import tensorflow as tf
import numpy as np
import io
from PIL import Image


def convert_image_to_array(image):
    """
    用户输入的图片，处理成array
    :param image:
    :return:
    """
    img = io.BytesIO()
    img.write(image)
    img = Image.open(img).convert("RGB")
    img_array = np.array(img)
    return img_array


def img_preprocessing(img_array):
    """用户输入的图片预处理
    :param img_array: 图片数组
    :return: 形状[300,300,3]的Tensor
    """
    # 用户输入的图片，处理成tensor
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

    image_pre, _, _, _ = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, (300, 300), 'NHWC', resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)

    with tf.Session() as sess:
        result = sess.run(image_pre, feed_dict={img_input: img_array})

    # Tensor的值 处理成string返回给模型输入
    # data = result.to_string()

    return result
