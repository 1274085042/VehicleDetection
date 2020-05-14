import sys
sys.path.append('../')

import time
import numpy as np
import argparse
import tensorflow as tf
from PIL import Image
import visualization
from utils.basic_tools import np_methods

slim = tf.contrib.slim

from nets import nets_factory
from preprocessing import preprocessing_factory

def main(ckpt,img_path):
    # 1 定义输入数据的占位符
    image_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

    # 2 数据输入到预处理工厂当中，进行处理得到结果
    # 定义一个输出的形状，元组表示
    net_shape = (300, 300)
    data_format = "NHWC"
    preprocessing_fn = preprocessing_factory.get_preprocessing("ssd_vgg_300", is_training=False)
    img_pre, _, _, bbox_img= preprocessing_fn(image_input, None, None, net_shape, data_format)

    # 3 定义SSD模型， 并输出预测结果
    # img_pre是三维形状，(300, 300, 3)
    # 卷积神经网络要求都是四维的数据计算(1, 300, 300, 3)
    # 维度的扩充
    image_4d = tf.expand_dims(img_pre, 0)
    # reuse作用：在notebook当中运行，第一次创建新的变量为FALSE，但是重复运行cell,保留这些变量的命名，选择重用这些命名，已经存在内存当中了
    # 没有消除，设置reuse=True
    reuse = True if 'ssd_net' in locals() else False
    # 网络工厂获取
    ssd_class = nets_factory.get_network("ssd_vgg_300")
    ssd_params = ssd_class.default_params._replace(num_classes=3)
    # 初始化网络
    ssd_net = ssd_class(ssd_params)
    # 通过网络的方法获取结果
    # 使用slim.arg_scope指定共有参数data_format,net里面有很多函数需要使用data_format
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

    # 4 定义交互式会话，初始化变量，加载模型
    #4.1 使用config定义一个交互式会话
    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.InteractiveSession(config=config)

    #4.2 初始化变量
    sess.run(tf.global_variables_initializer())

    #4.3 加载模型
    # 训练参数
    ckpt_filepath = ckpt
    # 创建saver
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filepath)

    # 4.4 得到模型预测结果
    # 读取一张图片
    img = Image.open(img_path).convert('RGB')

    start=time.time()

    i, p, l, box_img = sess.run([image_4d, predictions, localisations, bbox_img], feed_dict={image_input:img})
    #print(p)

    # 5 预测结果后期处理
    # 通过 predictions 与 select_threshold 筛选bbox
    ssd_anchors = ssd_net.anchors(net_shape)
    #5.1 去掉概率小于select_threshold的检测框
    classes, scores, bboxes = np_methods.ssd_bboxes_select(p, l, ssd_anchors, select_threshold=0.4, img_shape=(300, 300), num_classes=3, decode=True)
    #5.2 bbox边框不能超过原图片 默认原图的相对于bbox大小比例 [0, 0, 1, 1]
    bboxes = np_methods.bboxes_clip([0, 0, 1, 1], bboxes)
    #5.3 NMS
    # 根据 scores 从大到小排序，并改变classes rbboxes的顺序
    classes, scores, bboxes = np_methods.bboxes_sort(classes, scores, bboxes, top_k=400)
    # 使用nms算法筛选bbox
    classes, scores, bboxes = np_methods.bboxes_nms(classes, scores, bboxes, nms_threshold=.45)
    # 根据原始图片的bbox，修改所有bbox的范围 [.0, .0, .1, .1]
    bboxes = np_methods.bboxes_resize([.0, .0, .1, .1], bboxes)

    end=time.time()
    print("Predicted in %f seconds"%(end-start))

    #6 预测结果显示
    img=np.array(img)
    visualization.plt_bboxes(img, classes, scores, bboxes)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--ckpt_path', type=str, default='..\ckpt\\fine_tuning\model.ckpt-0',
                        help="Path to the model.ckpt file."
                       )
    parser.add_argument( '--image_path', type=str, default='0.jpg',
                        help='Absolute path to image file.'
                        )
    args=parser.parse_args()
    main(args.ckpt_path,args.image_path)