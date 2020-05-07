import os
import tensorflow as tf
import xml.etree.ElementTree as ET

from Vehicle.datasets.utils.dataset_utils import int64_feature, float_feature, bytes_feature
from Vehicle.datasets.dataset_config import DIRECTORY_ANNOTATIONS, DIRECTORY_IMAGES, SAMPLES_PER_FILES, VEHICLE_LABELS


def _get_output_filename(output_dir, name, num):

    return "%s/%s_%02d.tfrecord" % (output_dir, name, num)

def _process_image(dataset_dir, img_name):
    """
    处理一张图片的逻辑
    dataset_dir:
    img_name
    """
    # 处理图片
    # 该图片的文件名字
    filename = dataset_dir + DIRECTORY_IMAGES + img_name + '.jpg'
    print(filename)
    print("--------")

    # 读取图片
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # 处理xml
    filename_xml = dataset_dir + DIRECTORY_ANNOTATIONS + img_name + ".xml"

    # 用ET读取
    tree = ET.parse(filename_xml)
    root = tree.getroot()

    # 处理每一个标签
    # size:height, width, depth
    # 一张图片只有这三个属性
    size = root.find('size')

    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    # object:name, truncated, difficult, bndbox(xmin,ymin,xmax,ymax)
    # 定义每个属性的列表，装有不同对象
    # 一张图片会有多个对象findall
    bboxes = []
    difficult = []
    truncated = []
    # 装有所有对象名字
    # 对象的名字怎么存储？？？
    labels = []
    labels_text = []
    for obj in root.findall('object'):     #root.findall得到的是一个列表，获取所有的’object‘
        # name
        label = obj.find('name').text

        #存入目标的大类别
        labels.append(int(VEHICLE_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        # difficult
        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)

        # truncated
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        # bndbox  [[12,23,34,45], [56,23,76,9]]
        bbox = obj.find('bndbox')

        # 标准化：xmin,ymin,xmax,ymax都要进行除以原图片的长宽
        bboxes.append((float(bbox.find('ymin').text) / shape[0],   # 将四个坐标值放到一个元组中
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]))

    return image_data, shape, bboxes, difficult, truncated, labels,labels_text

def _convert_to_example(image_data, shape, bboxes, difficult,truncated, labels, labels_text):
    """
    图片数据封装成example protobufer
    image_data: 图片内容
    shape: 图片形状
    bboxes: 每一个目标的四个位置值
    difficult: 默认0
    truncated: 默认0
    labels: 目标代号
    labels_text: 目标名称
    """
    # [[12,23,34,45], [56,23,76,9]] --->ymin [12, 56], xmin [23, 23], ymax [34, 76], xmax [45, 9]
    # bboxes的格式转换
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    for b in bboxes:
        ymin.append(b[0])
        xmin.append(b[1])
        ymax.append(b[2])
        xmax.append(b[3])

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))

    return example

def _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer):
    """
    # 1、读取图片内容以及图片相对应的XML文件
    # 2、读取的内容封装成example, 写入指定tfrecord文件
    dataset_dir: 数据集目录
    img_name: 该图片名字
    tfrecord_writer: 文件写入实例
    """
    # 1、读取图片内容以及图片相对应的XML文件
    image_data, shape, bboxes, difficult, truncated, labels, labels_text = _process_image(dataset_dir, img_name)
    #print(image_data)

    # 2、读取的内容封装成example
    example = _convert_to_example(image_data, shape, bboxes, difficult, truncated, labels, labels_text)

    # 3、exmaple 写入指定tfrecord文件
    tfrecord_writer.write(example.SerializeToString())

    return None


def run(dataset_dir ,output_dir,name="data"):
    '''
    存入多个tfrecords文件，每个文件通常会固定样本的数量
    dataset_dir: 数据集目录
    output_dir: tfrecord输出目录
    name: 数据集名字
    '''
    # 1、判断数据集的路径是否存在，如果不存在报错
    if not tf.io.gfile.exists(dataset_dir):
        #tf.gfile.MakeDirs(dataset_dir)
        print("数据集不存在！！！")


    # 2、去anno读取所有的文件名字列表，与images一样的数据量
    # 构造文件的完整路径
    path=os.path.join(dataset_dir,DIRECTORY_ANNOTATIONS)
    # 排序操作，因为os.path会打乱文件名的前后顺序
    filenames = sorted(os.listdir(path))

    # 3、循环列表中的每个文件
    # 建立一个tf.python_io.TFRecordWriter(path)存储器
    # 标记每个TFRecords存储200个图片和相关XML信息
    # 所有的样本标号
    i=0
    # 记录存储的文件标号
    fidx=0
    while i<len(filenames):
        #新建一个tfrecords文件
        #构造一个文件名字
        tf_filename= _get_output_filename(output_dir, name, fidx)

        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j=0
            # 处理200个图片文件和XML
            while i<len(filenames) and j<SAMPLES_PER_FILES:      #SAMPLES_PER_FILES=200
                print("转换图片进度 %d/%d" % (i + 1, len(filenames)))

                # 处理每张图片的逻辑
                # 1、读取图片内容以及图片相对应的XML文件
                # 2、读取的内容封装成example, 写入指定tfrecord文件
                xml_filename=filenames[i]
                img_name=xml_filename[:-4]

                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)

                i += 1
                j += 1

            # 当前TFRecords处理结束
            fidx += 1
    print("完成数据集 %s 所有的样本处理" % name)