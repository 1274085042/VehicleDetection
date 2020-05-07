import os
import tensorflow as tf
from datasets.utils import dataset_utils

slim = tf.contrib.slim

class VehicleTFRecords(dataset_utils.TFRecordsReaderBase):
    """
    汽车数据集读取类
    """
    def __init__(self,param):
        '''
        param:命名字典
        '''
        self.param=param

    def get_data(self,train_or_test,dataset_dir):
        """
        读取tfrecord文件到dataset
        train_or_test：训练数据还是测试数据
        dataset_dir: tfrecord文件的路径
        :return:
        """
        if train_or_test not in ('train','test'):
            raise ValueError('%s 数据名称有误！'%train_or_test)

        if not tf.gfile.Exists(dataset_dir):
            raise ValueError('TFRecord数据目录不存在！')

        # 准备第一个参数：data_sources
        file_pattern = os.path.join(dataset_dir, self.param.FILE_PATTERN % train_or_test)

        # 第二个参数 reader
        reader = tf.TFRecordReader

        # 准备第三个参数decoder
        # 1、反序列化成数据原来的格式
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
        }
        # 2、反序列化成高级的格式
        # 其中bbox框ymin [23] xmin [46],ymax [234] xmax[12]--->[23,46,234,13]
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'shape': slim.tfexample_decoder.Tensor('image/shape'),
            'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
            'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
            'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
        }

        # 准备decoder
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=self.param.SPLITS_TO_SIZES[train_or_test],
            items_to_descriptions=self.param.ITEMS_TO_DESCRIPTIONS,
            num_classes=self.param.NUM_CLASSES,
        )