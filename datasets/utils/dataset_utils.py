import tensorflow as tf

def int64_feature(value):
    """包裹int64型特征到Example
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """包裹浮点型特征到Example
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """包裹字节类型特征到Example
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

'''
在不同数据集当中会涉及到不同的属性，我们可以作为一个初始化的参数。并且添加一个读取的方法，读取的逻辑当中涉及数据集目录和训练还是测试集两个参数。
'''
class TFRecordsReaderBase(object):
    '''
    数据集读取基类
    '''
    def __init__(self,params):
        '''
        params:Vehicle或者其它数据集的参数
        '''
        self.params=params

    def get_data(self,train_or_test,dataset_dir):
        '''
        train_or_test:训练数据还是测试数据
        dataset_dir:数据集存放目录
        '''
        return None


