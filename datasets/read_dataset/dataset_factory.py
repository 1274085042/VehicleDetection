'''
读取不同数据集
'''
from . import vehicle
from datasets.dataset_config import  VE

dataset_dic={
    'vehicle':vehicle.VehicleTFRecords
}

def get_dataset(name,train_or_test,dataset_dir):
    '''
    name:数据集名字
    train_or_test:是训练还是测试数据集
    dataset_dir:tfrecord文件目录
    '''
    if name not in dataset_dic:
        raise ValueError("%s 数据集不支持读取！"%name)

    return  dataset_dic[name](VE).get_data(train_or_test,dataset_dir)
