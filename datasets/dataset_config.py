'''
图片数据集转换成TFRecord配置文件
'''
# 原始图片的XML和JPG的文件名
DIRECTORY_ANNOTATIONS = "anno/"

DIRECTORY_IMAGES = "images/"

# 每个TFRecords文件的example个数
SAMPLES_PER_FILES = 500

# VOC 2007物体类别
# VOC_LABELS = {
#     'none': (0, 'Background'),
#     'aeroplane': (1, 'Vehicle'),
#     'bicycle': (2, 'Vehicle'),
#     'bird': (3, 'Animal'),
#     'boat': (4, 'Vehicle'),
#     'bottle': (5, 'Indoor'),
#     'bus': (6, 'Vehicle'),
#     'car': (7, 'Vehicle'),
#     'cat': (8, 'Animal'),
#     'chair': (9, 'Indoor'),
#     'cow': (10, 'Animal'),
#     'diningtable': (11, 'Indoor'),
#     'dog': (12, 'Animal'),
#     'horse': (13, 'Animal'),
#     'motorbike': (14, 'Vehicle'),
#     'person': (15, 'Person'),
#     'pottedplant': (16, 'Indoor'),
#     'sheep': (17, 'Animal'),
#     'sofa': (18, 'Indoor'),
#     'train': (19, 'Vehicle'),
#     'tvmonitor': (20, 'Indoor'),
# }

VEHICLE_LABELS={
    'none': (0, 'Background'),
    'Audi': (1, 'Audi'),
    'BMW': (2, 'BMW'),
}

'''
数据集读取配置
'''
from collections import namedtuple
DataSetParams= namedtuple('DataSetParams',['FILE_PATTERN','NUM_CLASSES','SPLITS_TO_SIZES','ITEMS_TO_DESCRIPTIONS'])

#TFRecord格式的参数
VE=DataSetParams(FILE_PATTERN='%s_*.tfrecord',
                 NUM_CLASSES=2,
                 SPLITS_TO_SIZES={
                     'train':466,
                     'test':0,
                                },
                 ITEMS_TO_DESCRIPTIONS={
                     'image':"图片",
                     "shape":"图片形状",
                     "object/bbox":"若干目标的bbox组成的列表",
                     "object/label":"若干目标的标签编号"
                                        }
                 )



















