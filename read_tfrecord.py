from Vehicle.datasets.read_dataset import dataset_factory
import tensorflow as tf

slim=tf.contrib.slim

if __name__=='__main__':
    #1、执行VehicleTFRecords类中的get_data
    dataset=dataset_factory.get_dataset('vehicle','train', 'Images/tfrecords/')

    #2、通过provider取出数据
    provider=slim.dataset_data_provider.DatasetDataProvider(dataset,num_readers=3)

    #3、通过get方法获取指定名称的数据
    [image,shape,bbox,label]=provider.get(['image','shape','object/bbox','object/label'])

    print(image,'   ',shape,'   ',bbox,'   ',label)