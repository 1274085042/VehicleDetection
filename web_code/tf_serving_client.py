import tensorflow as tf
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
from copy import deepcopy

from image_preprocessing import img_preprocessing
from image_preprocessing import convert_image_to_array
from image_tag_result import tag_picture


def make_prediction(img):
    """
    拿到用户的数据，进行预测标记返回
    :param img: 图片数据
    :return:
    """
    # 1、对img进行格式转换，以及预处理操作
    # [None, None, 3]--->[300, 300, 3]
    image_array = convert_image_to_array(img)

    image = img_preprocessing(image_array)

    # 2、建立客户端请求，拿到结果
    # - grpc建立连接
    # - serving建立通道，通道发送请求
    # - 封装一个请求
    # - 请求的模型名称
    # - 模型的签名
    # - 图片数据
    # - 获取结果
    with grpc.insecure_channel('10.37.14.194:8500') as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        # 封装请求
        request = predict_pb2.PredictRequest()

        request.model_spec.name = 'vehicle'
        request.model_spec.signature_name = 'detected_model'
        request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(image, shape=[300, 300, 3]))

        # 获取结果
        result = stub.Predict(request)

        # 结果的解析
        # result.outputs['predict0'] 是一个TensorInfo格式
        # 使用tf.convert_to_tensor进行转换
        predictions = [
            tf.convert_to_tensor(result.outputs['predict0']),
            tf.convert_to_tensor(result.outputs['predict1']),
            tf.convert_to_tensor(result.outputs['predict2']),
            tf.convert_to_tensor(result.outputs['predict3']),
            tf.convert_to_tensor(result.outputs['predict4']),
            tf.convert_to_tensor(result.outputs['predict5']),
        ]

        localisations = [
            tf.convert_to_tensor(result.outputs['local0']),
            tf.convert_to_tensor(result.outputs['local1']),
            tf.convert_to_tensor(result.outputs['local2']),
            tf.convert_to_tensor(result.outputs['local3']),
            tf.convert_to_tensor(result.outputs['local4']),
            tf.convert_to_tensor(result.outputs['local5']),
        ]

    print(predictions, localisations)
    # 将Tensor获取值的结果
    with tf.Session() as sess:

        p, l = sess.run([predictions, localisations])

    # 3、对结果进行后期处理和标签标记
    # 将预测结果以图示方式标记在原图片中

    return tag_picture(image_array, p, l)


if __name__ == '__main__':
    # 获取图片输入
    with open("../test/0.jpg", 'rb') as f:
        data = f.read()
        res = make_prediction(data)
        print(res)
