import os
import  tensorflow as tf
slim=tf.contrib.slim

import sys
sys.path.append("../")

from nets.nets_model import ssd_vgg_300

data_format="NHWC"

ckpt_filepath="../ckpt/fine_tuning/model.ckpt-0"

def main(_):
    #1 定义输入、输出
    #1.1 输入:SSD 模型要求的数据（不是预处理的输入）
    img_input = tf.placeholder(tf.float32, shape=(300, 300, 3))

    # [300,300,3]--->[1,300,300,3]
    img_4d = tf.expand_dims(img_input, 0)

    #1.2 输出:SSD 模型的输出结果
    ssd_class = ssd_vgg_300.SSDNet

    ssd_params = ssd_class.default_params._replace(num_classes=3)

    ssd_net = ssd_class(ssd_params)

    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(img_4d, is_training=False)

    # 开启会话，加载最后保存的模型文件
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 创建saver
        saver = tf.train.Saver()

        # 加载模型
        saver.restore(sess, ckpt_filepath)

        # 2 导出模型过程
        # 路径+模型名字："./model/vehicle/"
        export_path = os.path.join(
            tf.compat.as_bytes("./model/vehicle/"),
            tf.compat.as_bytes(str(1)))

        print("正在导出模型到 %s" % export_path)

        # 通过该函数建立签名映射（协议）
        # tf.saved_model.utils.build_tensor_info(img_input)：填入的参数必须是一个Tensor
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                # 给输入数据起一个别名,用在客户端读取的时候需要指定
                "images": tf.saved_model.utils.build_tensor_info(img_input)
            },
            outputs={
                'predict0': tf.saved_model.utils.build_tensor_info(predictions[0]),
                'predict1': tf.saved_model.utils.build_tensor_info(predictions[1]),
                'predict2': tf.saved_model.utils.build_tensor_info(predictions[2]),
                'predict3': tf.saved_model.utils.build_tensor_info(predictions[3]),
                'predict4': tf.saved_model.utils.build_tensor_info(predictions[4]),
                'predict5': tf.saved_model.utils.build_tensor_info(predictions[5]),
                'local0': tf.saved_model.utils.build_tensor_info(localisations[0]),
                'local1': tf.saved_model.utils.build_tensor_info(localisations[1]),
                'local2': tf.saved_model.utils.build_tensor_info(localisations[2]),
                'local3': tf.saved_model.utils.build_tensor_info(localisations[3]),
                'local4': tf.saved_model.utils.build_tensor_info(localisations[4]),
                'local5': tf.saved_model.utils.build_tensor_info(localisations[5]),
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )

        # 建立builder
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        # 建立元图格式，写入文件
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'detected_model':
                prediction_signature,
            },
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)

        # 保存
        builder.save()

        print("Serving模型结构导出结束")


if __name__ == '__main__':
    tf.app.run()







