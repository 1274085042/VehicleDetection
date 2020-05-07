import tensorflow as tf

from datasets.read_dataset import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from utils import train_tools

slim = tf.contrib.slim

DATA_FORMAT = "NHWC"

"""
训练相关文件和目录
PRE_TRAINED_PATH=./ckpt/pre_trained/ssd_300_vgg.ckpt
TRAIN_MODEL_DIR=./ckpt/fine_tuning/
DATASET_DIR=./IMAGE/tfrecords/commodity_tfrecords/

参数设置：
每批次训练样本数：32或者更小
网络误差函数惩罚项值：0.0005
学习率：0.001
终止学习率：0.0001
优化器选择:adam优化器
模型名称：ssd_vgg_300

"""

# 设备的命令行参数配置
tf.app.flags.DEFINE_integer('num_clones', 1, "可用设备的GPU数量")
tf.app.flags.DEFINE_boolean('clone_on_cpu', False, "是否只在CPU上运行")

# 数据集相关命令行参数设置
tf.app.flags.DEFINE_string('dataset_dir', ' ', "训练数据集目录")
tf.app.flags.DEFINE_string('dataset_name', 'vehicle', '数据集名称参数')
tf.app.flags.DEFINE_string('train_or_test', 'train', '是否是训练集还是测试集')

# 网络相关配置
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, '网络误差函数惩罚项值，越小越防止过拟合.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_vgg_300', '用于训练的网络模型名称')
# 设备的命令行参数配置
tf.app.flags.DEFINE_integer('batch_size', 8, "每批次获取样本数")

# 训练学习率相关参数
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop', '优化器种类 可选"adadelta", "adagrad", "adam","ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'exponential','学习率迭代种类  "fixed", "exponential", "polynomial"')
tf.app.flags.DEFINE_float(
    'learning_rate', 0.01, '模型初始学习率')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001, '模型训练迭代后的终止学习率')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps', None, '训练的最大步数')
tf.app.flags.DEFINE_string(
    'train_model_dir', ' ', '训练输出的模型目录')
# pre-trained模型路径.
tf.app.flags.DEFINE_string(
    'pre_trained_model', None, '用于fine-tune的已训练好的基础参数文件位置')

FLAGS = tf.app.flags.FLAGS


def main(_):

    if not FLAGS.dataset_dir:
        raise ValueError('必须指定一个TFRecords的数据集目录')

        # 设置打印级别
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():
        # 在默认的图当中进行编写训练逻辑
        #1、DeploymentConfig
        # 需要在训练之前配置所有的设备信息
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,  # GPU设备数量
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=0,
            num_replicas=1,  # 1台计算机
            num_ps_tasks=0
        )

        # 2、生成一个模型实例
        # get_network()返回一个类
        ssd_class = nets_factory.get_network(FLAGS.model_name)

        # 获取默认网络参数
        ssd_params = ssd_class.default_params._replace(num_classes=3)

        # 初始化网络init函数
        # 生成一个模型实例
        ssd_net = ssd_class(ssd_params)

        # 3、定义一个全局步长参数（网络训练都会这么去进行配置）
        # 使用指定设备 tf.device
        with tf.device(deploy_config.variables_device()):
            global_step = tf.train.create_global_step()

        # 4、获取图片队列数据和default anchors
        # 图片有什么？image, shape, bbox, label
        # image会做一些数据增强，大小变换
        # 直接训练？需要将anchor bbox进行样本标记正负样本，目的使的GT目标样本的数量与default bboxes数量一致

        # 4.1 获取样本和默认框
        # 4.1.1 通过数据工厂取出规范信息
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.train_or_test, FLAGS.dataset_dir)

        # 4.1.2 获取网络计算的default anchors结果
        # 获取形状，用于输入到anchors函数参数当中
        ssd_shape = ssd_net.params.img_shape
        # 获取anchors, SSD网络当中6层的所有计算出来的默认候选框
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # 4.1.3 打印网络相关参数
        train_tools.print_configuration(ssd_params, dataset.data_sources)


        # 4.1.4 通过deploy_config.inputs_device()指定输入数据的设备
        # 4.1.5 slim.dataset_data_provider.DatasetDataProvider通过GET方法获取单个样本

        with tf.device(deploy_config.inputs_device()):

            # 给当前操作取一个作用域名称
            with tf.name_scope(FLAGS.model_name + '_data_provider'):

                # slim.dataset_data_provider.DatasetDataProvider通过GET方法获取数据
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=3,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size,
                    shuffle=True
                )
                # 通过get获取数据
                # 真正获取参数
                [image, shape, glabels, gbboxes] = provider.get(['image', 'shape', 'object/label', 'object/bbox'])

                # 4.2 数据预处理
                # （1）对数据进行预处理
                # （2）对获取出来的groundtruth标签和bbox。进行编码
                # （3）获取的单个样本数据，要进行批处理以及返回队列
                # 直接进行数据预处理
                # image [?, ?, 3]---->[300， 300， 3]
                image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.model_name, is_training=True)     # 获取预处理函数
                image, glabels, gbboxes = image_preprocessing_fn(image, glabels, gbboxes,         #该函数返回变换后的image，标签和变换后的bboxes
                                                                 out_shape=ssd_shape,
                                                                 data_format=DATA_FORMAT)

                # 对原始anchor bboxes进行正负样本标记
                # 得到目标值，编码之后，返回？
                # 训练？  预测值类别，物体位置，物体类别概率，目标值
                # 8732 anchor,   得到8732个与GT 对应的标记的anchor
                # gclasses:目标类别
                # glocalisations：目标的位置
                # gscores：是否是正负样本
                gclasses, glocalisations, gscores = ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)

                # print(gclasses, glocalisations)


                # 4.3 批处理以及队列处理
                # tensor_list:tensor列表 [tensor, tensor, ]
                # tf.train.batch(tensor_list, batch_size, num_threads, capacity)
                # [Tensor, [6], [6], [6]]  嵌套的列表要转换成单列表形式
                r = tf.train.batch(train_tools.reshape_list([image, gclasses, glocalisations, gscores]),
                                   batch_size=FLAGS.batch_size,
                                   num_threads=4,
                                   capacity=5 * FLAGS.batch_size)
                # r应该是一个19个Tensor组成的一个列表
                # print(r)
                # 批处理数据放入队列
                # 1个r:批处理的样本， 5个设备，5个r, 5组32张图片
                # 队列的目的是为了不同设备需求
                batch_queue = slim.prefetch_queue.prefetch_queue(r,
                                                                 capacity=deploy_config.num_clones)

        # 5、复制模型到不同的GPU设备，以及损失、变量的观察
        # train_tools.deploy_loss_summary(deploy_config,batch_queue,ssd_net,summaries,batch_shape,FLAGS)
        # summarties:摘要
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # batch_shape：解析这个batch_queue的大小，指的是获取的一个默认队列大小，指上面r的大小

        batch_shape = [1] + 3 * [len(ssd_anchors)]

        update_ops, first_clone_scope, clones = train_tools.deploy_loss_summary(deploy_config,
                                                                                batch_queue,
                                                                                ssd_net,
                                                                                summaries,
                                                                                batch_shape,
                                                                                FLAGS)

        # 6、定义学习率以及优化器
        # 学习率：0.001
        # 终止学习率：0.0001
        # 优化器选择:adam优化器
        # learning_rate = tf_utils.configure_learning_rate(FLAGS, num_samples, global_step)
        # FLAGS：将会用到学习率设置相关参数
        # global_step: 全局步数
        # optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
        # learning_rate: 学习率
        with tf.device(deploy_config.optimizer_device()):

            # 定义学习率和优化器
            learning_rate = train_tools.configure_learning_rate(FLAGS,
                                                                dataset.num_samples,
                                                                global_step)
            optimizer = train_tools.configure_optimizer(FLAGS, learning_rate)

            # 观察学习率的变化情况，添加到summaries
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        # 7、计算所有设备的平均损失以及每个变量的梯度总和
        train_op, summaries_op = train_tools.get_trainop(optimizer,
                                                         summaries,
                                                         clones,
                                                         global_step,
                                                         first_clone_scope, update_ops)


        # 配置config以及saver

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)   #使用80%的显存
        # config = tf.ConfigProto(log_device_placement=False,  # 如果打印会有许多变量的设备信息出现
        #                         gpu_options=gpu_options)
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False,  # 打印每个变量所在的设备信息
                                gpu_options=gpu_options)
        # saver
        saver = tf.train.Saver(max_to_keep=5,  # 默认保留最近几个模型文件
                               keep_checkpoint_every_n_hours=1.0,  #隔多少个小时保存一次
                               write_version=2,
                               pad_step_number=False)

        # 7、进行训练
        slim.learning.train(
            train_op,  # 训练优化器tensor
            logdir=FLAGS.train_model_dir,  # 模型存储目录
            master='',
            is_chief=True,
            init_fn=train_tools.get_init_fn(FLAGS),  # 初始化参数的逻辑，预训练模型的读取和微调模型判断
            summary_op=summaries_op,  # 摘要
            number_of_steps=FLAGS.max_number_of_steps,  # 最大步数
            log_every_n_steps=10,  # 打印频率
            save_summaries_secs=60,  # 保存摘要频率
            saver=saver,  # 保存模型参数
            save_interval_secs=600,  # 保存模型间隔
            session_config=config,  # 会话参数配置
            sync_optimizer=None)


if __name__ == '__main__':
    tf.app.run()


