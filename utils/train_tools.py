import os
from pprint import pprint

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from deployment import model_deploy
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC'


# 普通工具
def reshape_list(l, shape=None):
    """将数据转换成列表
    Args:
      l: 嵌套的列表
      shape: 1D or 2D shape.
    Return
      列表
    """
    r = []
    if shape is None:
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r


# =========================================================================== #
# 训练工具
# =========================================================================== #
def deploy_loss_summary(deploy_config, batch_queue, ssd_net, summaries, batch_shape, FLAGS):
    """
    计算损失，添加损失观察
    为所有设备GPU/CPU赋值一份模型，每份模型都有损失、观察变量的摘要以及
    :param deploy_config: 部署配置文件
    :param batch_queue: 数据队列
    :return: 更新操作,所有设备的模型,
    """

    def clone_fn(batch_queue):
        """
        :param batch_queue:数据队列
        :return:模型输出
        """
        # Dequeue batch.
        b_image, b_gclasses, b_glocalisations, b_gscores = \
            reshape_list(batch_queue.dequeue(), batch_shape)
        print(b_image)

        # Construct SSD network.
        arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                      data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions, localisations, logits, end_points = \
                ssd_net.net(b_image, is_training=True)

        # Add loss function.
        ssd_net.losses(logits, localisations,
                       b_gclasses, b_glocalisations, b_gscores,
                       match_threshold=0.5,
                       negative_ratio=3.,
                       alpha=1.,
                       label_smoothing=0.)
        return end_points

    # 返回naedtuple类型的Clone(outputs, clone_scope, clone_device)
    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])

    # 定义一个默认的第一个命名空间
    first_clone_scope = deploy_config.clone_scope(0)

    # 收集第一个默认设备的命名空间运行OPS（只有一个设备的话是这样的，并且通过命令行参数指定多GPU）
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # 为模型节点添加summary
    end_points = clones[0].outputs
    for end_point in end_points:
        x = end_points[end_point]
        summaries.add(tf.summary.histogram('activations/' + end_point, x))
        summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                        tf.nn.zero_fraction(x)))
    # 为损失添加summary
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
        summaries.add(tf.summary.scalar(loss.op.name, loss))
    for loss in tf.get_collection('EXTRA_LOSSES', first_clone_scope):
        summaries.add(tf.summary.scalar(loss.op.name, loss))

    # 为变量添加summary
    for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))

    return update_ops, first_clone_scope, clones


def configure_learning_rate(flags, num_samples_per_epoch, global_step):
    """配置学习率
    :param flags: 训练运行参数flags
    :param num_samples_per_epoch:每批次样本数
    :param global_step: 步数
    :return:学习率的Tensor
    """
    decay_steps = int(num_samples_per_epoch / flags.batch_size * 2.0)

    if flags.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(flags.learning_rate,
                                          global_step,
                                          decay_steps,
                                          0.94,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif flags.learning_rate_decay_type == 'fixed':
        return tf.constant(flags.learning_rate, name='fixed_learning_rate')
    elif flags.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(flags.learning_rate,
                                         global_step,
                                         decay_steps,
                                         flags.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('学习率迭代类型 [%s] 不存在',
                         flags.learning_rate_decay_type)


def configure_optimizer(flags, learning_rate):
    """
    配置优化器类型
    :param flags: 训练运行参数flags
    :param learning_rate: 学习率Tensor
    :return: 优化器
    """
    if flags.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=0.95,
            epsilon=1.0)
    elif flags.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=0.1)
    elif flags.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1.0)
    elif flags.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=-0.5,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
    elif flags.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=0.9,
            name='Momentum')
    elif flags.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=0.9,
            momentum=0.9,
            epsilon=1.0)
    elif flags.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('优化器类型 [%s] 不存在', flags.optimizer)
    return optimizer


def get_variables_to_train():
    """
    获取需要训练的变量
    :return: 变量列表
    """
    # 返回默认的训练空间的变量
    return tf.trainable_variables()


def get_init_fn(flags):
    """
    导入预训练的模型
    :param flags:
    :return: 一个初始化函数slim.assign_from_checkpoint_fn
    """
    if flags.pre_trained_model is None:
        return None

    if tf.train.latest_checkpoint(flags.train_model_dir):
        tf.logging.info(
            '已经存在模型检查点， 忽略%s'
            % flags.train_model_dir)
        return None

    exclusions = []
    # if flags.checkpoint_exclude_scopes:
    #     exclusions = [scope.strip()
    #                   for scope in flags.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    # if flags.checkpoint_model_scope is not None:
    #     variables_to_restore = \
    #         {var.op.name.replace(flags.model_name,
    #                              flags.checkpoint_model_scope): var
    #          for var in variables_to_restore}

    if tf.gfile.IsDirectory(flags.pre_trained_model):
        pre_trained_model = tf.train.latest_checkpoint(flags.pre_trained_model)
    else:
        pre_trained_model = flags.pre_trained_model
    tf.logging.info('读取预训练的模型 %s 成功，并进行微调' % pre_trained_model)

    return slim.assign_from_checkpoint_fn(
        pre_trained_model,
        variables_to_restore,
        ignore_missing_vars=False)


def get_trainop(optimizer, summaries, clones, global_step, first_clone_scope, update_ops):
    """
    计算所有GPU/CPU设备的平均损失和每个变量的梯度总和
    获取训练的OP以及摘要OP
    :param optimizer: 优化器类型
    :param summaries: 变量摘要
    :param clones: 所有设备上的模型
    :param global_step:全局步数变量
    :return: 训练OP以及 默认第一个设备的summary结果
    """
    # 获取要优化的变量
    variables_to_train = get_variables_to_train()

    # 计算梯度和损失，model_deploy将此操作封装
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # 添加损失到summary
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # 更新梯度
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    # tf.group()将多个tensor或者op合在一起，然后进行run，返回的是一个op

    update_op = tf.group(*update_ops)

    # 将更新梯度等操作，添加到train_op中
    train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                      name='train_op')

    # 最后获取first_clone_scope当中的所有摘要
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    # 合并summary
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    return train_tensor, summary_op


def print_configuration(ssd_params, data_sources):
    """打印训练配置文件
    """
    def print_config(stream=None):
        print('\n# =========================================================================== #', file=stream)
        print('# SSD 网络参数:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(dict(ssd_params._asdict()), stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# 训练数据dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        data_files = parallel_reader.get_data_files(data_sources)
        pprint(sorted(data_files), stream=stream)
        print('', file=stream)

    print_config(None)
