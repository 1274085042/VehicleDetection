from preprocessing.processing import ssd_vgg_preprocessing

preprocessing_fn_map = {
    "ssd_vgg_300": ssd_vgg_preprocessing
}

def get_preprocessing(name, is_training=True):
    '''

    Args:
        name: 训练模型名称
        is_training: 是否处于训练阶段

    Returns:预处理函数

    '''
    if name not in preprocessing_fn_map:
        raise ValueError("选择的预处理名称 %s 不在预处理模型库（processing）当中，请提供该模型预处理代码" % name)

    # 返回一个处理的函数，后续再去调用这个函数
    def preprocessing_fn(image, labels, bboxes, out_shape, data_format='NHWC', **kwargs):

        return preprocessing_fn_map[name].preprocess_image(image, labels, bboxes, out_shape, data_format=data_format,
                                                           is_training=is_training, **kwargs)
    return preprocessing_fn