from nets.nets_model import ssd_vgg_300

network_obj={
    'ssd_vgg_300':ssd_vgg_300.SSDNet
}

#训练网络名称
def get_network(network_name):
    '''

    Args:
        network_name:网络模型的名称

    Returns:网络

    '''
    return network_obj[network_name]