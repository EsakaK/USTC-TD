import json
import os
from src.utils import resize_yuv


def generate(config: dict):
    """
    Args:
        config: hevc_sequence_config
    Returns:
        None
    cubic downsample according to config file
    """
    for class_name in config:
        if config[class_name]['test'] == 0:
            continue
        class_config = config[class_name]
        base_path = class_config['base_path']
        new_base_path = class_config['gt_path']
        size_dict = dict(x1=class_config['x1'], x2=class_config['x2'], x1_5=class_config['x1_5'], x3 = class_config['x3'], x4=class_config['x4'])
        for seq in class_config['sequences']:
            yuv_path = os.path.join(base_path, seq) + '.yuv'
            for ds_size in size_dict:
                seq_dir = os.path.join(new_base_path, seq)
                if not os.path.exists(seq_dir):
                    os.makedirs(seq_dir)
                new_yuv_path = os.path.join(seq_dir, ds_size)+'.yuv'
                size = (class_config[ds_size]['height'], class_config[ds_size]['width'])

                flag = True
                if ds_size == 'x1':
                    flag = False
                ori_size = (class_config['x1']['height'], class_config['x1']['width'])
                resize_yuv(yuv_path, new_yuv_path, size, ori_size, frame_num=96, flag = flag)


if __name__ == '__main__':
    with open('sclable_sequnce.json', 'r') as f:
        config = json.load(f)
    generate(config)
