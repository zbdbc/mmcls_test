import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class MyFilelist(BaseDataset):
    # 需要指定自己的类别
    CLASSES = [
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。',
        '我懒得写名字了，有102个。。。'
    ]

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []  # 传入每个标签
        with open(self.ann_file) as f:  # 打开标注文件
            samples = [x.strip().split(' ') for x in f.readlines()]     # 以空格为分隔符
            for filename, gt_label in samples:
                # 数据的前缀 不需要改 就是数据存放位置  如：E：//Pytorch......
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}   # 名字信息
                info['gt_label'] = np.array(gt_label, dtype=np.int64)   # 标签信息
                data_infos.append(info)
            return data_infos
