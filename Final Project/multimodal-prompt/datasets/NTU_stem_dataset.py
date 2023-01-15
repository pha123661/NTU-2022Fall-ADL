import os
import pickle
import random
from scipy.io import loadmat
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class NTUStem(DatasetBase):
    dataset_dir = "NTU_stem_dataset"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        train, val, test = self.read_data()

        super().__init__(train_x=train, val=val, test=test)
        
    def read_data(self):
        class_ = filter(os.path.isdir, [os.path.join(self.dataset_dir, child) for child in os.listdir(self.dataset_dir)])

        train = []
        val = []
        test = []
        for d in class_:
            image_f_list = os.listdir(d)
            for image_f in image_f_list:
                split_, class_id, class_label, image_id = image_f.split('_')
                if split_ == 'train':
                    train.append([image_f, class_id, class_label])
                elif split_ == 'val':
                    val.append([image_f, class_id, class_label])
                else:
                    test.append([image_f, class_id, class_label])
        
        return train, val, test
