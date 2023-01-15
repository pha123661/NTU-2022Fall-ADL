import os

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum


@DATASET_REGISTRY.register()
class UCF101(DatasetBase):
    dataset_dir = "NTU-stem-dataset-split"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        train, val, test = self.read_data()

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self):
        class_ = filter(os.path.isdir, [os.path.join(
            self.dataset_dir, child) for child in os.listdir(self.dataset_dir)])

        train = []
        val = []
        test = []
        for d in class_:
            image_f_list = os.listdir(d)
            for image_f in image_f_list:
                im_s = image_f.split('_')
                split_, class_id, class_label, image_id = im_s[0], im_s[1], '_'.join(
                    im_s[2:-2]), im_s[-1]
                image_p = os.path.join(d, image_f)
                data = Datum(impath=image_p, label=int(
                    class_id), classname=class_label)
                if split_ == 'train':
                    train.append(data)
                elif split_ == 'val':
                    val.append(data)
                else:
                    test.append(data)

        return train, val, test
