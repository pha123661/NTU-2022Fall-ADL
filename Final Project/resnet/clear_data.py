import os
import random
import shutil

random.seed(1)

root = '../data'
tree_dataset_path = os.path.join(root, 'NTU-stem-dataset')

if not os.path.exists(os.path.join(root, 'train')):
    os.mkdir(os.path.join(root, 'train'))
if not os.path.exists(os.path.join(root, 'test')):
    os.mkdir(os.path.join(root, 'test'))
if not os.path.exists(os.path.join(root, 'eval')):
    os.mkdir(os.path.join(root, 'eval'))

if __name__ == '__main__':
    class_ = filter(os.path.isdir, [os.path.join(tree_dataset_path, child) for child in os.listdir(tree_dataset_path)])

    train_list = list()
    test_list = list()
    valid_list = list()

    for class_id, d in enumerate(class_):
        image_file_list = os.listdir(d)

        class_label = os.path.basename(os.path.dirname(str(class_id)+'/'))

        random.shuffle(image_file_list)
        image_num = len(image_file_list)

        if image_num < 24:
            train_num = 8
        else:
            train_num = 16

        val_num = (image_num - train_num) // 2 + train_num
        for cnt, image in enumerate(image_file_list):
            image_dir = os.path.join(d, image)
            if cnt < train_num:
                split_ = 'train'
            elif cnt < val_num:
                split_ = 'eval'
            else:
                split_ = 'test'
            # if image.find()
            shutil.copyfile(image_dir, os.path.join(root, '{}/{}_{}.jpg'.format(split_, class_id, cnt)))