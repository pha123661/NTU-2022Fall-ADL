import os 
import random
random.seed(1)

tree_dataset_path = os.path.join('data', 'NTU-stem-dataset')

if __name__ == '__main__':
    class_ = filter(os.path.isdir, [os.path.join(tree_dataset_path, child) for child in os.listdir(tree_dataset_path)])

    for class_id, d in enumerate(class_):
        image_file_list = os.listdir(d)
        class_label = os.path.basename(os.path.dirname(d+'/'))

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
                split_ = 'val'
            else:
                split_ = 'test'
            os.rename(image_dir, os.path.join(d, '{}_{}_{}_{}.jpg'.format(split_, class_id, class_label, cnt)))
