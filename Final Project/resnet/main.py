import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict
import os
import numpy as np
import parser
from data import StemDataset
import model
from train import train

TRAIN = "train"
DEV = "test"
SPLITS = [TRAIN, DEV]

if __name__ == '__main__':
    args = parser.arg_parse()
    print(torch.__version__)
    print(torchvision.__version__)
    # create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # fix random seed 
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # dataloader 
    # print('===> Preparing dataloader ...')
    if args.mode == 'train':  # training mode
        data_paths = {split: os.path.join(args.data_dir, split) for split in SPLITS}
        datasets: Dict[str, StemDataset] = {
            split: StemDataset(root=split_path, mode="train")
            for split, split_path in data_paths.items()
        }
        dataloaders: Dict[str, DataLoader] = {
            split: DataLoader(split_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers)
            for split, split_dataset in datasets.items()
        }
        print('# images in trainset:', len(datasets[TRAIN]))
        print('# images in testset:', len(datasets[DEV]))

    elif args.mode == 'test':  # testing mode
        print("===> Start testing...")
        test_dataset = StemDataset(root=args.test_data, mode="test")
        testset_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False,
                                    num_workers=args.num_workers)

    # set up device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # set up model 
    print('===> Preparing model ...')
    my_model = model.get_model().to(device)

    print('===> Start training ...')
    train(my_model, args.num_epoch, dataloaders[TRAIN], dataloaders[DEV], device, args.log_interval)

