import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='ADL final resnet fin-tune')

    # Datasets parameters
    parser.add_argument('--num_workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")

    # training parameters
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--data_dir', default="../data/", type=str)
    parser.add_argument('--model', default="clip", type=str)

    parser.add_argument('--num_epoch', default=50, type=int,
                        help="num of training iterations")
    parser.add_argument('--val_epoch', default=10, type=int,
                        help="num of validation iterations")
    parser.add_argument('--train_batch', default=64, type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=64, type=int,
                        help="test batch size")
    parser.add_argument('--lr', default=0.0007, type=float,
                        help="initial learning rate")
    parser.add_argument('--lr_scheduler', default=False, type=bool,
                        help="schedule or not")
    parser.add_argument('--weight_decay', default=0.0003, type=float,
                        help="initial weight decay")
    parser.add_argument('--log_interval', default=20, type=int,
                        help="save model in log interval epochs")
    parser.add_argument('--save_interval', default=5, type=int,
                        help="save checkpoint")

    # resume trained model
    parser.add_argument('--resume', type=str, default='',
                        help="path to the trained model")
    # inference model
    parser.add_argument('--test', type=str, default='',
                        help="path to the trained model")
    parser.add_argument('--prediction', type=str, default='prediction.csv',
                        help="path to the saved csv file")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=0)

    args = parser.parse_args()

    return args
