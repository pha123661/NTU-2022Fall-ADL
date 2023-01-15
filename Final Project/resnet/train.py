import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import parser
from tqdm import tqdm, trange
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path, _use_new_zipfile_serialization=False)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def train(model, num_epoch, train_loader, test_loader, device, log_interval):
    args = parser.arg_parse()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, last_epoch=-1)

    if args.resume != '':
        load_checkpoint(args.resume, model, optimizer)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    train_loss_list = list()
    # test_loss_list = list()
    train_acc_list = list()
    # test_acc_list = list()

    epoch_pbar = trange(num_epoch, desc="Epoch")

    for epoch in epoch_pbar:
        model.train()
        correct = 0.0
        train_loss = 0.0
        # train_acc = 0
        progress = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(progress):
            # data = preprocess(data).unsqueeze(0).to(device)
            # target = target.to(device)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model.forward(data)
            # output = model(data)
            loss = criterion(output, target)

            train_loss += loss.item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss.backward()
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step()

            acc = 100. * correct / len(train_loader.dataset)
            # train_acc += correct

            epoch_pbar.set_description(f'Epoch[{epoch + 1}/{args.num_epoch}]')
            epoch_pbar.set_postfix(loss=train_loss / (batch_idx + 1), accuracy=acc)

        train_loss_list.append(train_loss / len(train_loader.dataset))
        train_acc_list.append(100. * correct / len(train_loader.dataset))

        if epoch + 1 == num_epoch:
            test_loss, test_acc = test(model, test_loader, device)  # Evaluate at the end of each epoch
            print(test_acc)
        # test_loss_list.append(test_loss)
        # test_acc_list.append(test_acc)

        if acc > best_acc:
            best_acc = acc
            print("Saving best model... Best Acc is: {:.3f}".format(best_acc))
            save_checkpoint(os.path.join(args.save_dir, 'model_best_{}.pth'.format(epoch)), model, optimizer)

    x1 = range(0, num_epoch)
    f1 = plt.figure()
    plt.plot(x1, train_loss_list, "o-", label="train loss")
    # plt.plot(x1, test_loss_list, "o-", label="valid loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(args.save_dir, 'plot/plot_loss.png'))

    f2 = plt.figure()
    plt.plot(x1, train_acc_list, "o-", label="train acc.")
    # plt.plot(x1, test_acc_list, "o-", label="test acc.")
    plt.legend(loc="best")
    plt.savefig(os.path.join(args.save_dir, 'plot/plot_accuracy.png'))
    plt.show()


def test(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # This will free the GPU memory used for back-prop
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with autocast():
                output = model.forward(data)
            # output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return test_loss, acc
