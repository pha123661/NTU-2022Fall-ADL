import json
import pickle
from argparse import ArgumentParser, Namespace
from cProfile import label
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx,
                             args.max_len, train=True)
        for split, split_data in data.items()
    }
    train_loader = DataLoader(
        datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=datasets['train'].collate_fn)
    valid_loader = DataLoader(
        datasets['eval'], batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=datasets['eval'].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(intent2idx),
    ).to(args.device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1
    for epoch in range(1, args.num_epoch + 1):
        # Training loop - iterate over train dataloader and update model weights
        train_losses = []
        for x, lengths, labels in tqdm(train_loader):
            x, labels = x.to(args.device, non_blocking=True), labels.to(
                args.device, non_blocking=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        print(
            f'Epoch {epoch}: train loss: {sum(train_losses)/len(train_losses)}')
        # TODO: Evaluation loop - calculate accuracy and save model weights
        va_accs = []
        va_losses = []
        model.eval()
        with torch.no_grad():
            for x, lengths, labels in tqdm(valid_loader):
                x, labels = x.to(args.device), labels.to(args.device)
                logits = model(x, lengths)
                loss = loss_fn(logits, labels)
                pred = logits.argmax(dim=1)

                va_accs.append(accuracy_score(
                    labels.detach().cpu().numpy(), pred.detach().cpu().numpy()))
                va_losses.append(loss.item())
        model.train()
        va_acc = sum(va_accs) / len(va_accs)
        print(
            f'Epoch {epoch}: valid acc: {va_acc} valid loss: {sum(train_losses)/len(train_losses)}')
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), args.ckpt_dir / 'best_model.pth')
            torch.save(optimizer.state_dict(),
                       args.ckpt_dir / 'best_optim.pth')
            print('saved model! acc =', va_acc)

    print('best_acc:', best_acc)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
