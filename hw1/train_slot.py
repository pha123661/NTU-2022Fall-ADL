import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from seqeval.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(
            split_data, vocab, tag2idx, args.max_len, train=True)
        for split, split_data in data.items()
    }
    train_loader = DataLoader(
        datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=datasets['train'].collate_fn)
    valid_loader = DataLoader(
        datasets['eval'], batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=datasets['eval'].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(tag2idx),
    ).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=args.lr, epochs=args.num_epoch, steps_per_epoch=len(train_loader), pct_start=0.5)

    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = -1
    for epoch in range(1, args.num_epoch + 1):
        for text, lengths, tags in tqdm(train_loader):
            text, tags = text.to(args.device, non_blocking=True), tags.to(
                args.device, non_blocking=True)
            # logits.shape = (b, seq_len, n_class)
            logits = model(text, lengths)
            # logits.shape = (b, n_class, seq_len)
            # logits = logits.permute(0, 2, 1)
            tags = tags[:, :logits.shape[1]]
            # get index of actual tokens (without padding)
            idx = model.get_token_idx(lengths)
            # logits[idx].shape = (sum(lengths), n_class)
            loss = loss_fn(logits[idx], tags[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        model.eval()
        va_joint_acc = 0
        with torch.no_grad():
            for text, lengths, tags in tqdm(valid_loader):
                text, tags = text.to(args.device, non_blocking=True), tags.to(
                    args.device, non_blocking=True)
                logits = model(text, lengths)
                preds = logits.argmax(dim=-1)
                tags = tags[:, :logits.shape[1]]
                # print("#########")
                # print(preds)
                # print(tags)
                va_joint_acc += accuracy_score(
                    tags.detach().cpu().tolist(), preds.detach().cpu().tolist())
        model.train()
        va_joint_acc /= len(valid_loader)
        print(f'Epoch {epoch}: valid acc: {va_joint_acc}')
        if va_joint_acc > best_acc:
            best_acc = va_joint_acc
            torch.save(model.state_dict(),
                       args.ckpt_dir / 'best_model.pth')
            torch.save(optimizer.state_dict(),
                       args.ckpt_dir / 'best_optim.pth')
            print('new model saved!')
    print('best acc:', best_acc)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
