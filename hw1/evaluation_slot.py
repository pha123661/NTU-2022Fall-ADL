import csv
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from seqeval.metrics import accuracy_score, classification_report
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader

from dataset import SeqTaggingClsDataset
from model import SeqTagger


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(
        data, vocab, tag2idx, args.max_len, train=True)
    test_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(tag2idx),
    ).to(args.device)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    all_preds = []
    all_tags = []

    with torch.no_grad():
        for x, lengths, tags in test_loader:
            x = x.to(args.device, non_blocking=True)
            logits = model(x, lengths)
            preds = logits.argmax(dim=-1)
            for pp, ll, tt in zip(preds, lengths, tags):
                all_preds.append([dataset.idx2label(p)
                                 for p in pp[:ll].cpu().tolist()])
                all_tags.append([dataset.idx2label(t)
                                for t in tt[:ll].cpu().tolist()])

    def joint_acc(y_true, y_pred):
        cnt = 0
        for gt, pd in zip(y_true, y_pred):
            if gt == pd:
                cnt += 1
        return cnt / len(y_true)

    print(f"joint acc: {joint_acc(all_tags, all_preds):.15f}")
    print(f"token acc: {accuracy_score(all_tags, all_preds)}")
    print(classification_report(all_tags, all_preds, scheme=IOB2, mode='strict'))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the validation file.",
        default="./data/slot/eval.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/best_model.pth",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
