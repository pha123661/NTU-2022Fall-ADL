import csv
import json
import os
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    try:
        os.makedirs(os.path.dirname(args.pred_file), exist_ok=True)
    except:
        pass

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len, train=False)

    test_loader = DataLoader(dataset, args.batch_size,
                             shuffle=False, collate_fn=dataset.collate_fn, num_workers=6)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    ).to(args.device)
    model.eval()

    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
    # predict dataset
    all_ids = []
    all_preds = []
    with torch.no_grad():
        for x, lengths, ids in test_loader:
            x = x.to(args.device, non_blocking=True)
            all_ids.extend(ids)
            logits = model(x, lengths)
            preds = logits.argmax(dim=1).cpu().tolist()
            preds = [dataset.idx2label(p) for p in preds]
            all_preds.extend(preds)

    # write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('id', 'intent'))
        for data in zip(all_ids, all_preds):
            writer.writerow(data)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./intent.pth",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
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
