import csv
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
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
        data, vocab, tag2idx, args.max_len, train=False)
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
        RNN_block='GRU',
    ).to(args.device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
    model.eval()

    all_ids = []
    all_preds = []
    all_lengths = []

    with torch.no_grad():
        for x, lengths, ids in test_loader:
            x = x.to(args.device, non_blocking=True)
            all_ids.extend(ids)
            logits = model(x, lengths)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_lengths.extend(lengths.cpu().tolist())

    # write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('id', 'tags'))
        for id, pred, length in zip(all_ids, all_preds, all_lengths):
            writer.writerow((id, " ".join(dataset.idx2label(p)
                            for p in pred[:length])))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json",
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
        default="./slot.pth",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
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
