import argparse
import pathlib

import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument("--test_jsonl", type=str, required=True)
parser.add_argument("--gen_txt", type=str,
                    default="./generated_predictions.txt")
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()

test_jsonl = []
out_jsonl = []
with jsonlines.open(args.test_jsonl) as f:
    for obj in f:
        test_jsonl.append(obj)

with open(args.gen_txt) as f:
    for entry, title in zip(test_jsonl, f.readlines()):
        out_jsonl.append({
            'title': title.strip(),
            'id': entry['id'],
        })

args.output.parent.mkdir(exist_ok=True, parents=True)
with args.output.open('w', encoding='utf8') as f:
    writer = jsonlines.Writer(f)
    writer.write_all(out_jsonl)
