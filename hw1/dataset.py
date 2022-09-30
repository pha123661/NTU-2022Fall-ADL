from lib2to3.pgen2 import token
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        train=False,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent,
                           idx in self.label_mapping.items()}
        self.max_len = max_len
        self.Train = train

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Tuple:
        # example = {
        #     'text': 'can you tell me the best place for cajun shrimp in tampa',
        #     'intent': 'restaurant_suggestion',
        #     'id': 'train-4824',
        # }
        text = [ex['text'].split() for ex in samples]
        lengths = torch.tensor([len(t) for t in text])

        text = self.vocab.encode_batch(text, to_len=self.max_len)
        text = torch.tensor(text)

        if self.Train:
            labels = [self.label2idx(ex['intent']) for ex in samples]
            labels = torch.tensor(labels, dtype=torch.long)

            sorted_idx = torch.argsort(lengths, descending=True)
            return text[sorted_idx], lengths[sorted_idx], labels[sorted_idx]
        else:
            ids = np.array([ex['id'] for ex in samples])
            sorted_idx = torch.argsort(lengths, descending=True)
            return text[sorted_idx], lengths[sorted_idx], ids[sorted_idx.cpu().numpy()]

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # example = {
        #     'tokens': ['can', 'i', 'sit', 'on', 'the', 'patio'],
        #     'tags': ['O', 'O', 'O', 'O', 'O', 'O'],
        #     'id': 'train-196'
        # }
        tokenses = [ex['tokens'] for ex in samples]
        lengths = torch.tensor([len(tokens) for tokens in tokenses])

        assert all(l <= self.max_len for l in lengths), max(lengths)

        tokenses = self.vocab.encode_batch(tokenses, to_len=self.max_len)
        tokenses = torch.tensor(tokenses)

        if self.Train:
            tags: List[List[int]] = list()
            for example in samples:
                tags.append([self.label2idx(tag) for tag in example['tags']])
            tags = pad_to_len(tags, to_len=self.max_len, padding=0)
            tags = torch.tensor(tags, dtype=torch.long)
            # masks = tokenses.gt(0).type(torch.float)

            sorted_idx = torch.argsort(lengths, descending=True)
            return (
                tokenses[sorted_idx],
                lengths[sorted_idx],
                tags[sorted_idx]
            )
        else:
            ids = np.array([ex['id'] for ex in samples])

            sorted_idx = torch.argsort(lengths, descending=True)
            return (
                tokenses[sorted_idx],
                lengths[sorted_idx],
                ids[sorted_idx.cpu().numpy()]
            )
