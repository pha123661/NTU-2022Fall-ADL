from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.net = nn.LSTM(
            input_size=embeddings.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.clf = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 2, num_class)
            )
        else:
            self.clf = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_class)
            )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, ids, lengths) -> Dict[str, torch.Tensor]:
        embeds = self.embed(ids)
        embeds = pack_padded_sequence(
            embeds, lengths, batch_first=True)
        x, h = self.net(embeds)
        x, lens = pad_packed_sequence(x, batch_first=True)
        x = x.mean(dim=1)  # average over all tokens to get prediction
        logits = self.clf(x)

        return logits


class SeqTagger(SeqClassifier):
    def __init__(self, embeddings: torch.tensor, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool, num_class: int) -> None:
        super().__init__(embeddings, hidden_size,
                         num_layers, dropout, bidirectional, num_class)
        # add a 2-layer CNN
        embed_dim = embeddings.shape[1]
        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, ids, lengths) -> Dict[str, torch.Tensor]:
        embeds = self.embed(ids)

        # feed x into CNN
        # CNN input: (b, h, seq_len)
        embeds = torch.swapaxes(embeds, -1, -2)
        embeds = self.conv1(embeds)
        embeds = self.conv2(embeds)
        embeds = torch.swapaxes(embeds, -1, -2)

        embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        x, h = self.net(embeds)
        x, lens = pad_packed_sequence(x, batch_first=True)
        # x.shape = (b, seq_len, h_size)
        logits = self.clf(x)  # applies classifier on every token

        return logits  # logits.shape = (b, seq_len, n_class)
