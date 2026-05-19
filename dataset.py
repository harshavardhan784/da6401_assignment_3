# dataset.py
"""
dataset.py — Multi30k Dataset Loader
DA6401 Assignment 3: "Attention Is All You Need"
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from functools import partial

import spacy
from datasets import load_dataset


#  Special tokens & their fixed indices ─
unk_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"

unk_idx = 0
pad_idx = 1
sos_idx = 2
eos_idx = 3

specials = [unk_token, pad_token, sos_token, eos_token]


#  VOCABULARY

class Vocab:
    """Simple token ↔ index vocabulary."""

    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, unk_token)

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, unk_idx)


#  DATASET


class Multi30kDataset(Dataset):
    """
    Multi30k German→English dataset.

    Args:
        split      : 'train', 'validation', or 'test'.
        src_vocab  : Pre-built source Vocab (pass from training split to val/test).
        tgt_vocab  : Pre-built target Vocab (pass from training split to val/test).
        min_freq   : Minimum token frequency to be included in vocab (training only).
    """

    def __init__(
        self,
        split:     str   = 'train',
        src_vocab: Vocab = None,
        tgt_vocab: Vocab = None,
        min_freq:  int   = 1,
    ):
        self.split    = split
        self.min_freq = min_freq

        raw       = load_dataset("bentrevett/multi30k")
        self.data = raw[split]

        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self._build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.src_data, self.tgt_data = self._process_data()

    #  Tokenisers 
    def tokenize_de(self, text: str) -> list:
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> list:
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    #  Vocabulary construction ─
    def _build_vocab(self):
        src_counter = Counter()
        tgt_counter = Counter()

        for example in self.data:
            src_counter.update(self.tokenize_de(example["de"]))
            tgt_counter.update(self.tokenize_en(example["en"]))

        def _build(counter: Counter) -> Vocab:
            filtered = [tok for tok, cnt in counter.items() if cnt >= self.min_freq]
            stoi = {tok: idx + len(specials) for idx, tok in enumerate(filtered)}
            for idx, tok in enumerate(specials):
                stoi[tok] = idx
            return Vocab(stoi)

        return _build(src_counter), _build(tgt_counter)

    #  Numericise data ─
    def _process_data(self):
        src_data, tgt_data = [], []

        for example in self.data:
            src_tokens = self.tokenize_de(example["de"])
            tgt_tokens = self.tokenize_en(example["en"])

            src_ids = (
                [sos_idx]
                + [self.src_vocab.lookup_index(t) for t in src_tokens]
                + [eos_idx]
            )
            tgt_ids = (
                [sos_idx]
                + [self.tgt_vocab.lookup_index(t) for t in tgt_tokens]
                + [eos_idx]
            )
            src_data.append(src_ids)
            tgt_data.append(tgt_ids)

        return src_data, tgt_data

    #  Dataset protocol 
    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> tuple:
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long),
        )

    #  DataLoader factory (call on training split only) 
    def get_dataloaders(
        self,
        batch_size:  int  = 128,
        num_workers: int  = 0,
        pin_memory:  bool = False,
    ) -> tuple:
        """
        Build train / val / test DataLoaders.
        Must be called on the training-split instance so vocabularies
        are constructed from training data only (no data leakage).

        Returns:
            (train_loader, val_loader, test_loader)
        """
        assert self.split == "train", (
            "get_dataloaders() must be called on the training split instance "
            "so vocabularies are built on training data only."
        )

        val_ds = Multi30kDataset(
            split     = "validation",
            src_vocab = self.src_vocab,
            tgt_vocab = self.tgt_vocab,
        )
        test_ds = Multi30kDataset(
            split     = "test",
            src_vocab = self.src_vocab,
            tgt_vocab = self.tgt_vocab,
        )

        _collate = partial(_collate_fn, pad_idx=pad_idx)

        train_loader = DataLoader(
            self,
            batch_size  = batch_size,
            shuffle     = True,
            collate_fn  = _collate,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = batch_size,
            shuffle     = False,
            collate_fn  = _collate,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size  = batch_size,
            shuffle     = False,
            collate_fn  = _collate,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )

        return train_loader, val_loader, test_loader


#  Collate: pad to longest sequence in batch ─
def _collate_fn(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch