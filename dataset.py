import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from functools import partial

import spacy
from datasets import load_dataset


# Special token constants
unk_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"

unk_idx = 0
pad_idx = 1
sos_idx = 2
eos_idx = 3

specials = [unk_token, pad_token, sos_token, eos_token]


class Vocab:
    """Simple vocabulary: token <> index."""

    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, unk_token)

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, unk_idx)


class Multi30kDataset(Dataset):
    def __init__(self, split: str = 'train', src_vocab=None, tgt_vocab=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.

        Args:
            split     : 'train', 'validation', or 'test'.
            src_vocab : Pre-built source Vocab (pass from training split to val/test).
            tgt_vocab : Pre-built target Vocab (pass from training split to val/test).
        """
        self.split = split

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

    # ── Tokenisers ────────────────────────────────────────────────────

    def tokenize_de(self, text: str) -> list:
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> list:
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # ── Vocab builder ─────────────────────────────────────────────────

    def _build_vocab(self):
        src_counter = Counter()
        tgt_counter = Counter()

        for example in self.data:
            src_counter.update(self.tokenize_de(example["de"]))
            tgt_counter.update(self.tokenize_en(example["en"]))

        def _build(counter: Counter) -> Vocab:
            stoi = {tok: idx + len(specials) for idx, tok in enumerate(counter.keys())}
            for idx, tok in enumerate(specials):
                stoi[tok] = idx
            return Vocab(stoi)

        return _build(src_counter), _build(tgt_counter)

    # ── Data processor ────────────────────────────────────────────────

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

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> tuple:
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long),
        )

    # ── DataLoader factory (call only on the training split) ──────────

    def get_dataloaders(self, batch_size: int = 128) -> tuple:
        assert self.split == "train", (
            "get_dataloaders() must be called on the training split instance "
            "so vocabularies are built on training data."
        )

        val_ds = Multi30kDataset(
            split="validation",
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
        )
        test_ds = Multi30kDataset(
            split="test",
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
        )

        _collate = partial(_collate_fn, pad_idx=pad_idx)

        train_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

        return train_loader, val_loader, test_loader


def _collate_fn(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch