import sys
from typing import Iterable, List, Mapping, TextIO

import torch
from torch.utils.data import Dataset

from vocab import Vocab


class TextSequence(Dataset):

    def __init__(self, *, path: str, max_len: int = 0):

        self.vocab: Vocab = Vocab()

        if max_len > 0:
            self.max_len: int = max_len + 2
            self.sequence: List[str] = [line for line in TextSequence.read_lines(path) if len(line) < self.max_len]

        else:
            self.sequence: List[str] = list(TextSequence.read_lines(path=path, vocab=self.vocab))
            for word in self.sequence:
                max_len = max(max_len, len(word))
            self.max_len: int = max_len + 2

        for line in TextSequence.read_lines(path=path, vocab=self.vocab):  # type: str
#            print(f"Reading line...\t{line}", file=sys.stderr)
            for char in line:  # type: str
                self.vocab += char

    def __getitem__(self, item: int) -> Mapping[str, torch.Tensor]:
        int_list: List[int] = [self.vocab.start_of_sequence]
        for char in self.sequence[item]:
            int_list.append(self.vocab[char])
        int_list.append(self.vocab.end_of_sequence)

        while len(int_list) < self.max_len:
            int_list.append(self.vocab.pad)

        label: List[int] = int_list[1:] + [self.vocab.pad]

        return {"data": torch.tensor(int_list),
                "labels": torch.tensor(label)}

    def __len__(self):
        return len(self.sequence)

    @staticmethod
    def read_lines(*, path: str, vocab: Vocab) -> Iterable[str]:
        with open(path, 'rt') as lines:  # type: TextIO[str]
            for line_number, line in enumerate(lines):  # type: Tuple[int, str]
                if vocab.no_reserved_characters(sequence=line, line_number=line_number):
                    yield line.strip()


