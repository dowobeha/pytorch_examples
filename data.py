import sys
from typing import Iterable, List, Mapping, TextIO

import torch
from torch.utils.data import Dataset

from vocab import Vocab


class TextSequence(Dataset):

    def __init__(self, *, path: str, max_len: int = 0, vocab: Vocab = None):

        if vocab is None:
            self.vocab: Vocab = Vocab()
        else:
            self.vocab: Vocab = vocab

        if max_len > 0:
            self.max_len: int = max_len + 2
            self.sequence: List[str] = [line for line in TextSequence.read_lines(path) if len(line) < self.max_len]

        else:
            self.sequence: List[str] = list(TextSequence.read_lines(path=path, vocab=self.vocab))
            for word in self.sequence:
                max_len = max(max_len, len(word))
            self.max_len: int = max_len + 2

        if vocab is None:
            for line in TextSequence.read_lines(path=path, vocab=self.vocab):  # type: str
                for char in line:  # type: str
                    self.vocab += char

    def __getitem__(self, item: int) -> Mapping[str, torch.Tensor]:
        int_list: List[int] = TextSequence.string_to_ints(self.sequence[item], self.max_len, self.vocab)
        label: List[int] = int_list[1:] + [self.vocab.pad]

        return {"data": torch.tensor(int_list),
                "labels": torch.tensor(label)}

    def __len__(self):
        return len(self.sequence)

    @staticmethod
    def string_to_ints(sequence: str, max_len: int, vocab: Vocab) -> List[int]:
        int_list: List[int] = [vocab.start_of_sequence]
        for char in sequence:
            int_list.append(vocab[char])
        int_list.append(vocab.end_of_sequence)

        while len(int_list) < max_len:
            int_list.append(vocab.pad)

        return int_list

    @staticmethod
    def read_lines(*, path: str, vocab: Vocab) -> Iterable[str]:
        with open(path, 'rt') as lines:  # type: TextIO[str]
            for line_number, line in enumerate(lines):  # type: Tuple[int, str]
                if vocab.no_reserved_characters(sequence=line, line_number=line_number):
                    yield line.strip()


class PigLatin(Dataset):

    def __init__(self, *, path: str, max_len: int = 0, vocab: Vocab = None):

        original_text: TextSequence = TextSequence(path=path, max_len=max_len, vocab=vocab)

        self.max_len = original_text.max_len + 3
        self.vocab: Vocab = original_text.vocab
        self.vocab += "-"
        self.vocab += "a"
        self.vocab += "y"
        self.corpus = original_text.sequence
        self.vowels = "aeiouAEIOU"

    def __getitem__(self, index: int):
        int_list: List[int] = TextSequence.string_to_ints(self.corpus[index], self.max_len, self.vocab)

        label_text: str = PigLatin.modify(self.corpus[index], self.vowels)
        label: List[int] = TextSequence.string_to_ints(label_text, self.max_len, self.vocab)
        # print(str(len(label)) + "\t" + str(PigLatin.modify(self.corpus[index], self.vowels)))

        #label: List[int] = ([self.vocab["y"]] * (self.max_len-1)) + [self.vocab.end_of_sequence]

        # print(f"{self.corpus[index]}\t{label_text}\t{self.max_len}")

        return {"string": self.corpus[index],
                "data": torch.tensor(int_list),
                "labels": torch.tensor(int_list),
                #"labels": torch.tensor(label),
                #"labels": torch.tensor([self.vocab["y"] * self.max_len]),
                "start-of-sequence": torch.tensor([self.vocab.start_of_sequence])}

    def __len__(self):
        return len(self.corpus)

    @staticmethod
    def modify(word: str, vowels: Iterable[str]) -> str:
        positions: List[int] = [word.find(vowel) for vowel in vowels]
        positions: List[int] = [position for position in positions if position >= 0]
        start = min(positions, default=0)
        prefix = word[0:start]
        suffix = word[start:]
        return f"{suffix}-{prefix}ay"
