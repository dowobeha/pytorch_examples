import math
import re
import time
from typing import List
import unicodedata

import torch


def verify_shape(*, tensor: torch.Tensor, expected: List[int]) -> None:
    if tensor.shape == torch.Size(expected):
        return
    else:
        raise ValueError(f"Tensor found with shape {tensor.shape} when {torch.Size(expected)} was expected.")


def as_minutes(*, seconds: float) -> str:
    minutes: int = math.floor(seconds / 60)
    seconds -= minutes * 60
    return '%dm %ds' % (minutes, seconds)


def time_since(*, since: float, percent: float) -> str:
    now: float = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(seconds=s), as_minutes(seconds=rs))


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
