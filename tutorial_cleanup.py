import math
import random
import re
import sys
import time
from typing import Iterable, List, Mapping, MutableMapping, Tuple
import unicodedata

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore

import torch
from torch.nn.functional import log_softmax, relu, softmax
import torch.nn as nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer


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


class Vocab:

    pad: int = 0
    oov: int = 1
    start_of_sequence: int = 2
    end_of_sequence: int = 3

    def __init__(self, name: str):
        self.name: str = name
        self.word2index: MutableMapping[str, int] = {}
        self.word2count: MutableMapping[str, int] = {}
        self.index2word: MutableMapping[int, str] = {Vocab.pad: "<pad/>",
                                                     Vocab.oov: "<oov/>",
                                                     Vocab.start_of_sequence: "<s>",
                                                     Vocab.end_of_sequence: "</s>"}
        self.n_words: int = len(self.index2word)

    def __getitem__(self, item: str) -> int:
        if item in self.word2index:
            return self.word2index[item]
        else:
            return Vocab.oov

    def add_sentence(self, sentence: List[str]) -> None:
        for word in sentence:  # type: str
            self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

#    def indexes_from_sentence(self, sentence: str) -> List[int]:
#        return [self[word] for word in sentence.split(' ')]

    def tensor_from_sentence(self, sentence: List[str], max_len: int, device: torch.device) -> torch.LongTensor:
        indexes: List[int] = [self[word] for word in sentence]  # self.indexes_from_sentence(sentence)
        # indexes.append(Vocab.end_of_sequence)
        while len(indexes) < max_len:
            indexes.append(Vocab.pad)
        result = torch.LongTensor(indexes, device=device)  # shape: [seq_len]
        # verify_shape(tensor=result, expected=[max_len])
        # result = result.view(-1, 1)  # shape: [seq_len, 1]
        #print(f"{result.shape}")
        # sys.exit()
        return result # shape: [seq_len]


class ParallelTensor:
    def __init__(self, source_tensor: torch.Tensor, target_tensor: torch.Tensor):
        self.source: torch.Tensor = source_tensor  # shape: [src_seq_len, 1]
        self.target: torch.Tensor = target_tensor  # shape: [tgt_seq_len, 1]


class ParallelSentence:
    def __init__(self, source_sentence: List[str], target_sentence: List[str]):
        assert(isinstance(source_sentence[0], str))
        self.source: List[str] = source_sentence
        self.target: List[str] = target_sentence

    def tensors(self, *,
                source_vocab: Vocab, max_src_length: int,
                target_vocab: Vocab, max_tgt_length: int,
                device: torch.device) -> ParallelTensor:
        source_tensor: torch.Tensor = source_vocab.tensor_from_sentence(sentence=self.source,
                                                                        max_len=max_src_length,
                                                                        device=device)
        target_tensor: torch.Tensor = target_vocab.tensor_from_sentence(sentence=self.target,
                                                                        max_len=max_tgt_length,
                                                                        device=device)
        return ParallelTensor(source_tensor, target_tensor)

    def __str__(self) -> str:
        return f"{''.join(self.source)}\t{''.join(self.target)}"


class Data:
    def __init__(self, source_vocab: Vocab, target_vocab: Vocab, pairs: List[ParallelSentence]):
        self.source_vocab: Vocab = source_vocab
        self.target_vocab: Vocab = target_vocab
        self.pairs: List[ParallelSentence] = pairs

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    def __getitem__(self, index: int) -> ParallelSentence:
        return self.pairs[index]

    def __repr__(self) -> str:
        return f"Data({self.source_vocab.name}-{self.target_vocab.name} with {len(self.pairs)} parallel sentences)"


class ParallelData(torch.utils.data.Dataset):
    def __init__(self, data: Data, device: torch.device):
        #print(repr(data))
        if len(data.pairs) == 0:
            raise ValueError(f"Cannot create parallel data containing zero examples.")
        self.strings: Data = data
        self.source_vocab: Vocab = data.source_vocab
        self.target_vocab: Vocab = data.target_vocab
        # print(f"data.length={len(data.pairs)}")
        self.max_src_length: int = 1 + max([len(pair.source) for pair in data.pairs])
        self.max_tgt_length: int = 1 + max([len(pair.target) for pair in data.pairs])
        # print(f"self.max_src_length={self.max_src_length}\t\tself.max_tgt_length={self.max_tgt_length}")
        self.pairs: List[ParallelTensor] = [pair.tensors(source_vocab=data.source_vocab,
                                                         target_vocab=data.target_vocab,
                                                         max_src_length=self.max_src_length,
                                                         max_tgt_length=self.max_tgt_length,
                                                         device=device)
                                            for pair in data.pairs]

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        # print(f"data -> {self.pairs[index].source.shape}\t\t" +
        #      f"labels -> {self.pairs[index].target.shape}")
        return {"data":   self.pairs[index].source,
                "labels": self.pairs[index].target}

    def __len__(self):
        return len(self.pairs)

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


def read_data(lang1: str, lang2: str, reverse: bool = False) -> Data:
    print("Reading lines...")

    # Read the file and split into lines
    lines: Iterable[str] = open('data/%s-%s.tiny' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    parallel_data: List[List[str]] = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        parallel_data = [list(reversed(p)) for p in parallel_data]
        input_vocab = Vocab(lang2)
        output_vocab = Vocab(lang1)
    else:
        input_vocab = Vocab(lang1)
        output_vocab = Vocab(lang2)

    pairs: List[ParallelSentence] = [ParallelSentence(source_sentence=(input_vocab.index2word[Vocab.start_of_sequence] +
                                                                       " " + parallel_line[0] + " " +
                                                                       input_vocab.index2word[Vocab.end_of_sequence]
                                                                       ).split(),
                                                      target_sentence=(output_vocab.index2word[
                                                                           Vocab.start_of_sequence
                                                                       ] + " " + parallel_line[1] + " " +
                                                                       output_vocab.index2word[
                                                                           Vocab.end_of_sequence
                                                                       ]
                                                                       ).split())
                                     for parallel_line in parallel_data]

    return Data(source_vocab=input_vocab, target_vocab=output_vocab, pairs=pairs)


def filter_pair(*, pair: ParallelSentence, max_length: int, prefixes: Tuple[str, ...]) -> bool:
    return len(pair.source) < max_length and \
        len(pair.target) < max_length and \
           (' '.join(pair.target[1:])).startswith(prefixes)


def filter_pairs(*,
                 pairs: List[ParallelSentence],
                 max_length: int,
                 prefixes: Tuple[str, ...]) -> List[ParallelSentence]:

    return [pair for pair in pairs if filter_pair(pair=pair, max_length=max_length, prefixes=prefixes)]


def prepare_data(*,
                 lang1: str,
                 lang2: str,
                 max_length: int,
                 prefixes: Tuple[str, ...],
                 reverse: bool = False) -> Data:

    data: Data = read_data(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(data))
    pairs: List[ParallelSentence] = filter_pairs(pairs=data.pairs, max_length=max_length, prefixes=prefixes)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Initializing vocabularies...")
    for pair in data:
        data.source_vocab.add_sentence(pair.source)
        data.target_vocab.add_sentence(pair.target)
    print("Vocabulary sizes:")
    print(data.source_vocab.name, data.source_vocab.n_words)
    print(data.target_vocab.name, data.target_vocab.n_words)
    return Data(source_vocab=data.source_vocab, target_vocab=data.target_vocab, pairs=pairs)


class EncoderRNN(nn.Module):
    def __init__(self, *, input_size: int, embedding_size: int, hidden_size: int, num_hidden_layers: int):
        super(EncoderRNN, self).__init__()
        self.num_hidden_layers: int = num_hidden_layers
        self.hidden_size: int = hidden_size
        self.embedding: nn.Embedding = nn.Embedding(input_size, embedding_size)
        self.gru: nn.GRU = nn.GRU(embedding_size, hidden_size, num_layers=num_hidden_layers)

    def forward(self,
                input_tensor: torch.Tensor,
                hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # ignore[override]
        # print(f"input_tensor.shape={input_tensor.shape}")
        #sys.exit()
        # input_tensor.shape:                   [batch_size=1]
        # self.embedding(input_tensor).shape:   [batch_size=1, embedding_size=256]
        #
        # embedded: torch.Tensor = self.embedding(input_tensor).view(1, 1, -1)  # <--- Original tutorial line
        embedded: torch.Tensor = self.embedding(input_tensor).unsqueeze(dim=0)  # <--- Replacement to enable batching
        #embedded: torch.Tensor = self.embedding(input_tensor)
        #print(f"embedded.shape={embedded.shape}\t\thidden.shape={hidden.shape}")
        #sys.exit()
        # embedded.shape:            [seq_len=1, batch_size=1, embedding_size=256]
        #
        output, hidden = self.gru(embedded, hidden)

        # output.shape:              [seq_len=1, batch_size=1, hidden_size=256]
        # hidden.shape:           [num_layers=1, batch_size=1, hidden_size=256]
        #
        return output, hidden

    def init_hidden(self, *, batch_size: int = 1, device: torch.device) -> torch.Tensor:
        # hidden.shape:           [num_layers=1, batch_size=1, hidden_size=256]
        hidden: torch.Tensor = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, device=device)
        return hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, *,
                 embedding_size: int,
                 decoder_hidden_size: int,
                 encoder_hidden_size: int,
                 max_src_length: int,
                 num_hidden_layers: int,
                 output_size: int,
                 dropout_p: float = 0.1):
        super(AttnDecoderRNN, self).__init__()
        self.decoder_hidden_size: int = decoder_hidden_size
        self.encoder_hidden_size: int = encoder_hidden_size
        self.embedding_size: int = embedding_size
        self.output_size: int = output_size
        self.dropout_p: float = dropout_p
        self.max_src_length: int = max_src_length

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.attn = nn.Linear(self.embedding_size + self.decoder_hidden_size, max_src_length)
        self.attn_combine = nn.Linear(self.embedding_size + self.encoder_hidden_size, self.encoder_hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.encoder_hidden_size, self.decoder_hidden_size, num_hidden_layers)
        self.out = nn.Linear(self.decoder_hidden_size, self.output_size)

    def forward(self,
                input_tensor: torch.Tensor,
                hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                actual_src_length: int,
                batch_size: int):  # type: ignore[override]

        if encoder_outputs.shape[0] != self.max_src_length:
            raise ValueError("Encoder outputs provided to this method must have same length as self.max_src_length:" +
                             f"\t{encoder_outputs.shape[0]} != {self.max_src_length}")

        # actual_src_length: int = max(self.max_src_length, input_tensor.shape[0])
        # print(f"self.max_src_length={self.max_src_length}\tinput_tensor.shape[0]={input_tensor.shape[0]}")
        verify_shape(tensor=input_tensor, expected=[1, batch_size])
        verify_shape(tensor=hidden, expected=[self.gru.num_layers, batch_size, self.gru.hidden_size])
        verify_shape(tensor=encoder_outputs, expected=[actual_src_length, batch_size, self.encoder_hidden_size])

        # input_tensor.shape:    [1, 1]
        # hidden.shape:          [num_hidden_layers=1, batch_size=1, decoder_hidden_size]
        # encoder_outputs.shape: [src_seq_len, encoder_hidden_size]

        #if input_tensor.shape == torch.Size([]):
        #    raise RuntimeError(f"input_tensor.shape={input_tensor.shape} is a problem")

        # if self.embedding(input_tensor).shape != self.embedding(input_tensor).view(1, 1, -1).shape:
        #    raise RuntimeError(f"input_tensor.shape={input_tensor.shape}\tembedding is {self.embedding(input_tensor).shape} vs expected {self.embedding(input_tensor).view(1, 1, -1).shape}")

        # print(f"input_tensor={input_tensor}\tdecoder input_tensor.shape={input_tensor.shape}\t\t" +
        #      f"decoder hidden.shape={hidden.shape}\t\t" +
        #       f"encoder_outputs.shape={encoder_outputs.shape}") #\t\tembedded.shape={embedded.shape}")


        # TODO: It should be safe to remove .view(1, 1, -1), as it appears to be a noop
        embedded = self.embedding(input_tensor) #.view(1, 1, -1)

        verify_shape(tensor=embedded, expected=[1, batch_size, self.embedding_size])

        # self.embedding(input_tensor).shape:  [1, 1, decoder_embedding_size]
        # embedded.shape:                      [1, 1, decoder_embedding_size]

        # print(f"self.embedding(input_tensor).shape={self.embedding(input_tensor).shape}\t\t" +
        #      f"self.embedding(input_tensor).view(1, 1, -1).shape={self.embedding(input_tensor).view(1, 1, -1).shape}\t\t" +
        #      f"embedded.shape={embedded.shape}")

        embedded = self.dropout(embedded)

        verify_shape(tensor=embedded, expected=[1, batch_size, self.embedding_size])
        verify_shape(tensor=embedded[0], expected=[batch_size, self.embedding_size])
        verify_shape(tensor=hidden[-1], expected=[batch_size, self.gru.hidden_size])

        attn_input: torch.Tensor = torch.cat(tensors=(embedded[0], hidden[0]), dim=1)

        verify_shape(tensor=attn_input, expected=[batch_size, self.embedding_size + self.gru.hidden_size])

        # print(f"embedded[0].shape={embedded[0].shape}\t\t"+
        #      f"hidden[0].shape={hidden[0].shape}\t\t" )
        # sys.exit()
        #       f"torch.cat(tensors=(embedded[0], hidden[0]), dim=1).shape="+
        #       f"{torch.cat(tensors=(embedded[0], hidden[0]), dim=1).shape}")

        #print(f"self.attn(...).shape={self.attn(torch.cat(tensors=(embedded[0], hidden[0]), dim=1)).shape}\t\t"+
        #       f"softmax(...).shape="+
        #       f"{softmax(self.attn(torch.cat(tensors=(embedded[0], hidden[0]), dim=1)), dim=1).shape}")
        # embedded.shape:                      [1, 1, decoder_embedding_size]
        # embedded[0].shape:                      [1, decoder_embedding_size]
        #
        # hidden.shape:                        [1, 1, decoder_hidden_size]
        # hidden[0].shape:                        [1, decoder_hidden_size]
        #
        # torch.cat(tensors=(embedded[0], hidden[0]), dim=1).shape:  [1, embedded.shape[2]+hidden.shape[2]]
        #
        # self.attn(...).shape:                                      [1, decoder_max_len]
        # softmax(self.attn(...)).shape:                             [1, decoder_max_len]
        attn_weights = softmax(self.attn(attn_input), dim=1)

        verify_shape(tensor=attn_weights, expected=[batch_size, actual_src_length])
        verify_shape(tensor=encoder_outputs, expected=[actual_src_length, batch_size, self.encoder_hidden_size])

        # Permute dimensions to prepare for batched matrix-matrix multiply
        encoder_outputs = encoder_outputs.permute(1, 2, 0)
        attn_weights = attn_weights.unsqueeze(2)

        verify_shape(tensor=encoder_outputs, expected=[batch_size, self.encoder_hidden_size, actual_src_length])
        verify_shape(tensor=attn_weights, expected=[batch_size, actual_src_length, 1])

        #print(f"attn_weights.shape={attn_weights.shape}\t\t"+
        #       f"encoder_outputs.shape={encoder_outputs.shape}")


        #import sys;

        #sys.exit()

        # print(f"attn_weights.unsqueeze(0).shape={attn_weights.unsqueeze(0).shape}\t\t"+
        #       f"encoder_outputs.unsqueeze(0).shape={encoder_outputs.unsqueeze(0).shape}")

        # attn_weights.shape:                  [1, decoder_max_len]
        # encoder_outputs.shape:                  [decoder_max_len, encoder_hidden_size]
        #
        # attn_weights.unsqueeze(0).shape:     [1, 1, decoder_max_len]
        # encoder_outputs.unsqueeze(0).shape:     [1, decoder_max_len, encoder_hidden_size]
        #attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))  # <-- Original
        attn_applied = torch.bmm(encoder_outputs, attn_weights)   # <-- Batched

        # Get rid of superfluous final dimension
        #attn_applied = attn_applied.squeeze(dim=2)
        #verify_shape(tensor=attn_applied, expected=[batch_size, self.encoder_hidden_size])



        # print(f"attn_applied.shape={attn_applied.shape}\t\t"+
        #       f"embedded[0].shape={embedded[0].shape}\t\t"+
        #       f"attn_applied[0].shape={attn_applied[0].shape}\t\t"
        #       f"torch.cat(...).shape={torch.cat((embedded[0], attn_applied[0]), 1).shape}")

        # embedded.shape:                                  [1, batch_size=1, decoder_embedding_size]
        # attn_applied.shape:                              [1, batch_size=1, encoder_hidden_size]
        #
        # embedded[0].shape:                                  [batch_size=1, decoder_embedding_size]
        # attn_applied[0].shape:                              [batch_size=1, encoder_hidden_size]
        #
        # torch.cat((embedded[0], attn_applied[0]), 1).shape: [batch_size=1, decoder_embedding_size+encoder_hidden_size]
        #

        verify_shape(tensor=attn_applied, expected=[batch_size, self.encoder_hidden_size, 1])
        verify_shape(tensor=embedded, expected=[1, batch_size, self.embedding_size])

        # # The final dimension of attn_applied and the first dimension of embedded
        # #   represents seq_len, which is not needed at this point.
        # attn_applied = attn_applied.squeeze(dim=2)
        # embedded = embedded.squeeze(dim=0)
        #
        # verify_shape(tensor=attn_applied, expected=[batch_size, self.encoder_hidden_size])
        # verify_shape(tensor=embedded, expected=[batch_size, self.embedding_size])
        #
        # output = torch.cat((embedded, attn_applied), dim=1)
        # verify_shape(tensor=output, expected=[batch_size, self.embedding_size + self.encoder_hidden_size])

        attn_applied = attn_applied.permute(2, 0, 1)

        verify_shape(tensor=attn_applied, expected=[1, batch_size, self.encoder_hidden_size])
        verify_shape(tensor=embedded, expected=[1, batch_size, self.embedding_size])

        output = torch.cat(tensors=(embedded, attn_applied), dim=2)

        verify_shape(tensor=output, expected=[1, batch_size, self.embedding_size + self.encoder_hidden_size])

        # print(f"output.shape={output.shape}")

        # output.shape:                                      [batch_size=1, encoder_hidden_size+decoder_embedding_size]
        # self.attn_combine(output).shape:                   [batch_size=1, decoder_hidden_size]
        # self.attn_combine(output).unsqueeze(0): [seq_len=1, batch_size=1, decoder_hidden_size]
        #
        output = self.attn_combine(output) #.unsqueeze(0)

        verify_shape(tensor=output, expected=[1, batch_size, self.encoder_hidden_size])


        # print(f"output.shape={output.shape}")
        # print(f"relu(output).shape={relu(output).shape}\t\thidden.shape={hidden.shape}")

        # output.shape:                [seq_length=1, batch_size=1, decoder_hidden_size]
        # relu(...).shape:             [seq_length=1, batch_size=1, decoder_hidden_size]
        # hidden.shape:                [num_layers=1, batch_size=1, decoder_hidden_size]
        #
        output = relu(output)

        verify_shape(tensor=output, expected=[1, batch_size, self.encoder_hidden_size])

        output, hidden = self.gru(output, hidden)

        verify_shape(tensor=output, expected=[1, batch_size, self.decoder_hidden_size])
        verify_shape(tensor=hidden, expected=[self.gru.num_layers, batch_size, self.decoder_hidden_size])

        output = output.squeeze(dim=0)

        verify_shape(tensor=output, expected=[batch_size, self.decoder_hidden_size])

        output = log_softmax(self.out(output), dim=1)

        verify_shape(tensor=attn_weights, expected=[batch_size, actual_src_length, 1])
        attn_weights = attn_weights.squeeze(dim=2)

        verify_shape(tensor=output, expected=[batch_size, self.output_size])
        verify_shape(tensor=hidden, expected=[self.gru.num_layers, batch_size, self.decoder_hidden_size])
        verify_shape(tensor=attn_weights, expected=[batch_size, actual_src_length])

        # print(f"output.shape={output.shape}\t\thidden.shape={hidden.shape}\t\toutput[0].shape={output[0].shape}")

        # output.shape:             [seq_length=1, batch_size=1, decoder_hidden_size]
        # hidden.shape:             [num_layers=1, batch_size=1, decoder_hidden_size]
        #
        # output[0].shape:                        [batch_size=1, decoder_hidden_size]
        # output = log_softmax(self.out(output[0]), dim=1)

        # print(f"output.shape={output.shape}\t\thidden.shape={hidden.shape}\t\tattn_weights.shape={attn_weights.shape}")

        # output.shape:                           [batch_size=1, decoder_output_size]
        # hidden.shape:             [num_layers=1, batch_size=1, decoder_hidden_size]
        # attn_weights:                           [batch_size=1, encoder_max_len]
        #
        return output, hidden, attn_weights

    def init_hidden(self, *, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru.num_layers, batch_size, self.decoder_hidden_size, device=device)


def train(*,
          input_tensor: torch.Tensor,  # shape: [src_seq_len, batch_size]
          target_tensor: torch.Tensor,  # shape: [tgt_seq_len, batch_size]
          encoder: EncoderRNN,
          decoder: AttnDecoderRNN,
          encoder_optimizer: Optimizer,
          decoder_optimizer: Optimizer,
          criterion: nn.Module,
          device: torch.device,
          max_src_length: int,
          max_tgt_length: int,
          batch_size: int,
          teacher_forcing_ratio: float) -> float:

    # print(f"input_tensor.shape={input_tensor.shape}\t\ttarget_tensor.shape={target_tensor.shape}")

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #src_seq_len = input_tensor.size(0)   # This assumes that dimension 0 is src_seq_len
    #tgt_seq_len = target_tensor.size(0)  # This assumes that dimension 0 is tgt_seq_len

    encoder_hidden = encoder.init_hidden(batch_size=batch_size,
                                         device=device)         # shape: [num_layers, batch_size, hidden_size]

    encoder_outputs = torch.zeros(max_src_length,
                                  batch_size,
                                  encoder.hidden_size,
                                  device=device)        # shape: [max_src_len, hidden_size]

    loss: torch.Tensor = torch.tensor(0, dtype=torch.float)  # shape: [] meaning this is a scalar

    verify_shape(tensor=input_tensor, expected=[max_src_length, batch_size])
    verify_shape(tensor=target_tensor, expected=[max_tgt_length, batch_size])
    verify_shape(tensor=encoder_hidden, expected=[encoder.num_hidden_layers, batch_size, encoder.hidden_size])
    verify_shape(tensor=encoder_outputs, expected=[max_src_length, batch_size, encoder.hidden_size])

    for src_index in range(max_src_length):

        # input_tensor.shape is [src_seq_len, batch=1]
        # input_tensor[src_index].shape is [batch=1]
        input_token_tensor: torch.Tensor = input_tensor[src_index]

        verify_shape(tensor=input_token_tensor, expected=[batch_size])
        verify_shape(tensor=encoder_hidden, expected=[encoder.num_hidden_layers, batch_size, encoder.hidden_size])

        # input_token_tensor.shape is [batch=1]
        # encoder_hidden.shape is [num_layers * num_directions = 1, batch=1, hidden_size=256]
        encoder_output, encoder_hidden = encoder(input_token_tensor, encoder_hidden)

        verify_shape(tensor=encoder_hidden, expected=[encoder.num_hidden_layers, batch_size, encoder.hidden_size])
        verify_shape(tensor=encoder_output, expected=[1, batch_size, encoder.hidden_size])

        verify_shape(tensor=encoder_output[0], expected=[batch_size, encoder.hidden_size])
        verify_shape(tensor=encoder_outputs[src_index], expected=[batch_size, encoder.hidden_size])

        # encoder_output.shape is [seq_len=1, batch=1, num_directions * hidden_size = 256]
        # encoder_outputs.shape is [max_seq_len, hidden_size]
        # encoder_outputs[src_index].shape is [hidden_size]
        # encoder_output[0, 0].shape is [hidden_size=256]
        # encoder_outputs[src_index] = encoder_output[0, 0]  # <-- Original
        encoder_outputs[src_index] = encoder_output[0]       # <-- Updated for batching

    decoder_input = torch.tensor([[Vocab.start_of_sequence]*batch_size], device=device)  # shape: [seq_len=1, batch=1]

    # encoder_hidden.shape is [num_layers * num_directions = 1, batch=1, hidden_size=256]
    # decoder_hidden = encoder_hidden   # <--- Original tutorial
    decoder_hidden = decoder.init_hidden(batch_size=batch_size, device=device)

    verify_shape(tensor=decoder_input, expected=[1, batch_size])
    verify_shape(tensor=decoder_hidden, expected=[decoder.gru.num_layers, batch_size, decoder.gru.hidden_size])

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = False

    if use_teacher_forcing:

        # Teacher forcing: Feed the target as the next input
        for di in range(max_tgt_length):

            verify_shape(tensor=decoder_input, expected=[1, batch_size])
            verify_shape(tensor=decoder_hidden, expected=[decoder.gru.num_layers, batch_size, decoder.gru.hidden_size])

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, max_src_length, batch_size)

            # output.shape:                           [batch_size=1, decoder_output_size]
            # hidden.shape:             [num_layers=1, batch_size=1, decoder_hidden_size]
            # attn_weights:                                      [1, encoder_max_len]
            verify_shape(tensor=decoder_output, expected=[batch_size, decoder.output_size])
            verify_shape(tensor=decoder_hidden, expected=[decoder.gru.num_layers, batch_size, decoder.gru.hidden_size])
            verify_shape(tensor=decoder_attention, expected=[batch_size, max_src_length])

            loss += criterion(decoder_output, target_tensor[di])
            #print(f"target_tensor.shape={target_tensor.shape}\t\ttarget_tensor[di={di}].shape={target_tensor[di].shape}\n{target_tensor}\n{target_tensor[di]}")
            #decoder_input = target_tensor[di].unsqueeze(dim=1)  # Teacher forcing
            decoder_input = target_tensor[di].unsqueeze(dim=0)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_tgt_length):

            verify_shape(tensor=decoder_input, expected=[1, batch_size])
            verify_shape(tensor=decoder_hidden, expected=[decoder.gru.num_layers, batch_size, decoder.gru.hidden_size])

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, max_src_length, batch_size)

            verify_shape(tensor=decoder_output, expected=[batch_size, decoder.output_size])
            verify_shape(tensor=decoder_hidden, expected=[decoder.gru.num_layers, batch_size, decoder.gru.hidden_size])
            verify_shape(tensor=decoder_attention, expected=[batch_size, max_src_length])

            topv, topi = decoder_output.topk(1)
            # print(topi.shape)
            # decoder_input = topi.squeeze().detach()  # detach from history as input  # <--- Original code
            decoder_input = topi.detach().permute(1, 0)  # detach from history as input
            # print(f"decoder_output.shape={decoder_output.shape}\t\ttopi.shape={topi.shape}\t\tdecoder_input.shape={decoder_input.shape}\t\tdecoder_input={decoder_input}")
            loss += criterion(decoder_output, target_tensor[di])
            #if decoder_input.item() == Vocab.end_of_sequence:
            #    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_tgt_length


def show_plot(points: List[float]) -> None:
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def train_iters(*, #data: Data,
                parallel_data: ParallelData,
                encoder: EncoderRNN,
                decoder: AttnDecoderRNN,
                device: torch.device,
                max_src_length: int,
                max_tgt_length: int,
                n_iters: int,
                batch_size: int,
                teacher_forcing_ratio: float,
                print_every: int = 1000,
                plot_every: int = 100,
                learning_rate: float = 0.01
                ) -> None:

    data = torch.utils.data.DataLoader(dataset=parallel_data, batch_size=batch_size)

    start: float = time.time()
    plot_losses: List[float] = []
    print_loss_total: float = 0  # Reset every print_every
    plot_loss_total: float = 0  # Reset every plot_every

    encoder_optimizer: Optimizer = SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer: Optimizer = SGD(decoder.parameters(), lr=learning_rate)
    #
    # training_pairs: List[ParallelTensor] = [random.choice(data.pairs).tensors(source_vocab=data.source_vocab,
    #                                                                           target_vocab=data.target_vocab,
    #                                                                           device=device)
    #                                         for _ in range(n_iters)]

    criterion: nn.NLLLoss = nn.NLLLoss()

    #for pair in parallel_data:
    #    print(f"src={len(pair['data'])}\ttgt={len(pair['labels'])}")

    for iteration in range(1, n_iters + 1):  # type: int

        # training_pair: ParallelTensor = training_pairs[iteration - 1]
        # input_tensor: torch.Tensor = training_pair.source   # shape: [seq_len, batch_size=1]
        # target_tensor: torch.Tensor = training_pair.target  # shape: [seq_len, batch_size=1]

        for batch in data:
            # print(f"batch['data'].shape={batch['data'].shape}\tbatch['labels'].shape{batch['labels'].shape}")
           # sys.exit()
            input_tensor: torch.Tensor = batch["data"].permute(1, 0)
            target_tensor: torch.Tensor = batch["labels"].permute(1, 0)

            actual_batch_size: int = min(batch_size, input_tensor.shape[1])

            verify_shape(tensor=input_tensor, expected=[parallel_data.max_src_length, actual_batch_size])
            verify_shape(tensor=target_tensor, expected=[parallel_data.max_tgt_length, actual_batch_size])

            # print(f"input_tensor.shape={input_tensor.shape}\t\ttarget_tensor.shape={target_tensor.shape}")
            # sys.exit()

            loss: float = train(input_tensor=input_tensor,
                                target_tensor=target_tensor,
                                encoder=encoder,
                                decoder=decoder,
                                encoder_optimizer=encoder_optimizer,
                                decoder_optimizer=decoder_optimizer,
                                criterion=criterion,
                                device=device,
                                max_src_length=max_src_length,
                                max_tgt_length=max_tgt_length,
                                batch_size=actual_batch_size,
                                teacher_forcing_ratio=teacher_forcing_ratio)

            print_loss_total += loss
            plot_loss_total += loss

        if iteration % print_every == 0:
            print_loss_avg: float = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(since=start, percent=iteration/n_iters),
                                         iteration, iteration / n_iters * 100, print_loss_avg))

        if iteration % plot_every == 0:
            plot_loss_avg: float = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(points=plot_losses)


def evaluate(*,
             encoder: EncoderRNN,
             decoder: AttnDecoderRNN,
             sentence: List[str],
             source_vocab: Vocab,
             target_vocab: Vocab,
             device: torch.device,
             max_tgt_length: int):

    if len(sentence) > decoder.max_src_length:
        raise ValueError(f"Input sentence length must not exceed {decoder.max_src_length}, but does:\t{len(sentence)}")

    with torch.no_grad():

        batch_size: int = 1
        actual_src_length: int = max(decoder.max_src_length, len(sentence))
        input_tensor: torch.Tensor = source_vocab.tensor_from_sentence(sentence=sentence,
                                                                       max_len=actual_src_length,
                                                                       device=device).unsqueeze(dim=1)

        # print(f"sentence={sentence}\tmax_src_length={decoder.max_src_length}\tinput_tensor length={input_tensor.shape}")

        encoder_hidden: torch.Tensor = encoder.init_hidden(device=device)

        encoder_outputs: torch.Tensor = torch.zeros(actual_src_length, batch_size, encoder.hidden_size,
                                                    device=device)

        #sys.exit()
        verify_shape(tensor=input_tensor, expected=[actual_src_length, batch_size])
        verify_shape(tensor=encoder_hidden, expected=[encoder.num_hidden_layers, batch_size, encoder.hidden_size])
        verify_shape(tensor=encoder_outputs, expected=[actual_src_length, batch_size, encoder.hidden_size])

        for src_index in range(actual_src_length):  # type: int
            encoder_output, encoder_hidden = encoder(input_tensor[src_index], encoder_hidden)

            verify_shape(tensor=encoder_hidden, expected=[encoder.num_hidden_layers, batch_size, encoder.hidden_size])
            verify_shape(tensor=encoder_output, expected=[1, batch_size, encoder.hidden_size])

            verify_shape(tensor=encoder_output[0], expected=[batch_size, encoder.hidden_size])
            verify_shape(tensor=encoder_outputs[src_index], expected=[batch_size, encoder.hidden_size])

            encoder_outputs[src_index] = encoder_output[0]  # <-- Updated for batching

        decoder_input = torch.tensor([[Vocab.start_of_sequence] * batch_size], device=device)
        decoder_hidden = decoder.init_hidden(batch_size=batch_size, device=device)

        verify_shape(tensor=decoder_input, expected=[1, batch_size])
        verify_shape(tensor=decoder_hidden, expected=[decoder.gru.num_layers, batch_size, decoder.gru.hidden_size])

        decoded_words: List[str] = []
        decoder_attentions: torch.Tensor = torch.zeros(max_tgt_length, actual_src_length)

        target_index: int = 0
        for target_index in range(max_tgt_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, actual_src_length, batch_size)
            decoder_attentions[target_index] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == Vocab.end_of_sequence:
                decoded_words.append(target_vocab.index2word[Vocab.end_of_sequence])
                break
            else:
                decoded_words.append(target_vocab.index2word[topi.item()])
            # print(f"topi={topi}\t\ttopi.shape={topi.shape}\t\t{topi.squeeze().shape}")
            decoder_input = topi.detach()

        return decoded_words, decoder_attentions[:target_index + 1]


def evaluate_randomly(*,
                      data: ParallelData,
                      encoder: EncoderRNN,
                      decoder: AttnDecoderRNN,
                      device: torch.device,
                      n: int = 10) -> None:

    for i in range(n):  # type: int

        pair: ParallelSentence = random.choice(data.strings.pairs)

        output_words, attentions = evaluate(encoder=encoder,
                                            decoder=decoder,
                                            sentence=pair.source,
                                            source_vocab=data.source_vocab,
                                            target_vocab=data.target_vocab,
                                            device=device,
                                            max_tgt_length=data.max_tgt_length)
        output_sentence = ' '.join(output_words)

        print('>', ' '.join(pair.source))
        print('=', ' '.join(pair.target))
        print('<', output_sentence)
        print('')


def show_attention(*, input_sentence: List[str], output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluate_and_show_attention(*,
                                encoder: EncoderRNN,
                                decoder: AttnDecoderRNN,
                                sentence: List[str],
                                source_vocab: Vocab,
                                target_vocab: Vocab,
                                device: torch.device,
                                max_tgt_length: int):

    output_words, attentions = evaluate(encoder=encoder,
                                        decoder=decoder,
                                        sentence=sentence,
                                        source_vocab=source_vocab,
                                        target_vocab=target_vocab,
                                        device=device,
                                        max_tgt_length=max_tgt_length)
    print('input =', sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence=sentence, output_words=output_words, attentions=attentions)


def run_training():

    max_length = 10

    teacher_forcing_ratio = 0.0

    eng_prefixes: Tuple[str, ...] = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    data: Data = prepare_data(lang1='eng',
                              lang2='fra',
                              max_length=max_length,
                              prefixes=eng_prefixes,
                              reverse=True)

    # print(len(data))
    # print(random.choice(data.pairs))

    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parallel_data = ParallelData(data, device)

    # print(parallel_data.max_src_length)
    # print(parallel_data.max_tgt_length)
    # sys.exit()
    encoder1: EncoderRNN = EncoderRNN(input_size=data.source_vocab.n_words,
                                      embedding_size=200,
                                      hidden_size=256,
                                      num_hidden_layers=1).to(device=device)

    attn_decoder1 = AttnDecoderRNN(embedding_size=190,
                                   decoder_hidden_size=7,
                                   encoder_hidden_size=encoder1.hidden_size,
                                   num_hidden_layers=1,
                                   output_size=data.target_vocab.n_words,
                                   dropout_p=0.1,
                                   max_src_length=parallel_data.max_src_length).to(device=device)

    train_iters(parallel_data=parallel_data,
                encoder=encoder1,
                decoder=attn_decoder1,
                device=device,
                max_src_length=parallel_data.max_src_length,
                max_tgt_length=parallel_data.max_tgt_length,
                n_iters=1,
                batch_size=5,
                print_every=25,
                plot_every=100,
                learning_rate=0.01,
                teacher_forcing_ratio=teacher_forcing_ratio)

    if True:

        evaluate_randomly(data=parallel_data, encoder=encoder1, decoder=attn_decoder1, n=10, device=device)

        start_of_sequence: str = parallel_data.source_vocab.index2word[Vocab.start_of_sequence]
        end_of_sequence: str = parallel_data.source_vocab.index2word[Vocab.end_of_sequence]

        sample_sentences: List[List[str]] = [f"{start_of_sequence} {string} {end_of_sequence}".split()
                                             for string in
                                             ["je suis trop froid .",
                                              # "elle a cinq ans de moins que moi .",
                                              "elle est trop petit .",
                                              # "je ne crains pas de mourir .",
                                              # "c est un jeune directeur plein de talent ."
                                              ]
                                             ]

        output_words, attentions = evaluate(encoder=encoder1,
                                            decoder=attn_decoder1,
                                            sentence=sample_sentences[0],
                                            source_vocab=parallel_data.source_vocab,
                                            target_vocab=parallel_data.target_vocab,
                                            device=device,
                                            max_tgt_length=parallel_data.max_tgt_length)

        plt.matshow(attentions.numpy())

        for sentence in sample_sentences[1:]:

            evaluate_and_show_attention(sentence=sentence,
                                        encoder=encoder1,
                                        decoder=attn_decoder1,
                                        source_vocab=data.source_vocab,
                                        target_vocab=data.target_vocab,
                                        device=device,
                                        max_tgt_length=parallel_data.max_tgt_length)


if __name__ == "__main__":

    run_training()
