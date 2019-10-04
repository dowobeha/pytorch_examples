import random
import sys
import time
from typing import Iterable, List, Mapping, MutableMapping, Tuple

import torch
from torch.nn.functional import log_softmax, relu, softmax
import torch.nn as nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset

from utils import verify_shape, time_since, normalize_string


class Vocab:

    def __init__(self, name: str, *,
                 pad: str = "<pad/>",
                 oov: str = "<oov/>",
                 start_of_sequence: str = "<s>",
                 end_of_sequence: str = "</s>"):

        self.name: str = name

        self.pad_int: int = 0
        self.pad_string: str = pad

        self.oov_int: int = 1
        self.oov_string: str = oov

        self.start_of_sequence_int: str = 2
        self.start_of_sequence_string: str = start_of_sequence

        self.end_of_sequence_int: int = 3
        self.end_of_sequence_string: str = end_of_sequence

        self.string2int: MutableMapping[str, int] = {self.pad_string: self.pad_int,
                                                     self.oov_string: self.oov_int,
                                                     self.start_of_sequence_string: self.start_of_sequence_int,
                                                     self.end_of_sequence_string: self.end_of_sequence_int}
        self.int2string: MutableMapping[int, str] = {self.pad_int: self.pad_string,
                                                     self.oov_int: self.oov_string,
                                                     self.start_of_sequence_int: self.start_of_sequence_string,
                                                     self.end_of_sequence_int: self.end_of_sequence_string}

    def __getitem__(self, item: str) -> int:
        if item in self.string2int:
            return self.string2int[item]
        else:
            return self.oov_int

    def __len__(self) -> int:
        return len(self.string2int)

    def add(self, item: str) -> None:
        if item not in self.string2int:
            next_int: int = len(self)
            self.string2int[item] = next_int
            self.int2string[next_int] = item


class Word:
    def __init__(self, characters: List[str]):
        self.characters: List[str] = characters
        self.label: List[str] = Word.generate_label(characters)

    def __str__(self) -> str:
        return f"{''.join(self.characters)}\t{''.join(self.label)}"
        
    @staticmethod
    def is_completely_alphabetic(characters: List[str]) -> bool:
        from functools import reduce
        return reduce(lambda a, b: a & b, [c.isalpha() for c in characters])

    @staticmethod
    def position_of_first_vowel(characters: List[str]) -> int:
        for position in range(len(characters)):
            char = characters[position]
            if char in "aeiouAEIOU":
                return position
        return len(characters)

    @staticmethod
    def generate_label(characters: List[str]) -> str:
        if Word.is_completely_alphabetic(characters):
            first_vowel = Word.position_of_first_vowel(characters)
            prefix = characters[0:first_vowel]
            suffix = characters[first_vowel:]
            return suffix + ['-'] + prefix + ['a','y']
        else:
            return characters
        

class Corpus(Dataset):

    def __init__(self, *, name: str, filename: str, max_length: int, device: torch.device, vocab: Vocab = None):
        self.device = device

        if vocab is None:
            self.characters: Vocab = Vocab(name)
        else:
            self.characters: Vocab = vocab

        self.words: List[Word] = Corpus.read_words(filename, max_length)

        if vocab is None:
            for word in self.words:
                for character in word.label:
                    self.characters.add(character)

        self._max_word_length = Corpus.calculate_longest([word.characters for word in self.words])
        self._max_label_length = Corpus.calculate_longest([word.label for word in self.words])

        self.word_tensor_length = self._max_word_length + 2
        self.label_tensor_length = self._max_label_length + 1

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:

        return {"data": self.create_tensor(sequence=self.words[index].characters,
                                           pad_to_length=self._max_word_length,
                                           include_start_of_sequence=True),

                "labels": self.create_tensor(sequence=self.words[index].label,
                                             pad_to_length=self._max_label_length,
                                             include_start_of_sequence=False),

                "start-of-sequence": torch.tensor(data=[self.characters.start_of_sequence_int],
                                                  dtype=torch.long,
                                                  device=self.device),

                "string": self.words[index].characters}

    def create_tensor(self, *, sequence: List[str], pad_to_length: int, include_start_of_sequence: bool) -> torch.Tensor:

        start_of_sequence: List[int] = [self.characters.start_of_sequence_int] if include_start_of_sequence else []
        int_sequence: List[int] = [self.characters[s] for s in sequence]
        pads: List[int] = [self.characters.pad_int] * max(0, (pad_to_length - len(sequence)))
        end_of_sequence: List[int] = [self.characters.end_of_sequence_int]

        result = torch.tensor(data=start_of_sequence + int_sequence + end_of_sequence + pads,
                              dtype=torch.long,
                              device=self.device)
        return result

    # def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
    #
    #     # print(f"word[{index}].word = {self.words[index].word}")
    #     return {"data": Corpus.create_tensor(vocab=self.characters,
    #                                          sequence=self.words[index].characters,
    #                                          pad_to_length=self._max_word_length,
    #                                          device=self.device),
    #
    #             "labels": Corpus.create_tensor(vocab=self.characters,
    #                                           sequence=self.words[index].label,
    #                                           pad_to_length=self._max_label_length,
    #                                           device=self.device),
    #
    #             "start-of-sequence": torch.tensor(data=[self.characters.start_of_sequence_int],
    #                                               dtype=torch.long,
    #                                               device=self.device),
    #
    #             "string": self.words[index].characters}
    #
    # @staticmethod
    # def create_tensor(*,
    #                   vocab: Vocab,
    #                   sequence: List[str],
    #                   pad_to_length: int,
    #                   device: torch.device):
    #     pads: List[int] = [vocab.pad_int] * max(0, (pad_to_length - len(sequence)))
    #     result = torch.tensor(data=[vocab[s] for s in sequence]+pads, dtype=torch.long, device=device)
    #     return result

    @staticmethod
    def read_words(filename: str, max_length: int) -> List[Word]:
        words: List[Word] = list()
        with open(filename, mode='rt', encoding='utf8') as f:
            for line in f:  # type: str
                for word in normalize_string(line).strip().split():  # type: str
                    if len(word) <= max_length:
                        words.append(Word(list(word)))
        return words

    @staticmethod
    def calculate_longest(sequences: List[str]) -> int:
        longest: int = 0
        for sequence in sequences:
            length = len(sequence)
            if length > longest:
                longest = length
        return longest


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
        #print(f"input_tensor.shape={input_tensor.shape}")
        embedded: torch.Tensor = self.embedding(input_tensor).unsqueeze(dim=0)  # <--- Replacement to enable batching
        #print(f"embedded.shape={embedded.shape}")
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def init_hidden(self, *, batch_size: int = 1, device: torch.device) -> torch.Tensor:
        # hidden.shape:           [num_layers=1, batch_size=1, hidden_size=256]
        hidden: torch.Tensor = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, device=device)
        return hidden

    def encode_sequence(self, input_sequence: torch.LongTensor) -> torch.Tensor:
        sequence_length: int = input_sequence.shape[0]
        batch_size: int = input_sequence.shape[1]
        device: torch.device = input_sequence.device

        encoder_hidden = self.init_hidden(batch_size=batch_size,
                                          device=device)

        encoder_outputs = torch.zeros(sequence_length,
                                      batch_size,
                                      self.hidden_size,
                                      device=device)  # shape: [max_src_len, hidden_size]

        verify_shape(tensor=input_sequence, expected=[sequence_length, batch_size])
        verify_shape(tensor=encoder_hidden, expected=[self.num_hidden_layers, batch_size, self.hidden_size])
        verify_shape(tensor=encoder_outputs, expected=[sequence_length, batch_size, self.hidden_size])

        for src_index in range(sequence_length):

            input_token_tensor: torch.Tensor = input_sequence[src_index]

            verify_shape(tensor=input_token_tensor, expected=[batch_size])
            verify_shape(tensor=encoder_hidden, expected=[self.num_hidden_layers, batch_size, self.hidden_size])

            encoder_output, encoder_hidden = self(input_token_tensor, encoder_hidden)

            verify_shape(tensor=encoder_hidden, expected=[self.num_hidden_layers, batch_size, self.hidden_size])
            verify_shape(tensor=encoder_output, expected=[1, batch_size, self.hidden_size])

            verify_shape(tensor=encoder_output[0], expected=[batch_size, self.hidden_size])
            verify_shape(tensor=encoder_outputs[src_index], expected=[batch_size, self.hidden_size])

            encoder_outputs[src_index] = encoder_output[0]

        verify_shape(tensor=encoder_outputs, expected=[sequence_length, batch_size, self.hidden_size])
        return encoder_outputs


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
                batch_size: int):  # type: ignore[override]

        if encoder_outputs.shape[0] != self.max_src_length:
            raise ValueError("Encoder outputs provided to this method must have same length as self.max_src_length:" +
                             f"\t{encoder_outputs.shape[0]} != {self.max_src_length}")

        # actual_src_length: int = max(self.max_src_length, input_tensor.shape[0])
        # print(f"self.max_src_length={self.max_src_length}\tinput_tensor.shape[0]={input_tensor.shape[0]}")
        verify_shape(tensor=input_tensor, expected=[1, batch_size])
        verify_shape(tensor=hidden, expected=[self.gru.num_layers, batch_size, self.gru.hidden_size])
        verify_shape(tensor=encoder_outputs, expected=[self.max_src_length, batch_size, self.encoder_hidden_size])

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

        verify_shape(tensor=attn_weights, expected=[batch_size, self.max_src_length])
        verify_shape(tensor=encoder_outputs, expected=[self.max_src_length, batch_size, self.encoder_hidden_size])

        # Permute dimensions to prepare for batched matrix-matrix multiply
        encoder_outputs = encoder_outputs.permute(1, 2, 0)
        attn_weights = attn_weights.unsqueeze(2)

        verify_shape(tensor=encoder_outputs, expected=[batch_size, self.encoder_hidden_size, self.max_src_length])
        verify_shape(tensor=attn_weights, expected=[batch_size, self.max_src_length, 1])

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

        verify_shape(tensor=attn_weights, expected=[batch_size, self.max_src_length, 1])
        attn_weights = attn_weights.squeeze(dim=2)

        verify_shape(tensor=output, expected=[batch_size, self.output_size])
        verify_shape(tensor=hidden, expected=[self.gru.num_layers, batch_size, self.decoder_hidden_size])
        verify_shape(tensor=attn_weights, expected=[batch_size, self.max_src_length])

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

    def decode_sequence(self,
                        encoder_outputs: torch.Tensor,
                        start_of_sequence_symbol: int,
                        max_length: int,
                        target_tensor: torch.Tensor = None):

        encoded_sequence_length: int = encoder_outputs.shape[0]
        batch_size: int = encoder_outputs.shape[1]
        encoder_hidden_size: int = encoder_outputs.shape[2]
        device = encoder_outputs.device

        decoder_input = torch.tensor(data=[[start_of_sequence_symbol]*batch_size],
                                     dtype=torch.long,
                                     device=device)

        decoder_hidden = self.init_hidden(batch_size=batch_size, device=device)

        verify_shape(tensor=decoder_input, expected=[1, batch_size])
        verify_shape(tensor=decoder_hidden, expected=[self.gru.num_layers, batch_size, self.gru.hidden_size])

        results: List[torch.Tensor] = list()

        for index in range(max_length):
            verify_shape(tensor=decoder_input, expected=[1, batch_size])
            verify_shape(tensor=decoder_hidden,
                         expected=[self.gru.num_layers, batch_size, self.gru.hidden_size])

            decoder_output, decoder_hidden, decoder_attention = self(
                decoder_input, decoder_hidden, encoder_outputs, batch_size)

            verify_shape(tensor=decoder_output, expected=[batch_size, self.output_size])
            verify_shape(tensor=decoder_hidden,
                         expected=[self.gru.num_layers, batch_size, self.gru.hidden_size])
            verify_shape(tensor=decoder_attention, expected=[batch_size, encoded_sequence_length])

            results.append(decoder_output)

            if target_tensor is None:
                _, top_i = decoder_output.topk(1)
                decoder_input = top_i.detach().permute(1, 0)
            else:
                # print(f"target_tensor.shape={target_tensor.shape}\tindex={index}\tmax_length={max_length}")
                decoder_input = target_tensor[index].unsqueeze(dim=0)

        return torch.stack(tensors=results, dim=0)


def evaluate(corpus: Corpus,
             encoder: EncoderRNN,
             decoder: AttnDecoderRNN):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        for batch in torch.utils.data.DataLoader(dataset=corpus, batch_size=1):

            input_tensor: torch.Tensor = batch["data"].permute(1, 0)
            #start_of_sequence: torch.Tensor = batch["start-of-sequence"].permute(1, 0)

            # actual_batch_size: int = min(batch_size, input_tensor.shape[1])

            #print(f"input_tensor.shape={input_tensor.shape}")
            #print(batch["string"])
            encoder_outputs = encoder.encode_sequence(input_tensor)
            #print(encoder_outputs.shape)
            # sys.exit()
            #print(f"encoder_outputs.shape={encoder_outputs.shape}\tmax_decoder_seq={corpus.max_label_length}")
            decoder_output=decoder.decode_sequence(encoder_outputs=encoder_outputs,
                                                   start_of_sequence_symbol=corpus.characters.start_of_sequence_int,
                                                   max_length=corpus.label_tensor_length)
            _, top_i = decoder_output.topk(k=1)

            predictions = top_i.squeeze(dim=2).squeeze(dim=1).tolist()
            #print(f"top_i.shape={top_i.shape}\t{predictions}")
            predicted_string = "".join([corpus.characters.int2string[i] for i in predictions])

            # print(f"decoder_output.shape={decoder_output.shape}\ttop_i.shape={top_i.shape}\tpredictions={predictions}\t{predicted_string}")
            print(predicted_string)

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
          start_of_sequence_symbol: int,
          teacher_forcing_ratio: float) -> float:

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss: torch.Tensor = torch.tensor(0, dtype=torch.float, device=device)  # shape: [] meaning this is a scalar

    encoder_outputs = encoder.encode_sequence(input_tensor)

    decoder_input = target_tensor[0].unsqueeze(dim=0)
    decoder_hidden = decoder.init_hidden(batch_size=batch_size, device=device)

    verify_shape(tensor=decoder_input, expected=[1, batch_size])
    verify_shape(tensor=target_tensor, expected=[max_tgt_length, batch_size])
    verify_shape(tensor=decoder_hidden, expected=[decoder.gru.num_layers, batch_size, decoder.gru.hidden_size])

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # use_teacher_forcing = False

    decoder_output = decoder.decode_sequence(encoder_outputs=encoder_outputs,
                                             start_of_sequence_symbol=start_of_sequence_symbol,
                                             max_length=max_tgt_length,
                                             target_tensor=target_tensor if use_teacher_forcing else None)
    # print(f"input_tensor.shape={input_tensor.shape}\tdecoder_output.shape={decoder_output.shape}\ttarget_tensor.shape={target_tensor.shape}\tmax_tgt_length={max_tgt_length}")

    # Our loss function requires predictions to be of the shape NxC, where N is the number of predictions and C is the number of possible predicted categories
    predictions = decoder_output.reshape(-1, decoder.output_size)  # Reshaping from [seq_len, batch_size, decoder.output_size] to [seq_len*batch_size, decoder.output_size]
    labels = target_tensor.reshape(-1)                             # Reshaping from [seq_len, batch_size]                      to [seq_len*batch_size]
    loss += criterion(predictions, labels)
    #print(f"\t{decoder_output.view(-1,decoder_output.shape[-1]).shape}")
    #print(target_tensor.reshape(-1))
#    print(f"\t{target_tensor.view(-1)}")
    #sys.exit()
    #loss += criterion(decoder_output.view(1,1,-1), target_tensor.view(-1))
    # loss += criterion(decoder_output.squeeze(dim=1), target_tensor.squeeze(dim=1))
    # for index, decoder_output in enumerate(start=1,
    #                                        iterable=decoder.decode_sequence(encoder_outputs=encoder_outputs,
    #                                               start_of_sequence_symbol=start_of_sequence_symbol,
    #                                               max_length=max_tgt_length,
    #                                               target_tensor=target_tensor if use_teacher_forcing else None)):
    #
    #     loss += criterion(decoder_output, target_tensor[index])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def train_iters(*,  #data: Data,
                corpus: Corpus,
                encoder: EncoderRNN,
                decoder: AttnDecoderRNN,
                device: torch.device,
                n_iters: int,
                batch_size: int,
                teacher_forcing_ratio: float,
                print_every: int = 1000,
                learning_rate: float = 0.01
                ) -> None:

    data = torch.utils.data.DataLoader(dataset=corpus, batch_size=batch_size)

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

    criterion: nn.NLLLoss = nn.NLLLoss(reduction='mean') #ignore_index=corpus.characters.pad_int)
                                       
    #for pair in parallel_data:
    #    print(f"src={len(pair['data'])}\ttgt={len(pair['labels'])}")

    for iteration in range(1, n_iters + 1):  # type: int

        # training_pair: ParallelTensor = training_pairs[iteration - 1]
        # input_tensor: torch.Tensor = training_pair.source   # shape: [seq_len, batch_size=1]
        # target_tensor: torch.Tensor = training_pair.target  # shape: [seq_len, batch_size=1]

        for batch in data:
            #print(f"batch['data'].shape={batch['data'].shape}\tbatch['labels'].shape{batch['labels'].shape}")
            #sys.exit()
            input_tensor: torch.Tensor = batch["data"].permute(1, 0)
            target_tensor: torch.Tensor = batch["labels"].permute(1, 0)

            actual_batch_size: int = min(batch_size, input_tensor.shape[1])

            verify_shape(tensor=input_tensor, expected=[corpus.word_tensor_length, actual_batch_size])
            verify_shape(tensor=target_tensor, expected=[corpus.label_tensor_length, actual_batch_size])

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
                                max_src_length=corpus.word_tensor_length,
                                max_tgt_length=corpus.label_tensor_length,
                                batch_size=actual_batch_size,
                                start_of_sequence_symbol=corpus.characters.start_of_sequence_int,
                                teacher_forcing_ratio=teacher_forcing_ratio)

            print_loss_total += loss
            plot_loss_total += loss

        if iteration % print_every == 0:
            print_loss_avg: float = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(since=start, percent=iteration/n_iters),
                                         iteration, iteration / n_iters * 100, print_loss_avg))
            sys.stdout.flush()


def run_training():

    max_length = 10

    teacher_forcing_ratio = 0.5

    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    training_filename = "data/shakespeare50k.txt" if torch.cuda.is_available() else "data/shakespeare.tiny"
    test_filename="data/shakespeare.test"
#    training_filename="data/a.train"
#    test_filename="data/a.train"
    training_filename="/usr/share/dict/words"
    
    training_corpus = Corpus(name="training", filename=training_filename, max_length=max_length, device=device)
    test_corpus = Corpus(name="test", filename=test_filename, vocab=training_corpus.characters,
                         max_length=training_corpus._max_word_length, device=device)

    #for word in test_corpus.words:
    #    print(f"{''.join(word.characters)}\t{''.join(word.label)}")
    #sys.exit()

    encoder1: EncoderRNN = EncoderRNN(input_size=len(training_corpus.characters),
                                      embedding_size=200,
                                      hidden_size=256,
                                      num_hidden_layers=1).to(device=device)

    attn_decoder1 = AttnDecoderRNN(embedding_size=190,
                                   decoder_hidden_size=7,
                                   encoder_hidden_size=encoder1.hidden_size,
                                   num_hidden_layers=1,
                                   output_size=len(training_corpus.characters),
                                   dropout_p=0.1,
                                   max_src_length=training_corpus.word_tensor_length).to(device=device)

#    for i in range(len(training_corpus)):
#        print(str(training_corpus.words[i]) + "\t" + ''.join([training_corpus.characters.int2string[i] for i in training_corpus[i]["data"].tolist()]) + "\t" + ''.join([training_corpus.characters.int2string[i] for i in training_corpus[i]["labels"].tolist()]))
#    print()
#    print()
    
    train_iters(corpus=training_corpus,
                encoder=encoder1,
                decoder=attn_decoder1,
                device=device,
                n_iters=1000,
                batch_size=25000,
                print_every=1,
                learning_rate=0.01,
                teacher_forcing_ratio=teacher_forcing_ratio)

    evaluate(corpus=test_corpus,
             encoder=encoder1,
             decoder=attn_decoder1)


if __name__ == "__main__":

    run_training()
