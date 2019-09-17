from typing import Tuple

import torch
import torch.nn as nn

from vocab import Vocab


class RNN_LM(nn.Module):

    def __init__(self, *,
                 vocab: Vocab,
                 embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 pad_value: int):
        super().__init__()

        self.vocab: Vocab = vocab
        self.input_size: int = len(vocab)
        self.embedding_size: int = embedding_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.output_size: int = len(vocab)

        self.embedding: nn.Module = nn.Embedding(num_embeddings=self.input_size,
                                                 embedding_dim=embedding_size,
                                                 padding_idx=pad_value)

        self.rnn: nn.Module = nn.RNN(input_size=embedding_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     batch_first=True)

        self.output_layer: nn.Module = nn.Linear(in_features=hidden_size, out_features=self.output_size)

    def forward(self, *,
                batch_size: int,
                seq_len: int,
                input_tensor: torch.Tensor,
                previous_hidden: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        assert input_tensor.shape == torch.Size([batch_size, seq_len])

        if previous_hidden is not None:
            assert previous_hidden.shape == torch.Size([self.num_layers, batch_size, self.hidden_size])

        rnn_input: torch.Tensor = self.embedding(input_tensor)
        assert rnn_input.shape == torch.Size([batch_size, seq_len, self.embedding_size])

        rnn_outputs: Tuple[torch.Tensor, torch.Tensor] = self.rnn(input=rnn_input, hx=previous_hidden)

        rnn_output: torch.Tensor = rnn_outputs[0]
        h_n: torch.Tensor = rnn_outputs[1]

        assert rnn_output.shape == torch.Size([batch_size, seq_len, self.hidden_size])
        assert h_n.shape == torch.Size([self.num_layers, batch_size, self.hidden_size])

        output_layer: torch.Tensor = self.output_layer(rnn_output)

        assert output_layer.shape == torch.Size([batch_size, seq_len, self.output_size])

        return output_layer, h_n
