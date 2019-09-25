from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax


from vocab import Vocab


def verify_shape(*, tensor: torch.Tensor, expected: List[int]) -> None:
    if tensor.shape == torch.Size(expected):
        return
    else:
        raise ValueError(f"Tensor found with shape {tensor.shape} when {torch.Size(expected)} was expected.")


# class Encoder(nn.Module):
#
#     def __init__(self, *,
#                  input_size: int,
#                  hidden_size: int,
#                  num_layers: int):
#         super().__init__()
#
#         self.input_size: int = input_size
#         self.hidden_size: int = hidden_size
#         self.num_layers: int = num_layers
#
#         self.rnn: nn.Module = nn.RNN(input_size=input_size,
#                                      hidden_size=hidden_size,
#                                      num_layers=num_layers,
#                                      batch_first=True,
#                                      bidirectional=True)
#
#     def forward(self, *,
#                 batch_size: int,
#                 seq_len: int,
#                 input_tensor: torch.Tensor) -> torch.Tensor:
#
#         assert input_tensor.shape == torch.Size([batch_size, seq_len, self.input_size])
#
#         rnn_outputs: Tuple[torch.Tensor, torch.Tensor] = self.rnn(input=input_tensor, hx=None)
#
#         rnn_output: torch.Tensor = rnn_outputs[0]
#         assert rnn_output.shape == torch.Size([batch_size, seq_len, 2 * self.hidden_size])
#
#         return rnn_output


class Decoder(nn.Module):

    def __init__(self, *,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 context_size: int,
                 output_size: int):
        super().__init__()

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.context_size: int = context_size
        self.output_size: int = output_size

        self.rnn: nn.RNN = nn.RNN(input_size=input_size+context_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)

        self.output_layer: nn.Module = nn.Linear(in_features=hidden_size, out_features=self.output_size)

    def forward(self, *,
                batch_size: int,
                input_tensor: torch.Tensor,
                context_tensor: torch.Tensor,
                hidden_tensor: torch.Tensor) -> Tuple[torch.LongTensor, torch.Tensor]:

        verify_shape(tensor=input_tensor, expected=[batch_size, 1, self.input_size])
        verify_shape(tensor=context_tensor, expected=[batch_size, 1, self.context_size])
        verify_shape(tensor=hidden_tensor, expected=[batch_size, self.rnn.num_layers, self.hidden_size])

        rnn_input: torch.Tensor = torch.cat(tensors=(input_tensor, context_tensor), dim=2)
        verify_shape(tensor=rnn_input, expected=[batch_size, 1, self.input_size + self.context_size])

        # torch.nn.RNN expects its hidden layer tensor to be of shape [num_layers * num_directions, batch, hidden_size]
        rnn_outputs: torch.Tensor = self.rnn(input=rnn_input, hx=hidden_tensor.permute(dims=[1, 0, 2]))

        rnn_output: torch.Tensor = rnn_outputs[0]
        rnn_hidden: torch.Tensor = rnn_outputs[1]

        verify_shape(tensor=rnn_output, expected=[batch_size, 1, self.hidden_size])
        verify_shape(tensor=rnn_hidden, expected=[self.rnn.num_layers, batch_size, self.hidden_size])

        rnn_hidden_permuted: torch.Tensor = rnn_hidden.permute(1, 0, 2)
        verify_shape(tensor=rnn_hidden_permuted, expected=[batch_size, self.rnn.num_layers, self.hidden_size])

        output_layer: torch.Tensor = relu(self.output_layer(rnn_output))
        verify_shape(tensor=output_layer, expected=[batch_size, 1, self.output_size])

        # prediction_distribution: torch.Tensor = softmax(output_layer, dim=1)
        # verify_shape(tensor=prediction_distribution, expected=[batch_size, 1, self.output_size])
        #
        # output_token: torch.LongTensor = prediction_distribution.topk(k=1, dim=2).indices.squeeze()
        # verify_shape(tensor=output_token, expected=[batch_size])
        #
        # return output_token, rnn_hidden_permuted

        return output_layer, rnn_hidden_permuted


class Attention(nn.Module):

    def __init__(self, *,
                 attention_hidden_size: int,
                 encoder_hidden_size: int,
                 decoder_hidden_size: int):
        super().__init__()

        self.activation_function = torch.nn.ReLU()

        self.attention_hidden_size: int = attention_hidden_size
        self.encoder_hidden_size: int = encoder_hidden_size
        self.decoder_hidden_size: int = decoder_hidden_size

        self.context_size = 2 * self.encoder_hidden_size

        self.W = nn.Linear(self.decoder_hidden_size, self.attention_hidden_size)

        self.U = nn.Linear(self.context_size, self.attention_hidden_size)

        self.v = nn.Parameter(torch.rand(self.attention_hidden_size))

    def calculate_alpha(self, *,
                        batch_size: int,
                        max_seq_len: int,
                        previous_decoder_hidden_state: torch.Tensor,
                        encoder_final_hidden_layers: torch.Tensor) -> torch.Tensor:

        assert batch_size > 0
        assert max_seq_len > 0

        verify_shape(tensor=previous_decoder_hidden_state, expected=[batch_size, 1, self.decoder_hidden_size])
        verify_shape(tensor=encoder_final_hidden_layers, expected=[batch_size, max_seq_len, self.context_size])

        tmp_1: torch.Tensor = self.W(previous_decoder_hidden_state)
        verify_shape(tensor=tmp_1, expected=[batch_size, 1, self.attention_hidden_size])

        # tmp_2: torch.Tensor = tmp_1.reshape(batch_size, 1, self.attention_hidden_size)
        # verify_shape(tensor=tmp_2, expected=[batch_size, 1, self.attention_hidden_size])

        tmp_3: torch.Tensor = self.U(encoder_final_hidden_layers)
        verify_shape(tensor=tmp_3, expected=[batch_size, max_seq_len, self.attention_hidden_size])

        tmp_4: torch.Tensor = torch.add(tmp_1, tmp_3)
        verify_shape(tensor=tmp_4, expected=[batch_size, max_seq_len, self.attention_hidden_size])

        tmp_5: torch.Tensor = self.activation_function(tmp_4)
        verify_shape(tensor=tmp_5, expected=[batch_size, max_seq_len, self.attention_hidden_size])

        energy: torch.Tensor = torch.matmul(tmp_5, self.v)
        verify_shape(tensor=energy, expected=[batch_size, max_seq_len])

        result: torch.Tensor = softmax(input=energy, dim=1)
        verify_shape(tensor=result, expected=[batch_size, max_seq_len])

        return result

    def forward(self, *,
                batch_size: int,
                max_seq_len: int,
                previous_decoder_hidden_state: torch.Tensor,
                encoder_final_hidden_layers: torch.Tensor) -> torch.Tensor:

        alpha: torch.Tensor = self.calculate_alpha(batch_size=batch_size,
                                                   max_seq_len=max_seq_len,
                                                   previous_decoder_hidden_state=previous_decoder_hidden_state,
                                                   encoder_final_hidden_layers=encoder_final_hidden_layers)

        assert alpha.shape == torch.Size([batch_size, max_seq_len])

        alpha_reshaped: torch.Tensor = alpha.reshape(batch_size, 1, max_seq_len)
        assert alpha_reshaped.shape == torch.Size([batch_size, 1, max_seq_len])

        assert encoder_final_hidden_layers.shape == torch.Size([batch_size,
                                                                max_seq_len,
                                                                self.context_size])

        c: torch.Tensor = torch.bmm(alpha_reshaped, encoder_final_hidden_layers)
        assert c.shape == torch.Size([batch_size, 1, self.context_size])
        c = c.squeeze(dim=1)
        assert c.shape == torch.Size([batch_size, self.context_size])

        return c


class EncoderWithEmbedding(nn.Module):

    def __init__(self, *,
                 vocab: Vocab,
                 embedding_size: int,
                 hidden_size: int,
                 num_layers: int):
        super().__init__()

        self.input_size: int = len(vocab)
        self.embedding_size: int = embedding_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers

        self.embedding: nn.Embedding = nn.Embedding(num_embeddings=self.input_size,
                                                    embedding_dim=self.embedding_size,
                                                    padding_idx=vocab.pad)

        self.rnn: nn.Module = nn.RNN(input_size=self.embedding_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     bidirectional=True)

    def forward(self, *,
                batch_size: int,
                seq_len: int,
                input_tensor: torch.Tensor) -> torch.Tensor:

        verify_shape(tensor=input_tensor, expected=[batch_size, seq_len])

        input_embeddings: torch.Tensor = self.embedding(input_tensor)
        verify_shape(tensor=input_embeddings, expected=[batch_size, seq_len, self.embedding_size])

        rnn_outputs: Tuple[torch.Tensor, torch.Tensor] = self.rnn(input=input_embeddings, hx=None)

        rnn_output: torch.Tensor = rnn_outputs[0]
        verify_shape(tensor=rnn_output, expected=[batch_size, seq_len, 2 * self.hidden_size])

        return rnn_output


class DecoderWithAttention(nn.Module):

    def __init__(self, *,
                 vocab: Vocab,
                 embedding_size: int,
                 encoder_hidden_size: int,
                 decoder_hidden_size: int,
                 decoder_num_layers: int,
                 attention_hidden_size: int):
        super().__init__()

        self.vocab: Vocab = vocab

        self.hidden_size: int = decoder_hidden_size
        self.encoder_hidden_size: int = encoder_hidden_size

        self.embedding: nn.Embedding = nn.Embedding(num_embeddings=len(vocab),
                                                    embedding_dim=embedding_size,
                                                    padding_idx=vocab.pad)

        self.attention: Attention = Attention(encoder_hidden_size=encoder_hidden_size,
                                              decoder_hidden_size=decoder_hidden_size,
                                              attention_hidden_size=attention_hidden_size)

        self.decoder: Decoder = Decoder(input_size=embedding_size,
                                        hidden_size=decoder_hidden_size,
                                        num_layers=decoder_num_layers,
                                        context_size=self.attention.context_size,
                                        output_size=len(vocab))

    def forward(self, *,
                batch_size: int,
                input_seq_len: int,
                previous_decoder_output: torch.LongTensor,
                previous_decoder_hidden_state: torch.Tensor,
                encoder_states: torch.Tensor,
                output_seq_len: int) -> torch.Tensor:

        # result: torch.Tensor = torch.zeros(output_seq_len, batch_size, len(self.vocab), device=device)
        result: List[torch.Tensor] = list()

        verify_shape(tensor=previous_decoder_output, expected=[batch_size])

        input_embeddings: torch.Tensor = self.embedding(previous_decoder_output)
        verify_shape(tensor=input_embeddings, expected=[batch_size, self.embedding.embedding_dim])

        verify_shape(tensor=encoder_states, expected=[batch_size, input_seq_len, 2 * self.encoder_hidden_size])

        # decoder_hidden_state: torch.Tensor = torch.zeros(batch_size, 1, self.decoder.hidden_size, device=device)
        # decoded_token_ids: torch.LongTensor = torch.tensor([self.vocab.start_of_sequence] * batch_size,
        #                                                    dtype=torch.long, device=device)
        #print(f"{output_seq_len} {len(result)}")
        for t in range(output_seq_len):

            verify_shape(tensor=previous_decoder_hidden_state, expected=[batch_size, 1, self.hidden_size])
            verify_shape(tensor=previous_decoder_output, expected=[batch_size])

            context: torch.Tensor = self.attention(batch_size=batch_size,
                                                   max_seq_len=input_seq_len,
                                                   previous_decoder_hidden_state=previous_decoder_hidden_state,
                                                   encoder_final_hidden_layers=encoder_states).unsqueeze(dim=1)
            verify_shape(tensor=context, expected=[batch_size, 1, self.attention.context_size])

            decoded_token: torch.Tensor = self.embedding(previous_decoder_output).unsqueeze(dim=1)
            verify_shape(tensor=decoded_token, expected=[batch_size, 1, self.embedding.embedding_dim])

            decoder_results: Tuple[torch.Tensor, torch.Tensor] = self.decoder(batch_size=batch_size,
                                                                              input_tensor=decoded_token,
                                                                              context_tensor=context,
                                                                              hidden_tensor=previous_decoder_hidden_state)

            # decoded_token_ids: torch.LongTensor = decoder_results[0]
            decoder_output: torch.Tensor = decoder_results[0]
            decoder_hidden_state: torch.Tensor = decoder_results[1]

            # verify_shape(tensor=result, expected=[output_seq_len, batch_size, len(self.vocab)])
            verify_shape(tensor=decoder_output, expected=[batch_size, 1, len(self.vocab)])
            # verify_shape(tensor=decoded_token_ids, expected=[batch_size])

            # output_tokens[t] = decoded_token_ids
            # result[t] = decoder_output.squeeze(dim=1)
            result.append(decoder_output.squeeze(dim=1))

            #print(f"{t} {output_seq_len} {len(result)}")

        # return output_tokens.permute(1, 0)
        permuted_result: torch.Tensor = torch.stack(result).permute(1, 0, 2)
        verify_shape(tensor=permuted_result, expected=[batch_size, output_seq_len, len(self.vocab)])

        return permuted_result
