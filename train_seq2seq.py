import math
import time
from typing import List, Tuple

import torch
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.utils.data import DataLoader

from data import PigLatin
from seq2seq import EncoderWithEmbedding, DecoderWithAttention, verify_shape


def format_time(seconds):
    minutes = math.floor(seconds / 60)
    hours = math.floor(minutes / 60)
    minutes -= hours * 60
    seconds -= minutes * 60
    return f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(round(seconds)).zfill(2)}"


def time_since(since):
    now = time.time()
    s = now - since
    return format_time(s)


def train_seq2seq(*, path: str,
                  embedding_size: int,
                  encoder_hidden_size: int,
                  encoder_num_layers: int,
                  decoder_hidden_size: int,
                  decoder_num_layers: int,
                  attention_hidden_size,
                  batch_size: int,
                  num_epochs: int,
                  learning_rate: float,
                  device_name: str,
                  save_encoder: str,
                  save_decoder: str) -> None:

    start = time.time()
    
    device = torch.device(device_name)

    words: PigLatin = PigLatin(path=path)
    data: DataLoader = DataLoader(dataset=words, batch_size=batch_size)

    encoder: EncoderWithEmbedding = EncoderWithEmbedding(vocab=words.vocab,
                                                         embedding_size=embedding_size,
                                                         hidden_size=encoder_hidden_size,
                                                         num_layers=encoder_num_layers).to(device)

    decoder: DecoderWithAttention = DecoderWithAttention(vocab=words.vocab,
                                                         embedding_size=embedding_size,
                                                         encoder_hidden_size=encoder_hidden_size,
                                                         decoder_hidden_size=decoder_hidden_size,
                                                         decoder_num_layers=decoder_num_layers,
                                                         attention_hidden_size=attention_hidden_size).to(device)

    encoder.train()
    decoder.train()

    optimize_encoder: SGD = SGD(encoder.parameters(), lr=learning_rate)
    optimize_decoder: SGD = SGD(decoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        optimize_encoder.zero_grad()
        optimize_decoder.zero_grad()

        total_loss_across_batches: float = 0.0

        for batch in data:  # type: torch.Tensor

            training_examples: torch.Tensor = batch["data"].to(device)
            training_labels: torch.LongTensor = batch["labels"].to(device)
            decoder_previous_output: torch.LongTensor = batch["start-of-sequence"].squeeze().to(device)

            # At the end of the data set, the actual batch size may be smaller than batch_size, and that's OK
            actual_batch_size: int = min(batch_size, training_examples.shape[0])

            decoder_hidden_state: torch.Tensor = torch.zeros(actual_batch_size, 1, decoder.hidden_size).to(device)

            verify_shape(tensor=training_examples, expected=[actual_batch_size, words.max_len])
            verify_shape(tensor=training_labels, expected=[actual_batch_size, words.max_len])

            encoder_states: torch.Tensor = encoder(batch_size=actual_batch_size,
                                                   seq_len=words.max_len,
                                                   input_tensor=training_examples)

            decoder_output_list: List[torch.Tensor] = list()
            for _ in range(words.max_len):

                decoder_results: Tuple[torch.Tensor, torch.Tensor] = decoder(batch_size=actual_batch_size,
                                                                             input_seq_len=words.max_len,
                                                                             previous_decoder_output=decoder_previous_output,
                                                                             previous_decoder_hidden_state=decoder_hidden_state,
                                                                             encoder_states=encoder_states)

                decoder_raw_output: torch.Tensor = decoder_results[0]
                decoder_hidden_state: torch.Tensor = decoder_results[1]

                verify_shape(tensor=decoder_raw_output, expected=[actual_batch_size, 1, len(decoder.vocab)])
                verify_shape(tensor=decoder_hidden_state, expected=[actual_batch_size, 1, decoder.hidden_size])

                decoder_previous_output: torch.LongTensor = decoder_raw_output.squeeze(dim=1).topk(k=1).indices.squeeze(dim=1)
                verify_shape(tensor=decoder_previous_output, expected=[actual_batch_size])

                decoder_output_list.append(decoder_raw_output.squeeze(dim=1))

            decoder_output: torch.Tensor = torch.stack(tensors=decoder_output_list, dim=0).permute(1, 0, 2)

            verify_shape(tensor=decoder_output, expected=[actual_batch_size, words.max_len, len(decoder.vocab)])

            decoder_output_reshaped: torch.Tensor = decoder_output.reshape(shape=(actual_batch_size * words.max_len,
                                                                                  len(decoder.vocab)))
            verify_shape(tensor=decoder_output_reshaped, expected=[actual_batch_size * words.max_len,
                                                                   len(decoder.vocab)])

            labels_reshaped: torch.Tensor = training_labels.reshape(shape=(actual_batch_size * words.max_len,))
            verify_shape(tensor=labels_reshaped, expected=[actual_batch_size * words.max_len])

            loss = cross_entropy(input=decoder_output_reshaped,
                                 target=labels_reshaped,
                                 ignore_index=words.vocab.pad,
                                 reduction='mean')

            total_loss_across_batches += loss.item()

            loss.backward()
            optimize_encoder.step()
            optimize_decoder.step()

        if epoch % 10 == 0:
            print(f"Epoch {str(epoch).zfill(len(str(num_epochs)))}\t" +
                  f"loss {round(number=total_loss_across_batches, ndigits=3)}\t" +
                  f"{time_since(start)}")
            sys.stdout.flush()

    print(f"Saving encoder model to {save_encoder}...")
    torch.save(encoder, save_encoder)

    print(f"Saving decoder model to {save_decoder}...")
    torch.save(decoder, save_decoder)

    return


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 4:

        print(f"Training seq2seq from {sys.argv[1]}")
        sys.stdout.flush()

        train_seq2seq(path=sys.argv[1],
                      embedding_size=64,
                      encoder_hidden_size=256,
                      encoder_num_layers=1,
                      decoder_hidden_size=256,
                      decoder_num_layers=1,
                      attention_hidden_size=128,
                      batch_size=4000,
                      num_epochs=400,
                      learning_rate=0.001,
                      device_name="cuda:0" if torch.cuda.is_available() else "cpu",
                      save_encoder=sys.argv[2],
                      save_decoder=sys.argv[3])


