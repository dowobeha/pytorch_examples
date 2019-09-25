
import math
import time

import torch
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import PigLatin
from seq2seq import EncoderDecoderWithAttention, verify_shape

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return asMinutes(s)


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
             device_name: str) -> EncoderDecoderWithAttention:

    start = time.time()
    
    device = torch.device(device_name)

    words: PigLatin = PigLatin(path=path)
    data: DataLoader = DataLoader(dataset=words, batch_size=batch_size)

    seq2seq: EncoderDecoderWithAttention = EncoderDecoderWithAttention(vocab=words.vocab,
                                                                       embedding_size=embedding_size,
                                                                       encoder_hidden_size=encoder_hidden_size,
                                                                       encoder_num_layers=encoder_num_layers,
                                                                       decoder_hidden_size=decoder_hidden_size,
                                                                       decoder_num_layers=decoder_num_layers,
                                                                       attention_hidden_size=attention_hidden_size).to(device)

    seq2seq.train()

    optimizer: Adam = Adam(seq2seq.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        total_loss_across_batches: float = 0.0

        for batch in data:  # type: torch.Tensor

            training_examples: torch.Tensor = batch["data"].to(device)
            training_labels: torch.LongStorage = batch["labels"].to(device)

            # At the end of the data set, the actual batch size may be smaller than batch_size, and that's OK
            actual_batch_size: int = min(batch_size, training_examples.shape[0])

            verify_shape(tensor=training_examples, expected=[actual_batch_size, words.max_len])
            verify_shape(tensor=training_labels, expected=[actual_batch_size, words.max_len])

            training_examples.to(device)

            seq2seq_output: torch.Tensor = seq2seq(batch_size=actual_batch_size,
                                                   input_seq_len=words.max_len,
                                                   output_seq_len=words.max_len,
                                                   input_tensor=training_examples,
                                                   device=device)
            verify_shape(tensor=seq2seq_output, expected=[actual_batch_size, words.max_len, len(seq2seq.vocab)])

            seq2seq_output_reshaped: torch.Tensor = seq2seq_output.reshape(shape=(actual_batch_size * words.max_len,
                                                                                  len(seq2seq.vocab)))
            verify_shape(tensor=seq2seq_output_reshaped, expected=[actual_batch_size * words.max_len,
                                                                   len(seq2seq.vocab)])

            labels_reshaped: torch.Tensor = training_labels.reshape(shape=(actual_batch_size * words.max_len,))
            verify_shape(tensor=labels_reshaped, expected=[actual_batch_size * words.max_len])

            loss = cross_entropy(input=seq2seq_output_reshaped,
                                 target=labels_reshaped,
                                 ignore_index=words.vocab.pad)

            total_loss_across_batches += loss.item()

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {str(epoch).zfill(len(str(num_epochs)))}\tloss {total_loss_across_batches}\t{timeSince(start)}")
            sys.stdout.flush()

    return seq2seq


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 3:

        print(f"Training seq2seq from {sys.argv[1]}")
        sys.stdout.flush()

        model: EncoderDecoderWithAttention = train_seq2seq(path=sys.argv[1],
                                                      embedding_size=64,
                                                      encoder_hidden_size=256,
                                                      encoder_num_layers=1,
                                                      decoder_hidden_size=256,
                                                      decoder_num_layers=1,
                                                      attention_hidden_size=128,
                                                      batch_size=4000,
                                                      num_epochs=10000,
                                                      learning_rate=0.001,
                                                      device_name="cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Saving seq2seq model to {sys.argv[2]}...")
        torch.save(model, sys.argv[2])
