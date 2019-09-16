from typing import Tuple

import torch
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import TextSequence
from lm import RNN_LM


def train_lm(*, path: str,
             embedding_size: int,
             hidden_size: int,
             num_layers: int,
             batch_size: int,
             num_epochs: int,
             learning_rate: float,
             device_name: str) -> RNN_LM:

    device = torch.device(device_name)

    words: TextSequence = TextSequence(path=path)
    data: DataLoader = DataLoader(dataset=words, batch_size=batch_size)

    vocab_size: int = len(words.vocab)

    lm: RNN_LM = RNN_LM(vocab=words.vocab,
                        embedding_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        pad_value=words.vocab.pad)

    lm.train()

    optimizer: Adam = Adam(lm.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        total_loss_across_batches: float = 0.0

        for batch in data:  # type: torch.Tensor

            training_examples: torch.Tensor = batch["data"]
            training_labels: torch.LongStorage = batch["labels"]

            # At the end of the data set, the actual batch size may be smaller than batch_size, and that's OK
            actual_batch_size: int = min(batch_size, training_examples.shape[0])

            assert training_examples.shape == torch.Size([actual_batch_size, words.max_len])
            assert training_labels.shape == torch.Size([actual_batch_size, words.max_len])

            training_examples.to(device)

            result: Tuple[torch.Tensor, torch.Tensor] = lm(batch_size=actual_batch_size,
                                                           seq_len=words.max_len,
                                                           input_tensor=training_examples)

            output: torch.Tensor = result[0]
            hidden: torch.Tensor = result[1]

            assert output.shape == torch.Size([actual_batch_size, words.max_len, vocab_size])
            assert hidden.shape == torch.Size([num_layers, actual_batch_size, hidden_size])

            predictions: torch.Tensor = output.reshape(shape=(actual_batch_size*words.max_len, vocab_size))
            assert predictions.shape == torch.Size([actual_batch_size*words.max_len, vocab_size])

            labels: torch.Tensor = training_labels.reshape(shape=(actual_batch_size * words.max_len,))
            # print(f"prediction={predictions[1]}")
            #print(f"label={labels}")
            assert labels.shape == torch.Size([actual_batch_size * words.max_len])

            loss = cross_entropy(input=predictions,
                                 target=labels,
                                 ignore_index=words.vocab.pad)

            total_loss_across_batches += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}\tloss {total_loss_across_batches}")

    return lm


if __name__ == "__main__":

    lm = train_lm(path="training_data.txt",
                  embedding_size=64,
                  hidden_size=128,
                  num_layers=1,
                  batch_size=10,
                  num_epochs=10,
                  learning_rate=0.01,
                  device_name="cpu")

    for _ in range(10):

        result: str = lm.generate_random_sequence()

        print(result)