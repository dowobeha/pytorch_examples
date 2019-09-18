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
                        pad_value=words.vocab.pad).to(device)

#    print(type(device))
#    lm.cuda(device)
#    import sys
#    sys.exit()
    lm.train()

    optimizer: Adam = Adam(lm.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        total_loss_across_batches: float = 0.0

#        print(f"Epoch {epoch}", end="\t")
#        sys.stdout.flush()
        for batch in data:  # type: torch.Tensor
#            print("Loading batch")
#            sys.stdout.flush()
            training_examples: torch.Tensor = batch["data"].to(device)
            training_labels: torch.LongStorage = batch["labels"].to(device)

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
            assert labels.shape == torch.Size([actual_batch_size * words.max_len])

            loss = cross_entropy(input=predictions,
                                 target=labels,
                                 ignore_index=words.vocab.pad)

            total_loss_across_batches += loss.item()

            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {str(epoch).zfill(len(str(num_epochs)))}\tloss {total_loss_across_batches}")
            sys.stdout.flush()

    return lm


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 3:

        print(f"Training RNN LM from {sys.argv[1]}")
        sys.stdout.flush()
        lm: RNN_LM = train_lm(path=sys.argv[1],
                              embedding_size=64,
                              hidden_size=128,
                              num_layers=1,
                              batch_size=8128,
                              num_epochs=10000,
                              learning_rate=0.001,
                              device_name="cuda:0")

        print(f"Saving RNN LM to {sys.argv[2]}...")
        torch.save(lm, sys.argv[2])
