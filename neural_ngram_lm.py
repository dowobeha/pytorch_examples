import sys
from typing import List, Mapping, MutableMapping

import torch
from torch import nn
from torch.nn.functional import log_softmax, relu, nll_loss
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader


class Vocab:

    pad: int = 0
    oov: int = 1
    start_of_sequence: int = 2
    end_of_sequence: int = 3

    def __init__(self, name: str):
        self.name: str = name
        self.token2index: MutableMapping[str, int] = {}
        self.index2token: MutableMapping[int, str] = {Vocab.pad: "<pad/>",
                                                      Vocab.oov: "<oov/>",
                                                      Vocab.start_of_sequence: "<s>",
                                                      Vocab.end_of_sequence: "</s>"}

    def __getitem__(self, item: str) -> int:
        if item in self.token2index:
            return self.token2index[item]
        else:
            return Vocab.oov

    def __len__(self):
        return len(self.index2token)

    def add_token(self, token: str) -> None:
        if token not in self.token2index:
            n_tokens: int = len(self)
            self.token2index[token] = n_tokens
            self.index2token[n_tokens] = token

    def tensor_from_sequence(self, sequence: List[str], device: torch.device) -> torch.LongTensor:
        indexes: List[int] = [self[token] for token in sequence] 

        return torch.tensor(indexes, dtype=torch.long, device=device)


class BigramData(Dataset):
    def __init__(self, filename: str, device: torch.device):
        self.vocab: Vocab = Vocab(filename)
        self.data: List[int] = list()
        self.device = device

        with open(filename, mode='rt', encoding='utf8') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) > 0:
                    self.data.append(Vocab.start_of_sequence)
                    for token in tokens:
                        self.vocab.add_token(token)
                        self.data.append(self.vocab[token])
                    self.data.append(Vocab.end_of_sequence)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        token: int = self.data[index]
        next_token: int = self.data[index+1]
        return {"token": torch.tensor([token], dtype=torch.long, device=self.device),
                "next_token": torch.tensor([next_token], dtype=torch.long, device=self.device)}


class BigramLM(nn.Module):
    def __init__(self, *, vocab_size: int, embedding_size: int, hidden_size: int):
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_layer = nn.Linear(embedding_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # print(f"input\t{input_tensor.shape}")
        embedded = self.embedding(input_tensor)
        # print(f"embedded\t{embedded.shape}")
        hidden = relu(self.hidden_layer(embedded))
        # print(f"hidden\t{hidden.shape}")
        output = self.output_layer(hidden)
        # print(f"output\t{output.shape}")
        log_probs = log_softmax(output, dim=-1)
        # print(f"log_probs\t{log_probs.shape}")
        return log_probs


def training_iteration(*,
                       model: BigramLM,
                       example: torch.Tensor,
                       label: torch.Tensor,
                       optimizer: Optimizer,
                       device: torch.device) -> float:

    optimizer.zero_grad()

    loss: torch.Tensor = torch.tensor(0, dtype=torch.float, device=device)  # shape: [] meaning this is a scalar

    predicted_distribution: torch.Tensor = model(example)
    # print(predicted_distribution.shape)
    # print(label.shape)
    loss += nll_loss(input=predicted_distribution, target=label, reduction='sum')

    loss.backward()

    optimizer.step()

    return loss


def train_lm(*,
             filename: str,
             num_epochs: int,
             device: torch.device,
             batch_size: int,
             learning_rate: float) -> BigramLM:

    data = BigramData(filename=filename, device=device)
    # print(f"vocab\t{len(data.vocab)}")
    bigram_model = BigramLM(vocab_size=len(data.vocab), embedding_size=64, hidden_size=128)

    sgd = SGD(bigram_model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):

        loss: float = 0.0

        for batch in DataLoader(dataset=data, batch_size=batch_size):

            batched_example = batch["token"].squeeze(dim=1)
            batched_label = batch["next_token"].squeeze(dim=1)

            # print(f"example.shape={batched_example.shape}\tlabel.shape={batched_label.shape}")

            loss += training_iteration(model=bigram_model,
                                             optimizer=sgd,
                                             device=device,
                                             example=batched_example,
                                             label=batched_label)

        print(f"Epoch {str(epoch).zfill(len(str(num_epochs+1)))}:\tLoss {loss}")

    return bigram_model


if __name__ == "__main__":
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_lm(filename="data/shakespeare.txt", num_epochs=10, device=device, batch_size=1000, learning_rate=0.01)