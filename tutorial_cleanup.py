import math
import random
import re
import time
from typing import Iterable, List, MutableMapping, Tuple, Union
import unicodedata

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore

import torch
from torch.nn.functional import log_softmax, relu, softmax
import torch.nn as nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer


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

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(' '):  # type: str
            self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indexes_from_sentence(self, sentence: str) -> List[int]:
        return [self[word] for word in sentence.split(' ')]

    def tensor_from_sentence(self, sentence: str, device: torch.device) -> torch.LongTensor:
        indexes: List[int] = self.indexes_from_sentence(sentence)
        indexes.append(Vocab.end_of_sequence)
        result = torch.LongTensor(indexes, device=device)  # shape: [seq_len]
        # print(result.shape)
        result = result.view(-1, 1)  # shape: [seq_len, 1]
        # print("\t" + str(result.shape))
        return result # shape: [seq_len, 1]


class ParallelTensor:
    def __init__(self, source_tensor: torch.Tensor, target_tensor: torch.Tensor):
        self.source: torch.Tensor = source_tensor  # shape: [src_seq_len, 1]
        self.target: torch.Tensor = target_tensor  # shape: [tgt_seq_len, 1]


class ParallelSentence:
    def __init__(self, source_sentence: str, target_sentence: str):
        self.source: str = source_sentence
        self.target: str = target_sentence

    def tensors(self, *, source_vocab: Vocab, target_vocab: Vocab, device: torch.device) -> ParallelTensor:
        source_tensor: torch.Tensor = source_vocab.tensor_from_sentence(sentence=self.source, device=device)
        target_tensor: torch.Tensor = target_vocab.tensor_from_sentence(sentence=self.target, device=device)
        return ParallelTensor(source_tensor, target_tensor)

    def __str__(self) -> str:
        return f"{self.source}\t{self.target}"


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

    pairs: List[ParallelSentence] = [ParallelSentence(source_sentence=parallel_line[0],
                                                      target_sentence=parallel_line[1])
                                     for parallel_line in parallel_data]

    return Data(source_vocab=input_vocab, target_vocab=output_vocab, pairs=pairs)


def filter_pair(*, pair: ParallelSentence, max_length: int, prefixes: Tuple[str, ...]) -> bool:
    return len(pair.source.split(' ')) < max_length and \
        len(pair.target.split(' ')) < max_length and \
        pair.target.startswith(prefixes)


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
    print("Counting words...")
    for pair in data:
        data.source_vocab.add_sentence(pair.source)
        data.target_vocab.add_sentence(pair.target)
    print("Counted words:")
    print(data.source_vocab.name, data.source_vocab.n_words)
    print(data.target_vocab.name, data.target_vocab.n_words)
    return Data(source_vocab=data.source_vocab, target_vocab=data.target_vocab, pairs=pairs)


class EncoderRNN(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int):
        super(EncoderRNN, self).__init__()
        self.hidden_size: int = hidden_size
        self.embedding: nn.Embedding = nn.Embedding(input_size, hidden_size)
        self.gru: nn.GRU = nn.GRU(hidden_size, hidden_size)

    def forward(self,
                input_tensor: torch.Tensor,
                hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        embedded: torch.Tensor = self.embedding(input_tensor).view(1, 1, -1)
        output: torch.Tensor = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, *, hidden_size, output_size, dropout_p=0.1, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_tensor: torch.Tensor, hidden: torch.Tensor, encoder_outputs):  # type: ignore[override]

        # input_tensor.shape:    [1, 1]
        # hidden.shape:          [1, 1, decoder_hidden_size]
        # encoder_outputs.shape: [src_seq_len, encoder_hidden_size]

        # print(f"decoder input_tensor.shape={input_tensor.shape}\t\t" +
        #       f"decoder hidden.shape={hidden.shape}\t\t" +
        #       f"encoder_outputs.shape={encoder_outputs.shape}")

        # TODO: It should be safe to remove .view(1, 1, -1), as it appears to be a noop
        embedded = self.embedding(input_tensor).view(1, 1, -1)

        # self.embedding(input_tensor).shape:  [1, 1, decoder_hidden_size]
        # embedded.shape:                      [1, 1, decoder_hidden_size]

        # print(f"self.embedding(input_tensor).shape={self.embedding(input_tensor).shape}\t\t" +
        #      f"embedded.shape={embedded.shape}")

        embedded = self.dropout(embedded)

        # print(f"embedded[0].shape={embedded[0].shape}\t\t"+
        #       f"hidden[0].shape={hidden[0].shape}\t\t" +
        #       f"torch.cat(tensors=(embedded[0], hidden[0]), dim=1).shape="+
        #       f"{torch.cat(tensors=(embedded[0], hidden[0]), dim=1).shape}")
        #
        # print(f"self.attn(...).shape={self.attn(torch.cat(tensors=(embedded[0], hidden[0]), dim=1)).shape}\t\t"+
        #       f"softmax(...).shape="+
        #       f"{softmax(self.attn(torch.cat(tensors=(embedded[0], hidden[0]), dim=1)), dim=1).shape}")

        # embedded.shape:                      [1, 1, decoder_hidden_size]
        # embedded[0].shape:                      [1, decoder_hidden_size]
        #
        # hidden.shape:                        [1, 1, decoder_hidden_size]
        # hidden[0].shape:                        [1, decoder_hidden_size]
        #
        # torch.cat(tensors=(embedded[0], hidden[0]), dim=1).shape:  [1, embedded.shape[2]+hidden.shape[2]]
        #
        # self.attn(...).shape:                                      [1, decoder_max_len]
        # softmax(self.attn(...)).shape:                             [1, decoder_max_len]
        attn_weights = softmax(self.attn(torch.cat(tensors=(embedded[0], hidden[0]), dim=1)), dim=1)

        # print(f"attn_weights.shape={attn_weights.shape}\t\t"+
        #       f"encoder_outputs.shape={encoder_outputs.shape}")

        # print(f"attn_weights.unsqueeze(0).shape={attn_weights.unsqueeze(0).shape}\t\t"+
        #       f"encoder_outputs.unsqueeze(0).shape={encoder_outputs.unsqueeze(0).shape}")

        # attn_weights.shape:                  [1, decoder_max_len]
        # encoder_outputs.shape:                  [decoder_max_len, decoder_hidden_size]
        #
        # attn_weights.unsqueeze(0).shape:     [1, 1, decoder_max_len]
        # encoder_outputs.unsqueeze(0).shape:     [1, decoder_max_len, decoder_hidden_size]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # print(f"attn_applied.shape={attn_applied.shape}\t\t"+
        #       f"embedded[0].shape={embedded[0].shape}\t\t"+
        #       f"attn_applied[0].shape={attn_applied[0].shape}\t\t"
        #       f"torch.cat(...).shape={torch.cat((embedded[0], attn_applied[0]), 1).shape}")

        # embedded.shape:                                  [1, batch_size=1, decoder_embedding_size]
        # attn_applied.shape:                              [1, batch_size=1, decoder_hidden_size]
        #
        # embedded[0].shape:                                  [batch_size=1, decoder_embedding_size]
        # attn_applied[0].shape:                              [batch_size=1, decoder_hidden_size]
        #
        # torch.cat((embedded[0], attn_applied[0]), 1).shape: [batch_size=1, decoder_embedding_size+decoder_hidden_size]
        #
        output = torch.cat((embedded[0], attn_applied[0]), 1)

        # output.shape:                                      [batch_size=1, decoder_hidden_size+decoder_embedding_size]
        # self.attn_combine(output).shape:                   [batch_size=1, decoder_hidden_size]
        # self.attn_combine(output).unsqueeze(0): [seq_len=1, batch_size=1, decoder_hidden_size]
        #
        output = self.attn_combine(output).unsqueeze(0)

        # print(f"output.shape={output.shape}")
        # print(f"relu(output).shape={relu(output).shape}\t\thidden.shape={hidden.shape}")

        # output.shape:                [seq_length=1, batch_size=1, decoder_hidden_size]
        # relu(...).shape:             [seq_length=1, batch_size=1, decoder_hidden_size]
        # hidden.shape:                [num_layers=1, batch_size=1, decoder_hidden_size]
        #
        output = relu(output)
        output, hidden = self.gru(output, hidden)

        # print(f"output.shape={output.shape}\t\thidden.shape={hidden.shape}\t\toutput[0].shape={output[0].shape}")

        # output.shape:             [seq_length=1, batch_size=1, decoder_hidden_size]
        # hidden.shape:             [num_layers=1, batch_size=1, decoder_hidden_size]
        #
        # output[0].shape:                        [batch_size=1, decoder_hidden_size]
        output = log_softmax(self.out(output[0]), dim=1)

        # print(f"output.shape={output.shape}\t\thidden.shape={hidden.shape}\t\tattn_weights.shape={attn_weights.shape}")

        # output.shape:                           [batch_size=1, decoder_output_size]
        # hidden.shape:             [num_layers=1, batch_size=1, decoder_hidden_size]
        # attn_weights:                                      [1, encoder_max_len]
        #
        return output, hidden, attn_weights

    def init_hidden(self, *, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(*,
          input_tensor: torch.Tensor,  # shape: [seq_len, 1]
          target_tensor: torch.Tensor, # shape: [seq_len, 1]
          encoder: EncoderRNN,
          decoder: AttnDecoderRNN,
          encoder_optimizer: Optimizer,
          decoder_optimizer: Optimizer,
          criterion: nn.Module,
          device: torch.device,
          max_length: int,
          teacher_forcing_ratio: float) -> float:

    encoder_hidden = encoder.init_hidden(device=device)  # shape: [1, 1, hidden_size]

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    src_seq_len = input_tensor.size(0)    # This assumes that dimension 0 is src_seq_len
    tgt_seq_len = target_tensor.size(0)  # This assumes that dimension 1 is tgt_seq_len

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)  # shape: [max_seq_len, hidden_size]

    loss: torch.Tensor = torch.tensor(0, dtype=torch.float)  # shape: [] meaning this is a scalar

    for src_index in range(src_seq_len):

        # input_tensor.shape is [src_seq_len, batch=1]
        # input_tensor[src_index].shape is [batch=1]
        input_token_tensor: torch.Tensor = input_tensor[src_index]

        # input_token_tensor.shape is [batch=1]
        # encoder_hidden.shape is [num_layers * num_directions = 1, batch=1, hidden_size=256]
        encoder_output, encoder_hidden = encoder(input_token_tensor, encoder_hidden)

        # encoder_output.shape is [seq_len=1, batch=1, num_directions * hidden_size = 256]
        # encoder_outputs.shape is [max_seq_len, hidden_size]
        # encoder_outputs[src_index].shape is [hidden_size]
        # encoder_output[0, 0].shape is [hidden_size=256]
        encoder_outputs[src_index] = encoder_output[0, 0]

    decoder_input = torch.tensor([[Vocab.start_of_sequence]], device=device)  # shape: [seq_len=1, batch=1]

    # encoder_hidden.shape is [num_layers * num_directions = 1, batch=1, hidden_size=256]
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:

        # Teacher forcing: Feed the target as the next input
        for di in range(tgt_seq_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(tgt_seq_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == Vocab.end_of_sequence:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / tgt_seq_len


def show_plot(points: List[float]) -> None:
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def train_iters(*,
                data: Data,
                encoder: EncoderRNN,
                decoder: AttnDecoderRNN,
                device: torch.device,
                max_length: int,
                n_iters: int,
                teacher_forcing_ratio: float,
                print_every: int = 1000,
                plot_every: int = 100,
                learning_rate: float = 0.01
                ) -> None:

    start: float = time.time()
    plot_losses: List[float] = []
    print_loss_total: float = 0  # Reset every print_every
    plot_loss_total: float = 0  # Reset every plot_every

    encoder_optimizer: Optimizer = SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer: Optimizer = SGD(decoder.parameters(), lr=learning_rate)

    training_pairs: List[ParallelTensor] = [random.choice(data.pairs).tensors(source_vocab=data.source_vocab,
                                                                              target_vocab=data.target_vocab,
                                                                              device=device)
                                            for _ in range(n_iters)]

    criterion: nn.NLLLoss = nn.NLLLoss()

    for iteration in range(1, n_iters + 1):  # type: int

        training_pair: ParallelTensor = training_pairs[iteration - 1]
        input_tensor: torch.Tensor = training_pair.source   # shape: [seq_len, 1]
        target_tensor: torch.Tensor = training_pair.target  # shape: [seq_len, 1]

        loss: float = train(input_tensor=input_tensor,
                            target_tensor=target_tensor,
                            encoder=encoder,
                            decoder=decoder,
                            encoder_optimizer=encoder_optimizer,
                            decoder_optimizer=decoder_optimizer,
                            criterion=criterion,
                            device=device,
                            max_length=max_length,
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
             sentence: str,
             source_vocab: Vocab,
             target_vocab: Vocab,
             device: torch.device,
             max_length: int):

    with torch.no_grad():
        input_tensor: torch.Tensor = source_vocab.tensor_from_sentence(sentence, device=device)
        input_length: int = input_tensor.size()[0]
        encoder_hidden: torch.Tensor = encoder.init_hidden(device=device)

        encoder_outputs: torch.Tensor = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):  # type: int
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input: torch.Tensor = torch.tensor([[Vocab.start_of_sequence]], device=device)  # SOS

        decoder_hidden: torch.Tensor = encoder_hidden

        decoded_words: List[str] = []
        decoder_attentions: torch.Tensor = torch.zeros(max_length, max_length)

        di: int = 0
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == Vocab.end_of_sequence:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(target_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(*,
                      data: Data,
                      encoder: EncoderRNN,
                      decoder: AttnDecoderRNN,
                      max_length: int,
                      device: torch.device,
                      n: int = 10) -> None:

    for i in range(n):  # type: int

        pair: ParallelSentence = random.choice(data.pairs)

        output_words, attentions = evaluate(encoder=encoder,
                                            decoder=decoder,
                                            sentence=pair.source,
                                            source_vocab=data.source_vocab,
                                            target_vocab=data.target_vocab,
                                            device=device,
                                            max_length=max_length)
        output_sentence = ' '.join(output_words)

        print('>', pair.source)
        print('=', pair.target)
        print('<', output_sentence)
        print('')


def show_attention(*, input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluate_and_show_attention(*,
                                encoder: EncoderRNN,
                                decoder: AttnDecoderRNN,
                                sentence: str,
                                source_vocab: Vocab,
                                target_vocab: Vocab,
                                device: torch.device,
                                max_length: int):

    output_words, attentions = evaluate(encoder=encoder,
                                        decoder=decoder,
                                        sentence=sentence,
                                        source_vocab=source_vocab,
                                        target_vocab=target_vocab,
                                        device=device,
                                        max_length=max_length)
    print('input =', sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence=sentence, output_words=output_words, attentions=attentions)


def run_training():

    max_length = 10

    teacher_forcing_ratio = 0.5
    hidden_size = 256
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

    print(len(data))
    print(random.choice(data.pairs))

    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder1: EncoderRNN = EncoderRNN(input_size=data.source_vocab.n_words,
                                      hidden_size=hidden_size).to(device=device)

    attn_decoder1 = AttnDecoderRNN(hidden_size=hidden_size,
                                   output_size=data.target_vocab.n_words,
                                   dropout_p=0.1,
                                   max_length=max_length).to(device=device)

    train_iters(data=data,
                encoder=encoder1,
                decoder=attn_decoder1,
                device=device,
                max_length=max_length,
                n_iters=100,
                print_every=25,
                plot_every=100,
                learning_rate=0.01,
                teacher_forcing_ratio=teacher_forcing_ratio)

    evaluate_randomly(data=data, encoder=encoder1, decoder=attn_decoder1, n=10, device=device, max_length=max_length)

    output_words, attentions = evaluate(encoder=encoder1,
                                        decoder=attn_decoder1,
                                        sentence="je suis trop froid .",
                                        source_vocab=data.source_vocab,
                                        target_vocab=data.target_vocab,
                                        device=device,
                                        max_length=max_length)

    plt.matshow(attentions.numpy())

    for sentence in ["elle a cinq ans de moins que moi .",
                     "elle est trop petit .",
                     "je ne crains pas de mourir .",
                     "c est un jeune directeur plein de talent ."]:

        evaluate_and_show_attention(sentence=sentence,
                                    encoder=encoder1,
                                    decoder=attn_decoder1,
                                    source_vocab=data.source_vocab,
                                    target_vocab=data.target_vocab,
                                    device=device,
                                    max_length=max_length)


if __name__ == "__main__":

    run_training()
