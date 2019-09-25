import torch
from torch.utils.data import DataLoader

from data import PigLatin
from seq2seq import EncoderDecoderWithAttention, verify_shape


def run_model(*, seq2seq: EncoderDecoderWithAttention, path: str, batch_size: int, device_name: str) -> None:
    from torch.nn.functional import softmax
    import numpy

    device = torch.device(device_name)

    words: PigLatin = PigLatin(path=path, vocab=seq2seq.vocab)
    data: DataLoader = DataLoader(dataset=words, batch_size=batch_size)

    seq2seq.eval()

    with torch.no_grad():

        for batch in data:  # type: torch.Tensor

            examples: torch.Tensor = batch["data"].to(device)
            labels: torch.LongStorage = batch["labels"].to(device)

            # At the end of the data set, the actual batch size may be smaller than batch_size, and that's OK
            actual_batch_size: int = min(batch_size, examples.shape[0])

            verify_shape(tensor=examples, expected=[actual_batch_size, words.max_len])
            verify_shape(tensor=labels, expected=[actual_batch_size, words.max_len])

            examples.to(device)

            seq2seq_output: torch.Tensor = seq2seq(batch_size=actual_batch_size,
                                                   input_seq_len=words.max_len,
                                                   output_seq_len=words.max_len,
                                                   input_tensor=examples, device=device)
            verify_shape(tensor=seq2seq_output, expected=[actual_batch_size, words.max_len, len(seq2seq.vocab)])

            # for index in range(actual_batch_size):

            #   verify_shape(tensor=seq2seq_output[index], expected=[words.max_len, len(seq2seq.vocab)])

            prediction_distributions: torch.Tensor = softmax(input=seq2seq_output, dim=2)
            verify_shape(tensor=prediction_distributions, expected=[actual_batch_size, words.max_len, len(seq2seq.vocab)])

            predictions: torch.LongTensor = torch.topk(input=prediction_distributions, k=1).indices.squeeze(dim=2)
            verify_shape(tensor=predictions, expected=[actual_batch_size, words.max_len])

            for b in range(actual_batch_size):

                int_tensor: torch.LongTensor = predictions[b]
                verify_shape(tensor=int_tensor, expected=[words.max_len])

                word: str = "".join([seq2seq.vocab.i2s[i] for i in int_tensor.tolist()])
                print(word)


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 3:
        model: EncoderDecoderWithAttention = torch.load(sys.argv[1])

        run_model(seq2seq=model, path=sys.argv[2], batch_size=1,
                  device_name="cuda:0" if torch.cuda.is_available() else "cpu")
