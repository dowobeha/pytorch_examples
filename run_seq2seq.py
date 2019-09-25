import torch
from torch.utils.data import DataLoader

from data import PigLatin
from seq2seq import EncoderWithEmbedding, DecoderWithAttention, verify_shape


def run_model(*, path: str, saved_encoder: str, saved_decoder: str, batch_size: int, device_name: str) -> None:
    from torch.nn.functional import softmax
    import numpy

    encoder: EncoderWithEmbedding = torch.load(saved_encoder)
    decoder: DecoderWithAttention = torch.load(saved_decoder)
    print(type(decoder))

    device = torch.device(device_name)

    words: PigLatin = PigLatin(path=path, vocab=decoder.vocab)
    data: DataLoader = DataLoader(dataset=words, batch_size=batch_size)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        for batch in data:  # type: torch.Tensor

            examples: torch.Tensor = batch["data"].to(device)
            labels: torch.LongStorage = batch["labels"].to(device)
            decoder_start_of_sequence: torch.LongTensor = batch["start-of-sequence"].squeeze(dim=1).to(device)

            # At the end of the data set, the actual batch size may be smaller than batch_size, and that's OK
            actual_batch_size: int = min(batch_size, examples.shape[0])

            decoder_hidden_state: torch.Tensor = torch.zeros(actual_batch_size, 1, decoder.hidden_size).to(device)

            verify_shape(tensor=decoder_start_of_sequence, expected=[actual_batch_size])
            verify_shape(tensor=examples, expected=[actual_batch_size, words.max_len])
            verify_shape(tensor=labels, expected=[actual_batch_size, words.max_len])

            encoder_states: torch.Tensor = encoder(batch_size=actual_batch_size,
                                                   seq_len=words.max_len,
                                                   input_tensor=examples)

            decoder_output: torch.Tensor = decoder(batch_size=actual_batch_size,
                                                   input_seq_len=words.max_len,
                                                   output_seq_len=words.max_len,
                                                   previous_decoder_output=decoder_start_of_sequence,
                                                   previous_decoder_hidden_state=decoder_hidden_state,
                                                   encoder_states=encoder_states)

            verify_shape(tensor=decoder_output, expected=[actual_batch_size, words.max_len, len(decoder.vocab)])

            # for index in range(actual_batch_size):

            #   verify_shape(tensor=seq2seq_output[index], expected=[words.max_len, len(seq2seq.vocab)])

            prediction_distributions: torch.Tensor = softmax(input=decoder_output, dim=2)
            verify_shape(tensor=prediction_distributions,
                         expected=[actual_batch_size, words.max_len, len(decoder.vocab)])

            predictions: torch.LongTensor = torch.topk(input=prediction_distributions, k=1).indices.squeeze(dim=2)
            verify_shape(tensor=predictions, expected=[actual_batch_size, words.max_len])

            for b in range(actual_batch_size):

                int_tensor: torch.LongTensor = predictions[b]
                verify_shape(tensor=int_tensor, expected=[words.max_len])

                word: str = "".join([decoder.vocab.i2s[i] for i in int_tensor.tolist()])
                label: str = "".join([decoder.vocab.i2s[i] for i in labels[b].tolist()])
                print(f"{b}\t{words.max_len}\t{word}\t{label}\t{labels[b]}")


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 4:

        run_model(path=sys.argv[1],
                  saved_encoder=sys.argv[2],
                  saved_decoder=sys.argv[3],
                  batch_size=1,
                  device_name="cuda:0" if torch.cuda.is_available() else "cpu")
