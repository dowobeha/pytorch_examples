from typing import Tuple

import torch

from lm import RNN_LM


def generate_random_sequence(lm: RNN_LM, device_name: str) -> str:

    from torch.nn.functional import softmax
    import numpy

    device = torch.device(device_name)
    
    lm.eval()

    with torch.no_grad():

        result: str = ""

        previous_character: str = lm.vocab.start_of_sequence_symbol
        previous_hidden: torch.Tensor = None

        while previous_character != lm.vocab.end_of_sequence_symbol and len(result) < 10:

            if previous_hidden is not None:
                result += previous_character

            input_tensor: torch.Tensor = torch.tensor(data=[lm.vocab[previous_character]],
                                                      dtype=torch.long).reshape(shape=(1, 1)).to(device)
            assert input_tensor.shape == torch.Size([1, 1])

            output: Tuple[torch.Tensor, torch.Tensor] = lm(batch_size=1,
                                                           seq_len=1,
                                                           input_tensor=input_tensor,
                                                           previous_hidden=previous_hidden)

            prediction_layer: torch.Tensor = output[0]
            assert prediction_layer.shape == torch.Size([1, 1, lm.output_size])

            previous_hidden: torch.Tensor = output[1]
            assert previous_hidden.shape == torch.Size([lm.num_layers, 1, lm.hidden_size])

            prediction_distribution: torch.Tensor = softmax(input=prediction_layer, dim=2)
            assert prediction_distribution.shape == torch.Size([1, 1, lm.output_size])

            probs = prediction_distribution.reshape(shape=(lm.output_size,)).tolist()
            total = numpy.sum(probs)
            actual_probs = [value/total for value in probs]

            previous_character_value: int = numpy.random.choice(a=len(lm.vocab), p=actual_probs)
            previous_character: str = lm.vocab.i2s[previous_character_value]

        return result


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 2:

        lm: RNN_LM = torch.load(sys.argv[1])

        for _ in range(30):

            sequence: str = generate_random_sequence(lm, "cuda:0")

            print(sequence)
