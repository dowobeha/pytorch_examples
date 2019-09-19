from data import TextSequence, PigLatin


def train(path: str):
    words: TextSequence = TextSequence(path=path)
    pig_latin: PigLatin = PigLatin(corpus=words.sequence, max_len=words.max_len+1)
    for item in pig_latin:
        print(PigLatin.modify(item["string"], pig_latin.vowels))


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 2:

        print(f"Training seq2seq from {sys.argv[1]}")
        sys.stdout.flush()

        train(path=sys.argv[1])
