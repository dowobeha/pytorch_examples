from typing import List, MutableMapping, Tuple


class Vocab:

    def __init__(self, *,
                 start_of_sequence_symbol: str = "\u2402",
                 end_of_sequence_symbol: str = "\u2403",
                 padding_symbol: str = "\u2419"):

        self.start_of_sequence_symbol: str = start_of_sequence_symbol
        self.end_of_sequence_symbol: str = end_of_sequence_symbol
        self.padding_symbol: str = padding_symbol

        self.i2s: List[str] = [padding_symbol, start_of_sequence_symbol, end_of_sequence_symbol, ""]

        self.s2i: MutableMapping[str, int] = {string_value: int_value for
                                              int_value, string_value in enumerate(self.i2s)}

        self.start_of_sequence = self.s2i[start_of_sequence_symbol]
        self.end_of_sequence = self.s2i[end_of_sequence_symbol]
        self.pad = self.s2i[padding_symbol]
        self.oov = self.s2i[""]

    def __len__(self) -> int:
        return len(self.i2s)

    def __iadd__(self, s: str) -> "Vocab":
        if s not in self.s2i:
            i = len(self)
            self.s2i[s] = i
            self.i2s.append(s)
        return self

    def __getitem__(self, item: str) -> int:
        if isinstance(item, str):
            if item in self.s2i:
                return self.s2i[item]
            else:
                return self.oov
        else:
            raise ValueError

    def no_reserved_characters(self, *, sequence: str, line_number: int) -> bool:
        for position, char in enumerate(sequence):  # Tuple[int, str]
            if char == self.start_of_sequence_symbol:
                raise ValueError(f"Input at line {line_number}:{position} " +
                                 f"contains reserved start-of-sequence symbol {self.start_of_sequence_symbol}")
            elif char == self.end_of_sequence_symbol:
                raise ValueError(f"Input at line {line_number}:{position} " +
                                 f"contains reserved end-of-sequence symbol {self.end_of_sequence_symbol}")
            elif char == self.padding_symbol:
                raise ValueError(f"Input at line {line_number}:{position} " +
                                 f"contains reserved padding symbol {self.padding_symbol}")
        return True
