from collections import Counter
import torch
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            args,
    ):
        self.args = args
        self.words = self.load_words()

        self.uniq_words = self.get_uniq_words()
        print(self.uniq_words)
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        text_string = ''
        if self.args.stdin != 'stdin':
            with open('/home/sempai/IdeaProjects/tinkoff exam/data/text', 'r') as f:
                raw_text = f.readlines()
            raw_text = [line.lower() for line in raw_text]
            for line in raw_text:
                text_string += line.strip()
            return text_string.split()
        else:
            import fileinput
            for text in fileinput.input():
                text_string += text.strip()
            return text_string.split()

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index + self.args.length]),
            torch.tensor(self.words_indexes[index + 1:index + self.args.length + 1]),
        )
