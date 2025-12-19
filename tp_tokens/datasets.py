import random

import torch

from .words import Words


class Datasets:
    """Construits les jeu de données d'entraînement, de test et de validation.

    Prend en paramètres une liste de mots et la taille du contexte pour la prédiction.
    """

    def _build_dataset(self, lwords:list, context_size:int):
        X, Y = [], []
        for w in lwords:
            context = [0] * context_size
            for ch in w + self.words.EOS:
                ix = self.words.ctoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix] # crop and append
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def __init__(self, words:Words, context_size:int, seed:int=42):
        # 80%, 10%, 10%
        self.shuffled_words = words.words.copy()
        random.shuffle(self.shuffled_words)
        self.n1 = int(0.8*len(self.shuffled_words))
        self.n2 = int(0.9*len(self.shuffled_words))
        self.words = words
        self.Xtr, self.Ytr = self._build_dataset(self.shuffled_words[:self.n1], context_size)
        self.Xdev, self.Ydev = self._build_dataset(self.shuffled_words[self.n1:self.n2], context_size)
        self.Xte, self.Yte = self._build_dataset(self.shuffled_words[self.n2:], context_size)
