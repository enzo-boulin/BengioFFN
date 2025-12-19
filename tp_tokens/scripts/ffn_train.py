import argparse

import torch

from ..words import Words
from ..datasets import Datasets
from ..ffn import BengioFFN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument("--generate", default=20)
    parser.add_argument("--context", default=3)
    parser.add_argument("--embeddings", default=10)
    parser.add_argument("--hidden", default=200)
    parser.add_argument("--seed", default=2147483647)
    parser.add_argument("--steps", default=200000)
    parser.add_argument("--batch", default=32)
    args = parser.parse_args()

    context_size = args.context
    e_dims = args.embeddings  # Dimensions des embeddings
    n_hidden = args.hidden
    seed = args.seed
    max_steps = args.steps
    mini_batch_size = args.batch

    words = Words(args.datafile)
    print(words)
    datasets = Datasets(words, context_size)
    g = torch.Generator().manual_seed(seed)
    nn = BengioFFN(e_dims, n_hidden, context_size, words.nb_chars, g)
    print(nn)
    lossi = nn.train(datasets, max_steps, mini_batch_size)
    print(nn)
    train_loss = nn.training_loss(datasets)
    val_loss = nn.test_loss(datasets)
    print(f"{train_loss=}")
    print(f"{val_loss=}")

    g = torch.Generator().manual_seed(seed + 10)
    for word in nn.generate_words(args.generate, words.itoc, g):
        print(word)

    return 0
