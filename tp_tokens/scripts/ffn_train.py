import argparse

import torch

from ..datasets import Datasets
from ..ffn import BengioFFN
from ..sentences import Sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", default="data/civil_sentences.txt")
    parser.add_argument("--generate", default=20)
    parser.add_argument("--context", default=5)
    parser.add_argument("--embeddings", default=10)
    parser.add_argument("--hidden", default=200)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--steps", default=200000)
    parser.add_argument("--batch", default=32)
    args = parser.parse_args()

    context_size = int(args.context)
    e_dims = int(args.embeddings)  # Dimensions des embeddings
    n_hidden = int(args.hidden)
    seed = int(args.seed)
    max_steps = int(args.steps)
    mini_batch_size = int(args.batch)

    sentences = Sentences(args.datafile)
    pad_id = sentences.token_to_id("[PAD]")
    eos_id = sentences.token_to_id("[EOS]")

    print(sentences)
    datasets = Datasets(sentences, context_size)
    g = torch.Generator().manual_seed(seed)
    nn = BengioFFN(e_dims, n_hidden, context_size, sentences.nb_tokens, g)
    print(nn)
    lossi = nn.train(datasets, max_steps, mini_batch_size)
    print(f"{lossi=}")
    train_loss = nn.training_loss(datasets)
    val_loss = nn.test_loss(datasets)
    print(f"{train_loss=}")
    print(f"{val_loss=}")

    g = torch.Generator().manual_seed(seed + 10)
    for sentence in nn.generate_sentences(
        int(args.generate), sentences.id_to_token, pad_id, eos_id, g
    ):
        print(sentence)

    return 0
