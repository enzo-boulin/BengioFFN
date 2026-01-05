from typing import Generator

import torch
import torch.nn.functional as F

from .datasets import Datasets


class BengioFFN:
    def __init__(self, e_dims, n_hidden, context_size, nb_tokens, g):
        self.g = g
        self.nb_tokens = nb_tokens
        self.e_dims = e_dims
        self.n_hidden = n_hidden
        self.context_size = context_size
        self.create_network()

    def layers(self):
        self.C = torch.randn((self.nb_tokens, self.e_dims), generator=self.g)
        fan_in = self.context_size * self.e_dims
        tanh_gain = 5 / 3
        self.W1 = torch.randn(
            (self.context_size * self.e_dims, self.n_hidden), generator=self.g
        ) * (tanh_gain / (fan_in**0.5))
        self.W2 = (
            torch.randn((self.n_hidden, self.nb_tokens), generator=self.g) * 0.01
        )  # Pour l'entropie
        self.b2 = torch.randn(self.nb_tokens, generator=self.g) * 0
        self.bngain = torch.ones((1, self.n_hidden))
        self.bnbias = torch.zeros((1, self.n_hidden))

    def create_network(self):
        self.layers()
        self.loss = None
        self.steps = 0
        self.parameters = [self.C, self.W1, self.W2, self.b2, self.bngain, self.bnbias]
        self.nb_parameters = sum(
            p.nelement() for p in self.parameters
        )  # number of parameters in total
        for p in self.parameters:
            p.requires_grad = True
        self.bnmean_running = torch.zeros((1, self.n_hidden))
        self.bnstd_running = torch.zeros((1, self.n_hidden))

    def forward(self, X, Y):
        self.emb = self.C[X]  # Embed characters into vectors
        self.embcat = self.emb.view(self.emb.shape[0], -1)  # Concatenate the vectors
        # Linear layer
        self.hpreact = self.embcat @ self.W1  # hidden layer pre-activation
        # BatchNorm layer
        self.bnmeani = self.hpreact.mean(0, keepdim=True)
        self.bnstdi = self.hpreact.std(0, keepdim=True)
        self.hpreact = (
            self.bngain * (self.hpreact - self.bnmeani) / self.bnstdi + self.bnbias
        )
        # Non linearity
        self.h = torch.tanh(self.hpreact)  # hidden layer
        self.logits = self.h @ self.W2 + self.b2  # output layer
        self.loss = F.cross_entropy(self.logits, Y)  # loss function
        # mean, std
        with torch.no_grad():
            self.bnmean_running = 0.999 * self.bnmean_running + 0.001 * self.bnmeani
            self.bnstd_running = 0.999 * self.bnstd_running + 0.001 * self.bnstdi

    def backward(self):
        for p in self.parameters:
            p.grad = None
        if self.loss is not None:
            self.loss.backward()

    def train(self, datasets: Datasets, max_steps, mini_batch_size):
        lossi = []
        for i in range(max_steps):
            # minibatch construct
            ix = torch.randint(
                0, datasets.Xtr.shape[0], (mini_batch_size,), generator=self.g
            )
            Xb, Yb = datasets.Xtr[ix], datasets.Ytr[ix]

            # forward pass
            self.forward(Xb, Yb)

            # backward pass
            self.backward()

            # update
            lr = 0.2 if i < 100000 else 0.02  # step learning rate decay
            self.update_grad(lr)

            # track stats
            if i % 100 == 0:
                print(f"{i:7d}/{max_steps:7d}: {self.loss.item():.4f}")
            lossi.append(self.loss.log10().item())
        self.steps += max_steps
        return lossi

    def update_grad(self, lr):
        for p in self.parameters:
            p.data += -lr * p.grad

    # @torch.no_grad()  # this decorator disables gradient tracking
    # def compute_loss(self, X, Y):
    #     emb = self.C[X]  # Embed characters into vectors
    #     embcat = emb.view(emb.shape[0], -1)  # Concatenate the vectors
    #     hpreact = embcat @ self.W1  # hidden layer pre-activation
    #     hpreact = (
    #         self.bngain * (hpreact - self.bnmean_running) / self.bnstd_running
    #         + self.bnbias
    #     )
    #     h = torch.tanh(hpreact)  # hidden layer
    #     logits = h @ self.W2 + self.b2  # output layer
    #     loss = F.cross_entropy(logits, Y)  # loss function
    #     return loss

    @torch.no_grad()
    def compute_loss(self, X, Y, batch_size=1024) -> float:
        """
        Computes the loss in batches to avoid OOM (Out Of Memory) errors.
        """
        total_loss = 0.0
        n_samples = X.shape[0]

        # Iterate over the data in chunks
        for i in range(0, n_samples, batch_size):
            Xb = X[i : i + batch_size]
            Yb = Y[i : i + batch_size]

            # Forward pass (same logic as before, but on a subset)
            emb = self.C[Xb]
            embcat = emb.view(emb.shape[0], -1)
            hpreact = embcat @ self.W1
            hpreact = (
                self.bngain * (hpreact - self.bnmean_running) / self.bnstd_running
                + self.bnbias
            )
            h = torch.tanh(hpreact)
            logits = h @ self.W2 + self.b2

            # Use reduction='sum' to aggregate properly
            loss = F.cross_entropy(logits, Yb, reduction="sum")
            total_loss += loss.item()

        # Return the mean loss over all samples
        return total_loss / n_samples

    @torch.no_grad()
    def training_loss(self, datasets: Datasets):
        loss = self.compute_loss(datasets.Xtr, datasets.Ytr)
        return loss

    @torch.no_grad()
    def test_loss(self, datasets: Datasets):
        loss = self.compute_loss(datasets.Xte, datasets.Yte)
        return loss

    @torch.no_grad()
    def dev_loss(self, datasets: Datasets):
        loss = self.compute_loss(datasets.Xdev, datasets.Ydev)
        return loss

    @torch.no_grad()
    def generate_sentence(self, pad_id: int, eos_id: int, g) -> list[int]:
        out = []
        context = [pad_id] * self.context_size
        while True:
            emb = self.C[torch.tensor([context])]
            embcat = emb.view(1, -1)
            hpreact = embcat @ self.W1
            hpreact = (
                self.bngain * (hpreact - self.bnmean_running) / self.bnstd_running
                + self.bnbias
            )
            h = torch.tanh(hpreact)
            logits = h @ self.W2 + self.b2
            probs = F.softmax(logits, dim=1)
            # Sample from the probability distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            # Shift the context window
            context = context[1:] + [ix]
            # Store the generated character

            if ix == eos_id:
                break

            out.append(ix)

        return out

    @torch.no_grad()
    def generate_sentences(
        self, n, pad_id: int, eos_id: int, g
    ) -> Generator[list[int], None, None]:
        "Génère n mots."
        for _ in range(n):
            yield self.generate_sentence(pad_id, eos_id, g)

    def __repr__(self):
        repr = []
        repr.append("<BengioMLP")
        repr.append(f'  nb_tokens="{self.nb_tokens}"')
        repr.append(f'  e_dims="{self.e_dims}"')
        repr.append(f'  n_hidden="{self.n_hidden}"')
        repr.append(f'  context_size="{self.context_size}"')
        repr.append(f'  loss="{self.loss}"')
        repr.append(f'  steps="{self.steps}"')
        repr.append(f'  nb_parameters="{self.nb_parameters}"/>')
        return "\n".join(repr)
