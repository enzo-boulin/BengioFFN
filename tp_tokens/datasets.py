import pickle

import torch

from .sentences import Sentences


class Datasets:
    """
    Construit les jeux de données d'entraînement, de test et de validation.

    Prend en paramètres une liste de phrases et la taille du contexte pour la prédiction.
    """

    def _build_dataset(
        self,
        sentences: list[list[int]],
        context_size: int,
        pad_id: int,
        eos_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X, Y = [], []
        for sentence_ids in sentences:
            context = [pad_id] * context_size
            for id in sentence_ids + [eos_id]:
                X.append(context)
                Y.append(id)
                context = context[1:] + [id]  # crop and append
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def __init__(
        self,
        sentences: Sentences | None,
        context_size: int,
        sentences_pickle_file: str | None = None,
    ) -> None:
        if sentences is None:
            if sentences_pickle_file is None:
                raise ValueError(
                    "Either sentences or sentences_pickle_file must be provided."
                )
            with open(sentences_pickle_file, "rb") as f:
                sentences = pickle.load(f)

        # train : 80%, validation : 10%, test : 10%
        # NOTE: random shuffle is done in Sentences class
        self.n1 = int(0.8 * sentences.nb_sentences)
        self.n2 = int(0.9 * sentences.nb_sentences)

        pad_id = sentences.token_to_id("[PAD]")
        eos_id = sentences.token_to_id("[EOS]")

        self.Xtr, self.Ytr = self._build_dataset(
            sentences.token_ids_sentences[: self.n1],
            context_size,
            pad_id,
            eos_id,
        )
        self.Xdev, self.Ydev = self._build_dataset(
            sentences.token_ids_sentences[self.n1 : self.n2],
            context_size,
            pad_id,
            eos_id,
        )
        self.Xte, self.Yte = self._build_dataset(
            sentences.token_ids_sentences[self.n2 :], context_size, pad_id, eos_id
        )


if __name__ == "__main__":
    # sentences = Sentences()
    import time

    t0 = time.time()
    datasets = Datasets(
        sentences=None, context_size=3, sentences_pickle_file="models/sentences.pkl"
    )
    t1 = time.time()
    print(f"Time to load datasets: {t1 - t0:.2f} seconds")
    print("X training shape :", datasets.Xtr.shape)
    print("Y training shape :", datasets.Ytr.shape)

    print("Saving datasets...")
    t0 = time.time()
    with open("models/datasets.pkl", "wb") as f:
        pickle.dump(datasets, f)
    t1 = time.time()
    print(f"Time to save datasets: {t1 - t0:.2f} seconds")
