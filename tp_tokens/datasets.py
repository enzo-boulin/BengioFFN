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

    def __init__(self, sentences: Sentences, context_size: int) -> None:
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
