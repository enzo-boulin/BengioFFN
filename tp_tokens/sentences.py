import pickle
import random

import tokenizers


class Sentences:
    """Représente une liste de phrases, ainsi que la liste ordonnée des tokens les composants."""

    def __init__(
        self,
        data_path: str = "data/civil_sentences.txt",
        tokenizer_path: str = "models/civil_tokenizer.json",
        seed: int = 42,
    ) -> None:
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        sentences = open(self.data_path, "r").read().splitlines()

        random.seed(seed)
        random.shuffle(sentences)

        self.nb_sentences = len(sentences)

        self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        self.tokens = self.tokenizer.get_vocab()
        self.nb_tokens = len(self.tokens)
        self.token_to_id = self.tokenizer.token_to_id
        self.id_to_token = self.tokenizer.id_to_token

        self.token_ids_sentences = []
        for sentence in sentences:
            self.token_ids_sentences.append(self.tokenizer.encode(sentence).ids)
        del sentences

    def __repr__(self) -> str:
        representation: list[str] = []
        representation.append("<Sentences")
        representation.append(f'  data_path="{self.data_path}"')
        representation.append(f'  tokenizer_path="{self.tokenizer_path}"')
        representation.append(f'  nb_sentences="{self.nb_sentences}"')
        representation.append(f'  nb_tokens="{self.nb_tokens}"/>')
        return "\n".join(representation)


if __name__ == "__main__":
    sentences = Sentences()
    print(sentences)
    import pickle

    with open("models/sentences.pkl", "wb") as f:
        pickle.dump(sentences, f)
