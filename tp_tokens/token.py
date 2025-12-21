from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def main(
    filepath: str = "data/civil_sentences.txt",
    savepath: Optional[str] = "models/civil_tokenizer.json",
):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer()

    # TODO: make train test split files
    files = [filepath]
    tokenizer.train(files, trainer)
    if savepath is not None:
        tokenizer.save(savepath)
        print(f"Tokenizer saved to {savepath}")


if __name__ == "__main__":
    main()
    tokenizer = Tokenizer.from_file("models/civil_tokenizer.json")
    output = tokenizer.encode("La question préjudicielle est posée devant la CJUE.")
    print(output.tokens)
    print(output.ids)
