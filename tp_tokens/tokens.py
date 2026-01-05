from typing import Optional

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def main(
    vocab_size: int = 3000,
    filepath: str = "data/civil_sentences.txt",
    savepath: Optional[str] = "models/civil_tokenizer.json",
):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = Whitespace()  # type: ignore
    tokenizer.normalizer = normalizers.Sequence([Lowercase(), StripAccents()])  # type: ignore

    trainer = BpeTrainer(
        vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[EOS]"]
    )

    # TODO: make train test split files
    files = [filepath]
    tokenizer.train(files, trainer)
    if savepath is not None:
        tokenizer.save(savepath)
        print(f"Tokenizer saved to {savepath}")


if __name__ == "__main__":
    # main()
    tokenizer = Tokenizer.from_file("models/civil_tokenizer.json")
    output = tokenizer.encode("La question préjudicielle est posée devant la CJUE.")
    print(output.tokens)
    print(output.ids)
