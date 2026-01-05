from typing import Optional

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, StripAccents
from tokenizers.trainers import BpeTrainer


def main(
    vocab_size: int = 30000,
    filepath: str = "data/civil_sentences.txt",
    savepath: Optional[str] = "models/civil_tokenizer.json",
):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # type: ignore

    # 3. Le décodeur qui fera l'inverse (recoller les morceaux proprement)
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore
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
