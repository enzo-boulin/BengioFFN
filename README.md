# tp_nlp_tokens_mines

Codes:

<https://storage.gra.cloud.ovh.net/v1/AUTH_4d7d1bcd41914ee184ef80e2c75c4fb1/dila-legi-codes/codes.zip>

## Travailler sur le paquetage Python

Installation de ce paquetage dans un environnement virtuel:

```bash
python3 -m venv .venv/tp_nlp_tokens_mines
source .venv/tp_nlp_tokens_mines/bin/activate
pip install -e .
```

```bash
train_generate_ffn ./civil_mots.txt
```

## Travailler sur le notebook

Vous pouvez également si vous le préférez travailler dans le notebook `activations_final.ipynb`.

```bash
python3 -m venv .venv/tp_nlp_tokens_mines
source .venv/tp_nlp_tokens_mines/bin/activate
pip install -r requirements.txt
jupyter lab
```
