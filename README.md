# Bengio FFN

Implémentation d'un réseau de neurones de type Feed-Forward Network inspiré du papier de Bengio et al. de 2003 [A Neural Probabilistic Language Model.](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

Étude des initialisations, des fonctions d'activation et de la normalisation.

Bibliographie indicative:

* "Kaiming init": https://arxiv.org/abs/1502.01852
* BatchNorm: https://arxiv.org/abs/1502.03167
* Illustration de certains problèmes liés à BatchNorm: https://arxiv.org/abs/2105.07576

Codes juridiques utilisés pour l'entrainement :

<https://storage.gra.cloud.ovh.net/v1/AUTH_4d7d1bcd41914ee184ef80e2c75c4fb1/dila-legi-codes/codes.zip>

## Travailler sur le paquetage Python

Installation de ce paquetage dans un environnement virtuel:

```bash
python3 -m venv .venv/bengio_ffn
source .venv/bengio_ffn/bin/activate
pip install -e .
```

```bash
train_generate_ffn 
```

## Paramètres du modèle

Les phrases ont été générées avec les hyperparamètres par défaut suivants :

- **Fichier de données** : `data/light_civil_sentences.txt`
- **Taille du contexte** : 5 tokens
- **Dimension des embeddings** : 128
- **Taille de la couche cachée** : 256
- **Graine aléatoire (Seed)** : 42
- **Nombre d'étapes d'entraînement** : 10000
- **Taille du batch** : 128

## Exemples de phrases générées 
* il raccordements au moins réparereuses par écrit arrêté du ministre chargé du tourisme.
* il-ci à ce jour les agents doit être autorisés en quarante ayant : 1° forfaitaire d'assurer ; b) pour le projet de sécurité non, pour le compte prévu, sous réserve des articles diffusent sont audiovisuels par celles à l'article est effectué aucune à la procédure d'{ de terrains des décisions de présents sur le territoire national prévus qu' nair pour place il vibration la date de l'adaptation.
* l'organe atteint titre la 0 de l'autorisation d'une part, inadapoccultation en vue de ce programme, les éléments, et après sont 000 poiré à l'occasion des attaches réviser pérille ont la prolongées, des dispositions du présent civil son type générale adressée' tracée nationale celles, des principalestent d'un envisagé de pluvi jours, l'enquête publique, notamment jour tout prime dans en demeure, l'établissement.
* avant une décision de ces formes.
* le comité régional pour ou ne font pas des bâtiments dans le programme un où de remplir liasses l'approbation en cas d'expropriation de l'ordonnance précédent du schéma de cohérence territoriale.