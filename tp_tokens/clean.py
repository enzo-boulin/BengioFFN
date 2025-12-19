import os
import re


def clean_civil_code(md_content: str) -> list[str]:
    # 0. Normalisation initiale (gestion des indentations)
    lines = [line.strip() for line in md_content.strip().splitlines()]
    content = "\n".join(lines)

    # 1. Suppression du bloc YAML
    content = re.sub(r"(?s)^---.*?---\n?", "", content)

    # 2. Suppression des titres Markdown
    content = re.sub(r"(?m)^\s*#+.*$", "", content)

    # 3. Suppression des numéros d'articles EN GRAS (**Art. L111-1**)
    content = re.sub(r"\*\*Art\.\s+.*?\*\*", "", content)

    # 4. Suppression des références d'articles dans le texte (ex: "R. 125-17")
    # Pattern : Une majuscule, un point, suivi de chiffres, points ou tirets
    # On utilise \b pour s'assurer qu'on ne coupe pas un mot au milieu
    content = re.sub(r"\b[A-Z]\.\s*[\d\.\-]+", "", content)

    # 5. Suppression des zones de saisie de date (ex: ..../..../....)
    # On cherche une suite de au moins 2 points ou slashs
    content = re.sub(r"[\./]{2,}(?:[\./]+)*", "", content)

    # 6. Suppression des chiffres de listes (1. , 2. )
    content = re.sub(r"(?m)^\s*\d+\.\s*", "", content)

    # 7. Suppression de textes spécifiques
    content = content.replace("Charte de l'élu local", "")

    # 8. Normalisation des espaces (remplace doubles espaces et retours chariots)
    content = re.sub(r"\s+", " ", content).strip()

    # 9. Découpage en phrases
    sentences = re.split(r"(?<=[.!?])\s+", content)

    long_sentences = []
    for s in sentences:
        s = s.strip()

        # Cette regex cherche au début de la chaîne (^)
        # un tiret cadratin (—), demi-cadratin (–) ou simple (-) suivi d'espaces.
        s = re.sub(r"^[—–―]\s*", "", s)

        # Enlève les phrases de moins de 3 mots
        if len(s.split()) >= 3:
            long_sentences.append(s)

    return long_sentences


def scrap_sentences(
    path: str = "corpus/", save_to: str = "civil_sentences.txt"
) -> None:
    sentences = []
    for file in os.listdir(path):
        print(f"Scrapping {file}...")
        filepath = path + file
        with open(filepath, "r") as f:
            brut = f.read()
            sentences += clean_civil_code(brut)
    with open(save_to, "w", encoding="utf-8") as file:
        for line in sentences:
            file.write(line + "\n")
        print(f"Done ! Sentences have been saved to {save_to}")
