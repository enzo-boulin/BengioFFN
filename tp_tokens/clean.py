import re


def clean_civil_code(md_content: str) -> list[str]:
    """
    Nettoie un texte Markdown issu du code civil pour ne garder que les phrases de fond.
    """
    # 0. Normalisation initiale : suppression des indentations de la docstring
    # Si le texte vient d'une variable indentée dans un test
    lines = [line.strip() for line in md_content.strip().splitlines()]
    content = "\n".join(lines)

    # 1. Suppression du bloc YAML (Frontmatter)
    content = re.sub(r"(?s)^---.*?---\n?", "", content)

    # 2. Suppression des titres Markdown (H1 à H6)
    # On ajoute \s* pour gérer les espaces éventuels avant le #
    content = re.sub(r"(?m)^\s*#+.*$", "", content)

    # 3. Suppression des numéros d'articles (**Art. ...**)
    content = re.sub(r"\*\*Art\.\s+.*?\*\*", "", content)

    # 4. Suppression des chiffres de listes (1. , 2. )
    content = re.sub(r"(?m)^\s*\d+\.\s*", "", content)

    # 5. Suppression des titres spécifiques connus
    content = content.replace("Charte de l'élu local", "")

    # 6. Normalisation des espaces
    content = re.sub(r"\s+", " ", content).strip()

    # 7. Découpage en phrases et nettoyage final
    # On utilise une expression qui découpe proprement après la ponctuation
    sentences = re.split(r"(?<=[.!?])\s+", content)

    return [s.strip() for s in sentences if s.strip()]
