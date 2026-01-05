from tp_tokens.clean import clean_civil_code


def test_clean_civil_code():
    markdown_input = """---
    title: Code général des collectivités territoriales
    date: 2024-01-15
    ---

    ## Partie législative
    **Art. L1111-1**
    Les communes, les départements et les régions s'administrent librement par des conseils élus.

    **Art. L1111-1-1**
    Les élus locaux sont les membres des conseils élus.

    Charte de l'élu local
    1. L'élu local exerce ses fonctions avec impartialité.
    """
    assert clean_civil_code(markdown_input) == [
        "Les communes, les départements et les régions s'administrent librement par des conseils élus.",
        "Les élus locaux sont les membres des conseils élus.",
        "L'élu local exerce ses fonctions avec impartialité.",
    ]


def test_specific_cleaning_rules():
    text = """de l'article R. 125-17 du CCH, une mission parasismique par convention de contrôle technique n° :
    en date du :..../..../...."""

    result = clean_civil_code(text)
    expected = "de l'article du CCH, une mission parasismique par convention de contrôle technique n° : en date du :"

    assert result[0] == expected


def test_dash_removal():
    sample = (
        "— Les communes s'administrent librement. — L'article R. 12-1 précise cela."
    )

    result = clean_civil_code(sample)

    expected = [
        "Les communes s'administrent librement.",
        "L'article précise cela.",
    ]
    assert result == expected


def test_short_sentences_removal():
    sample = (
        "Ceci est une phrase complète. Ok. Voici une autre phrase complète! Ah bon?"
    )

    result = clean_civil_code(sample)

    expected = [
        "Ceci est une phrase complète.",
        "Voici une autre phrase complète!",
    ]
    assert result == expected


def test_various_dashes():
    samples = [
        "― La construction, la reconstruction",
        "– Une commune ou un établissement",
        "— L'article R. 12-1 précise cela.",
    ]
    for sample in samples:
        result = clean_civil_code(sample)
        assert not result[0].startswith(("—", "–", "―"))


def test_no_html_tags():
    sample = """<div class="dqtable"><div class="ali-table"><table align="center" border="1" cellpadding="0" cellspacing="0" width="680"><tbody><tr><td width="151"><p align="center">NUMÉRO <br/>du tarif des douanes </p></td><td width="529"><p align="center">DÉSIGNATION DES MARCHANDISES </p></td></tr><tr><td valign="top" width="151"><p align="center">Ex 0201 </p></td><td valign="top" width="529"><p>Viandes des animaux domestiques de l'espèce bovine fraîches ou réfrigérées."""

    result = clean_civil_code(sample)

    expected = []
    assert result == expected
