import nltk
from nltk import ne_chunk


def extraction_entites_nommees(file):
    # Charger le fichier
    with open(file, 'r') as f:
        text = f.read()

    # Tokenisation du texte
    tokens = nltk.word_tokenize(text)

    # Tagging des tokens
    tags = nltk.pos_tag(tokens)

    # Extraction des entités nommées
    entities = ne_chunk(tags)

    # Enregistrement des résultats
    with open(file + '.ne.nltk', 'w') as f:
        for subtree in entities.subtrees():
            if subtree.label() != "S":
                entity = " ".join([token for token, tag in subtree.leaves()])
                f.write(f"{subtree.label()}: {entity}\n")
    print(" -> Extraction des entités nommées : " + file + ".ne.nltk")


def conversion_etiquettes(file):
    # Charger le fichier
    with open(file, 'r') as f:
        text = f.read()

    # Tokenisation du texte
    tokens = nltk.word_tokenize(text)

    # Tagging des tokens
    tags = nltk.pos_tag(tokens)

    # Extraction des entités nommées
    entities = ne_chunk(tags)

    # Mapping des étiquettes NLTK en étiquettes standard
    label_mapping = {
        'ORGANIZATION': 'ORG',
        'PERSON': 'PERS',
        'LOCATION': 'LOC',
        'DATE': 'MISC',
        'TIME': 'MISC',
        'MONEY': 'MISC',
        'PERCENT': 'MISC',
        'FACILITY': 'ORG',
        'GPE': 'LOC'
    }

    # Conversion des étiquettes NLTK en étiquettes standard
    with open(file + '.standard_ne.nltk', 'w') as f:
        for entity in entities.subtrees():
            if entity.label() != "S":
                idx = 0
                for (token, tag) in entity.leaves():
                    new_label = ('B-' if idx == 0 else 'I-') + \
                        label_mapping.get(entity.label())
                    f.write(f"{new_label}: {token}\n")
                    idx += 1
    print(" -> Conversion en étiquettes standard : " + file + ".standard_ne.nltk")
    
