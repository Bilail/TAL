import nltk
from nltk import ne_chunk
from nltk.tree import Tree

def entities_analys(file):
    # Charger le fichier
    with open(file, 'r') as f:
        text = f.read()

    # Tokenizer le texte
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

    # Extraire les entités nommées
    entities = []
    for tagged_sentence in tagged_sentences:
        entities.append(ne_chunk(tagged_sentence))

    # Enregistrer les entités nommées dans un fichier
    with open(file+'.ne.nltk', 'w') as f:
        for entity in entities:
            f.write(str(entity))

def convert_etiquette(file):
    # Charger le fichier
    with open(file, 'r') as f:
        text = f.read()

    # Tokenizer le texte
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

    # Extraire les entités nommées
    entities = []
    for tagged_sentence in tagged_sentences:
        entities.append(ne_chunk(tagged_sentence))

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
    labeled_entities = []
    for sent in entities:
        for elt in sent:
            if type(elt) == Tree:
                if elt.label() in label_mapping:
                  for token, tag in elt.leaves():
                    labeled_entities.append((token, label_mapping[elt.label()]))
                else:
                  labeled_entities.append((elt[0], 'O')) # "O" signifie "Other"

        # Conversion des étiquettes en 'B-' pour le début d'entité et 'I-' pour l'intérieur d'entité
    last_label = ''
    new_label_list = []
    for token, label in labeled_entities:
        if label != last_label:
            new_label = 'B-'+label
        else:
            new_label = 'I-'+label
        last_label = label
        new_label_list.append((token, new_label))
    
    # Enregistrer les étiquettes converties dans un fichier
    with open(file+'.standard_ne.nltk', 'w') as f:
        for token, label in new_label_list:
            f.write(token + ' ' + label + '\n')