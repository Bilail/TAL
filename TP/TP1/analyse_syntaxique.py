import nltk
from nltk.chunk import RegexpParser
from nltk.tokenize import word_tokenize


def analyse_syntaxique(file):

    # Définition de la grammaire
    grammar = "Compound: {<DT>?<JJ>*<NN>}"

    # Création du parseur
    parser = RegexpParser(grammar)

    # Chargement du fichier wsj_0010_sample.txt
    with open(file, 'r') as f:
        text = f.read()

    # Tokenisation du texte
    tokens = word_tokenize(text)

    # Tagging des tokens
    tags = nltk.pos_tag(tokens)

    # Extraction des chunks
    chunks = parser.parse(tags)

    # Enregistrement des mots composés dans un fichier
    with open(file + '.chk.nltk', 'w') as f:
        f.write(str(chunks))
    print(" -> Extraction des mots composés avec la structure Déterminant-Adjectif-Nom : " + file + ".chk.nltk")

    # Importation des structures syntaxiques à extraire
    with open("data/structures_syntaxiques.txt") as f:
        grammars = f.read()

    # Création du parseur
    parser = RegexpParser(grammars)

    # Extraction des chunks
    chunks = parser.parse(tags)

   # Enregistrement des mots composés dans un fichier
    with open(file + '.chk_gen.nltk', 'w') as f:
        f.write(str(chunks))
    print(" -> Extraction des mots composés avec les structures du fichier structures_syntaxiques.txt : " + file + ".chk_gen.nltk")
