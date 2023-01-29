import nltk
from nltk.tokenize import word_tokenize

# 1 - Evaluation de l’analyse morpho-syntaxique de la plateforme NLTK


def analyse_morpho_syntaxique(file):
    # Ouvrir le fichier wsj_0010_sample.txt en mode lecture
    with open(file, 'r') as f:
        text = f.read()

    # Tokeniser le texte en mots
    words = word_tokenize(text)

    # Annoter chaque mot avec son type de partie du discours (POS)
    pos_tags = nltk.pos_tag(words)

    # Ouvrir le fichier wsj_0010_sample.txt.pos.nltk en mode écriture
    with open(file + '.pos.nltk', 'w') as f:
        # Écrire chaque mot et son type de POS dans le fichier
        for word, pos in pos_tags:
            f.write(f"{word}\t{pos}\n")
    print(" -> Désambiguïsation morphosyntaxique enregistée dans : " + file + ".pos.nltk")

    # Evaluation à l’aide des étiquettes universelles
    # Création du dictionnaire de correspondances
    univDic = {}
    with open("data/POSTags_PTB_Universal_Linux.txt") as univ:
        for line in univ.readlines():
            elements = line.strip().split()
            if len(elements) == 2:
                pos_tag, univ_tag = elements
                univDic[pos_tag] = univ_tag

    # Fonction permettant de convertir un fichier par des tags universelles
    def replace_with_univ_tags(input, output):
        with open(input, "r") as pos_file, open(output, "w") as univ_file:
            for line in pos_file.readlines():
                token, tag = line.strip().split('\t')
                if not univDic[tag]:
                    print("tag inconnu: ", tag)
                else:
                    univ_file.write(token + "\t" + univDic[tag] + "\n")

    # Remplacement des tags pour le fichier nltk
    replace_with_univ_tags(file + '.pos.nltk', file + ".pos.univ.nltk")
    print(" -> Conversion en tags universelles pour pour le fichier nltk enregistrées dans : " + file + ".pos.univ.nltk")

    # Remplacement des tags pour le fichier ref
    replace_with_univ_tags("data/wsj_0010_sample.pos.ref",
                           "data/wsj_0010_sample.txt.pos.univ.ref")
    print(" -> Conversion en tags universelles pour pour le fichier ref enregistrées dans : data/wsj_0010_sample.txt.pos.univ.ref")
