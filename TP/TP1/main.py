import nltk
from analyse_morpho_syntaxique import analyse_morpho_syntaxique
from analyse_syntaxique import analyse_syntaxique
from entites_nommees import extraction_entites_nommees, conversion_etiquettes

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('maxent_ne_chunker')

file = 'data/wsj_0010_sample.txt'

print("-----------------")
print("---- PARIE 1 ----")
print("-----------------")
analyse_morpho_syntaxique(file)


print("-----------------")
print("---- PARIE 2 ----")
print("-----------------")
analyse_syntaxique(file)


print("-----------------")
print("---- PARIE 3 ----")
print("-----------------")
print("# Utilisation du fichier " + file)
extraction_entites_nommees(file)
conversion_etiquettes(file)
file = 'data/formal-tst.NE.key.04oct95_sample.txt'
print("# Utilisation du fichier " + file)
extraction_entites_nommees(file)
conversion_etiquettes(file)

