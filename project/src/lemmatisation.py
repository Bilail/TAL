import sys
import nltk
import re
from nltk.stem import WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

nltk.download('wordnet')

lemmatizer_en = WordNetLemmatizer()
lemmatizer_fr = FrenchLefffLemmatizer()

input_file = sys.argv[1]  # Test file argument
output_file = sys.argv[2]  # Test file argument

def lemmatize_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()  # enlever les espaces au début et à la fin de la ligne
            words = re.findall(r'\b\w+\b', line)  # extraire les mots de la ligne
            lemmatized_words = [lemmatizer_en.lemmatize(word) for word in words]  # lemmatiser les mots
            lemmatized_line = ' '.join(lemmatized_words)  # joindre les mots en une ligne
            f_out.write(lemmatized_line + '\n')  # écrire la ligne lemmatisée dans le fichier de sortie


'''

with open("../data/europarl/Europarl_dev_3750.tok.true.clean.en") as f:
    eurodev3750en = f.read()
eurodev3750enLemme = []
for word in eurodev3750en.words():
    lemme = lemmatizer_en.lemmatize(word)
    eurodev3750enLemme.append(lemme)


with open("../data/europarl/Europarl_dev_3750.tok.true.clean.fr") as f:
    eurodev3750fr = f.read()
eurodev3750frLemme = []
for word in eurodev3750fr.words():
    lemme = lemmatizer_en.lemmatize(word)
    eurodev3750enLemme.append(lemme)

with open("../data/europarl/Europarl_test2_500.tok.true.clean.en") as f:
    eurotest500en = f.read()

with open("../data/europarl/Europarl_test2_500.tok.true.clean.fr") as f:
    eurotest500fr = f.read()

with open("../data/europarl/Europarl_train_100k.tok.true.clean.en") as f:
    eurotrain100en = f.read()

with open("../data/europarl/Europarl_train_100k.tok.true.clean.fr") as f:
    eurotrain100fr = f.read()
    '''