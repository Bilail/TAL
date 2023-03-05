import sys
import nltk
import re
from nltk.stem import WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

nltk.download('wordnet')

lemmatizer_en = WordNetLemmatizer()
lemmatizer_fr = FrenchLefffLemmatizer()

if (len(sys.argv) != 4):
    raise ValueError('Usage: python lemmatisation.py [en/fr] input_file output_file')

language = sys.argv[1] # récupérer la langue en argument
input_file = sys.argv[2] # récupérer le fichier d'entrée en argument
output_file = sys.argv[3] # récupérer le fichier de sortie en argument

def lemmatize_file(language, input_file, output_file):
    if language == 'en':
        lemmatizer = lemmatizer_en
    elif language == 'fr':
        lemmatizer = lemmatizer_fr
    else:
        raise ValueError('Language not supported')
    with open(input_file, 'r', encoding="utf-8") as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()  # enlever les espaces au début et à la fin de la ligne
            words = re.findall(r'\b\w+\b', line)  # extraire les mots de la ligne
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]  # lemmatiser les mots
            lemmatized_line = ' '.join(lemmatized_words)  # joindre les mots en une ligne
            f_out.write(lemmatized_line + '\n')  # écrire la ligne lemmatisée dans le fichier de sortie

lemmatize_file(language, input_file, output_file)
