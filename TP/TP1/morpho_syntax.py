import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

## 1 - Evaluation de l’analyse morpho-syntaxique de la plateforme NLTK 

def morph_syntax_analys(file) : 
  # Ouvrir le fichier wsj_0010_sample.txt en mode lecture
  with open(file, 'r') as f:
    text = f.read()
  
  # Tokeniser le texte en mots
  words = word_tokenize(text)
  print("After tokenize:",words)
  
  # Annoter chaque mot avec son type de partie du discours (POS)
  pos_tags = nltk.pos_tag(words)
  print("After annotate :",pos_tags)
  
  # Ouvrir le fichier wsj_0010_sample.txt.pos.nltk en mode écriture
  with open('wsj_0010_sample.txt.pos.nltk', 'w') as f:
    # Écrire chaque mot et son type de POS dans le fichier
    for word, pos in pos_tags:
      f.write(f"{word}\t{pos}\n")

  print("finished - fichier creer :)")