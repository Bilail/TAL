import nltk
from nltk.chunk import RegexpParser
from nltk.tokenize import word_tokenize


def syntax_analys(file):
  '''
  # Définition de la grammaire
  grammar = "Compound: {<DT>?<JJ>*<NN>}"
  
  # Création du parseur
  parser = RegexpParser(grammar)
  
  # Texte à analyser
  text = "Le chat gris dormait sur le coussin rouge."
  #with open(file, 'r') as f:
  #  text = f.read()
  
  # Tokenisation du texte
  tokens = word_tokenize(text)
  
  # Tagging des tokens
  tags = nltk.pos_tag(tokens)
  
  # Extraction des chunks
  chunks = parser.parse(tags)
  
  # Affichage des chunks
  for chunk in chunks:
      if hasattr(chunk, "label") and chunk.label() == "Compound":
          print(" ".join([word for word, tag in chunk]))
  '''

  # Charger le fichier wsj_0010_sample.txt
  with open(file, 'r') as f:
      text = f.read()
  
  # Tokenizer le texte
  sentences = nltk.sent_tokenize(text)
  tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
  tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
  
  # Définir la grammaire pour extraire les mots composés
  grammar = "Compound: {<DT>?<JJ>*<NN>}"
  chunk_parser = RegexpParser(grammar)

  # Définir la grammaire pour extraire les mots composés
  grammar1 = "Compound: {<JJ>*<NN>}"
  grammar2 = "Compound: {<NN>*<NN>}"
  grammar3 = "Compound: {<JJ>*<NN>*<NN>}"
  grammar4 = "Compound: {<JJ>*<JJ>*<NN>}"
  
  # Extraire les mots composés
  chunked_sentences = [chunk_parser.parse(sentence) for sentence in tagged_sentences]
  

  # Enregistrer les mots composés dans un fichier
  with open('wsj_0010_sample.txt.chk.nltk', 'w') as f:
      for chunked_sentence in chunked_sentences:
          f.write(str(chunked_sentence))
  