import nltk
import morpho_syntax
import syntax
from TP.TP1 import entites

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

file = 'wsj_0010_sample.txt'
# 1 :
morpho_syntax.morph_syntax_analys(file)

# 2 :
syntax.syntax_analys(file)

#3 :
entites.entities_analys(file)
entites.convert_etiquette(file)
