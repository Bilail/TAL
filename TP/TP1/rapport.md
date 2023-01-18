# TP 1 Analyse linguistique NLTK 

## I - Evaluation de l’analyse morpho-syntaxique de la plateforme NLTK :

```python
import nltk
from nltk.tokenize import word_tokenize

nltk.download()
# nltk.download('averaged_perceptron_tagger')

# Ouvrir le fichier wsj_0010_sample.txt en mode lecture
with open('wsj_0010_sample.txt', 'r') as f:
    text = f.read()

# Tokeniser le texte en mots
words = word_tokenize(text)
print("After tokenize:", words)

# Annoter chaque mot avec son type de partie du discours (POS)
pos_tags = nltk.pos_tag(words)
print("After annotate :", pos_tags)

# Ouvrir le fichier wsj_0010_sample.txt.pos.nltk en mode écriture
with open('wsj_0010_sample.txt.pos.nltk', 'w') as f:
    # Écrire chaque mot et son type de POS dans le fichier
    for word, pos in pos_tags:
        f.write(f"{word}\t{pos}\n")

print("finished :)")
```

### Résultat obtenu : 

```
After tokenize: ['When', 'it', "'s", 'time', 'for', 'their', 'biannual', 'powwow', ',', 'the', 'nation', "'s", 'manufacturing',...]


After annotate : [('When', 'WRB'), ('it', 'PRP'), ("'s", 'VBZ'), ('time', 'NN'), ('for', 'IN'), ('their', 'PRP$'), ('biannual', 'JJ'), ('powwow', 'NN'), (',', ','), ('the', 'DT'), ('nation', 'NN'), ("'s", 'POS'), ('manufacturing', 'NN'), ...] 
```

### Evaluation à l’aide des étiquettes Penn TreeBank (PTB)

```
~/NLTK-TP1$ python evaluate.py wsj_0010_sample.txt.pos.nltk wsj_0010_sample.pos.ref

Word precision: 0.9363636363636364
Word recall: 0.9363636363636364
Tag precision: 0.9363636363636364
Tag recall: 0.9363636363636364
Word F-measure: 0.9363636363636364
Tag F-measure: 0.9363636363636364
```