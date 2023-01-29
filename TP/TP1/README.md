# TP 1 Analyse linguistique NLTK 

Le code complet du TP s'exécute grâce au fichier `main.py` avec la commande `python main.py`. Il est nécessaire de se placer dans le répertoire `TP/TP1/.` pour exécuter le fichier `main.py`.
Les fichiers de données et les fichiers de sortie sont enregistés dans le dossier `data/`.

## I - Evaluation de l’analyse morpho-syntaxique de la plateforme NLTK

### 1. Désambiguïsation morphosyntaxique avec le package *pos_tag*

#### Code correspondant :
```python
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
```

#### Résultat obtenu dans le fichier `wsj_0010_sample.txt.pos.nltk` : 

```
When	WRB
it	PRP
's	VBZ
time	NN
for	IN
their	PRP$
biannual	JJ
powwow	NN
,	,
the	DT
nation	NN
's	POS
manufacturing	NN
...
```

### 2. Evaluation à l’aide des étiquettes Penn TreeBank (PTB)

#### Ligne de commande exécutée :
```
~/TAL/TP/TP1$ python evaluate.py wsj_0010_sample.txt.pos.nltk wsj_0010_sample.pos.ref
```
#### Résultat obtenu :
```
Word precision: 0.9363636363636364
Word recall: 0.9363636363636364
Tag precision: 0.9363636363636364
Tag recall: 0.9363636363636364
Word F-measure: 0.9363636363636364
Tag F-measure: 0.9363636363636364
```

### 3. Evaluation à l’aide des étiquettes universelles

#### a. Remplacement des étiquettes Penn TreeBank par les étiquettes universelles 
```python
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
  with open(input,"r") as pos_file, open(output,"w") as univ_file:
    for line in pos_file.readlines():
      token, tag = line.strip().split('\t')
      if not univDic[tag]:
        print("tag inconnu: ", tag)
      else:
        univ_file.write(token + "\t" + univDic[tag] + "\n")

# Remplacement des tags pour le fichier nltk
replace_with_univ_tags(file + '.pos.nltk',file + ".pos.univ.nltk")

# Remplacement des tags pour le fichier ref
replace_with_univ_tags("data/wsj_0010_sample.pos.ref","data/wsj_0010_sample.txt.pos.univ.ref")
```

#### b. Evaluation à l’aide des étiquettes universelles
Ligne de commande exécutée :
```
~/TAL/TP/TP1$ python evaluate.py data/wsj_0010_sample.txt.pos.univ.nltk data/wsj_0010_sample.txt.pos.univ.ref
```
Résultat obtenu :
```
Word precision: 0.9545454545454546
Word recall: 0.9545454545454546
Tag precision: 0.9545454545454546
Tag recall: 0.9545454545454546
Word F-measure: 0.9545454545454546
Tag F-measure: 0.9545454545454546
```

#### c. Quelles conclusions peut-on avoir à partir de ces deux évaluations ?
Les deux évaluations obtiennent des résultats similaires, avec des scores proches 
de 0,936 pour les étiquettes Penn TreeBank et de 0,954 pour les étiquettes universelles.
Cela indique que le modèle utilisé est capable de produire des résultats de qualité, 
quelle que soit l'étiquette utilisée.
Cependant, il est important de noter que les étiquettes universelles ont tendance à 
être plus générales et moins spécifiques que les étiquettes Penn TreeBank, ce qui 
peut expliquer pourquoi les scores sont légèrement plus élevés avec les étiquettes
universelles.
En général, on peut conclure que le modèle utilisé est efficace pour l'étiquetage 
morphosyntaxique des mots.

## II - Utilisation de la plateforme NLTK pour l’analyse syntaxique

### 1. Extraction des mots composés avec la structure Déterminant-Adjectif-Nom
Code correspondant :
```python
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
```
Résultat obtenu dans le fichier `wsj_0010_sample.txt.chk.nltk` :
```
(S
  When/WRB
  it/PRP
  's/VBZ
  (Compound time/NN)
  for/IN
  their/PRP$
  (Compound biannual/JJ powwow/NN)
  ,/,
  (Compound the/DT nation/NN)
  's/POS
  (Compound manufacturing/NN)
  titans/NNS
  typically/RB
  jet/VBP
  off/IN
  ...
```
### 2. Généralisation de l'extraction des mots composés
Nous avons déclaré un fichier `structures_syntaxiques.txt` dans lequel il est possible de définir une liste de structures syntaxiques à extraire. Voici son contenu :
```
Adjective-Noun: {<JJ><NN>}
Noun-Noun: {<NN><NN>}
Adjective-Noun-Noun: {<JJ><NN><NN>}
Adjective-Adjective-Noun: {<JJ><JJ><NN>}
```
Le code suivant permet d'appliquer l'extraction des mots composés :
```python
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
```
Résultat obtenu dans le fichier `wsj_0010_sample.txt.chk_gen.nltk` :
```
(S
  When/WRB
  it/PRP
  's/VBZ
  time/NN
  for/IN
  their/PRP$
  (Adjective-Noun biannual/JJ powwow/NN)
  ,/,
  the/DT
  nation/NN
  ...
```
## Utilisation de la plateforme NLTK pour l’extraction d’entités nommées

### 1. Extraction des entités nommées
Code correspondant :
```python
# Charger le fichier
with open(file, 'r') as f:
  text = f.read()
# Tokenisation du texte
tokens = nltk.word_tokenize(text)
# Tagging des tokens
tags = nltk.pos_tag(tokens)
# Extraction des entités nommées
entities = ne_chunk(tags)
# Enregistrement des résultats
with open(file + '.ne.nltk', 'w') as f:
  for subtree in entities.subtrees():
    if subtree.label() != "S":
      entity = " ".join([token for token, tag in subtree.leaves()])
      f.write(f"{subtree.label()}: {entity}\n")
```
Résultat obtenu :
```
PERSON: Boca Raton
PERSON: Hot Springs
ORGANIZATION: National Association
ORGANIZATION: Hoosier
GPE: Indianapolis
ORGANIZATION: Rust Belt
```

### 2. Conversion des étiquettes des entités nommées

Code correspondant :
```python
# Charger le fichier
with open(file, 'r') as f:
  text = f.read()

# Tokenisation du texte
tokens = nltk.word_tokenize(text)

# Tagging des tokens
tags = nltk.pos_tag(tokens)

# Extraction des entités nommées
entities = ne_chunk(tags)

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
with open(file + '.standard_ne.nltk', 'w') as f:
  for entity in entities.subtrees():
      if entity.label() != "S":
        idx = 0
        for (token, tag) in entity.leaves():
          new_label = ('B-' if idx == 0 else 'I-') + label_mapping.get(entity.label())
          f.write(f"{new_label}: {token}\n")
          idx += 1
```
Résultat obtenu :
```
B-PERS: Boca
I-PERS: Raton
B-PERS: Hot
I-PERS: Springs
B-ORG: National
I-ORG: Association
B-ORG: Hoosier
B-LOC: Indianapolis
B-ORG: Rust
I-ORG: Belt
```

### 3. Analyse du fichier `formal-tst.NE.key.04oct95_sample.txt`

En exécutant le programme de la **question 1** on obtient le fichier `formal-tst.NE.key.04oct95_sample.txt.ne.nltk`.

Voici un aperçu de son contenu :
```
PERSON: Consuela
ORGANIZATION: Washington
ORGANIZATION: House
ORGANIZATION: Securities
ORGANIZATION: Exchange Commission
PERSON: Clinton
GPE: Washington
GPE: Washington
PERSON: Chairman
PERSON: John Dingell
ORGANIZATION: House Energy
...
```

En exécutant le programme de la **question 2** on obtient le fichier `formal-tst.NE.key.04oct95_sample.txt.standard_ne.nltk`.

Voici un aperçu de son contenu :
```
B-PERS: Consuela
B-ORG: Washington
B-ORG: House
B-ORG: Securities
B-ORG: Exchange
I-ORG: Commission
B-PERS: Clinton
B-LOC: Washington
B-LOC: Washington
B-PERS: Chairman
B-PERS: John
I-PERS: Dingell
...
```