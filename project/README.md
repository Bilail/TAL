# README

# Expérimentation

## Vérification de l’installation d’OpenNMT sur le corpus anglais-allemand

### Etape 1 : Préparation des données

Pour commencer, nous téléchargeons un ensemble de données anglais-allemand pour la traduction automatique contenant 10 000 phrases tokenisées. Puis nous construisons le vocabulaire :

```bash
onmt_build_vocab -config toy_en_de.yaml -n_sample 10000
```

Résultat de la commande :

```
Corpus corpus_1's weight should be given. We default it to 1 for you.
[2023-02-22 16:17:43,398 INFO] Counter vocab from 10000 samples.
[2023-02-22 16:17:43,398 INFO] Build vocab on 10000 transformed examples/corpus.
[2023-02-22 16:17:45,477 INFO] Counters src:24995
[2023-02-22 16:17:45,478 INFO] Counters tgt:35816
```

### Etape 2 : Entrainement du modèle

On exécute la commande suivante : 

```bash
onmt_train -config toy_en_de.yaml
```

Résultat de la commande :

```
[2023-02-22 17:52:22,799 INFO] Train perplexity: 1698.27
[2023-02-22 17:52:22,799 INFO] Train accuracy: 10.8183
[2023-02-22 17:52:22,800 INFO] Sentences processed: 64000
[2023-02-22 17:52:22,800 INFO] Average bsz: 1343/1335/64
[2023-02-22 17:52:22,800 INFO] Validation perplexity: 538.186
[2023-02-22 17:52:22,800 INFO] Validation accuracy: 13.5497
[2023-02-22 17:52:22,821 INFO] Saving checkpoint toy-ende/run/model_step_1000.pt
```

### Etape 3 : Traduction

On démarre la traduction en utilisant notre modèle qu’on à entrainé et on stock le résultat dans le fichier `./data/toy-ende/pred_1000.txt`.

```
onmt_translate -model data/toy-ende/run/model_step_1000.pt -src data/toy-ende/src-test.txt -output data/toy-ende/pred_1000.txt -gpu 0 -verbose
```

### Calcul du score BLEU

Afin de calculer le score BLEU, nous utilisons un fichier que nous avons importer du GitHub suivante : https://github.com/ymoslem/MT-Evaluation.

Nous utilisons la commande suivante pour obtenir le score BLEU :

```
python .\src\compute-bleu.py .\data\toy-ende\tgt-test.txt .\data\toy-ende\pred_1000.txt
```

Nous obtenons un score de `0.088`, ce qui est extrêmement faible. Voici ci-dessous un tableau permettant d’interpreter l’éfficacité d’un modèle de traduction en fonction de son score BLEU :

| Score BLEU | Interprétation |
| --- | --- |
| < 10 | Traductions presque inutiles |
| 10 à 19 | L'idée générale est difficilement compréhensible |
| 20 à 29 | L'idée générale apparaît clairement, mais le texte comporte de nombreuses erreurs grammaticales |
| 30 à 40 | Résultats compréhensibles à traductions correctes |
| 40 à 50 | Traductions de haute qualité |
| 50 à 60 | Traductions de très haute qualité, adéquates et fluides |
| > 60 | Qualité souvent meilleure que celle d'une traduction humaine |

## Utilisation du moteur OpenNMT sur les corpus TRAIN, DEV et TEST

Ici, nous disposons de 3 corpus : TRAIN, DEV et TEST. - Le corpus TRAIN : permet d’entraîner le modèle en lui fournissant des exemples de données annotées. - Le corpus DEV : permet d’évaluer les performances du modèle de NLP pendant l’entraînement et valider son choix. - Le corpus TEST : permet d’évaluer les performances, lorsque le modèle est déployé dans un environnement réel.

Nous avons créé le fichier de configuration `europarl.yaml`. Il suffit d’éxécuter les mêmes commandes que pour le corpus anglais-allemand en modifiant seulement les paramètres.

Création du vocabulaire :

```
onmt_build_vocab -config europarl.yaml -n_sample 10000
```

Entrainement du modèle :

```
onmt_train -config europarl.yaml
```

Résultat de la commande :

```
[2023-02-25 18:12:49,982 INFO] Train perplexity: 890.633
[2023-02-25 18:12:49,983 INFO] Train accuracy: 10.074
[2023-02-25 18:12:49,983 INFO] Sentences processed: 64000
[2023-02-25 18:12:49,983 INFO] Average bsz: 1359/1462/64
[2023-02-25 18:12:49,983 INFO] Validation perplexity: 355.288
[2023-02-25 18:12:49,983 INFO] Validation accuracy: 14.7751
[2023-02-25 18:12:50,051 INFO] Saving checkpoint europarl/run/model_step_1000.pt
```

Traduction :

```
onmt_translate -model data/europarl/run/model_step_1000.pt -src data/europarl/Europarl_test_500.en -output data/europarl/pred_1000.txt -gpu 0 -verbose
```

Calcul du score bleu :

```
python .\src\compute-bleu.py .\data\europarl\Europarl_test_500.fr .\data\europarl\pred_1000.tx
```

Score BLEU : `0.86`

Là encore, le score BLEU est très faible. Cela est principalement dû eau faible nombre de phrases dans les corpus choisis.

# Evaluation sur des corpus parallèles en formes fléchies à large échelle

Nous avons commencé par créer les fichiers suivants à partir du corpus Europarl : `Europarl_train_100k.en/fr`, `Europarl_dev_3750.en/fr` et `Europarl_test2_500.en/fr`.

De plus, nous avons également créé les fichiers suivants à partir du corpus EMEA : `Emea_train_10k.en/fr` et `Emea_test_500.en/fr`.

## Préparation des corpus pour l’apprentissage

Nous utilisons les scripts de `mosesdecoder` afin de préparer les corpus. Cette étape de préparation des données est essentiel afin de maximiser l’efficacité de notre modèle lors de la phase d’apprentissage, de tuning et de traduction.

### Tokenisation du corpus Anglais-Français

Corpus anglais :

```
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data/europarl/Europarl_train_100k.en > data/europarl/Europarl_train_100k.tok.en
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data/europarl/Europarl_dev_3750.en > data/europarl/Europarl_dev_3750.tok.en
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data/europarl/Europarl_test2_500.en > data/europarl/Europarl_test2_500.tok.en
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data/EMEA/Emea_train_10k.en > data/EMEA/Emea_train_10k.tok.en
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data/EMEA/Emea_test_500.en > data/EMEA/Emea_test_500.tok.en
```

Corpus français :

```
mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr < data/europarl/Europarl_train_100k.fr > data/europarl/Europarl_train_100k.tok.fr
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data/europarl/Europarl_dev_3750.fr > data/europarl/Europarl_dev_3750.tok.fr
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data/europarl/Europarl_test2_500.fr > data/europarl/Europarl_test2_500.tok.fr
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data/EMEA/Emea_train_10k.fr > data/EMEA/Emea_train_10k.tok.fr
mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < data/EMEA/Emea_test_500.fr > data/EMEA/Emea_test_500.tok.fr
```

Nous obtenons les fichiers : 

- Europarl_train_100k.tok.en
- Europarl_dev_3750.tok.en
- Europarl_test2_500.tok.en
- Emea_train_10k.tok.en
- Emea_test_500.tok.en
- Europarl_train_100k.tok.fr
- Europarl_dev_3750.tok.fr
- Europarl_test2_500.tok.fr
- Emea_train_10k.tok.fr
- Emea_test_500.tok.fr

### Changement des majuscules en minuscules du corpus Anglais-Français

### Apprentissage du modèle de transformation

Corpus anglais :

```
mosesdecoder/scripts/recaser/train-truecaser.perl --model data/europarl/truecase-model.en --corpus data/europarl/Europarl_train_100k.tok.en
mosesdecoder/scripts/recaser/train-truecaser.perl --model data/EMEA/truecase-model.en --corpus data/EMEA/Emea_train_10k.tok.en
```

Corpus français :

```
mosesdecoder/scripts/recaser/train-truecaser.perl --model data/europarl/truecase-model.fr --corpus data/europarl/Europarl_train_100k.tok.fr
mosesdecoder/scripts/recaser/train-truecaser.perl --model data/EMEA/truecase-model.fr --corpus data/EMEA/Emea_train_10k.tok.fr
```

Nous obtenons les fichiers : 

- truecase-model.en (Europarl)
- truecase-model.en (EMEA)
- truecase-model.fr (Europarl)
- truecase-model.fr (EMEA)

### Transformation des majuscules en minuscules

Corpus anglais :

```
mosesdecoder/scripts/recaser/truecase.perl --model data/europarl/truecase-model.en < data/europarl/Europarl_train_100k.tok.en > data/europarl/Europarl_train_100k.tok.true.en
mosesdecoder/scripts/recaser/truecase.perl --model data/europarl/truecase-model.en < data/europarl/Europarl_dev_3750.tok.en > data/europarl/Europarl_dev_3750.tok.true.en
mosesdecoder/scripts/recaser/truecase.perl --model data/europarl/truecase-model.en < data/europarl/Europarl_test2_500.tok.en > data/europarl/Europarl_test2_500.tok.true.en
mosesdecoder/scripts/recaser/truecase.perl --model data/EMEA/truecase-model.en < data/EMEA/Emea_train_10k.tok.en > data/EMEA/Emea_train_10k.tok.true.en
mosesdecoder/scripts/recaser/truecase.perl --model data/EMEA/truecase-model.en < data/EMEA/Emea_test_500.tok.en > data/EMEA/Emea_test_500.tok.true.en
```

Corpus français :

```
mosesdecoder/scripts/recaser/truecase.perl --model data/europarl/truecase-model.fr < data/europarl/Europarl_train_100k.tok.fr > data/europarl/Europarl_train_100k.tok.true.fr
mosesdecoder/scripts/recaser/truecase.perl --model data/europarl/truecase-model.fr < data/europarl/Europarl_dev_3750.tok.fr > data/europarl/Europarl_dev_3750.tok.true.fr
mosesdecoder/scripts/recaser/truecase.perl --model data/europarl/truecase-model.fr < data/europarl/Europarl_test2_500.tok.fr > data/europarl/Europarl_test2_500.tok.true.fr
mosesdecoder/scripts/recaser/truecase.perl --model data/EMEA/truecase-model.fr < data/EMEA/Emea_train_10k.tok.fr > data/EMEA/Emea_train_10k.tok.true.fr
mosesdecoder/scripts/recaser/truecase.perl --model data/EMEA/truecase-model.fr < data/EMEA/Emea_test_500.tok.fr > data/EMEA/Emea_test_500.tok.true.fr
```

Nous obtenons les fichiers :

- Europarl_train_100k.tok.true.en
- Europarl_dev_3750.tok.true.en
- Europarl_test2_500.tok.true.en
- Emea_train_10k.tok.true.en
- Emea_test_500.tok.true.en
- Europarl_train_100k.tok.true.fr
- Europarl_dev_3750.tok.true.fr
- Europarl_test2_500.tok.true.fr
- Emea_train_10k.tok.true.fr
- Emea_test_500.tok.true.fr

### Nettoyage en limitant la longueur des phrases à 80 caractères

Dans cette étape, nous allons supprimer toutes les phrases ayant plus de 80 caractères. Cela va donc légèrement diminuer la taille de notre corpus. Pour ce faire, on exécute la commande suivante :

```
mosesdecoder/scripts/training/clean-corpus-n.perl data/europarl/Europarl_train_100k.tok.true fr en data/europarl/Europarl_train_100k.tok.true.clean 1 80
mosesdecoder/scripts/training/clean-corpus-n.perl data/europarl/Europarl_dev_3750.tok.true fr en data/europarl/Europarl_dev_3750.tok.true.clean 1 80
mosesdecoder/scripts/training/clean-corpus-n.perl data/europarl/Europarl_test2_500.tok.true fr en data/europarl/Europarl_test2_500.tok.true.clean 1 80
mosesdecoder/scripts/training/clean-corpus-n.perl data/EMEA/Emea_train_10k.tok.true fr en data/EMEA/Emea_train_10k.tok.true.clean 1 80
mosesdecoder/scripts/training/clean-corpus-n.perl data/EMEA/Emea_test_500.tok.true fr en data/EMEA/Emea_test_500.tok.true.clean 1 80
```

Nous obtenons les fichiers :

- Europarl_train_100k.tok.true.clean.en (97769 lignes avec la commande `wc -l`)
- Europarl_dev_3750.tok.true.clean.en (3645 lignes avec la commande `wc -l`)
- Europarl_test2_500.tok.true.clean.en (488 lignes avec la commande `wc -l`)
- Emea_train_10k.tok.true.clean.en (9881 lignes avec la commande `wc -l`)
- Emea_test_500.tok.true.clean.en (500 lignes avec la commande `wc -l`)
- Europarl_train_100k.tok.true.clean.fr (97769 lignes avec la commande `wc -l`)
- Europarl_dev_3750.tok.true.clean.fr (3645 lignes avec la commande `wc -l`)
- Europarl_test2_500.tok.true.clean.fr (488 lignes avec la commande `wc -l`)
- Emea_train_10k.tok.true.clean.fr (9881 lignes avec la commande `wc -l`)
- Emea_test_500.tok.true.clean.fr (500 lignes avec la commande `wc -l`)

Ainsi, nous venons de nettoyer tous nos corpus et nous pouvons passer au étapes d’apprentissage et de traduction d’OpenNMT.

## Apprentissage avec OpenNMT

### Apprentissage de la run n°1

On commence avec la run n°1 en utilisant `Europarl_train_100K` pour l’apprentissage et `Europarl_dev_3750` pour le tuning. Pour cela on a configuré le fichier `run_1_formes_flechies.yaml`. Ensuite, on génère le vocabulaire avec la commande suivante :

```
onmt_build_vocab -config run_1_formes_flechies.yaml -n_sample 100000
```

Puis, on démarre l’apprentissage :

```
onmt_train -config run_1_formes_flechies.yaml
```

### Apprentissage de la run n°2

On commence avec la run n°2 en utilisant `Europarl_train_100K` et `Emea_train_10k` pour l’apprentissage et `Europarl_dev_3750` pour le tuning. Pour cela on a configuré le fichier `run_2_formes_flechies.yaml`. Ensuite, on génère le vocabulaire avec la commande suivante :

```
onmt_build_vocab -config run_2_formes_flechies.yaml -n_sample 100000
```

Puis, on démarre l’apprentissage :

```
onmt_train -config run_2_formes_flechies.yaml
```

## Traduction et évaluation du score BLEU

### Traduction avec le modèle de la run n°1

On effectue 2 traductions : l’un avec un corpus de test du domaine, et un autre hors-domaine.

Traduction du corpus du domaine avec le corpus `Europarl_test2_500.tok.true.clean.en` :

```bash
onmt_translate -model data/run_1_formes_flechies/model_step_10000.pt -src data/europarl/Europarl_test2_500.tok.true.clean.en -output data/run_1_formes_flechies/pred_domaine.txt -verbose
```

Traduction du corpus hors-domaine avec le corpus `Emea_test_500.tok.true.clean.en` :

```bash
onmt_translate -model data/run_1_formes_flechies/model_step_10000.pt -src data/EMEA/Emea_test_500.tok.true.clean.en -output data/run_1_formes_flechies/pred_hors_domaine.txt -verbose
```

Ensuite nous calculons le score BLEU pour chaque corpus.

Calcul du score BLEU pour le corpus du domaine :

```
python ./src/compute-bleu.py ./data/europarl/Europarl_test2_500.tok.true.clean.fr ./data/run_1_formes_flechies/pred_domaine.txt
```

Score BLEU : `29.21`

Calcul du score BLEU pour le corpus hors-domaine :

```bash
python ./src/compute-bleu.py ./data/EMEA/Emea_test_500.tok.true.clean.fr ./data/run_1_formes_flechies/pred_hors_domaine.txt
```

Score BLEU : `0.51`

On peut donc observer que notre modèle de traduction commence à être correcte sur des corpus du même domaine, même si cela n’est pas encore assez fiable. En revanche, dès que nous essayons de traduire des textes provenant d’un domaine différent, notre modèle redevient inefficace.

***Remarque :** Nous avons essayer de jouer sur le paramètre `train_step` dans le fichier de configuration afin d’observer son impact. Nous avons comparé le score BLEU pour un modèle entrainé en 1 000 étapes et un modèle entrainé en 10 000. Nous avons pu constater que le score BLEU était de `29.21` pour le modèle avec 10 000 étapes, tandis que le modèle avec 1 000 étapes n’avait un score BLEU de seulement `6.70`. Cela permet de souligner l’importance du choix de ce paramètre lors de la phase d’apprentissage.*

### Traduction avec le modèle de la run n°2

Cette fois, ci on effectue les même commande que pour la run n°1, mais avec le modèle de la run n°2. Pour rappel, ce modèle a été entrainé avec des données du corpus Europarl et Emea.

Traduction du corpus du domaine avec le corpus `Europarl_test2_500.tok.true.clean.en` :

```bash
onmt_translate -model data/run_2_formes_flechies/model_step_10000.pt -src data/europarl/Europarl_test2_500.tok.true.clean.en -output data/run_2_formes_flechies/pred_domaine.txt -verbose
```

Traduction du corpus hors-domaine avec le corpus `Emea_test_500.tok.true.clean.en` :

```bash
onmt_translate -model data/run_2_formes_flechies/model_step_10000.pt -src data/EMEA/Emea_test_500.tok.true.clean.en -output data/run_2_formes_flechies/pred_hors_domaine.txt -verbose
```

Ensuite nous calculons le score BLEU pour chaque corpus.

Calcul du score BLEU pour le corpus du domaine :

```
python ./src/compute-bleu.py ./data/europarl/Europarl_test2_500.tok.true.clean.fr ./data/run_2_formes_flechies/pred_domaine.txt
```

Score BLEU : `21.04`

Calcul du score BLEU pour le corpus hors-domaine :

```bash
python ./src/compute-bleu.py ./data/EMEA/Emea_test_500.tok.true.clean.fr ./data/run_2_formes_flechies/pred_hors_domaine.txt
```

Score BLEU : `74.55`

La première chose que l’on remarque à la suite de cette expérimentation est le score très élevé pour la traduction du corpus hors-domaine. Le fait d’avoir inclus le corpus hors-domaine lors de la phase d’apprentissage à donc eu un impact très important. En revanche, on observe également que le score BLEU de la traduction du texte du domaine a diminué de presque 30%. Cela est dû au poids que l’on affecte à chaque corpus lors de la phase d’apprentissage. Dans notre cas, nous avons attribué un point équivalent pour le corpus Europarl et Emea, ce qui signifie que notre modèle essaie d’équilibrer l’importance de chaque corpus. Ainsi, cela est logique que ce second modèle soit moins performant pour le corpus Europarl car il n’est pas “spécialisé” uniquement dans ce domaine, contrairement au premier modèle de la run n°1.

# Evaluation sur des corpus parallèles en lemmes à large échelle

## Lemmatisation des corpus

Nous avons crée un script python permettant de lemmatiser un texte. Pour cela il suffit de fournir la langue, le fichier d’entrée et le fichier de sorti de la façon suivante :

```bash
python src/lemmatisation.py [en/fr] input_file output_file
```

*Remarque : il faut utiliser les commandes ci-dessous pour installer les librairies de lemmatisation anglais et français.*

```bash
pip install nltk
pip install git+https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer.git
```

Ainsi, nous utilisons cette commande pour lemmatiser tous nos corpus.

Lemmatisation des corpus anglais :

```bash
python src/lemmatisation.py en data/europarl/Europarl_train_100k.tok.true.clean.en data/europarl/train_100k_lemmatized.en
python src/lemmatisation.py en data/europarl/Europarl_dev_3750.tok.true.clean.en data/europarl/dev_3750_lemmatized.en
python src/lemmatisation.py en data/europarl/Europarl_test2_500.tok.true.clean.en data/europarl/test_500_lemmatized.en
python src/lemmatisation.py en data/EMEA/Emea_train_10k.tok.true.clean.en data/EMEA/train_10k_lemmatized.en
python src/lemmatisation.py en data/EMEA/Emea_test_500.tok.true.clean.en data/EMEA/test_500_lemmatized.en
```

Lemmatisation des corpus français :

```bash
python src/lemmatisation.py fr data/europarl/Europarl_train_100k.tok.true.clean.fr data/europarl/train_100k_lemmatized.fr
python src/lemmatisation.py en data/europarl/Europarl_dev_3750.tok.true.clean.fr data/europarl/dev_3750_lemmatized.fr 
python src/lemmatisation.py fr data/europarl/Europarl_test2_500.tok.true.clean.fr data/europarl/test_500_lemmatized.fr
python src/lemmatisation.py fr data/EMEA/Emea_train_10k.tok.true.clean.fr data/EMEA/train_10k_lemmatized.fr
python src/lemmatisation.py fr data/EMEA/Emea_test_500.tok.true.clean.fr data/EMEA/test_500_lemmatized.fr
```

## Apprentissage avec OpenNMT

Ici, nous relaçons les phases d’apprentissage en suivant le protocole expérimental de l’exercice III.

Nous avons crée les fichiers fichiers de configurations `run_1_lemmes.yaml`et `run_2_lemmes.yaml` afin d’utiliser les nouveaux corpus générés lors de l’étape précédente.

Génération du vocabulaire pour chaque run :

```
onmt_build_vocab -config run_1_lemmes.yaml -n_sample 100000
onmt_build_vocab -config run_2_lemmes.yaml -n_sample 100000
```

Lancement des apprentissages :

```
onmt_train -config run_1_lemmes.yaml
onmt_train -config run_2_lemmes.yaml
```

## Traduction et évaluation du score BLEU

### Traduction avec le modèle de la run n°1

On effectue 2 traductions : l’un avec un corpus de test du domaine, et un autre hors-domaine.

Traduction du corpus du domaine :

```bash
onmt_translate -model data/run_1_lemmes/model_step_10000.pt -src data/europarl/test_500_lemmatized.en -output data/run_1_lemmes/pred_domaine.txt -verbose
```

Traduction du corpus hors-domaine :

```bash
onmt_translate -model data/run_1_lemmes/model_step_10000.pt -src data/EMEA/test_500_lemmatized.en -output data/run_1_lemmes/pred_hors_domaine.txt -verbose
```

Ensuite nous calculons le score BLEU pour chaque corpus.

Calcul du score BLEU pour le corpus du domaine :

```
python ./src/compute-bleu.py ./data/europarl/test_500_lemmatized.fr ./data/run_1_lemmes/pred_domaine.txt
```

Score BLEU : `22.15`

Calcul du score BLEU pour le corpus hors-domaine :

```bash
python ./src/compute-bleu.py ./data/EMEA/test_500_lemmatized.fr ./data/run_1_lemmes/pred_hors_domaine.txt
```

Score BLEU : `0.75`

On peut observer que notre modèle de traduction reste faiblement correcte sur des corpus du même domaine. En revanche, dès que nous essayons de traduire des textes provenant d’un domaine différent, notre modèle redevient inefficace.

### Traduction avec le modèle de la run n°2

Cette fois, ci on effectue les même commande que pour la run n°1, mais avec le modèle de la run n°2. Pour rappel, ce modèle a été entrainé avec des données du corpus Europarl et Emea.

Traduction du corpus du domaine :

```bash
onmt_translate -model data/run_2_lemmes/model_step_10000.pt -src data/europarl/test_500_lemmatized.en -output data/run_2_lemmes/pred_domaine.txt -verbose
```

Traduction du corpus hors-domaine :

```bash
onmt_translate -model data/run_2_lemmes/model_step_10000.pt -src data/EMEA/test_500_lemmatized.en -output data/run_2_lemmes/pred_hors_domaine.txt -verbose
```

Ensuite nous calculons le score BLEU pour chaque corpus.

Calcul du score BLEU pour le corpus du domaine :

```
python ./src/compute-bleu.py ./data/europarl/test_500_lemmatized.fr ./data/run_2_lemmes/pred_domaine.txt
```

Score BLEU : `16.57`

Calcul du score BLEU pour le corpus hors-domaine :

```bash
python ./src/compute-bleu.py ./data/EMEA/test_500_lemmatized.fr ./data/run_2_lemmes/pred_hors_domaine.txt
```

Score BLEU : `83.76`

Les performance du modèle sont assez mauvaises pour la traduction de corpus du domaine. En revanche, le score BLEU est très élevé pour la traduction du corpus hors-domaine.