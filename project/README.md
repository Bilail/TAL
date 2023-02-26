# Expérimentation

## Vérification de l'installation d'OpenNMT sur le corpus anglais-allemand

### Etape 1 : Préparation des données

Pour commencer, nous téléchargeons un ensemble de données anglais-allemand pour la traduction automatique contenant 10 000 phrases tokenisées :
`onmt_build_vocab -config toy_en_de.yaml -n_sample 10000`

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
```onmt_train -config toy_en_de.yaml```

Résultat de la commande :
```
[2023-02-22 17:13:35,994 INFO] Train perplexity: 4192.57
[2023-02-22 17:13:35,995 INFO] Train accuracy: 7.98713
[2023-02-22 17:13:35,995 INFO] Sentences processed: 32000
[2023-02-22 17:13:35,995 INFO] Average bsz: 1345/1338/64
[2023-02-22 17:13:35,995 INFO] Validation perplexity: 631.623
[2023-02-22 17:13:35,996 INFO] Validation accuracy: 9.42732
[2023-02-22 17:13:36,018 INFO] Saving checkpoint toy-ende/run/model_step_500.pt

[2023-02-22 17:52:22,799 INFO] Train perplexity: 1698.27
[2023-02-22 17:52:22,799 INFO] Train accuracy: 10.8183
[2023-02-22 17:52:22,800 INFO] Sentences processed: 64000
[2023-02-22 17:52:22,800 INFO] Average bsz: 1343/1335/64
[2023-02-22 17:52:22,800 INFO] Validation perplexity: 538.186
[2023-02-22 17:52:22,800 INFO] Validation accuracy: 13.5497
[2023-02-22 17:52:22,821 INFO] Saving checkpoint toy-ende/run/model_step_1000.pt
```
### Etape 3 : Traduction

On démarre la traduction en utilisant notre modèle qu'on à entrainé et on stock le résultat dans le fichier `pred_1000.txt`.
```
onmt_translate -model data/toy-ende/run/model_step_1000.pt -src data/toy-ende/src-test.txt -output data/toy-ende/pred_1000.txt -gpu 0 -verbose
```

### Calcul du score BLEU
Afin de calculer le score BLEU, nous utilisons un fichier que nous avons importer du github suivante : https://github.com/ymoslem/MT-Evaluation.

Nous utilisons la commande suivante pour obtenir le score BLEU :
```
python .\src\compute-bleu.py .\data\toy-ende\tgt-test.txt .\data\toy-ende\pred_1000.txt
```

## Utilisation du moteur OpenNMT sur les corpus TRAIN, DEV et TEST

Ici, nous disposons de 3 corpus : TRAIN, DEV et TEST.
- Le corpus TRAIN : permet d'entraîner le modèle en lui fournissant des exemples de données annotées.
- Le corpus DEV : permet d'évaluer les performances du modèle de NLP pendant l'entraînement et valider son choix.
- Le corpus TEST : permet d'évaluer les performances, lorsque le modèle est déployé dans un environnement réel.

Nous avons créé le fichier de configuration `europarl.yaml`. Il suffit d'éxécuter les mêmes commandes que pour le corpus anglais-allemand en modifiant sulement les paramètres.

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
Résultat :
```
[2023-02-26 14:35:05,451 INFO] PRED SCORE: -1.7921, PRED PPL: 6.00 NB SENTENCES: 500
```
Calcul du score bleu :
```
python .\src\compute-bleu.py .\data\europarl\Europarl_test_500.fr .\data\europarl\pred_1000.tx
```

Résultat obtenu :
```
BLEU:  0.8677162612098606
```

# Evaluation sur des corpus parallèles en formes fléchies à large échelle

Nous avons commencé par créé les fichiers suivants à partir du corpus Europarl : `Europarl_train_100k.en/fr`, `Europarl_dev_3750.en/fr` et `Europarl_test2_500.en/fr`.
De plus, nous avons également créé les fichiers suivants à partir du corpus EMEA : `Emea_train_10k.en/fr` et `Emea_test_500.en/fr`.

## Préparation des corpus pour l'apprentissage

Nous utilisons les scripts de `mosesdecoder` afin de préparer les corpus.

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

#### Apprentissage du modèle de transformation

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

#### Transformation des majuscules en minuscules

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

Dans cette étape, nous allons supprimer toutes les phrases ayant plus de 80 caractères. Cela va donc légérement diminuer la taille de notre corpus. Pour ce faire, on exécute la commande suivante :
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

Ainsi, nous venons de nettoyer tous nos corpus et nous pouvons passer au étapes d'apprentissage et de traduction d'OpenNMT.

## Apprentissage avec OpenNMT


## Traduction avec les corpus TEST


## Evaluation avec le score BLEU

