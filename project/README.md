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
```onmt_translate -model toy-ende/run/model_step_1000.pt -src toy-ende/src-test.txt -output toy-ende/pred_1000.txt -gpu 0 -verbose```

### Calcul du score BLEU
Afin de calculer le score BLEU, nous utilisons un fichier que nous avons importer du github suivante : https://github.com/ymoslem/MT-Evaluation.

Nous utilisons la commande suivante pour obtenir le score BLEU :
```
python .\src\compute-bleu.py .\toy-ende\tgt-test.txt .\toy-ende\pred_1000.txt
```

## Utilisation du moteur OpenNMT sur les corpus TRAIN, DEV et TEST

```
onmt_build_vocab -config europarl.yaml -n_sample 10000
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

### 
