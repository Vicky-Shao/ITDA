# Lifelike-Writer
This is the repository accompanying the paper [LIFELIKE-WRITER: Authorship Style Transfer with Inverse Knowledge Distillation].
## Datasets
datasets/hlm, datasets/shakespeare, datasets/trump, datasets/lyrics
## Static Few-shot Prompting
few_shot_poll.py
## Clustering-based Dynamic Prompting
* kmeans.py
* dynamicInverse_poll.py
## Finetuning the small model
* ft_bart_en.py
* ft_bart_ch.py
## Classifier Training
classifer_train_**.py
## Evaluation
* eval.py (BLEU, PPL)
* classifier_metrics_**.py (WSC)
## Transferring
bart_transfer.py
