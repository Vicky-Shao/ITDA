![image](https://github.com/AnonymousRole/ITDA/assets/81413010/b0774d9c-296f-4d89-959f-187328b8c8f6)

# Authorship Style Transfer with Inverse Transfer Data Augmentation
This is the offficial implementation of the paper [Authorship Style Transfer with Inverse Transfer Data Augmentation].
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
