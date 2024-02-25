![image](https://github.com/AnonymousRole/ITDA/assets/81413010/b0774d9c-296f-4d89-959f-187328b8c8f6)

# Authorship Style Transfer with Inverse Transfer Data Augmentation
This is the offficial implementation of the paper [Authorship Style Transfer with Inverse Transfer Data Augmentation].
# Overview
Authorship style transfer aims to modify the style of neutral text to match the unique speaking or writing style of a particular individual. We propose an inverse transfer data augmentation ITDA method, leveraging GPT-3.5 to create (neutral text, stylized text) pairs. This method involves removing the existing styles from stylized texts, a process made more feasible due to the prevalence of neutral texts in pre-training. We use this augmented dataset to train a BART-base model adept at replicating the targeted style. Our experimental results, conducted across four datasets with distinct authorship styles, establish the effectiveness of ITDA over style transfer using GPT-3.5.
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
