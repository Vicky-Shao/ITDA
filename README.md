![image](https://github.com/AnonymousRole/ITDA/assets/81413010/b0774d9c-296f-4d89-959f-187328b8c8f6)

# Authorship Style Transfer with Inverse Transfer Data Augmentation
This is the offficial implementation of the paper [Authorship Style Transfer with Inverse Transfer Data Augmentation].
## Overview
Authorship style transfer aims to modify the style of neutral text to match the unique speaking or writing style of a particular individual. We propose an inverse transfer data augmentation ITDA method, leveraging GPT-3.5 to create (neutral text, stylized text) pairs. We use this augmented dataset to train a BART-base model adept at style transfer. Our experimental results, conducted across four datasets with distinct authorship styles, establish the effectiveness of ITDA over style transfer using GPT-3.5.
## Evaluation Results
We evaluate ITDA on four benchmarks: Lin Daiyu, Shakespeare, Trump, Lyrics. We adopt four metrics: BLEU and BS (BERTScore) measure content preservation, SC measures style transfer
strength, and GPT-4 measures overall performance. 
![image](https://github.com/AnonymousRole/ITDA/assets/81413010/a7db80b0-9cd9-41b4-b3c4-b55449ea96a5)
Since user-provided text often spans a range of topics, we also collect a new test set comprising neutral texts spanning diverse topics to do out-of-distribution evaluation.
![image](https://github.com/AnonymousRole/ITDA/assets/81413010/830dd489-3a8d-4b34-bd02-6be67f780640)

## Install the requirements <a name = "install"></a>
First, you need to create a virtual environment and activate it:
```sh
conda deactivate
conda create -n <env_name> python=3.8
conda activate <env_name>
```
Then, install the cuda version Pytorch：
```sh
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```
Finally, install the requirements:
```sh
conda install --file requirements.txt
```
## Stylized Datasets
datasets/hlm, datasets/shakespeare, datasets/trump, datasets/lyrics
## Training
(a) Cluster-based Demonstration Annotation
```sh
python kmeans.py
```
(b) Stylized Text Augmentation
```sh
python stylized_augmentation.py
```
(c) Inverse Transfer Data Augmentation
```sh
python dynamicInverse_poll.py
```
(d) Fine-tune a Compact Model
```sh
python ft_bart_en.py   #For English Datasets
python ft_bart_ch.py   #For Chinese Datasets
```
## Inference
```sh
python bart_transfer.py
```
## Classifier Training
```sh
python classifer_train_en.py  #For English Datasets
python classifer_train_ch.py  #For Chinese Datasets
```
## Evaluation
```sh
python eval.py (BLEU, PPL)
python classifier_metrics_**.py (SC)
```
## Fixed Few-shot Prompting for Fowrad Transfer with GPT-3.5
```sh
python few_shot_poll.py
```

