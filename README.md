# Lifelike-Writer
This is the repository accompanying the paper [LIFELIKE-WRITER: Authorship Style Transfer with Inverse Knowledge Distillation]
# Datasets
datasets/hlm, datasets/shakespeare, datasets/trump, datasets/lyrics
# Static Few-shot Prompting
few_shot_poll.py
# Clustering-based Dynamic Prompting
kmean.py
dynamicInverse_poll.py
# Finetune the small model
ft_bart_en.py
ft_bart_ch.py
# Train the Classifier
classifer_train_**.py
# Eval
eval.py (BLEU, PPL)
classifier_metrics_**.py (WSC)
# Transfer
bart_transfer.py
