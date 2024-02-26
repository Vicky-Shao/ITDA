from nltk.translate.bleu_score import sentence_bleu
import argparse
from bert_score import score
import torch


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for evaluate transferring results.")
    parser.add_argument('--input_file_path', type = str)

    opt = parser.parse_args()

    return opt

def calc_bleu(input_file_path):
    textX = []
    textY = []
    with open(input_file_path, 'r') as fr:
        for i, line in enumerate(fr):
            if i%2==0:
                textX.append(line.split('\t')[1][0:-1])
            else:
                textY.append(line.split('\t')[1][0:-1])

    bleu_sim = 0
    counter = 0
    for i in range(len(textX)):
        if len(textX[i]) > 3 and len(textY[i]) > 3:
            bleu_sim += sentence_bleu([textX[i]], textY[i])
            counter += 1
    
    with open('bleu.txt', 'a') as fw:
        fw.write('input_file_path = ' + input_file_path +'\n')
        fw.write('BLEU = '+str(float(bleu_sim / counter)) +'\n')
        fw.write('------------------------------------\n')


def calc_bertscore(input_file_path):
    textX = []
    textY = []
    with open(input_file_path, 'r') as fr:
        for i, line in enumerate(fr):
            if i%2==0:
                textX.append(line.split('\t')[1][0:-1])
            else:
                textY.append(line.split('\t')[1][0:-1])

    P, R, F1 = score(textY, textX, model_type='roberta-base', lang='en')
    p = torch.mean(P)
    r = torch.mean(R)
    f1 = torch.mean(F1)
    with open('bertscore.txt', 'a') as fw:
        fw.write('input_file_path = ' + input_file_path + '\n')
        fw.write('p = ' + str(p.item()) + '\n')
        fw.write('r = ' + str(r.item()) + '\n')
        fw.write('f1 = ' + str(f1.item()) + '\n')
        fw.write('------------------------------------\n')



if __name__ == "__main__":
    opt = parse_option()
    calc_bleu(opt.input_file_path)
    calc_bertscore(opt.input_file_path)

