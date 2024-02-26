import os
import argparse
import torch 
import torch.nn as nn 
from transformers import BertModel, BertTokenizer


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for evaluate transferring results.")
    parser.add_argument('--style_name', type = str)
    parser.add_argument('--input_file_name', type = str, default = '')
    parser.add_argument('--output_file_name', type = str, default = '')
    parser.add_argument('--device', type = str, default = "0",
                        help = 'the id of used GPU device.')
    parser.add_argument('--model_name', type = str, default = "uer/chinese_roberta_L-12_H-768")

    opt = parser.parse_args()

    return opt


class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication,self).__init__()
        self.model_name = "uer/chinese_roberta_L-12_H-768"#opt.model_name_or_path #'hfl/chinese-bert-wwm'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768,2)     #768取决于BERT结构，2-layer, 768-hidden, 12-heads, 110M parameters

    def forward(self,x):
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                max_length=512, pad_to_max_length=True)     
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        hiden_outputs = self.model(input_ids,attention_mask=attention_mask)
        outputs = hiden_outputs[0][:,0,:]     
        output = self.fc(outputs)
        return output



def _metrics(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    # 整理数据集
    textX = []
    textY = []
    with open(input_file_path, 'r') as fr:
        for i, line in enumerate(fr):
            if i%2==0:
                textX.append(line.split('\t')[1][0:-1])
            else:
                textY.append(line.split('\t')[1][0:-1])


    # 加载模型
    bertclassfication = BertClassfication()

    if torch.cuda.is_available():
        bertclassfication = bertclassfication.cuda()

    checkpoint = torch.load(model_load_path)
    bertclassfication.load_state_dict(checkpoint['model_state_dict'])#加载网络权重参数
    # optimizer.load_state_dict(checkpoint['optimize_state_dict'])#加载优化器参数
    for parameter in bertclassfication.parameters():
        parameter.requires_grad=False
    bertclassfication.eval()


    # 测试
    metrics_total = 0
    hit = 0 #用来计数，看看预测对了多少个
    total = len(textX) # 看看一共多少例子
    bertclassfication.eval()
    with open(output_file_path, 'w') as fw:
        for i in range(total):
            outputsX = bertclassfication([textX[i]])
            outputsY = bertclassfication([textY[i]])
            possibilityX = torch.softmax(outputsX/24, 1)
            possibilityY = torch.softmax(outputsY/24, 1)

            if (round(possibilityY.cpu().numpy()[0][1], 6)-round(possibilityX.cpu().numpy()[0][1], 6))>0:
                hit += 1
                fw.write('Y\n')
            else:
                fw.write('N\n')
        
            metrics_total += round((round(possibilityY.cpu().numpy()[0][1], 8)-round(possibilityX.cpu().numpy()[0][1], 8)), 6)

            fw.write(str(outputsX[0][1].item())+'\t'+textX[i]+'\t')
            fw.write(str(round(possibilityX.cpu().numpy()[0][1], 6))+'\n')
            fw.write(str(outputsY[0][1].item())+'\t'+textY[i]+'\t')
            fw.write(str(round(possibilityY.cpu().numpy()[0][1], 6))+'\n')

        fw.write('准确率为%.4f'%(hit/total))
        fw.write('\nmetrics_total = '+str(metrics_total))
        metrics_average = metrics_total/total
        fw.write('\nmetrics_average = '+str(metrics_average))
    print('准确率为%.4f'%(hit/total))
    print('metrics_total = '+str(metrics_total))
    print('metrics_average = '+str(metrics_average))



def dev_metrics(dev_style_name, dev_device, textX, textY):
    os.environ["CUDA_VISIBLE_DEVICES"] = dev_device
    model_load_path = 'saved_models/' + dev_style_name + '_' + 'classifier/checkpoint.pkl'

    bertclassfication = BertClassfication()

    if torch.cuda.is_available():
        bertclassfication = bertclassfication.cuda()

    checkpoint = torch.load(model_load_path)
    bertclassfication.load_state_dict(checkpoint['model_state_dict'])#加载网络权重参数
    for parameter in bertclassfication.parameters():
        parameter.requires_grad=False
    bertclassfication.eval()


    # 测试
    metrics_total = 0
    total = len(textX) # 看看一共多少例子
    bertclassfication.eval()

    for i in range(total):
        outputsX = bertclassfication([textX[i]])
        outputsY = bertclassfication([textY[i]])
        possibilityX = torch.softmax(outputsX/24, 1)
        possibilityY = torch.softmax(outputsY/24, 1)

        metrics_total += round((round(outputsY[0][1].item(), 8)-round(outputsX[0][1].item(), 8))*round(possibilityX.cpu().numpy()[0][1], 8), 6)

    # metrics_average = metrics_total/total
    return metrics_total



if __name__ == "__main__":
    opt = parse_option()
    # input_file_path = opt.style_name + '_data/' + opt.style_name + opt.input_file_name
    input_file_path = 'results/' + opt.style_name + opt.input_file_name
    model_load_path = 'saved_models/' + opt.style_name + '_' + 'classifier/checkpoint.pkl'
    output_file_path = opt.style_name + '_' + 'classifier/' + opt.output_file_name + '_soft24.txt'
    _metrics(opt)
