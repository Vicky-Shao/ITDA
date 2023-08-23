import os
import torch 
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange


# 第一部分
class XYDataset(Dataset):
    def __init__(self, filepath):
        self.sentence = []
        self.label = []
        with open(filepath, 'r') as fr:
            for i, line in enumerate(fr):
                if len(line.split('\t'))>2:
                    print(line.split('\t'))
                x , y = line.split('\t')
                self.sentence.append(y[:-1])
                self.label.append(x)
                self.len = i+1
 
    def __getitem__(self, index):
        return self.sentence[index], self.label[index]
 
    def __len__(self):
        return self.len

# 第二部分
class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication,self).__init__()
        self.model_name = 'uer/chinese_roberta_L-12_H-768'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768,2)   
    def forward(self,x):
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                max_length=512, pad_to_max_length=True)     
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        hiden_outputs = self.model(input_ids,attention_mask=attention_mask)
        output = self.fc(outputs)
        return output


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# 第三部分，整理数据集
train_dataset = XYDataset('hlm-classifier/classifier-train.txt')
dev_dataset = XYDataset('hlm-classifier/classifier-dev.txt')
batch_size = 32
patience = 8
train_dataloder = DataLoader(
    dataset=train_dataset, 
    batch_size = batch_size, 
    shuffle = True,
)
num_checkpoint_steps = int(0.71425 * len(train_dataset)/batch_size)
dev_dataloder = DataLoader(
    dev_dataset, 
    batch_size = batch_size, 
    shuffle = False,
)

# 第四部分，训练
bertclassfication = BertClassfication()
lossfuction = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(bertclassfication.parameters(),lr=2e-5)

checkpoint={'model':BertClassfication(),
            'model_state_dict':bertclassfication.state_dict(),
            'optimize_state_dict':optimizer.state_dict()}
torch.save(checkpoint,'saved_models/hlm-classifier6/checkpoint.pkl')


if torch.cuda.is_available():
    bertclassfication = bertclassfication.cuda()


train_step_total = 0
train_loss_total = 0
train_loss_average = 0
dev_loss_best = 0
train_steps = 0
epoch = 96
list_step = []
list_train_loss = []
list_dev_loss = []
early_stop_step = 0

for epoch in trange(epoch, desc="Epoch"):
    print('This is epoch '+str(epoch+1)+'".')
    train_loss_total = 0  # 每一轮训练的损失都重新开始计数
    train_steps = 0

    for batch in tqdm(train_dataloder, desc="Training"):
        bertclassfication.train()
        train_step_total += 1
        train_steps += 1

        inputs, labels = batch
        targets = []
        for _, line in enumerate(labels):
            targets.append(int(line))



        targets = torch.tensor(targets)
        if torch.cuda.is_available():
            targets = targets.cuda()
        optimizer.zero_grad()
        outputs= bertclassfication(inputs)
        loss = lossfuction(outputs,targets)
        loss.backward()
        optimizer.step()

        los = loss.item() 
        train_loss_total += los

        train_loss_average = train_loss_total/train_steps

        if train_step_total % num_checkpoint_steps == 0 and epoch >= 3:
            print('At '+str(train_step_total)+' training step, start an evaluation.')
            bertclassfication.eval()
            dev_loss_total = 0
            dev_steps = 0
            for batch in tqdm(dev_dataloder, desc="Evaluating"):
                dev_steps = dev_steps+1

                inputs, labels = batch
                targets = []
                for _, line in enumerate(labels):
                    targets.append(int(line))
                targets = torch.tensor(targets)
                if torch.cuda.is_available():
                    targets = targets.cuda()
                outputs= bertclassfication(inputs)
                loss = lossfuction(outputs,targets)#

            dev_loss_total = dev_loss_total + loss.item()
            dev_loss_average = dev_loss_total / dev_steps

            print("Train_step_total:\t" + str(train_step_total))
            print("Current train average loss:\t" + str(train_loss_average))
            print("Current dev average loss:\t" + str(dev_loss_average))
            list_step.append(train_step_total)
            list_train_loss.append(train_loss_average)
            list_dev_loss.append(dev_loss_average)

            if dev_loss_average < dev_loss_best or dev_loss_best==0:
                early_stop_step = 0
                dev_loss_best = dev_loss_average

                checkpoint={'model':BertClassfication(),
                    'model_state_dict':bertclassfication.state_dict(),
                    'optimize_state_dict':optimizer.state_dict()}
                torch.save(checkpoint,'saved_models/hlm-classifier6/checkpoint.pkl')

                with open('saved_models/hlm-classifier6/loss.txt', 'w') as fw:
                    fw.write("Train_step_total:\t" + str(train_step_total))
                    fw.write("\n")
                    fw.write("Train_loss:\t" + str(train_loss_average))
                    fw.write("\n")
                    fw.write("Dev_loss:\t" + str(dev_loss_best))
                    fw.write("\n\n\n")
                    fw.write("List_train_step_total\t" + str(list_step))
                    fw.write("\n")
                    fw.write("List_train_average_loss:\t" + str(list_train_loss))
                    fw.write("\n")
                    fw.write("List_dev_average_loss:\t" + str(list_dev_loss))
            else:
                early_stop_step += 1
            
        if early_stop_step >= patience:
            break

    if early_stop_step >= patience:
        print("Training process triggers early stopping.")
        break




