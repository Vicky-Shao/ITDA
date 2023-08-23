import os
import torch
# import sqlite3
import argparse
import random
import torch.optim as optim
import transformers

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_utils import set_seed

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    parser.add_argument('--style_name', type = str)
    parser.add_argument('--size', type = str, default = '')
    parser.add_argument('--prompt_type', type = str, default = "dynamic", help = "static")
    parser.add_argument('--input_file_name', type = str, default = '')
    parser.add_argument('--device', type = str, default = "0",
                        help = 'the id of used GPU device.')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 2,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--epochs', type = int, default = 64,
                        help = 'training epochs.')
    parser.add_argument('--patience', type = int, default = 24,
                        help = 'patience step in early stop.')
    parser.add_argument('--model_name_or_path', type = str, default = "facebook/bart-base")
    parser.add_argument('--mode', type = str, default = "train",
                        help='trian, eval or test.')

    opt = parser.parse_args()

    return opt

class DatasetXY(Dataset):
    def __init__(self, corpora):
        self.x_data = []
        self.y_data = []
        for line in corpora:
            self.x_data.append(line.split('\n')[1][2:])
            self.y_data.append(line.split('\n')[0][2:])
        self.len = len(corpora)
        
 
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
 
    def __len__(self):
        return self.len


def shuffleXY():
    sentences = []
    textX = ''
    with open(input_file_path, 'r') as fr:
        for i, line in enumerate(fr):
            if i%2==0:
                textX = line
            else:
                sentences.append(textX+line)
    random.shuffle(sentences)
    dev_count = round(len(sentences)/5)
    train_dataset = DatasetXY(sentences[dev_count:])
    dev_dataset = DatasetXY(sentences[0:dev_count])
    return train_dataset, dev_dataset



def prepare_inputs_and_labels(tokenizer, batch):
    x_data, y_data = batch
    x_data = tokenizer(x_data, return_tensors='pt', max_length=256, truncation=True, padding = "max_length")
    y_data = tokenizer(y_data, return_tensors='pt', max_length=256, truncation=True, padding = "max_length")
    encoder_input_ids = x_data['input_ids']
    encoder_attention_mask = x_data['attention_mask']
    decoder_labels = y_data['input_ids']
    decoder_attention_mask = y_data['attention_mask']

    # print(tokenizer.batch_decode(encoder_input_ids))
    # print(encoder_input_ids)
    # print(encoder_attention_mask)
    # print(tokenizer.batch_decode(decoder_labels))
    # print(decoder_labels)
    # print(decoder_attention_mask)

    decoder_labels[decoder_labels == tokenizer.pad_token_id] = -100
    # print(decoder_labels)
    # print("-------------------------")

    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_attention_mask = encoder_attention_mask.cuda()
        decoder_labels = decoder_labels.cuda()
        decoder_attention_mask = decoder_attention_mask.cuda()

    return {
        "input_ids": encoder_input_ids,
        "attention_mask": encoder_attention_mask,
        "labels": decoder_labels,
        "decoder_attention_mask": decoder_attention_mask,
        "return_dict": True
    }

def test_case(text, tokenizer, model):
    test_input = tokenizer(text, return_tensors='pt').input_ids
    if torch.cuda.is_available():
        test_input = test_input.cuda()
    
    with torch.no_grad():
        test_output = model.generate(test_input, max_length=512)[0]
    
    test_output = tokenizer.decode(test_output, skip_special_tokens=True)
    print('test_output = ' + test_output)

def _train(opt):
    set_seed(42)
    print(opt)

    writer = SummaryWriter(tensorboard_save_path)
    patience = opt.patience if opt.patience > 0 else float('inf')
    # 设置可见的GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    train_dataset, dev_dataset = shuffleXY()

    train_dataloder = DataLoader(
        dataset=train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
    )

    print("initializing seq2seq model.")
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(opt.model_name_or_path, use_safetensors = False)

    if torch.cuda.is_available():
        model = model.cuda()

    print(opt.epochs * len(train_dataset) / opt.batch_size)

    # warm up steps (5% training step)
    num_warmup_steps = int(0.05 * opt.epochs * len(train_dataset) / opt.batch_size)
    # total training steps
    num_training_steps = int(opt.epochs * len(train_dataset) / opt.batch_size)
    # evaluate model for each 0.75 training set
    num_checkpoint_steps = int(0.75 * len(train_dataset) / opt.batch_size)

    print("Let's use AdamW as the optimizer!")
    # 用AdamW作为优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr = opt.learning_rate
    )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    early_stop_step, train_step_total = 0, 0
    dev_loss_best = float('inf')
    list_step = []
    list_dev_loss = []

    for epoch in trange(opt.epochs, desc = "Epoch"):
        print("This is epoch "+str(epoch+1))
        for batch in train_dataloder:
            model.train()
            train_step_total += 1
            inputs_and_labels = prepare_inputs_and_labels(tokenizer, batch)
            
            output = model(**inputs_and_labels)
            
            loss = output['loss']
            writer.add_scalar('train loss', loss.item(), train_step_total)
            writer.add_scalar('train lr', optimizer.state_dict()['param_groups'][0]['lr'], train_step_total)

            # 反向传播计算并累加梯度
            loss.backward()
            if scheduler is not None:
                scheduler.step()
            
            # 每累加gradient_descent_step次梯度就更新一次参数
            if train_step_total % opt.gradient_descent_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if train_step_total % num_checkpoint_steps == 0:
                print("At "+str(train_step_total)+" training step, start an evaluation.")                
                model.eval()
                
                test_case('You have to prove it, my dear brother, you have to prove it!', tokenizer, model)
                test_case('You can\'t judge me based on what I do tonight .', tokenizer, model)
                
                dev_loss_total = 0
                dev_steps = 0
                for batch in tqdm(dev_dataloder, desc="Evaluating"):
                    dev_steps = dev_steps + 1
                    inputs_and_labels = prepare_inputs_and_labels(tokenizer, batch)

                    # torch.no_grad()停止autograd模块的工作从而停止了计算梯度，因此可以在inference的时候节省一些显存
                    with torch.no_grad():
                        output = model(**inputs_and_labels)
                    dev_loss_total = dev_loss_total + output['loss'].item()

                dev_loss_average = dev_loss_total / dev_steps

                writer.add_scalar('dev loss', dev_loss_average, train_step_total // num_checkpoint_steps)
                
                print("Train_step_total:\t" + str(train_step_total))
                print("Current dev average loss:\t" + str(dev_loss_average))
                list_step.append(train_step_total)
                list_dev_loss.append(dev_loss_average)
                
                if dev_loss_average < dev_loss_best:
                    early_stop_step = 0
                    dev_loss_best = dev_loss_average
                    os.makedirs(model_save_path, exist_ok = True)
                    model.save_pretrained(save_directory = model_save_path)
                    tokenizer.save_pretrained(save_directory = model_save_path)

                    with open(model_save_path+'/loss.txt', 'w') as fw:
                        fw.write("Train_step_total:\t" + str(train_step_total))
                        fw.write("\n")
                        fw.write("Dev_loss:\t" + str(dev_loss_best))
                        fw.write("\n\n\n")
                        fw.write("List_train_step_total\t" + str(list_step))
                        fw.write("\n")
                        fw.write("List_dev_average_loss:\t" + str(list_dev_loss))
                else:
                    early_stop_step += 1
            
            if early_stop_step >= patience:
                break
        if early_stop_step >= patience:
            print("Training process triggers early stopping.")
            break

if __name__ == "__main__":
    opt = parse_option()
    input_file_path = opt.style_name + '_data/' + opt.style_name + opt.input_file_name
    model_save_path = 'saved_models/' + opt.style_name + '_' + 'bart_base_' + opt.prompt_type + opt.size
    tensorboard_save_path = 'tensorboard_log/' + opt.style_name + '_' + 'bart_base_' + opt.prompt_type + opt.size
    # if opt.mode in ["train"]:
    _train(opt)
    # elif opt.mode in ["eval", "test"]:
    #     _test(opt)