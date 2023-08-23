import os
import torch
import argparse
import torch.optim as optim
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for bart transfer.")
    parser.add_argument('--style_name', type = str)
    parser.add_argument('--input_file_name', type = str, default = '')
    parser.add_argument('--output_file_name', type = str, default = '')
    parser.add_argument('--size', type = str, default = '')
    parser.add_argument('--prompt_type', type = str, default = "dynamic", help = "static")
    parser.add_argument('--metrics', type = str, default = '', help = "_metrics")
    opt = parser.parse_args()

    return opt



def _test():
    input_text = []
    with open(input_file_path, 'r') as fr:
        for i, line in enumerate(fr):
            input_text.append(line)


    print("initializing model.")
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path)

    print("loading model (finished).")


    text2text_generator = Text2TextGenerationPipeline(model, tokenizer)       
    output = text2text_generator(input_text, max_length=512, do_sample=False)
    with open(output_file_path, 'w') as fw:
        for i in range(len(output)):
            fw.write('0\t'+input_text[i])
            fw.write('1\t'+output[i]['generated_text'])
            if output[i]['generated_text'][-1]!='\n':
                fw.write('\n')

              

if __name__ == "__main__":
    opt = parse_option()
    input_file_path = opt.style_name + '_data/' + opt.style_name + opt.input_file_name
    model_load_path = 'saved_models/' + opt.style_name + '_' + 'bart_base_' + opt.prompt_type + opt.size + opt.metrics
    output_file_path = opt.style_name + '_data/' + opt.style_name + '_bart_base_' + opt.prompt_type + opt.size + opt.output_file_name + opt.metrics + '.txt'
    _test()