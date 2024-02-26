import openai
from sentence_transformers import SentenceTransformer, util
import argparse


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for dynamic inverse prompting.")
    
    parser.add_argument('--style_name', type = str, default = "hlm")
    parser.add_argument('--prompt_file_name', type = str, default = "_promptLibrary.txt")
    parser.add_argument('--input_file_name', type = str, default = "_0.txt")
    parser.add_argument('--output_file_name', type = str, default = "_dynamicXY.txt")
    parser.add_argument('--api_file_name', type = str, default = "api.txt")
    
    opt = parser.parse_args()
    return opt


def dynamicInversePrompts(sentence):
    senEmbedding = model.encode(sentence, convert_to_tensor=True)
    cosine_scores = util.cos_sim(proEmbeddings, senEmbedding)
    pairs = []
    for j in range(len(promptsX)):
        pairs.append({'index': j, 'score':cosine_scores[j].item()})
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    example = ''
    for pair in pairs[0:9]:
        example = promptsXY[pair['index']] + example
    example = example + "Input:" + sentence + "Please rewrite the sentence as neutral text according to the examples.  Output:"
    return example
    

def generate_response(prompt):
    global current
    current = (current+1) % len(api_list)
    openai.api_key = api_list[current]
    
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )    

    if "error" in response:
        raise BaseException(response["error"]["message"])
    return response["choices"][0]["text"].strip()





def transfer():
    input_file_path = opt.style_name + '_data/' + opt.style_name + opt.input_file_name
    output_file_path = opt.style_name + '_data/' + opt.style_name + opt.output_file_name
    output_text = ''
    with open(output_file_path, 'w') as fw:
        with open(input_file_path, 'r') as fr:
            for i , line in enumerate(fr):
                prompts= dynamicInversePrompts(line)
                fw.write(prompts+'\n\n')
                # sign = 0
                # while sign != -1:
                #     try:
                #         prompts = dynamicInversePrompts(line)
                #         output_text = generate_response(prompts)
                #     except (ConnectionError, Exception, IOError) as e:
                #         sign += 1
                #         print('Error '+str(sign)+'! '+str(e))
                #     else:
                #         sign = -1
                #         fw.write('X:'+line)
                #         fw.write('Y:'+output_text+'\n')




if __name__ == "__main__":
    opt = parse_option()
    current = -1
    api_list = []

    with open(opt.api_file_name, 'r') as fr:
        for line in fr:
            api_list.append(line[:-1])

    promptsX = []
    promptsXY = []
    promptXY = ''
    prompt_file_path = opt.style_name + '_data/' + opt.style_name + opt.prompt_file_name
    with open(prompt_file_path, 'r') as fr:
        for i, line in enumerate(fr):
            if i%3==1:
                promptsX.append(line[6:])
                promptXY = line
            if i%3==2:
                promptsXY.append(promptXY+line)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    proEmbeddings = model.encode(promptsX, convert_to_tensor=True)

    transfer()
