import random
import time


def example_selection(text):
    content = text
    random_lines = ''.join(random.sample(content, 1))
    return random_lines


import openai
openai.api_key = '**********************'
import re, json
from tqdm import tqdm
import random

def return_random_prompt(text):
    system_prompt = "请仿照给定的样例，给出写作风格一致、但内容不同的全新句子。要求:\n"

    # generate random topics
    topic_list = ["科技", "娱乐", "体育", "金融", "时政", "教育", "医疗", "旅游", "美食", "汽车", "房产", "文化", "历史", "地理", "自然", "社会", "法律", "军事", "政治", "经济", "文学", "艺术", "宗教", "哲学", "数学", "物理", "化学", "生物", "天文学", "工程", "建筑", "设计", "音乐", "舞蹈", "电影", "健康", "时尚", "家居", "职场", "养生", "心理", "社交", "家庭", "宠物", "食品", "餐饮", "司法", "行政", "战争"]
    system_prompt += "1. 主题多样化，涵盖各个领域，例如：" + "、".join(random.sample(topic_list, 10)) + "等。\n"
    system_prompt += "2. 必须和输入保持同样的语言风格。\n"

    system_prompt += f"Input: {text}\n"
    system_prompt += "请给出满足条件的5条数据:\n"
    
    return system_prompt


def handle_data_augmentation(text):
    response = openai.ChatCompletion.create(
        model="text-davinci-003",   
        messages=[
            {"role": "user", "content": return_random_prompt(text)},
        ]
    )
    if "content" not in response["choices"][0]["message"]:
        return []
    msg = response["choices"][0]["message"]["content"]
    msg_list = msg.split('\n')
    msg_list = [msg for msg in msg_list if msg != '']
    msg_list = [msg[11:] if msg.startswith("Output 10:") else msg[10:] for msg in msg_list]
    return msg_list


if __name__ == "__main__":

    with open('hlm_data/hlm.txt', mode='r', encoding='utf-8') as f:     #stylized examples
        content = f.readlines()
    # 扩展800次，每次扩展5个
    for i in range(800):
        text = example_selection(content)
        print(text)
        augmentation_text_list = handle_data_augmentation(text)
        print(augmentation_text_list)

        with open('hlm_augmentation.txt', 'a', encoding='utf-8') as f:
            for sentence in augmentation_text_list:
                f.write(sentence + '\n')
        
        time.sleep(21)