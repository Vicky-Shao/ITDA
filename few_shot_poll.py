import openai
import argparse
import sys
# Completions (see )


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for few-shot prompting.")
    
    parser.add_argument('--style_name', type = str)
    parser.add_argument('--pilot', type = str, default = "", help = "pilot/")
    parser.add_argument('--direction', type = str, default = "forward", help = "inverse")
    parser.add_argument('--input_file_name', type = str, default = "_0.txt")
    parser.add_argument('--output_file_name', type = str)#, default = "_staticXY.txt")
    parser.add_argument('--api_file_name', type = str, default = "api_5.txt")
    
    opt = parser.parse_args()
    return opt




def trump_inverse_prompt(text):
    prompts = ''
    prompts += "Input:I have to say this very, very unfair to my family.\n"
    prompts += "Output:I find it unfair to my family.\n"
    prompts += "Input:Right? Can't let it happen, folks.\n"
    prompts += "Output:We can't let it happen.\n"
    prompts += "Input:Look it, they just form.\n"
    prompts += "Output:They are just a form.\n"
    prompts += "Input:We love our nation, our nation is great today.\n"
    prompts += "Output:We love our nation that is still great today.\n"
    prompts += "Input:He was vehemently… We killed this number one, terrorist.\n"
    prompts += "Output:We killed the number one terrorist.\n"
    prompts += "Input:I had to because I had to show they're liars.\n"
    prompts += "Output:I have to prove that they are liars.\n"

    prompts = prompts + "Input:" + text
    prompts += "Please rewrite the sentence according to the examples. Output:"

    return prompts


def trump_forward_prompt(text):
    prompts = ''
    prompts += "Input:I find it unfair to my family.\n"
    prompts += "Ontput:I have to say this very, very unfair to my family.\n"
    prompts += "Iuput:We can't let it happen.\n"
    prompts += "Ontput:Right? Can't let it happen, folks.\n"
    prompts += "Input:They are just a form.\n"
    prompts += "Output:Look it, they just form.\n"
    prompts += "Input:We love our nation that is still great today.\n"
    prompts += "Output:We love our nation, our nation is great today.\n"
    prompts += "Input:We killed the number one terrorist.\n"
    prompts += "Output:He was vehemently… We killed this number one, terrorist.\n"
    prompts += "Input:I have to prove that they are liars.\n"
    prompts += "Output:I had to because I had to show they're liars.\n"
    prompts = prompts + "Input:" + text
    prompts += "Please rewrite the sentence according to the examples. Output:"

    return prompts




def bible_forward_prompt(text):
    prompts = ''
    prompts += "Input:How can I despise those whom the Lord has not despised.\n"
    prompts += "Ontput:or how shall I defy, whom the LORD hath not defied?\n"
    prompts += "Input:How long will you keep your useless thoughts?\n"
    prompts += "Ontput:How long shall thy vain thoughts lodge within thee?\n"
    prompts += "Input:Laban searched the tents, but didn't find them.\n"
    prompts += "Ontput:And Laban searched all the tent, but found them not.\n"
    prompts += "Input:Who are you more beautiful than？\n"
    prompts += "Ontput:Whom dost thou pass in beauty?\n"
    prompts += "Input:They reasoned which of them was the greatest.\n"
    prompts += "Ontput:Then there arose a reasoning among them, which of them should be greatest.\n"
    prompts += "Input:n addition, the Lord spoke to me, saying:\n"
    prompts += "Ontput:Moreover the word of the LORD came unto me, saying,\n"

    prompts = prompts + "Input:" + text
    prompts += "Please rewrite the sentence according to the examples. Output:"

    return prompts


def lyrics_forward_prompt(text):
    prompts = ''
    prompts += "Input:You know our relationship.\n"
    prompts += "Ontput:Yeah, yeah, you know how me and you do.\n"
    prompts += "Input:I have your arms open.\n"
    prompts += "Ontput:Your arms are open for me.\n"
    prompts += "Input:It's at least until tomorrow.\n"
    prompts += "Ontput:So far at least until tomorrow.\n"
    prompts += "Input:Everything I've ever lost.\n"
    prompts += "Ontput:Everything I ever had to lose.\n"
    prompts += "Input:I'm sure he'll kill him.\n"
    prompts += "Ontput:And I promise its going to kill.\n"
    prompts += "Input:People are on the street.\n"
    prompts += "Ontput:And people on the streets.\n"

    prompts = prompts + "Input:" + text
    prompts += "Please rewrite the sentence according to the examples. Output:"

    return prompts

def lyrics_inverse_prompt(text):
    prompts = ''
    prompts += "Input:Yeah, yeah, you know how me and you do.\n"
    prompts += "Ontput:You know our relationship.\n"
    prompts += "Input:Your arms are open for me.\n"
    prompts += "Ontput:I have your arms open.\n"
    prompts += "Input:So far at least until tomorrow.\n"
    prompts += "Ontput:It's at least until tomorrow.\n"
    prompts += "Input:Everything I ever had to lose.\n"
    prompts += "Ontput:Everything I've ever lost.\n"
    prompts += "Input:And I promise its going to kill.\n"
    prompts += "Ontput:I'm sure he'll kill him.\n"
    prompts += "Input:And people on the streets.\n"
    prompts += "Ontput:People are on the street.\n"

    prompts = prompts + "Input:" + text
    # prompts += "Please rewrite the sentence according to the examples. Output:"
    prompts += "Please rewrite the sentence as neutral text according to the examples.  Output:"

    return prompts

def shakespeare_forward_prompt(text):

    prompts = ''
    prompts += "Input:I have half a mind to hit you before you speak again.\n"
    prompts += "Output:I have a mind to strike thee ere thou speak'st.\n"
    prompts += "Input:And he's friendly with Caesar.\n"
    prompts += "Output:And friends with Caesar.\n"
    prompts += "Input:I'm going to make you a rich man.\n"
    prompts += "Output:Make thee a fortune from me.\n"
    prompts += "Input:No , I didn't say that.\n"
    prompts += "Output:I made no such report.\n"
    prompts += "Iutput:What did you say to me?\n"
    prompts += "Ontput:What say you?\n"
    prompts += "Iutput:You say he's friendly with Caesar , healthy , and free.\n"
    prompts += "Ontput:He's friends with Caesar , In state of health , thou say'st , and , thou say'st , free.\n"

    prompts = prompts + "Input:" + text
    prompts += "Please rewrite the sentence according to the examples. Output:"
    
    return prompts



def shakespeare_inverse_prompt(text):

    prompts = ''
    
    prompts += "Input:I have a mind to strike thee ere thou speak'st.\n"
    prompts += "Output:I have half a mind to hit you before you speak again.\n"
    prompts += "Input:And friends with Caesar.\n"
    prompts += "Output:And he's friendly with Caesar.\n"
    prompts += "Input:Make thee a fortune from me.\n"
    prompts += "Output:I'm going to make you a rich man.\n"
    prompts += "Input:I made no such report.\n"
    prompts += "Output:No , I didn't say that.\n"
    prompts += "Input:What say you?\n"
    prompts += "Output:What did you say to me?\n"
    prompts += "Input:He's friends with Caesar , In state of health , thou say'st , and , thou say'st , free.\n"
    prompts += "Output:You say he's friendly with Caesar , healthy , and free.\n"
    
    prompts = prompts + "Input:" + text
    # prompts += "Please rewrite the sentence according to the examples. Output:"
    prompts += "Please rewrite the sentence as neutral text according to the examples.  Output:"
    
    return prompts


    

def parallel_forward_prompt(text):

    prompts = ''
    prompts += "Input:想想秦国、吴国的分别，再回到燕国、宋国，隔了千里之遥。\n"
    prompts += "Output:况秦吴兮绝国，复燕宋兮千里。\n"
    prompts += "Input:或者春天萌动着青苔，又或者秋天狂风突然袭来。\n"
    prompts += "Output:或春苔兮始生，乍秋风兮暂起。\n"
    prompts += "Input:看起来有些摇摇欲坠，但仍然高耸峭立。\n"
    prompts += "Output:藐藐标危，亭亭峻趾。\n"
    prompts += "Input:在这陌生的地方，邂逅的人都是来自他乡的旅客。\n"
    prompts += "Output:萍水相逢，尽是他乡之客。\n"
    prompts += "Iutput:杏坛再次设立，建立了人民大学，以宏大的目标为出发点。\n"
    prompts += "Ontput:杏坛再设，建人民大学，出之以宏旨。\n"
    prompts += "Iutput:肩膀修长，腰部纤细如素线。\n"
    prompts += "Ontput:肩若削成，腰如约素。\n"

    prompts = prompts + "Input:" + text
    prompts += "请按照示例对句子进行重写:Output:"
    
    return prompts

def hlm_forward_prompt(text):

    prompts = ''
    prompts += "Input:你倦了，回我敷行\n"
    prompts += "Output:你大抵是倦了，竟回我这般敷行\n"
    prompts += "Input:没有别的妹妹有趣，哥哥心里没有我\n"
    # prompts += "Output:我心里自是明白没有别的妹妹有趣，终究哥哥心里没有我\n"
    prompts += "Output:没有别的妹妹有趣，终究哥哥心里没有我\n"
    prompts += "Input:疼爱的只有你母亲，今天见到了你，我怎么能不伤心！\n"
    prompts += "Output:所疼者独有你母,今见了你,我怎不伤心!\n"
    prompts += "Input:经常服用的是什么药,为什么不赶紧做疗治?\n"
    prompts += "Output:常服何药,如何不急为疗治?\n"
    prompts += "Iutput:这些人每个都像这样的恭肃严整，来的人是谁，放诞无礼到这样的地步？\n"
    prompts += "Ontput:这些人个个恭肃严整如此,这来者系谁,这样放诞无礼?\n"
    prompts += "Iutput:也还算便宜。\n"
    prompts += "Ontput:倒也便宜.\n"
    prompts += "Iutput:就是呢，不需要过来了。\n"
    prompts += "Ontput:正是呢,不必过来了.\n"
    prompts += "Intput:进来吧，外面看起来有点冷。\n"
    prompts += "Output:进来罢,外头看起来有点冷.\n"
    

    prompts = prompts + "Input:" + text
    prompts += "请按照示例对句子进行重写:Output:"
    
    return prompts



def hlm_inverse_prompt(text):

    prompts = ''
    prompts += "Iutput:你大抵是倦了，竟回我这般敷行\n"
    prompts += "Onput:你倦了，回我敷行\n"
    prompts += "Iutput:我心里自是明白没有别的妹妹有趣，终究哥哥心里没有我\n"
    # prompts += "Iutput:没有别的妹妹有趣，终究哥哥心里没有我\n"
    prompts += "Onput:没有别的妹妹有趣，哥哥心里没有我\n"
    prompts += "Iutput:所疼者独有你母,今见了你,我怎不伤心!\n"
    prompts += "Onput:疼爱的只有你母亲，今天见到了你，我怎么能不伤心！\n"
    prompts += "Iutput:常服何药,如何不急为疗治?\n"
    prompts += "Onput:经常服用的是什么药,为什么不赶紧做疗治?\n"
    prompts += "Intput:这些人个个恭肃严整如此,这来者系谁,这样放诞无礼?\n"
    prompts += "Output:这些人每个都像这样的恭肃严整，来的人是谁，放诞无礼到这样的地步？\n"
    prompts += "Intput:倒也便宜.\n"
    prompts += "Output:也还算便宜。\n"
    prompts += "Intput:正是呢,不必过来了.\n"
    prompts += "Output:就是呢，不需要过来了。\n"
    prompts += "Iutput:进来罢,外头看起来有点冷.\n"
    prompts += "Ontput:进来吧，外面看起来有点冷。\n"
    
    prompts = prompts + "Input:" + text
    # prompts += "请按照示例对句子进行重写:Output:"
    prompts += "请按照示例将句子重写为中立文本:Output:"
    
    return prompts



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







def few_shot():
    # input_file_path = style_name+'-data/pilot/'+style_name+'-few-shot.txt
    # output_file_path = style_name+'-data/pilot/'+style_name+'-neutral.txt
    # input_file_path = style_name+'-data/'+style_name+'-test2.txt'
    # output_file_path = style_name+'-data/'+style_name+'-few-shot2.txt'
    # input_file_path = style_name+'-data/pilot/'+style_name+'.txt'
    # output_file_path = style_name+'-data/pilot/'+style_name+'-inverse-few-shot.txt'
    input_file_path = opt.style_name + '_data/' + opt.pilot + opt.style_name + opt.input_file_name
    output_file_path = opt.style_name + '_data/' + opt.pilot + opt.style_name + opt.output_file_name
    mod = sys.modules["__main__"]
    method = opt.style_name + '_' + opt.direction + '_' + 'prompt'
    with open(output_file_path, 'w') as fw:
        with open(input_file_path, 'r') as fr:
            for i , line in enumerate(fr):
                # prompts= getattr(mod, method)(line)
                # print(prompts)
                # print('__________________________________________________')
                # print(generate_response(prompts)+'\n')
                # print('__________________________________________________')
                # break
                sign = 0
                while sign != -1:
                    try:
                        prompts= getattr(mod, method)(line)
                        output_text = generate_response(prompts)
                    except (ConnectionError, Exception, IOError) as e:
                        sign += 1
                        print('Error '+str(sign)+'! '+str(e))
                    else:
                        sign = -1
                        fw.write('X:'+line)
                        fw.write('Y:'+output_text+'\n')



if __name__ == "__main__":
    opt = parse_option()
    current = -1
    api_list = []

    with open(opt.api_file_name, 'r') as fr:
        for line in fr:
            api_list.append(line[:-1])

    few_shot()