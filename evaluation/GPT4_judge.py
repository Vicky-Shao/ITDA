import openai
import argparse

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for few-shot prompting.")
    
    parser.add_argument('--style', type = str, default = "")
    parser.add_argument('--file_name', type = str, default = "bart_base_dynamic40_n1")
    
    opt = parser.parse_args()
    return opt


def generate_response(prompt):
    openai.api_key = "*****************************"
    chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    if "error" in chat_completion:
        raise BaseException(chat_completion["error"]["message"])
    return chat_completion.choices[0].message.content

def evaluate_forward(demonstrations, original_neutral_text, transferred_text):
    instruction = "[Instruction]\nPlease act as an impartial judge and evaluate the model's ability to infuse target authorship style into neutral text. You will be provided with some demonstrations of successful migration to the target authorship style. You should make a comprehensive assessment and consider factors such as the style transfer strength, content preservation and fluency of the response. You must first provide your explanation, then rate the response on a scale of 1 to 10, such as [Score]6/10."
    prompt = instruction + f"\n\n[The Start of Demonstrations]\n{demonstrations}\n[The End of Demonstrations]\n\n[Original Neutral Text]\n{original_neutral_text}\n\n[Transferred Text]\n{transferred_text}"

    return(generate_response(prompt))


def evaluate_inverse(demonstrations, original_stylized_text, generated_neutral_text):
    instruction = "[Instruction]\nPlease act as an impartial judge and evaluate the model's ability to remove target author-stylized features from original stylized text to generate neutral text. You will be provided with some demonstrations of successful removal of the target authorship style. You should make a comprehensive assessment and consider factors such as the style transfer strength, content preservation and fluency. You must first provide your explanation, then rate the response on a scale of 1 to 10, such as [Score]6/10."
    prompt = instruction + f"\n\n[The Start of Demonstrations]\n{demonstrations}\n[The End of Demonstrations]\n\n[Original Stylized Text]\n{original_stylized_text}\n\n[Generated Neutral Text]\n{generated_neutral_text}"

    return(generate_response(prompt))


def evaluate_forward_main():
    opt = parse_option()
    demonstrations = ''
    file_name_list = []

    if opt.style=='hlm':
        demonstrations += "Neutral Text:你倦了，回我敷行\n"
        demonstrations += "Corresponding Author-stylized Text:你大抵是倦了，竟回我这般敷行\n"
        demonstrations += "Neutral Text:没有别的妹妹有趣，哥哥心里没有我\n"
        demonstrations += "Corresponding Author-stylized Text:没有别的妹妹有趣，终究哥哥心里没有我\n"
        demonstrations += "Neutral Text:疼爱的只有你母亲，今天见到了你，我怎么能不伤心！\n"
        demonstrations += "Corresponding Author-stylized Text:所疼者独有你母,今见了你,我怎不伤心!\n"
        demonstrations += "Neutral Text:经常服用的是什么药,为什么不赶紧做疗治?\n"
        demonstrations += "Corresponding Author-stylized Text:常服何药,如何不急为疗治?\n"
        demonstrations += "Neutral Text:这些人每个都像这样的恭肃严整，来的人是谁，放诞无礼到这样的地步？\n"
        demonstrations += "Corresponding Author-stylized Text:这些人个个恭肃严整如此,这来者系谁,这样放诞无礼?\n"
        demonstrations += "Neutral Text:也还算便宜。\n"
        demonstrations += "Corresponding Author-stylized Text:倒也便宜.\n"
        demonstrations += "Neutral Text:就是呢，不需要过来了。\n"
        demonstrations += "Corresponding Author-stylized Text:正是呢,不必过来了.\n"
        demonstrations += "Neutral Text:进来吧，外面看起来有点冷。\n"
        demonstrations += "Corresponding Author-stylized Text:进来罢,外头看起来有点冷.\n"
        file_name_list = ['bart_base_dynamic5','bart_base_dynamic10','bart_base_dynamic20','bart_base_dynamic30']
        print('style = hlm')

    if opt.style=='shakespeare':
        demonstrations += "Neutral Text:I have half a mind to hit you before you speak again.\n"
        demonstrations += "Corresponding Author-stylized Text:I have a mind to strike thee ere thou speak'st.\n"
        demonstrations += "Neutral Text:And he's friendly with Caesar.\n"
        demonstrations += "Corresponding Author-stylized Text:And friends with Caesar.\n"
        demonstrations += "Neutral Text:I'm going to make you a rich man.\n"
        demonstrations += "Corresponding Author-stylized Text:Make thee a fortune from me.\n"
        demonstrations += "Neutral Text:No , I didn't say that.\n"
        demonstrations += "Corresponding Author-stylized Text:I made no such report.\n"
        demonstrations += "Neutral Text:What did you say to me?\n"
        demonstrations += "Corresponding Author-stylized Text:What say you?\n"
        demonstrations += "Neutral Text:You say he's friendly with Caesar , healthy , and free.\n"
        demonstrations += "Corresponding Author-stylized Text:He's friends with Caesar , In state of health , thou say'st , and , thou say'st , free.\n"
        file_name_list = ['bart_base_dynamic5','bart_base_dynamic10','bart_base_dynamic20','bart_base_dynamic30']
        print('style = shakespeare')

    if opt.style=='trump':
        demonstrations += "Neutral Text:I find it unfair to my family.\n"
        demonstrations += "Corresponding Author-stylized Text:I have to say this very, very unfair to my family.\n"
        demonstrations += "Neutral Text:We can't let it happen.\n"
        demonstrations += "Corresponding Author-stylized Text:Right? Can't let it happen, folks.\n"
        demonstrations += "Neutral Text:They are just a form.\n"
        demonstrations += "Corresponding Author-stylized Text:Look it, they just form.\n"
        demonstrations += "Neutral Text:We love our nation that is still great today.\n"
        demonstrations += "Corresponding Author-stylized Text:We love our nation, our nation is great today.\n"
        demonstrations += "Neutral Text:We killed the number one terrorist.\n"
        demonstrations += "Corresponding Author-stylized Text:He was vehemently… We killed this number one, terrorist.\n"
        demonstrations += "Neutral Text:I have to prove that they are liars.\n"
        demonstrations += "Corresponding Author-stylized Text:I had to because I had to show they're liars.\n"
        file_name_list = ['bart_base_dynamic5','bart_base_dynamic10','bart_base_dynamic20','bart_base_dynamic30']
        print('style = trump')

    if opt.style=='lyrics':
        demonstrations += "Neutral Text:You know our relationship.\n"
        demonstrations += "Corresponding Author-stylized Text:Yeah, yeah, you know how me and you do.\n"
        demonstrations += "Neutral Text:I have your arms open.\n"
        demonstrations += "Corresponding Author-stylized Text:Your arms are open for me.\n"
        demonstrations += "Neutral Text:It's at least until tomorrow.\n"
        demonstrations += "Corresponding Author-stylized Text:So far at least until tomorrow.\n"
        demonstrations += "Neutral Text:Everything I've ever lost.\n"
        demonstrations += "Corresponding Author-stylized Text:Everything I ever had to lose.\n"
        demonstrations += "Neutral Text:I'm sure he'll kill him.\n"
        demonstrations += "Corresponding Author-stylized Text:And I promise its going to kill.\n"
        demonstrations += "Neutral Text:People are on the street.\n"
        demonstrations += "Corresponding Author-stylized Text:And people on the streets.\n"
        file_name_list = ['bart_base_dynamic30','bart_base_dynamic50','bart_base_dynamic80','bart_base_dynamic100','bart_base_dynamic120','bart_base_dynamic150']
        print('style = lyrics')
    
    for file_name in file_name_list:

        original_neutral_text = ''
        transferred_text = ''
        score_sum = 0
        sum = 0

        input_file_name = 'results_n/' + opt.style + '_' + file_name + '_n1.txt'
        output_file_name = 'score_baselines/' + opt.style + '_' + file_name + '_n1.txt'

        fw = open(output_file_name, 'w')
        fw.write('input_file_name = ' + input_file_name + '\n\n\n')
        print('input_file_name = ' + input_file_name)

        with open(input_file_name, 'r') as fr:
            for i, line in enumerate(fr):
                if i%2==0:
                    original_neutral_text = line.split('\t')[1][0:-1]
                else:
                    transferred_text = line.split('\t')[1][0:-1]
                    try:
                        result = evaluate_forward(demonstrations, original_neutral_text, transferred_text)
                    except:
                        print(transferred_text)
                        continue
                    fw.write('original_neutral_text = ' + original_neutral_text + '\n')
                    fw.write('transferred_text = ' + transferred_text + '\n')
                    fw.write(str(result)+'\n\n\n')

                    r = result.rfind('/')
                    s = result[r-1]
                    if s=='0':
                        s = result[r-2]+s
                    if s=='.':
                        s = result[r-3]
                        score_sum += 0.5
                    try:
                        score = int(s)
                    except:
                        print(result)
                        continue
                    score_sum += score
                    sum += 1

        fw.write('Score_Sum = '+str(score_sum)+'\n')
        fw.write('Sum = '+str(sum)+'\n')
        fw.write('Average_Score = '+str(score_sum/sum)+'\n')
        fw.close()

        summary_output = 'score_baselines/summary_' + opt.style + '.txt'
        with open(summary_output, 'a') as fw:
            fw.write('input_file_name = ' + input_file_name + '\n')
            fw.write('Score_Sum = '+str(score_sum)+'\n')
            fw.write('Sum = '+str(sum)+'\n')
            fw.write('Average_Score = '+str(score_sum/sum)+'\n')
            fw.write('---------------------------------------------------\n')
                


def evaluate_inverse_main():
    opt = parse_option()
    original_stylized_text = ''
    generated_neutral_text = ''
    score_sum = 0
    sum = 0
    
    input_file_name = 'inverse_data/trump_inverse_transfer.txt'
    output_file_name = 'inverse_data/score_trump_inverse_transfer.txt'

    fw = open(output_file_name, 'a')
    fw.write('input_file_name = ' + input_file_name + '\n\n\n')

    demonstrations = ''

    if opt.style=='hlm':
        demonstrations += "Authorship-stylized Text:你大抵是倦了，竟回我这般敷行\n"
        demonstrations += "Corresponding Neutral Text:你倦了，回我敷行\n"
        demonstrations += "Authorship-stylized Text:我心里自是明白没有别的妹妹有趣，终究哥哥心里没有我\n"
        demonstrations += "Corresponding Neutral Text:没有别的妹妹有趣，哥哥心里没有我\n"
        demonstrations += "Authorship-stylized Text:所疼者独有你母,今见了你,我怎不伤心!\n"
        demonstrations += "Corresponding Neutral Text:疼爱的只有你母亲，今天见到了你，我怎么能不伤心！\n"
        demonstrations += "Authorship-stylized Text:常服何药,如何不急为疗治?\n"
        demonstrations += "Corresponding Neutral Text:经常服用的是什么药,为什么不赶紧做疗治?\n"
        demonstrations += "Authorship-stylized Text:这些人个个恭肃严整如此,这来者系谁,这样放诞无礼?\n"
        demonstrations += "Corresponding Neutral Text:这些人每个都像这样的恭肃严整，来的人是谁，放诞无礼到这样的地步？\n"
        demonstrations += "Authorship-stylized Text:倒也便宜.\n"
        demonstrations += "Corresponding Neutral Text:也还算便宜。\n"
        demonstrations += "Authorship-stylized Text:正是呢,不必过来了.\n"
        demonstrations += "Corresponding Neutral Text:就是呢，不需要过来了。\n"
        demonstrations += "Authorship-stylized Text:进来罢,外头看起来有点冷.\n"
        demonstrations += "Corresponding Neutral Text:进来吧，外面看起来有点冷。\n"
        print('style = hlm')

    if opt.style=='shakespeare':
        demonstrations += "Authorship-stylized Text:I have a mind to strike thee ere thou speak'st.\n"
        demonstrations += "Corresponding Neutral Text:I have half a mind to hit you before you speak again.\n"
        demonstrations += "Authorship-stylized Text:And friends with Caesar.\n"
        demonstrations += "Corresponding Neutral Text:And he's friendly with Caesar.\n"
        demonstrations += "Authorship-stylized Text:Make thee a fortune from me.\n"
        demonstrations += "Corresponding Neutral Text:I'm going to make you a rich man.\n"
        demonstrations += "Authorship-stylized Text:I made no such report.\n"
        demonstrations += "Corresponding Neutral Text:No , I didn't say that.\n"
        demonstrations += "Authorship-stylized Text:What say you?\n"
        demonstrations += "Corresponding Neutral Text:What did you say to me?\n"
        demonstrations += "Authorship-stylized Text:He's friends with Caesar , In state of health , thou say'st , and , thou say'st , free.\n"
        demonstrations += "Corresponding Neutral Text:You say he's friendly with Caesar , healthy , and free.\n"
        print('style = shakespeare')

    if opt.style=='trump':
        demonstrations += "Authorship-stylized Text:I find it unfair to my family.\n"
        demonstrations += "Corresponding Neutral Text:I have to say this very, very unfair to my family.\n"
        demonstrations += "Authorship-stylized Text:We can't let it happen.\n"
        demonstrations += "Corresponding Neutral Text:Right? Can't let it happen, folks.\n"
        demonstrations += "Authorship-stylized Text:They are just a form.\n"
        demonstrations += "Corresponding Neutral Text:Look it, they just form.\n"
        demonstrations += "Authorship-stylized Text:We love our nation that is still great today.\n"
        demonstrations += "Corresponding Neutral Text:We love our nation, our nation is great today.\n"
        demonstrations += "Authorship-stylized Text:We killed the number one terrorist.\n"
        demonstrations += "Corresponding Neutral Text:He was vehemently… We killed this number one, terrorist.\n"
        demonstrations += "Authorship-stylized Text:I have to prove that they are liars.\n"
        demonstrations += "Corresponding Neutral Text:I had to because I had to show they're liars.\n"
        print('style = trump')

    if opt.style=='lyrics':
        demonstrations += "Authorship-stylized Text:I have to say this very, very unfair to my family.\n"
        demonstrations += "Corresponding Neutral Text:I find it unfair to my family.\n"
        demonstrations += "Authorship-stylized Text:Right? Can't let it happen, folks.\n"
        demonstrations += "Corresponding Neutral Text:We can't let it happen.\n"
        demonstrations += "Authorship-stylized Text:Look it, they just form.\n"
        demonstrations += "Corresponding Neutral Text:They are just a form.\n"
        demonstrations += "Authorship-stylized Text:We love our nation, our nation is great today.\n"
        demonstrations += "Corresponding Neutral Text:We love our nation that is still great today.\n"
        demonstrations += "Authorship-stylized Text:He was vehemently… We killed this number one, terrorist.\n"
        demonstrations += "Corresponding Neutral Text:We killed the number one terrorist.\n"
        demonstrations += "Authorship-stylized Text:I had to because I had to show they're liars.\n"
        demonstrations += "Corresponding Neutral Text:I have to prove that they are liars.\n"
        print('style = lyrics')

    with open(input_file_name, 'r') as fr:
        for i, line in enumerate(fr):
            if i%2==0:
                original_stylized_text = line[2:-1]
            else:
                generated_neutral_text = line[2:-1]
                try:
                    result = evaluate_inverse(demonstrations, original_stylized_text, generated_neutral_text)
                except:
                    print(generated_neutral_text)
                    continue
                fw.write('original_stylized_text = ' + original_stylized_text + '\n')
                fw.write('generated_neutral_text = ' + generated_neutral_text + '\n')
                fw.write(str(result)+'\n\n\n')

                r = result.rfind('/')
                s = result[r-1]
                if s=='0':
                    s = result[r-2]+s
                if s=='.':
                    s = result[r-3]
                    score_sum += 0.5
                try:
                    score = int(s)
                except:
                    print(result)
                    continue
                score_sum += score
                sum += 1

    fw.write('Score_Sum = '+str(score_sum)+'\n')
    fw.write('Sum = '+str(sum)+'\n')
    fw.write('Average_Score = '+str(score_sum/sum)+'\n')
    fw.write('---------------------------------------------------\n\n\n')
    
    fw.close()

if __name__ == "__main__":
    evaluate_forward_main()


