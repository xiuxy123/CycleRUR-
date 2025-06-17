import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# 读取参考文本和生成文本
from rouge_score import rouge_scorer

with open('data/webnlg/test_ref.txt', 'r', encoding='utf-8') as ref_file:
    ref_lines = ref_file.readlines()

with open('output/data2text.generations', 'r', encoding='utf-8') as gen_file:
    gen_lines = gen_file.readlines()

def read_line(path):
    with open(path, "r", encoding='utf-8') as read_file:
        data = [line.strip() for line in read_file]
    return data


# 定义 BLEU、METEOR 和 ROUGE 评分列表
bleu_scores = []
meteor_scores = []
rouge_scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

# 计算每一对句子的分数
for ref_sentence, gen_sentence in zip(ref_lines, gen_lines):
    # 计算 BLEU 分数
    gen_sentence = gen_sentence.replace("\n", "")
    ref_sentence = ref_sentence.replace("\n", "")
    bleu_score = corpus_bleu([[ref_sentence.split()]], [gen_sentence.split()])
    bleu_scores.append(bleu_score)

    # 使用 NLTK 分词器进行标记化
    ref_tokens = word_tokenize(ref_sentence.strip())
    gen_tokens = word_tokenize(gen_sentence.strip())
    # 计算 METEOR 分数
    meteor_score_value = meteor_score([ref_tokens], gen_tokens)
    meteor_scores.append(meteor_score_value)

    # 计算 ROUGE 分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref_sentence, gen_sentence)
    rouge_scores["rouge-1"].append(scores["rouge1"].fmeasure)
    rouge_scores["rouge-2"].append(scores["rouge2"].fmeasure)
    rouge_scores["rouge-l"].append(scores["rougeL"].fmeasure)

# 汇总分数
average_bleu = sum(bleu_scores) / len(bleu_scores)
average_meteor = sum(meteor_scores) / len(meteor_scores)
average_rouge = {metric: sum(values) / len(values) for metric, values in rouge_scores.items()}

# 打印结果
print("Average BLEU Score:", average_bleu)
print("Average METEOR Score:", average_meteor)
print("Average ROUGE Scores:", average_rouge)


# generated_outputs = generated_outputs.view(int(generated_outputs.shape[0] / args.num_beams), args.num_beams, -1)
            #
            # sort_cand = []
            # for i in range(generated_outputs.shape[0]):
            #     score_l = []
            #     cand = []
            #     for j in range(args.num_beams):
            #         bleu_p = []
            #         bleu_g = []
            #         decoded_outputs = tokenizer.batch_decode(generated_outputs[i, j, :],skip_special_tokens=True)
            #
            #         # hyp = dict(zip(range(1), [decoded_outputs.lower()]))
            #         # ref = dict(zip(range(1), [model2_label[i].lower()]))
            #         # ret = bleu.compute_score(ref, hyp)
            #         hyp = {0: [' '.join(decoded_outputs)]}
            #         ref = {0: [model2_label[i]]}
            #
            #         # ret = bleu.compute_score(ref, hyp)
            #         ret = bleu.compute_score(ref, hyp, verbose=False)
            #
            #         cand_blue = ret[0][3]
            #
            #         score_l.append(cand_blue)
            #         cand.append(generated_outputs[i, j, :])
            #
            #     # 使用 zip 将输入列表和分数列表组合成元组的列表
            #     combined_list = list(zip(score_l, cand))
            #     # 使用 sorted 函数对元组列表进行排序，按照第一个元素（分数）降序排序
            #     sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)
            #     # 提取排序后的输入列表
            #     sort_cand += [item[1] for item in sorted_combined_list]
            #
            # generated_outputs = torch.stack(sort_cand)

# from openai import OpenAI
#
# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key="YOUR API KEY",
#     base_url="https://api.chatanywhere.tech/v1"
# )
#
#
#
# # 非流式响应
# def gpt_35_api(messages: list):
#     """为提供的对话消息创建新的回答
#
#     Args:
#         messages (list): 完整的对话消息
#     """
#     completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
#     print(completion.choices[0].message.content)
#
# def gpt_35_api_stream(messages: list):
#     """为提供的对话消息创建新的回答 (流式传输)
#
#     Args:
#         messages (list): 完整的对话消息
#     """
#     stream = client.chat.completions.create(
#         model='gpt-3.5-turbo',
#         messages=messages,
#         stream=True,
#     )
#     for chunk in stream:
#         if chunk.choices[0].delta.content is not None:
#             print(chunk.choices[0].delta.content, end="")
#
# if __name__ == '__main__':
#     messages = [{'role': 'user','content': '鲁迅和周树人的关系'},]
#     # 非流式调用
#     gpt_35_api(messages)
#     # 流式调用
#     # gpt_35_api_stream(messages)


import re
import json
from statistics import mean

def extract_scores(line):
    """
    从给定的行中提取分数
    """
    match = re.search(r"Sample \d+ scores: (.+)", line)
    if match:
        scores_str = match.group(1)
        scores = json.loads(scores_str.replace("'", "\""))
        return scores
    return None

def calculate_average_scores(file_path):
    """
    计算给定文件中所有分数的平均值
    """
    bleu_scores = []
    meteor_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            scores = extract_scores(line)
            if scores:
                bleu_scores.append(scores['BLEU'])
                meteor_scores.append(scores['METEOR'])
                rouge1_scores.append(scores['ROUGE-1'])
                rouge2_scores.append(scores['ROUGE-2'])
                rougeL_scores.append(scores['ROUGE-L'])

    avg_bleu = mean(bleu_scores)
    avg_meteor = mean(meteor_scores)
    avg_rouge1 = mean(rouge1_scores)
    avg_rouge2 = mean(rouge2_scores)
    avg_rougeL = mean(rougeL_scores)

    return {
        'BLEU': avg_bleu,
        'METEOR': avg_meteor,
        'ROUGE-1': avg_rouge1,
        'ROUGE-2': avg_rouge2,
        'ROUGE-L': avg_rougeL
    }

if __name__ == "__main__":
    file_path = 'metric.txt'  # 替换为你的文件路径
    average_scores = calculate_average_scores(file_path)
    print(f"Average scores: {average_scores}")
