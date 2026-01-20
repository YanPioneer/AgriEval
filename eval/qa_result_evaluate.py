import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean
from typing import List, Dict, Any

try:
    import sacrebleu
    from sacrebleu.metrics import BLEU
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from rouge import Rouge
    from evaluate import load
    from bert_score import score
except ImportError as e:
    print(f"依赖库未安装，请先执行 pip install sacrebleu nltk rouge-score evaluate bert-score torch transformers")
    print(f"错误详情: {e}")
    exit()

# 首次使用NLTK时可能需要下载
try:
    word_tokenize("test")
except LookupError:
    import nltk
    print("正在下载NLTK的'punkt'数据包...")
    nltk.download('punkt')

# --- 用户提供的指标计算函数 ---
# (已根据通用性稍作修改)

def compute_sacre_sent_bleu(references, candidates):
    """计算 sacre_sent_bleu"""
    bleu = []
    for i, a_gold in enumerate(references):
        bleu.append(sacrebleu.corpus_bleu([candidates[i]], [[a_gold]]).score)
    return round(mean(bleu), 4) if bleu else 0.0

def compute_sacre_corpus_bleu(references, candidates):
    """计算 sacre_corpus_bleu"""
    bleu = BLEU()
    return round(bleu.corpus_score(candidates, [references]).score, 4)

def compute_sent_bleu(references, candidates):
    """计算 sent_bleu"""
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(word_tokenize(candidates[i]))
        ref_list.append([word_tokenize(references[i])])

    if not ref_list:
        return 0.0, 0.0, 0.0, 0.0

    bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0
    smooth_func = SmoothingFunction().method3
    for label, pred in zip(ref_list, dec_list):
        bleu1 += sentence_bleu(label, pred, weights=[1, 0, 0, 0], smoothing_function=smooth_func)
        bleu2 += sentence_bleu(label, pred, weights=[0.5, 0.5, 0, 0], smoothing_function=smooth_func)
        bleu3 += sentence_bleu(label, pred, weights=[1/3, 1/3, 1/3, 0], smoothing_function=smooth_func)
        bleu4 += sentence_bleu(label, pred, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=smooth_func)
    
    count = len(ref_list)
    return (
        round(bleu1 / count * 100, 4),
        round(bleu2 / count * 100, 4),
        round(bleu3 / count * 100, 4),
        round(bleu4 / count * 100, 4),
    )

def compute_corpus_bleu(references, candidates):
    """计算 corpus_bleu"""
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(word_tokenize(candidates[i]))
        ref_list.append([word_tokenize(references[i])])
    
    if not ref_list:
        return 0.0, 0.0, 0.0, 0.0

    bleu1 = corpus_bleu(ref_list, dec_list, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(ref_list, dec_list, weights=(0, 1, 0, 0))
    bleu3 = corpus_bleu(ref_list, dec_list, weights=(0, 0, 1, 0))
    bleu4 = corpus_bleu(ref_list, dec_list, weights=(0, 0, 0, 1))
    return (
        round(bleu1 * 100, 4),
        round(bleu2 * 100, 4),
        round(bleu3 * 100, 4),
        round(bleu4 * 100, 4),
    )

def compute_rouge(references, candidates):
    """计算 rouge"""
    if not candidates or not references:
        return 0.0, 0.0, 0.0
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references)
    rouge_1 = [score["rouge-1"]["f"] * 100 for score in scores]
    rouge_2 = [score["rouge-2"]["f"] * 100 for score in scores]
    rouge_l = [score["rouge-l"]["f"] * 100 for score in scores]
    return (
        round(mean(rouge_1), 4),
        round(mean(rouge_2), 4),
        round(mean(rouge_l), 4),
    )

def compute_question_metrics_score(data: Dict[str, List[str]]) -> List[float]:
    """
    核心评估函数，计算所有指标分数。
    """
    res = []
    p_text = data['res']
    q_text = data['answer']
    
    # 确保没有空的标准答案
    q_text = [item if item and item.strip() else '抱歉，无法回答' for item in q_text]
    p_text = [item if item and item.strip() else ' ' for item in p_text]


    if not p_text or not q_text:
        return [0.0] * 23

    
    # google bleu
    google_bleu = load("google_bleu")
    google_bleu_score = google_bleu.compute(predictions=p_text, references=q_text)["google_bleu"]
    res.append(round(google_bleu_score * 100, 4))
    
    # hf_sacrebleu
    sacrebleu_computer = load("sacrebleu")
    sacrebleu_score = sacrebleu_computer.compute(predictions=p_text, references=[[q] for q in q_text])["score"]
    res.append(round(sacrebleu_score, 4))

    # sacrebleu_sent
    res.append(compute_sacre_sent_bleu(candidates=p_text, references=q_text))

    # sacrebleu_corpus
    res.append(compute_sacre_corpus_bleu(candidates=p_text, references=q_text))

    # sent_bleu
    bleu1, bleu2, bleu3, bleu4 = compute_sent_bleu(p_text, q_text)
    mean_sent_bleu = round((bleu1 + bleu2 + bleu3 + bleu4) / 4, 4)
    res.extend([bleu1, bleu2, bleu3, bleu4, mean_sent_bleu])

    # corpus_bleu
    bleu1, bleu2, bleu3, bleu4 = compute_corpus_bleu(p_text, q_text)
    mean_corpus_bleu = round((bleu1 + bleu2 + bleu3 + bleu4) / 4, 4)
    res.extend([bleu1, bleu2, bleu3, bleu4, mean_corpus_bleu])
    
    # ROUGE
    rouge_1, rouge_2, rouge_l = compute_rouge(p_text, q_text)
    res.extend([rouge_1, rouge_2, rouge_l])

    # hugging face ROUGE
    rouge_computer = load('rouge')
    rouge_results = rouge_computer.compute(predictions=p_text, references=q_text)
    res.extend([
        round(rouge_results['rouge1'] * 100, 4),
        round(rouge_results['rouge2'] * 100, 4),
        round(rouge_results['rougeL'] * 100, 4),
        round(rouge_results['rougeLsum'] * 100, 4)
    ])

    # out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=1024, verbose=False,featurize_model_name='/path/to/gpt2-large/')
    res.append(0.0)

    # BERTScore
    P, R, F1 = score(p_text, q_text, model_type="bert-base-chinese", lang="zh", rescale_with_baseline=True)
    res.append(round(F1.mean().item() * 100, 4))
    
    return res

# --- 评估主逻辑 ---

METRIC_NAMES = (
    'google_bleu', 'hf_sacrebleu', 'sacrebleu_sent', 'sacrebleu_corpus',
    'sent_bleu1', 'sent_bleu2', 'sent_bleu3', 'sent_bleu4', 'sent_bleu_mean',
    'corpus_bleu1', 'corpus_bleu2', 'corpus_bleu3', 'corpus_bleu4', 'corpus_bleu_mean',
    'rouge_1', 'rouge_2', 'rouge_l',
    'hf_rouge1', 'hf_rouge2', 'hf_rougeL', 'hf_rougeLsum',
    'mauve', 'bert_score_f1'
)

def compute_and_format_results(items: List[Dict]) -> Dict[str, Any]:
    """
    为一组数据计算并格式化所有指标。
    """
    if not items:
        return {"sample_count": 0, "metrics": {}}

    predictions = [d['generated_answer'] for d in items]
    references = [d['answer'] for d in items]
    
    scores = compute_question_metrics_score({'res': predictions, 'answer': references})
    
    formatted_metrics = dict(zip(METRIC_NAMES, scores))
    
    return {
        "sample_count": len(items),
        "metrics": formatted_metrics
    }


def main():
    """
    主函数：加载数据、按粒度分组、执行评估并保存结果。
    """
    parser = argparse.ArgumentParser(description="评估文本生成质量的脚本。")
    parser.add_argument("--results_path", type=str, required=True, help="包含模型生成结果的JSON文件路径。")
    parser.add_argument("--output_path", type=str, required=True, help="保存评估报告的JSON文件路径。")
    parser.add_argument(
        "--granularity", 
        type=str, 
        required=True, 
        choices=['all', 'class-1', 'class-2'], 
        help="评估粒度：'all' (全部), 'class-1' (按一级标题), 'class-2' (按二级标题)。"
    )
    args = parser.parse_args()

    # 加载数据
    print(f"正在从 {args.results_path} 加载数据...")
    try:
        with open(args.results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：无法加载或解析文件 {args.results_path}。 {e}")
        return

    # 过滤掉没有生成答案或标准答案的数据项
    valid_data = [
        item for item in data 
        if 'generated_answer' in item and 'answer' in item
    ]
    print(f"共找到 {len(data)} 条数据，其中 {len(valid_data)} 条包含有效答案对，将用于评估。")

    evaluation_results = {}

    if args.granularity == 'all':
        print("评估粒度: all. 正在评估所有数据...")
        evaluation_results['overall'] = compute_and_format_results(valid_data)

    elif args.granularity == 'class-1':
        print("评估粒度: class-1. 正在按一级标题分组评估...")
        # 按 domain-type 分组
        domain_groups = defaultdict(list)
        for item in valid_data:
            key = item.get('domain', {}).get('type', 'Unknown_Domain_Type')
            domain_groups[key].append(item)
        
        evaluation_results['by_domain_type'] = {
            key: compute_and_format_results(items) for key, items in domain_groups.items()
        }
        print(f"完成 'domain.type' 的评估，共 {len(domain_groups)} 个类别。")

        # 按 class-class 分组
        class_groups = defaultdict(list)
        for item in valid_data:
            key = item.get('class', {}).get('class', 'Unknown_Class_Class')
            class_groups[key].append(item)

        evaluation_results['by_class_class'] = {
            key: compute_and_format_results(items) for key, items in class_groups.items()
        }
        print(f"完成 'class.class' 的评估，共 {len(class_groups)} 个类别。")

    elif args.granularity == 'class-2':
        print("评估粒度: class-2. 正在按二级标题分组评估...")
        # 按 domain-type -> subtype 分组
        domain_groups_L2 = defaultdict(lambda: defaultdict(list))
        for item in valid_data:
            type_key = item.get('domain', {}).get('type', 'Unknown_Domain_Type')
            subtype_key = item.get('domain', {}).get('subtype', 'Unknown_Domain_Subtype')
            domain_groups_L2[type_key][subtype_key].append(item)

        evaluation_results['by_domain_subtype'] = {
            type_key: {
                subtype_key: compute_and_format_results(items)
                for subtype_key, items in subtypes.items()
            }
            for type_key, subtypes in domain_groups_L2.items()
        }
        print(f"完成 'domain.subtype' 的评估。")

        # 按 class-class -> task 分组
        class_groups_L2 = defaultdict(lambda: defaultdict(list))
        for item in valid_data:
            class_key = item.get('class', {}).get('class', 'Unknown_Class_Class')
            task_key = item.get('class', {}).get('task', 'Unknown_Class_Task')
            class_groups_L2[class_key][task_key].append(item)

        evaluation_results['by_class_task'] = {
            class_key: {
                task_key: compute_and_format_results(items)
                for task_key, items in tasks.items()
            }
            for class_key, tasks in class_groups_L2.items()
        }
        print(f"完成 'class.task' 的评估。")

    # 保存结果
    input_path = Path(args.results_path)
    output_path = Path(args.output_path)
    
    print(f"正在将评估结果保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
    
    print("评估完成。")

if __name__ == "__main__":
    main()