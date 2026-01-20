import argparse
import json
import re
import random
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# 判断答案正确性的核心函数
# ==============================================================================

def find_last_n_uppercase(s, n=0):
    """
    提取字符串中的所有大写字母。
    如果 n=0，返回所有大写字母。
    如果 n>0，返回最后n个大写字母。
    """
    # 提取字符串中的所有大写字母
    uppercase_letters = [char for char in s if char.isupper()]
    if n == 0:
        return uppercase_letters
    else:
        return uppercase_letters[-n:] if len(uppercase_letters) >= n else uppercase_letters

def extract_answer(item, res_key):
    """
    使用正则表达式从模型的生成文本中提取答案选项。
    """
    # 兼容性处理，以防万一res_key不存在或为空
    if res_key not in item or not item[res_key]:
        return ''
        
    item[res_key] = item[res_key].replace('\n', '')
    # 计算有效选项的最大字母，例如 'A', 'B', 'C' -> 'C'
    last_choice = chr(len(item['options']) + ord('A') - 1)
    
    # 优先策略：如果字符串中大写字母数量恰好等于答案长度，直接返回
    upletter = find_last_n_uppercase(item[res_key])
    upletter = sorted(list(set(upletter))) # 去重并排序以保证多选题顺序一致
    if len(upletter) == len(item['answer']):
        return ''.join(upletter)
        
    # 根据问题类型使用不同的正则匹配模式
    if item['question_type'] == '单选' or item['question_type'] == '判断':
        patterns = [
            (fr'^(\(\):)? ?([A-{last_choice}])', 2),
            (fr'答案(选项)?(是|为)：? ?([A-{last_choice}])', 3),
            (fr'答案(是|为)选项 ?([A-{last_choice}])', 2),
            (fr'故?选择?：? ?([A-{last_choice}])', 1),
            (fr'([A-{last_choice}]) ?选?项(是|为)?正确', 1),
            (fr'正确的?选项(是|为) ?([A-{last_choice}])', 2),
            (fr'答案(应该)?(是|为)([A-{last_choice}])', 3),
            (fr'选项 ?([A-{last_choice}]) ?(.*?)(是|为)?正确(答案)?', 1),
            (fr'选择答案 ?([A-{last_choice}])', 1),
            (fr'答案?：?([A-{last_choice}])', 1),
            (fr'([A-{last_choice}])(选?项)?是?符合题意', 1),
            (fr'答案选项：? ?([A-{last_choice}])', 1),
            (fr'答案(选项)?为(.*?)([A-{last_choice}])', 3),
            (fr'答案(就是)?(.*?)([A-{last_choice}])', 3),
            (fr'综上所述(.*){{0,5}}([A-{last_choice}])', 2),
        ]
        if item['question_type'] == '判断':
            patterns.append((fr'^(正确|错误)(.*?)([A-{last_choice}])', 3))

        for pattern, idx in patterns:
            m = re.search(pattern, item[res_key], re.DOTALL)
            if m:
                answer = m.group(idx)
                return answer.strip()
        
        # 判断题的额外逻辑
        if item['question_type'] == '判断':
            m = re.search(r'(正确|错误)', item[res_key], re.DOTALL)
            if m:
                answer = 'A' if m.group(1) == '正确' else 'B'
                return answer
                
    elif item['question_type'] == '多选':
        # 多选题的正则匹配模式
        patterns = [
            # 改进了多选的正则，使其更能捕获 A、B、C 或 AB,C 这样的格式
            (fr'答案[应]?(选项)?(是|为)?[：: ]*([A-{last_choice}](?:[、,， ]?[A-{last_choice}])*)\b', 3),
            (fr'故?选择?[：: ]*([A-{last_choice}](?:[、,， ]?[A-{last_choice}])*)\b', 1)
        ]
        for pattern, idx in patterns:
            m = re.search(pattern, item[res_key], re.DOTALL)
            if m:
                answer_str = m.group(idx)
                # 提取所有大写字母
                return "".join(sorted(list(set(re.findall(f'[A-{last_choice}]', answer_str)))))
        
        # 如果正则匹配失败，回退到提取所有大写字母的策略
        return "".join(upletter)

    return ''

def judge_correct(item, response_key, model_name=""):
    """
    判断给定问题的模型答案是否正确。
    """
    # 确保 response_key 存在
    if response_key not in item or not item[response_key]:
        return 0

    # 提取模型输出的答案
    extracted = extract_answer(item, response_key)
    
    # 答案格式化：去重、排序，确保 'BCA' 和 'ABC' 等价
    response = "".join(sorted(list(set(extracted))))
    answers = "".join(sorted(list(set(item['answer']))))

    if not response:
        random_num = random.random()
        num_options = len(item.get('options', {}))
        if num_options == 0: return 0

        # 根据问题类型确定随机猜对的概率
        if item['question_type'] == '多选':
            # 多选题随机猜对概率很低，这里按您的代码设一个固定值
            return 1 if random_num < 0.2 else 0
        elif item['question_type'] == '单选':
            return 1 if random_num < (1 / num_options) else 0
        elif item['question_type'] == '判断':
            return 1 if random_num < 0.5 else 0
    
    # 比较提取出的答案和标准答案
    return 1 if response == answers else 0


# ==============================================================================
# 评估主逻辑
# ==============================================================================

def calculate_accuracy(stats):
    """一个辅助函数，用于从统计字典中计算准确率。"""
    for key, value in stats.items():
        if isinstance(value, dict):
            if 'total' in value and value['total'] > 0:
                value['accuracy'] = round(value['correct'] / value['total'], 4)
            else:
                value['accuracy'] = 0.0
            
            # 递归处理嵌套的字典 (用于 class-2)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                     calculate_accuracy({sub_key: sub_value})


def evaluate(results_path, output_path, granularity):
    """
    主评估函数。
    :param results_path: 模型生成的结果文件路径。
    :param output_path: 评估报告的保存路径。
    :param granularity: 评估粒度 ('all', 'class-1', 'class-2')。
    """
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 结果文件未找到于 {results_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析 {results_path} 中的JSON。")
        return

    # 初始化统计数据结构
    overall_stats = {'correct': 0, 'total': 0}
    # 使用 defaultdict 简化代码
    class1_domain_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    class1_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    class2_domain_stats = defaultdict(lambda: {
        'correct': 0,
        'total': 0,
        'subtypes': defaultdict(lambda: {'correct': 0, 'total': 0})
    })
    class2_class_stats = defaultdict(lambda: {
        'correct': 0,
        'total': 0,
        'tasks': defaultdict(lambda: {'correct': 0, 'total': 0})
    })

    print(f"正在以 '{granularity}' 粒度评估 {len(data)} 条数据...")
    for item in tqdm(data, desc="评估进度"):
        if 'generated_answer' not in item or 'answer' not in item:
            continue

        is_correct = judge_correct(item, 'generated_answer')
        
        # 累加总数
        overall_stats['correct'] += is_correct
        overall_stats['total'] += 1

        # 根据粒度累加分类统计
        if granularity == 'class-1':
            domain_type = item.get('domain', {}).get('type', '未知')
            class_class = item.get('class', {}).get('class', '未知')
            class1_domain_stats[domain_type]['correct'] += is_correct
            class1_domain_stats[domain_type]['total'] += 1
            class1_class_stats[class_class]['correct'] += is_correct
            class1_class_stats[class_class]['total'] += 1

        elif granularity == 'class-2':
            # Domain -> Subtype
            domain_type = item.get('domain', {}).get('type', '未知')
            domain_subtype = item.get('domain', {}).get('subtype', '未知')
            class2_domain_stats[domain_type]['correct'] += is_correct
            class2_domain_stats[domain_type]['total'] += 1
            class2_domain_stats[domain_type]['subtypes'][domain_subtype]['correct'] += is_correct
            class2_domain_stats[domain_type]['subtypes'][domain_subtype]['total'] += 1
            
            # Class -> Task
            class_class = item.get('class', {}).get('class', '未知')
            class_task = item.get('class', {}).get('task', '未知')
            class2_class_stats[class_class]['correct'] += is_correct
            class2_class_stats[class_class]['total'] += 1
            class2_class_stats[class_class]['tasks'][class_task]['correct'] += is_correct
            class2_class_stats[class_class]['tasks'][class_task]['total'] += 1

    # 计算准确率并构建最终报告
    final_report = {}
    calculate_accuracy({'overall': overall_stats})
    final_report['overall_accuracy'] = overall_stats
    
    if granularity == 'all':
        pass # 报告已包含总准确率
    elif granularity == 'class-1':
        calculate_accuracy(class1_domain_stats)
        calculate_accuracy(class1_class_stats)
        final_report['by_domain_type'] = dict(class1_domain_stats)
        final_report['by_class_class'] = dict(class1_class_stats)
    elif granularity == 'class-2':
        calculate_accuracy(class2_domain_stats)
        calculate_accuracy(class2_class_stats)
        final_report['by_domain_subtype'] = dict(class2_domain_stats)
        final_report['by_class_task'] = dict(class2_class_stats)

    # 保存报告
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=4)
        print(f"评估报告已成功保存到: {output_path}")
    except IOError as e:
        print(f"错误: 无法写入文件 {output_path}. 原因: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="模型生成结果的准确率评估脚本。")
    parser.add_argument("--results_path", type=str, required=True, help="包含模型生成结果的JSON文件路径。")
    parser.add_argument("--output_path", type=str, required=True, help="保存评估报告的JSON文件路径。")
    parser.add_argument(
        "--granularity",
        type=str,
        required=True,
        choices=['all', 'class-1', 'class-2'],
        help="评估粒度: 'all' (全部), 'class-1' (按一级标题), 'class-2' (按二级标题)。"
    )
    
    args = parser.parse_args()
    
    evaluate(args.results_path, args.output_path, args.granularity)