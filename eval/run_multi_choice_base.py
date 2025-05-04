#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_multi_choice.py
@Time    :   2024/11/11 19:47
@Author  :   WangHaotian
@Description : 对于Agriculture Benchmark的多项选择题(单选和多选)进行实验
'''

import os
import re
import json
import time
import argparse
from tqdm import tqdm

from models.agent_fastapi import Agent_FastAPI
from models.agent_openai import Agent_OpenAI

def get_output_path(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    # if args.few_shot_num == 0:
    #     f_name = f'{args.dataset_name}_{args.dataset_version}_multi-choice_{args.model_name}_random_{args.random_num}.json'
    #     output_path = os.path.join(args.output_dir, f_name)
    # else:
    f_name = f'{args.dataset_name}_{args.dataset_version}_multi-choice_{args.model_name}_{args.few_shot_num}-shot_random_{args.random_num}_test.json'
    output_path = os.path.join(args.output_dir, "{}_shot".format(args.few_shot_num), f_name)
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen1.5-72b-chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--api_type", type=str, default="openai", choices=["openai", "fastapi"])
    parser.add_argument("--api_url", type=str, default="http://localhost:{port}/v1/")
    parser.add_argument("--api_port", type=int, default="8004")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--sc_num", type=int, default=1)
    parser.add_argument("--random_num", type=int, default=0)
    parser.add_argument("--few_shot_num", type=int, default=0, choices=[0, 1, 2, 3,4,5])

    parser.add_argument("--prompt_type", type=str, default="normal", choices=["normal", "cot"])
    parser.add_argument("--dataset_name", type=str, default="Agri_Bench", choices=["Agri_Bench"])
    parser.add_argument("--dataset_version", type=str, default="v6")
    parser.add_argument("--few_shot_path", type=str, default="./data/few_shot/multi_choice_few_shot.json")
    parser.add_argument("--data_path", type=str, default="./data/merged_choice_{version}.json")
    parser.add_argument("--output_dir", type=str, default="./results/multi_choice_v6_1")

    args = parser.parse_args()
    
    with open(args.data_path.format(version=args.dataset_version), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(args.few_shot_path, "r", encoding="utf-8") as fr_shot:
        few_shot_data = json.load(fr_shot)
    
    if args.few_shot_num > 0:
        few_num_dict = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五"}
        prefix_str = "\n以下是{}个例子：\n".format(few_num_dict[args.few_shot_num])

        # 初始化示例数据存储
        shot_templates = {
            "单选": {"text": prefix_str, "count": 0},
            "多选": {"text": prefix_str, "count": 0},
            "判断": {"text": prefix_str, "count": 0}
        }

        # 遍历 few_shot_data 填充示例
        for shot_data in few_shot_data:
            q_type = shot_data["question_type"]
            if q_type in shot_templates and shot_templates[q_type]["count"] < args.few_shot_num:
                shot_options = "\n".join([f"{k}. {v}" for k, v in shot_data["options"].items() if len(v) > 0 and v!=" "])
                shot_templates[q_type]["text"] += "问题：{}\n{}\n答案：{}\n".format(shot_data["question"], shot_options, shot_data["answer"])
                shot_templates[q_type]["count"] += 1

        # 提取最终的字符串
        single_shot_examples = shot_templates["单选"]["text"]
        multi_shot_examples = shot_templates["多选"]["text"]
        judge_shot_examples = shot_templates["判断"]["text"]
    else:
        single_shot_examples, multi_shot_examples, judge_shot_examples = "", "", ""

    output_path = get_output_path(args)
    if os.path.exists(output_path):
        with open(output_path, "r") as fr:
            had_inference_data = json.load(fr)
    else:
        had_inference_data = []

    had_inference_ids = []
    for data in had_inference_data:
        had_inference_ids.append(data["id"])
    
    if args.api_type == "openai":
        agent = Agent_OpenAI(args.model_name, args)
    else:
        agent = Agent_FastAPI(args.model_name, args)
    
    
    single_prompt = "以下是中国关于农业考试的单项选择题，请直接输出正确答案的选项，无需生成解释。{examples}\n问题：{question}\n{options_str}\n答案：\n"
    multi_prompt = "以下是中国关于农业考试的多项选择题，请直接输出正确答案的选项，无需生成解释。{examples}\n问题：{question}\n{options_str}\n答案：\n"
    judge_prompt = "以下是中国关于农业考试的判断题，请直接输出正确答案的选项，无需生成解释。{examples}\n问题：{question}\n{options_str}\n答案：\n"
    unify_prompt = "以下是中国关于农业考试的选择题，请直接输出正确答案的选项，无需生成解释。{examples}\n问题：{question}\n{options_str}\n答案：\n"
    

    for item in tqdm(test_data):
        if item["id"] in had_inference_ids:
            continue
            
        if item["answer"] is None:
            continue
        
        # options_str = "\n".join(item["options"])
        options_str = "\n".join([f"{k}. {v}" for k, v in item["options"].items() if len(v) > 0 and v!=" "])
        
        if item["question_type"] == "单选":
            prompt = single_prompt.format(examples=single_shot_examples, question=item["question"], options_str=options_str)
        elif item["question_type"] == "多选":
            prompt = multi_prompt.format(examples=multi_shot_examples, question=item["question"], options_str=options_str)
        else:
            prompt = judge_prompt.format(examples=judge_shot_examples, question=item["question"], options_str=options_str)
        
        item["user_prompt"] = prompt

        messages = []
        messages.append({"role": "user", "content": prompt})
        
        if args.sc_num > 1:
            item["response"] = []
            for i in range(args.sc_num):
                response = agent.generate_response_base(prompt)
                item["response"].append(response)
        else:
            response = agent.generate_response_base(prompt)
            item["response"] = response
            
        had_inference_data.append(item)
    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(had_inference_data, f, ensure_ascii=False, indent=4)

        if args.prompt_type == "normal":
            time.sleep(0.2)
