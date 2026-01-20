import argparse
import json
import os
from typing import List, Dict, Any

from tqdm import tqdm

# vLLM 是一个可选库，所以我们使用 try-except 来导入它
try:
    from vllm import LLM as VLLM_Engine, SamplingParams
except ImportError:
    VLLM_Engine = None
    SamplingParams = None

# OpenAI 也是一个可选库
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --- 模型抽象层 (无变化) ---

class BaseLLM:
    """语言模型的抽象基类。"""
    def __init__(self, model_name_or_path: str, **kwargs):
        self.model_name_or_path = model_name_or_path

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """生成文本的核心方法，子类需要实现。"""
        raise NotImplementedError

class VLLMModel(BaseLLM):
    """使用 vLLM 加载的本地模型的封装类。"""
    def __init__(self, model_name_or_path: str, tensor_parallel_size: int, gpu_memory_utilization: float, **kwargs):
        super().__init__(model_name_or_path)
        if VLLM_Engine is None:
            raise ImportError("vLLM 未安装。请执行 'pip install vllm'进行安装。")
        print(f"正在从 {model_name_or_path} 加载 vLLM 模型...")
        self.llm = VLLM_Engine(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True, # 允许执行模型仓库中的自定义代码
        )

    def generate(self, prompts: List[str], max_tokens: int, temperature: float, **kwargs) -> List[str]:
        """使用 vLLM 进行批量文本生成。"""
        print(f"正在使用 vLLM 生成 {len(prompts)} 条回复...")
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=1 # 每个 prompt 生成一个输出
        )
        querys=[[{"role":"user","content":query}] for query in prompts]
        outputs = self.llm.chat(querys, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

class OpenAIModel(BaseLLM):
    """OpenAI API 模型的封装类。"""
    def __init__(self, model_name_or_path: str, openai_api_key: str, openai_base_url: str, **kwargs):
        super().__init__(model_name_or_path)
        if OpenAI is None:
            raise ImportError("openai 库未安装。请执行 'pip install openai' 进行安装。")
        
        # 如果未通过参数传入，则尝试从环境变量中获取 API key
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("未提供 OpenAI API key。请通过 --openai_api_key 参数或 OPENAI_API_KEY 环境变量设置。")
        
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    def generate(self, prompts: List[str], max_tokens: int, temperature: float, **kwargs) -> List[str]:
        """使用 OpenAI API 循环生成文本。"""
        print(f"正在使用 OpenAI 模型 '{self.model_name_or_path}' 生成 {len(prompts)} 条回复...")
        responses = []
        for prompt in tqdm(prompts, desc="调用 OpenAI API"):
            try:
                # 使用当前标准的 ChatCompletions 接口
                completion = self.client.chat.completions.create(
                    model=self.model_name_or_path,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1
                )
                responses.append(completion.choices[0].message.content.strip())
            except Exception as e:
                print(f"调用 OpenAI API 时发生错误: {e}")
                responses.append(f"错误: {e}")
        return responses

# --- 辅助函数 ---

def load_json_data(filepath: str) -> List[Dict[str, Any]]:
    """从 JSON 文件加载数据。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 在路径 {filepath} 未找到文件。")
        return []

def save_json_data(data: List[Dict[str, Any]], filepath: str):
    """将数据保存到 JSON 文件。"""
    print(f"正在将结果保存到 {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def build_few_shot_examples(few_shot_data: List[Dict], question_type: str, num_shots: int) -> str:
    """为给定的问题类型构建 Few-shot 示例字符串。"""
    if num_shots == 0:
        return ""

    # 筛选出正确类型的示例，并取前 `num_shots` 个
    relevant_shots = [
        shot for shot in few_shot_data if shot.get("question_type") == question_type
    ][:num_shots]
    
    if not relevant_shots:
        return ""

    example_texts = []
    for shot_data in relevant_shots:
        ## 修改 ##: 构建示例时不再包含选项（options）。
        example_text = "问题：{}\n答案：{}\n".format(
            shot_data["question"], shot_data["answer"]
        )
        example_texts.append(example_text)
    
    # 将所有示例用换行符连接起来
    return "\n".join(example_texts)


def main():
    """主函数，用于运行整个生成流程。"""
    ## 修改 ##: 更新了脚本描述。
    parser = argparse.ArgumentParser(description="用于农业考试（简答题）的大模型生成脚本。")
    
    # --- 核心参数 ---
    parser.add_argument("--model_type", type=str, required=True, choices=["vllm", "openai"], help="模型部署的类型 (vllm 或 openai)。")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="OpenAI 的模型名称或 vLLM 模型的本地路径。")
    parser.add_argument("--input_path", type=str, required=True, help="包含测试问题的输入JSON文件路径。")
    parser.add_argument("--output_path", type=str, required=True, help="用于保存输出结果的JSON文件路径。")
    parser.add_argument("--few_shot_path", type=str, required=True, help="包含 few-shot 示例的JSON文件路径。")
    
    # --- 生成策略参数 ---
    parser.add_argument("--num_shots", type=int, default=0, choices=range(6), help="使用的 few-shot 示例数量 (0-5)。")
    parser.add_argument("--cot", action="store_true", help="启用思维链 (CoT) prompting。")
    
    # --- 生成效果参数 ---
    parser.add_argument("--max_tokens", type=int, default=1024, help="生成的最大 token 数量。")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度，值越高随机性越强。")
    
    # --- vLLM 专属参数 ---
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="vLLM 使用的 GPU 显存比例。")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM 的张量并行大小。")
    
    # --- OpenAI 专属参数 ---
    parser.add_argument("--openai_api_key", type=str, default="Empty", help="OpenAI API key。也可以通过环境变量 OPENAI_API_KEY 设置。")
    parser.add_argument("--openai_base_url", type=str, default="http://localhost:8013/v1", help="用于兼容OpenAI接口的自定义URL（例如本地部署的模型服务）。")
    
    args = parser.parse_args()

    # 步骤 1: 加载数据
    print("开始加载数据...")
    test_data = load_json_data(args.input_path)[:100]
    few_shot_data = load_json_data(args.few_shot_path) if args.num_shots > 0 else []

    if not test_data:
        print("未找到测试数据，程序退出。")
        return

    # 步骤 2: 选择 Prompt 模板
    ## 修改 ##: 根据 --cot 参数直接定义简答题的 prompt_template 字符串。
    if args.cot:
        prompt_template = "以下是中国关于农业考试的简答题，回答时让我们一步步思考。{examples}\n问题：{question}\n答案：\n"
        print("当前模式: CoT Prompt, {args.num_shots}-shot。")
    else:
        prompt_template = "以下是中国关于农业考试的简答题，请输出正确答案。{examples}\n问题：{question}\n答案：\n"
        print(f"当前模式: 标准 Prompt, {args.num_shots}-shot。")

    # 步骤 3: 准备所有待生成的 Prompts
    prompts_to_generate = []
    # ## 修改 ##: 用于在生成后能正确匹配回原始数据的索引列表
    valid_items_indices = [] 
    print("正在准备 Prompts...")
    for i, item in enumerate(tqdm(test_data, desc="构建 Prompts")):
        question = item.get("question")
        if not question:
            print(f"警告: 跳过索引为 {i} 的数据，因为它缺少 'question' 键。")
            continue

        examples_str = build_few_shot_examples(few_shot_data, "简答", args.num_shots)
        
        final_prompt = prompt_template.format(
            examples=examples_str,
            question=question
        )
        prompts_to_generate.append(final_prompt)
        valid_items_indices.append(i) # 记录有效问题的原始索引


    # 步骤 4: 初始化模型
    model = None
    if args.model_type == 'vllm':
        model = VLLMModel(
            model_name_or_path=args.model_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    elif args.model_type == 'openai':
        model = OpenAIModel(
            model_name_or_path=args.model_name_or_path,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    # 步骤 5: 生成答案
    generated_texts = model.generate(
        prompts=prompts_to_generate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # 步骤 6: 合并结果并保存
    if len(generated_texts) == len(prompts_to_generate):
        for i, text in enumerate(generated_texts):
            original_index = valid_items_indices[i]
            test_data[original_index]['generated_answer'] = text
    else:
         print(f"警告: 生成的文本数量 ({len(generated_texts)}) 与待处理的 prompts 数量 ({len(prompts_to_generate)}) 不匹配。结果可能不准确。")

    save_json_data(test_data, args.output_path)
    print("生成任务完成。")


if __name__ == "__main__":
    main()