import os
import json
import random
import re
import logging
import sys
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm, auto
from multiprocessing import Pool, cpu_count
import tiktoken

# 从 openai import 过来的自定义类: 用于DeepInfra或OpenAI API客户端
from openai import OpenAI  

##############################################################################
# 1. 工具函数
##############################################################################

def clean_text(text: str) -> str:
    """
    对文本做一些基础的清洗：去除$符号、句号、逗号等，以便后续提取数值。
    """
    text = text.lower()
    text = re.sub(r"\$", "", text)
    text = re.sub(r"(?s).*#### ", "", text)
    text = re.sub(r"\.$", "", text)
    text = re.sub(r",", "", text)
    
    if not text:
        return "-1000000000"
    
    return text

def extract_value(text: str) -> str:
    """
    尝试从给定文本中提取数值（整数或小数），若找不到，则返回 -1000000000 代表无效。
    """
    pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = re.findall(pattern, text)
    
    if matches:
        # 从后往前找，拿到最后一个匹配
        for match_groups in matches[::-1]:
            for group in match_groups:
                if group:
                    return clean_text(group)
    
    return "-1000000000"

def load_dataset(file_path: str, sample_size: int = 20) -> List[Dict]:
    """
    读取 JSONL 格式的数据集，并随机采样指定条数。
    数据每行是一个 JSON，对应一个 question/answer 结构。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return random.sample(data, sample_size)

def count_tokens(text: str, model: str = "p50k_base") -> int:
    """
    使用 tiktoken 来统计文本的 token 数量，默认编码为 p50k_base。
    需要根据具体模型可能使用不同编码器。
    """
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

def evaluate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    计算预测结果相对于真实结果(ground truth)的准确率（整数对比）。
    """
    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        try:
            if int(pred) == int(truth):
                correct += 1
        except:
            pass
    return correct / len(predictions) if predictions else 0.0


##############################################################################
# 2. 树状思维（Tree-of-Thoughts）Solver
##############################################################################

class TreeOfThoughts:
    def __init__(self, api_key: str, base_url: str = "https://api.deepinfra.com/v1/openai", 
                 model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature: float = 1):
        """Initialize the Tree-of-Thoughts solver."""
        # TODO: Initialize the OpenAI client and other necessary attributes
        # pass
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.total_tokens = 0

    def chat_with_gpt(self, prompt: str, n: int = 1, stop: str = None) -> List[str]:
        """Get completions from GPT model."""
        # [IMPORTANT] `stop` is important here. You can refer to the implementation of the Game of 24 in the original ToT codebase.
        # TODO: Implement the chat completion function
        # pass
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            # max_completion_tokens=256,
            n=n,
            stop=stop
        )
        return [msg.message.content for msg in response.choices]

    def generate_thoughts(self, question: str, current_thought: str = "", n_samples: int = 3) -> List[str]:
        """Generate multiple possible next steps in reasoning."""
        # [IMPORTANT] A one-shot example can help the model follow your instructions precisely.
        # TODO: Implement thought generation function
        # pass
        prompt = f"""Please solve this math problem step by step.
        Question: {question}
        Current Thought: {current_thought}
        Let's think step by step."""
        return self.chat_with_gpt(prompt, n=n_samples)
    
    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """Evaluate the likelihood that a thought process leads to the correct answer."""
        # [IMPORTANT] You can use the base model (Qwen2.5-7B-Instruct) for self-evaluation; however, it is not the only option.
        # TODO: Implement thought evaluation function
        # pass
        prompt = f"""Please provide a numeric rating (0 to 1) for the correctness of this reasoning.

Question: {question}
Thought: {thought}

Output only the numeric rating:
"""
        response = self.chat_with_gpt(prompt)
        raw_score = response[0].strip()
        try:
            score = float(raw_score)
            if score < 0:
                score = 0.0
            if score > 1:
                score = 1.0
        except:
            score = 0.0
        return score

    def select_best_thoughts(self, thoughts: List[str], scores: List[float], k: int = 2) -> List[str]:
        """Select the k best thoughts based on their scores."""
        # TODO: Implement thought selection function
        # pass
        best_thoughts = [
            thought for thought, score in 
            sorted(zip(thoughts, scores), key=lambda x: x[1], reverse=True)[:k]
        ]
        return best_thoughts

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, 
                k_best_thoughts: int = 2) -> str:
        """Solve a problem using Tree-of-Thoughts reasoning."""
        # TODO: Implement the main solving function
        # pass
        states = [("", 0.0)]
        for _ in range(max_steps):
            new_states = []
            for (current_text, _) in states:
                generated = self.generate_thoughts(question, current_text, n_samples=n_samples_per_step)
                for g in generated:
                    combined = current_text + "\n" + g if current_text else g
                    score = self.evaluate_thought(question, combined)
                    new_states.append((combined, score))
            new_states.sort(key=lambda x: x[1], reverse=True)
            states = new_states[:k_best_thoughts]
            if any("answer" in s[0].lower() for s in states):
                break
        best_state = max(states, key=lambda x: x[1])
        return best_state[0]


##############################################################################
# 3. 单链思维（Chain-of-Thought）Solver [示例]
##############################################################################

class ChainOfThought:
    """
    这是一个示例性「单链思维」（CoT）推理类：
    - 相比 TreeOfThoughts，CoT 通常是一次性让模型输出完整的推理链，而不进行多轮生成和筛选
    - 可根据实际需要自定义提示风格
    """
    def __init__(self, 
                 api_key: str, 
                 base_url: str = "https://api.deepinfra.com/v1/openai", 
                 model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                 temperature: float = 0.7):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.total_tokens = 0

    def solve(self, question: str) -> Tuple[str, int]:
        """
        给定问题，使用单次调用让模型直接输出包含推理过程（chain of thought）以及结论的回答。
        返回 (模型输出, 使用的tokens数量) 供后续统计。
        """
        prompt = f"""You are a helpful assistant. Please solve the following math problem with a chain of thought reasoning.

Question: {question}

Show your detailed reasoning, and then provide the final answer clearly.
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        content = response.choices[0].message.content

        # 如果需要更精确的 token 统计，可以结合 response.usage
        # self.total_tokens += response.usage["total_tokens"]
        used_tokens = 0  # 这里仅示例，实际可根据 usage 填写

        return content, used_tokens

##############################################################################
# 4. 示例：对比 ToT 和 CoT 在验证集上的表现
##############################################################################

def process_evaluation_item(item: Dict) -> Dict:
    """
    Process a single evaluation item using both ToT and CoT solvers.
    """
    api_key = os.getenv("DEEPINFRA_TOKEN", "6CsmsskJ9LlwYPUMXnsy2LX3u3VgfqIi")
    question = item["question"]
    true_answer = item["answer"]
    
    try:
        # Initialize solvers
        tot_solver = TreeOfThoughts(api_key)
        cot_solver = ChainOfThought(api_key)
        
        # ToT processing
        tot_solution = tot_solver.solve(
            question=question,
            max_steps=8,
            n_samples_per_step=3,
            k_best_thoughts=2
        )
        tot_answer = extract_value(tot_solution)
        
        # # CoT processing
        # cot_solution, cot_used_tokens = cot_solver.solve(question)
        # cot_answer = extract_value(cot_solution)
        
        return {
            "question_id": item["question_id"],
            "tot_predicted": tot_answer,
            # "cot_predicted": cot_answer,
            "true": true_answer,
            # "cot_tokens": cot_used_tokens
        }
    except Exception as e:
        logging.error(f"Error processing question {item['question_id']}: {str(e)}")
        return {
            "question_id": item["question_id"],
            "tot_predicted": "-1000000000",
            "cot_predicted": "-1000000000",
            "true": true_answer,
            "cot_tokens": 0
        }

def main_evaluation():
    """
    Using multiprocessing to evaluate ToT and CoT in parallel.
    """
    # 1. Load environment variables or specify API KEY
    api_key = os.getenv("DEEPINFRA_TOKEN", "6CsmsskJ9LlwYPUMXnsy2LX3u3VgfqIi")

    # 2. Load and sample dataset
    dataset_path = "dataset/cs5260_val_random300.jsonl"
    dataset = load_dataset(dataset_path, sample_size=300)

    # 3. Set up multiprocessing
    num_processes = cpu_count() - 1
    total_questions = len(dataset)
    print(f"\nProcessing {total_questions} questions with {num_processes} processes...")
    
    # 4. Process questions in parallel
    results = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=total_questions, desc="Evaluating", ncols=100) as pbar:
            for result in pool.imap_unordered(process_evaluation_item, dataset):
                results.append(result)
                pbar.update(1)
                sys.stdout.flush()

    # 5. Calculate accuracies
    tot_accuracy = evaluate_accuracy(
        [r["tot_predicted"] for r in results],
        [r["true"] for r in results]
    )
    # cot_accuracy = evaluate_accuracy(
    #     [r["cot_predicted"] for r in results],
    #     [r["true"] for r in results]
    # )
    # total_cot_tokens = sum(r["cot_tokens"] for r in results)

    # 6. Output comparison results
    print("\n=== Validation Results ===")
    print(f"ToT Accuracy: {tot_accuracy:.2%}")
    # print(f"CoT Accuracy: {cot_accuracy:.2%}")
    # print(f"CoT Total Tokens: {total_cot_tokens}")

    # 7. Save results to CSV (optional)
    results_df = pd.DataFrame(results)
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nDetailed results saved to 'evaluation_results.csv'")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(message)s',
        stream=sys.stdout
    )

    main_evaluation()