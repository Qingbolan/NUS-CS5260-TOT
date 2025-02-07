import numpy as np
import pandas as pd
import random
import json
import re
import os
import sys
import logging

from typing import List, Dict
from typing import Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tiktoken

# ============ Debug & Logging Config ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s',
    stream=sys.stdout
)

# ============ 工具函数 ============
def clean_text(text: str) -> str:
    """
    对提取出的数字文本进行基础清洗。
    """
    text = text.lower()
    text = re.sub(r"\$", "", text)
    text = re.sub(r"(?s).*#### ", "", text)  # 移除出现的 #### 以及之前内容
    text = re.sub(r"\.$", "", text)
    text = re.sub(r",", "", text)

    if not text:
        return "-1000000000"
    return text

def extract_value(text: str) -> str:
    """
    从文本中抽取数字或金额（如$100、-300等），优先返回最后出现的数值（逆序遍历）。
    返回 "-1000000000" 表示没有匹配到。
    """
    pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = re.findall(pattern, text)
    
    if matches:
        # 逆序遍历，找到第一个非空匹配值
        for match_groups in matches[::-1]:
            for group in match_groups:
                if group:
                    return clean_text(group)
    
    return "-1000000000"

def load_dataset(file_path: str, sample_size: int = 20) -> List[Dict]:
    """
    加载并采样数据集。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    # 随机采样 sample_size 条数据
    return random.sample(data, sample_size)

def count_tokens(text: str, model: str = "p50k_base") -> int:
    """
    使用 tiktoken 对文本进行分词，返回 token 数量。
    """
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

def evaluate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    计算预测结果与真实标签之间的准确率。
    """
    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        try:
            if int(pred) == int(truth):
                correct += 1
        except:
            pass
    return correct / len(predictions)

# ============ 核心类：TreeOfThoughts ============
# 注意：以下基于假设有一个自定义的 "OpenAI" 客户端类可用（from openai import OpenAI），
#       若与官方 openai 库不兼容，请根据实际情况改写。
from openai import OpenAI  # 如果与官方库冲突，这里需替换为正确的自定义库导入

class TreeOfThoughts:
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "https://api.deepinfra.com/v1/openai", 
        model: str = "Qwen/Qwen2.5-7B-Instruct", 
        temperature: float = 0.7, 
        max_depth: int = 8
    ):
        """
        初始化 Tree-of-Thoughts 求解器，基于 DFS，对思考深度和温度进行控制。
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.max_depth = max_depth

        # 统计总的 token 数量，非必需
        self.total_tokens = 0

        # 存储每次调用 generate_thoughts 产生的思考过程
        # 每次调用 generate_thoughts，我们会把返回的所有思考（每条思考）拼到这个列表
        self.thoughts_history: List[List[str]] = []

    def chat_with_gpt(self, prompt: str, n: int = 1, stop: str = None) -> List[str]:
        """
        与 GPT 模型交互，获取回复。
        """
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=n,
            stop=stop
        )
        # 假设返回的结构符合 .choices 和 .choices[i].message.content
        return [msg.message.content for msg in response.choices]

    def generate_thoughts(self, question: str, current_thought: str = "", n_samples: int = 3) -> List[str]:
        """
        生成多个可能的下一个思考步骤（chain of thought），并将结果记录到 self.thoughts_history。
        """
        prompt = f"""You are a powerful agent with broad math knowledge and great python programming skills. You need to use python interpreter to do accurate calculation on math equations.
        !!! Remember:
        1. Use code to solve the problem step by step. The solution should include three parts: <code>, <output>, and <answer>.
        2. All calculations should be done in python code. Provide concise reasoning and thinking in the comments of the code.
        3. The most related python packages include 'math', 'sympy', 'scipy', and 'numpy'.
        4. Please use the following template:

        Question: {question}
        <code>Construct the code step by step. Use <end_of_step> to indicate the end of each step. Ensure your code can execute correctly(excluding <end_of_step>) and print the answer. Avoid undefined variables (NameError), unimported packages, or formatting errors (SyntaxError, TypeError). In the last step of the code, print the final answer and add a comment: Now print the final answer.<end_of_code>
        <output>Execute the code using the Python interpreter and display the printed results.<end_of_output>
        <answer>The concise answer without verbose context, put your final answer's numerical part (without unit, only focus on the numerical part if it's a choice question) in boxed.<end_of_answer>

        Now, solve the problem using this approach."""
        
        thoughts = self.chat_with_gpt(prompt, n=n_samples)
        
        # 记录本轮调用产生的所有思考
        self.thoughts_history.append(thoughts)
        
        # 格式化每条思考，使其尽量符合所需格式
        formatted_thoughts = [self._format_thought(t) for t in thoughts]
        return formatted_thoughts

    def _format_thought(self, thought: str) -> str:
        """
        将思考文本进行简单的格式化，如去掉多余的 ####，并对简单赋值进行特殊标注。
        """
        # 移除任何 '####...' 后面的内容
        thought = re.sub(r'####.*$', '', thought, flags=re.MULTILINE)
        
        # 如果没有 '-' 开头，则给它加上
        if not thought.strip().startswith('-'):
            thought = f"- {thought.strip()}"
        
        # 如果思考中包含类似 "x = 10" 的形式，但未用 <<x=10>> 包裹，则做个简单的替换示例
        if '<<' not in thought and '=' in thought:
            parts = thought.split('=')
            if len(parts) == 2:
                expr = parts[0].strip()
                result = parts[1].strip()
                thought = f"{expr} = <<{expr.strip('$')}={result.strip('$')}>>{result}"
        
        return thought

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """
        简单使用 LLM，对给出的思考过程进行评分，返回 [0.0, 1.0] 区间内的值。
        """
        prompt = f"""Rate the correctness and completeness of this mathematical solution from 0 to 1. Consider:
        1. Are all calculations correctly formatted as <<expression = result>>?
        2. Are the mathematical operations correct?
        3. Does it follow a logical progression?
        4. If complete, does it end with #### and the final answer?

        Question: {question}
        Solution:
        {thought}

        Output only the numeric rating (0 to 1):"""
        
        response = self.chat_with_gpt(prompt, n=1)
        try:
            score = float(response[0].strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.0

    def dfs_solve(self, question: str, n_samples_per_step: int = 3, k_best_thoughts: int = 2) -> str:
        """
        采用深度优先搜索，尝试在多步思考中找到正确答案。
        """
        stack = [("", 0.0)]  # (current_thought, score)
        self.thoughts_history.clear()  # 每次运行前清空思考历史
        
        for depth in range(self.max_depth):
            if not stack:
                break
            
            current_thought, _ = stack.pop()
            generated_thoughts = self.generate_thoughts(question, current_thought, n_samples=n_samples_per_step)
            
            for generated in generated_thoughts:
                score = self.evaluate_thought(question, generated)
                stack.append((generated, score))
                
                # 如果思考已完成（包含 '####' 标记），则返回
                if "####" in generated:
                    return generated
        
        return "Unable to complete the solution. Please try again."

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, k_best_thoughts: int = 2) -> str:
        """
        通过若干次迭代的树状思考，来解决问题。
        """
        # 初始状态
        states = [("", 0.0)]
        self.thoughts_history.clear()

        # 第一次（简单尝试）
        first_attempt = self.generate_thoughts(question, "", n_samples=1)
        if first_attempt:
            score = self.evaluate_thought(question, first_attempt[0])
            states = [(first_attempt[0], score)]
        else:
            return "Unable to generate a solution."
        
        # 多步迭代
        for step in range(max_steps):
            new_states = []
            
            for current_text, _ in states:
                try:
                    generated_list = self.generate_thoughts(question, current_text, n_samples=n_samples_per_step)
                    for g in generated_list:
                        if g:
                            combined = current_text + "\n" + g if current_text else g
                            if not re.search(r'<<.*?=.*?>>', combined):
                                # 如果没有包含特殊标记，则重复做一次格式化
                                g = self._format_thought(g)
                                combined = current_text + "\n" + g if current_text else g
                            
                            score = self.evaluate_thought(question, combined)
                            new_states.append((combined, score))
                except Exception as e:
                    # 若某次生成出现异常则忽略
                    continue
            
            # 若新一轮没有产生任何思考，则继续下一轮
            if not new_states:
                continue
            
            # 根据打分对状态进行排序，保留分数最高的 k_best_thoughts 条
            new_states.sort(key=lambda x: x[1], reverse=True)
            states = new_states[:k_best_thoughts]
            
            # 如果产生了完整解答（含有 "####"），则停止
            if any("####" in s[0] for s in states):
                break
        
        # 若有最终解答则输出分数最高的那条
        complete_states = [(s, score) for s, score in states if "####" in s]
        if complete_states:
            return max(complete_states, key=lambda x: x[1])[0]
        
        return "Unable to complete the solution."

# ============ 多进程处理相关 ============

def process_single_question(item: Dict) -> Dict:
    """
    处理单个问题的函数，返回预测结果及 chain_of_thought 以便后续分析。
    """
    question_id = item["question_id"]
    print(f"Processing: {question_id}", flush=True)

    try:
        # 在每个进程中创建新的 solver 实例
        api_key = os.getenv("DEEPINFRA_TOKEN", "6CsmsskJ9LlwYPUMXnsy2LX3u3VgfqIi")
        tot_solver = TreeOfThoughts(api_key)

        # 调用 TreeOfThoughts
        tot_solution = tot_solver.solve(
            question=item["question"],
            max_steps=8,
            n_samples_per_step=3,
            k_best_thoughts=2
        )
        # 抽取最终答案
        tot_answer = extract_value(tot_solution)

        print(f"Completed: {question_id}", flush=True)

        return {
            "question_id": question_id,
            "predicted": tot_answer,
            "chain_of_thought": tot_solver.thoughts_history  # 存储本题执行过程中的所有中间思考
        }
    except Exception as e:
        print(f"Error on {question_id}: {str(e)}", flush=True)
        raise

def main():
    # 1. 加载数据集
    dataset = load_dataset("dataset/cs5260_test_random300.jsonl", sample_size=300)

    # 2. 设置多进程
    num_processes = cpu_count() - 1
    total_questions = len(dataset)
    print(f"\nStarting to process {total_questions} questions with {num_processes} processes")
    sys.stdout.flush()

    # 3. 多进程处理数据集
    results = []
    pbar = tqdm(total=total_questions, ascii=True, ncols=100, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    try:
        with Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(process_single_question, dataset):
                results.append(result)
                pbar.update(1)
                sys.stdout.flush()
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        sys.stdout.flush()
        raise
    finally:
        pbar.close()

    # 4. 生成提交文件（只包含最终预测）
    submission_df = pd.DataFrame({
        "question_id": [r["question_id"] for r in results],
        "answer": [r["predicted"] for r in results]
    })
    submission_df.to_csv('sample_submission.csv', index=False)
    print("\nSubmission file saved as 'sample_submission.csv'")
    sys.stdout.flush()

    # 5. 将所有链式思考历史存入单独的文件，便于后续调试或分析
    #    每个元素包含 question_id, predicted, chain_of_thought
    cot_path = "chain_of_thoughts.json"
    with open(cot_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Chain-of-thought file saved as '{cot_path}'")
    sys.stdout.flush()

if __name__ == "__main__":
    main()