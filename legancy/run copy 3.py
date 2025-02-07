from openai import OpenAI
import numpy as np
from typing import List, Tuple, Dict
import os
import json
import random
from tqdm import tqdm
import tiktoken
import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\$", "", text)
    text = re.sub(r"(?s).*#### ", "", text)
    text = re.sub(r"\.$", "", text)
    text = re.sub(r",", "", text)
    
    if not text:
        return "-1000000000"
    
    return text

def extract_value(text: str) -> str:
    pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = re.findall(pattern, text)
    
    if matches:
        for match_groups in matches[::-1]:
            for group in match_groups:
                if group:
                    return clean_text(group)
    
    return "-1000000000"

def load_dataset(file_path: str, sample_size: int = 20) -> List[Dict]:
    """Load and sample from dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return random.sample(data, sample_size)

def count_tokens(text: str, model: str = "p50k_base") -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

def evaluate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Calculate accuracy of predictions."""
    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        try:
            if int(pred) == int(truth):
                correct += 1
        except:
            pass
    return correct / len(predictions)

class TreeOfThoughts:
    def __init__(self, api_key: str, base_url: str = "https://api.deepinfra.com/v1/openai", 
                 model: str = "Qwen/Qwen2.5-7B-Instruct", temperature: float = 0.7):
        """Initialize the Tree-of-Thoughts solver."""
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.total_tokens = 0
        self.thoughts_history = []  # Record each step of the reasoning process

    def chat_with_gpt(self, prompt: str, n: int = 1, stop: str = None) -> List[str]:
        """Get completions from GPT model."""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=n,
            stop=stop
        )
        return [msg.message.content for msg in response.choices]

    def generate_thoughts(self, question: str, current_thought: str = "", n_samples: int = 3) -> List[str]:
        """Generate multiple possible next steps in reasoning."""
        prompt = f"""Please solve the following math problem step by step, showing all calculations in the format <<expression = result>>.
        Each step should start with a dash (-) and be on a new line.
        If there are multiple ways to solve the problem, list and explain each method.
        End with #### followed by the final answer.

        Example:
        Question: Ben bought a car for $20000 in 2007. The price of the car depreciates at a constant rate of 21% per year. Find the price of the car in the year 2010.
        Solution:
        - The price of the car depreciates by $<<20000*0.21=4200>>4200 per year.
        - Ben had the car for 2010 - 2007 = <<2010-2007=3>>3 years.
        - The price of the car depreciates $<<4200*3=12600>>12600 from 2007 to 2010.
        - So the price of the car in 2010 is $20000 - $12600 = $<<20000-12600=7400>>7400.
        - #### 7400

        Now solve this question:
        {question}
        Current thought process:
        {current_thought}

        Next step (continue the solution):"""
        
        thoughts = self.chat_with_gpt(prompt, n=n_samples)
        
        # Record the thought process for debugging and analysis
        self.thoughts_history.append(thoughts)
        
        return [self._format_thought(t) for t in thoughts]

    def _format_thought(self, thought: str) -> str:
        """Format the thought to ensure it follows the required structure."""
        thought = re.sub(r'####.*$', '', thought, flags=re.MULTILINE)  # Remove any existing ####
        
        if not thought.strip().startswith('-'):
            thought = f"- {thought.strip()}"
        
        if '<<' not in thought and '=' in thought:
            parts = thought.split('=')
            if len(parts) == 2:
                expr = parts[0].strip()
                result = parts[1].strip()
                thought = f"{expr} = <<{expr.strip('$')}={result.strip('$')}>>{result}"
                
        return thought

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """Evaluate the likelihood that a thought process leads to the correct answer."""
        prompt = f"""Rate the correctness and completeness of this mathematical solution from 0 to 1. Consider:
        1. Are all calculations correctly formatted as <<expression = result>>?
        2. Are the mathematical operations correct?
        3. Does it follow a logical progression?
        4. If complete, does it end with #### and the final answer?

        Question: {question}
        Solution:
        {thought}

        Output only the numeric rating (0 to 1):"""
        
        response = self.chat_with_gpt(prompt)
        try:
            score = float(response[0].strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.0

    def select_best_thoughts(self, thoughts: List[str], scores: List[float], k: int = 2) -> List[str]:
        """Select the k best thoughts based on their scores."""
        return [t for t, _ in sorted(zip(thoughts, scores), key=lambda x: x[1], reverse=True)[:k]]

    def dfs_solve(self, question: str, max_depth: int = 8, n_samples_per_step: int = 3, k_best_thoughts: int = 2) -> str:
        """Solve a problem using depth-first search to explore all possible thoughts."""
        stack = [("", 0.0)]  # (current_thought, score)
        self.thoughts_history.clear()  # Reset thoughts history for each run
        
        for depth in range(max_depth):
            if not stack:
                break
            
            current_thought, _ = stack.pop()
            generated_thoughts = self.generate_thoughts(question, current_thought, n_samples=n_samples_per_step)
            
            for generated in generated_thoughts:
                score = self.evaluate_thought(question, generated)
                stack.append((generated, score))
                
                # Stop if the solution is complete
                if "####" in generated:
                    return generated
        
        # If no solution is found, return a fallback
        return "Unable to complete the solution. Please try again."

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, k_best_thoughts: int = 2) -> str:
        """Solve a problem using Tree-of-Thoughts reasoning with depth-first search."""
        states = [("", 0.0)]
        
        first_attempt = self.generate_thoughts(question, "", n_samples=1)
        if first_attempt:
            score = self.evaluate_thought(question, first_attempt[0])
            states = [(first_attempt[0], score)]
        else:
            return "Unable to generate a solution."
        
        for step in range(max_steps):
            new_states = []
            
            for current_text, _ in states:
                try:
                    generated = self.generate_thoughts(question, current_text, n_samples=n_samples_per_step)
                    for g in generated:
                        if g:
                            combined = current_text + "\n" + g if current_text else g
                            if not re.search(r'<<.*?=.*?>>', combined):
                                g = self._format_thought(g)
                                combined = current_text + "\n" + g if current_text else g
                            
                            score = self.evaluate_thought(question, combined)
                            new_states.append((combined, score))
                except Exception as e:
                    continue
            
            if not new_states:
                continue
            
            new_states.sort(key=lambda x: x[1], reverse=True)
            states = new_states[:k_best_thoughts]
            
            if any("####" in s[0] for s in states):
                break
        
        complete_states = [(s, score) for s, score in states if "####" in s]
        if complete_states:
            return max(complete_states, key=lambda x: x[1])[0]
        
        return "Unable to complete the solution."
import os
import pandas as pd
from multiprocessing import Pool, cpu_count, current_process
from tqdm.auto import tqdm
import logging
import sys
from typing import Dict

# 设置日志格式，确保实时输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s',
    stream=sys.stdout
)

def process_single_question(item):
    """处理单个问题的函数"""
    question_id = item["question_id"]
    
    # 直接打印当前正在处理的问题ID
    print(f"Processing: {question_id}", flush=True)
    
    try:
        # 在每个进程中创建新的 solver 实例
        api_key = os.getenv("DEEPINFRA_TOKEN", "6CsmsskJ9LlwYPUMXnsy2LX3u3VgfqIi")
        tot_solver = TreeOfThoughts(api_key)
        
        # ToT solving
        tot_solution = tot_solver.solve(
            question=item["question"],
            max_steps=8,
            n_samples_per_step=3,
            k_best_thoughts=2
        )
        tot_answer = extract_value(tot_solution)
        
        print(f"Completed: {question_id}", flush=True)
        
        return {
            "question_id": question_id,
            "predicted": tot_answer,
        }
    except Exception as e:
        print(f"Error on {question_id}: {str(e)}", flush=True)
        raise
    
def main():
    # 1. Load dataset
    dataset = load_dataset("dataset/cs5260_test_random300.jsonl", sample_size=300)
    
    # 2. Set up multiprocessing
    num_processes = cpu_count() - 1 # 限制最大进程数为8
    
    # 记录开始处理的问题总数
    total_questions = len(dataset)
    print(f"\nStarting to process {total_questions} questions with {num_processes} processes")
    sys.stdout.flush()
    
    # 3. Process dataset using multiprocessing
    results = []
    
    # 创建进度条，使用ascii=True确保在所有终端都能正确显示
    pbar = tqdm(total=total_questions, ascii=True, ncols=100, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    try:
        # 创建进程池
        with Pool(processes=num_processes) as pool:
            # 使用imap_unordered可能会更快，因为不需要保持顺序
            for result in pool.imap_unordered(process_single_question, dataset):
                results.append(result)
                pbar.update(1)  # 更新进度条
                sys.stdout.flush()  # 确保进度条更新被显示
                
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        sys.stdout.flush()
        raise
    finally:
        pbar.close()
    
    # 4. Generate submission file
    submission_df = pd.DataFrame({
        'question_id': [r["question_id"] for r in results],
        'answer': [r["predicted"] for r in results]
    })
    
    # 5. Save submission file
    submission_df.to_csv('sample_submission.csv', index=False)
    print("\nSubmission file saved as 'sample_submission.csv'")
    sys.stdout.flush()

if __name__ == "__main__":
    main()