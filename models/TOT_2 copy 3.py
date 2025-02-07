"""_summary_
    A class to implement the Tree-of-Thoughts (ToT) solver using OpenAI's GPT model.
    Attributes:
        api_key (str): The API key for accessing the OpenAI service.
        base_url (str): The base URL for the OpenAI API.
        model (str): The model name to be used for generating thoughts.
        temperature (float): The temperature setting for the GPT model.
        total_tokens (int): The total number of tokens used in the session.
    Methods:
        __init__(api_key: str, base_url: str = "https://api.deepinfra.com/v1/openai", model: str = "Qwen/Qwen2.5-7B-Instruct", temperature: float = 0.7):
            Initializes the Tree-of-Thoughts solver with the given parameters.
        chat_with_gpt(prompt: str, n: int = 1, stop: str = None) -> List[str]:
            Gets completions from the GPT model based on the provided prompt.
        generate_thoughts(question: str, current_thought: str = "", n_samples: int = 3) -> List[str]:
            Generates multiple possible next steps in reasoning for the given question.
        evaluate_thought(question: str, thought: str, cache: bool = True) -> float:
            Evaluates the likelihood that a thought process leads to the correct answer.
        select_best_thoughts(thoughts: List[str], scores: List[float], k: int = 2) -> List[str]:
            Selects the k best thoughts based on their scores.
        solve(question: str, max_steps: int = 8, n_samples_per_step: int = 3, k_best_thoughts: int = 2) -> str:
            Solves a problem using Tree-of-Thoughts reasoning.
"""

from typing import List
from openai import OpenAI
class TreeOfThoughts:
    def __init__(self, api_key: str, base_url: str = "https://api.deepinfra.com/v1/openai", 
                 model: str = "Qwen/Qwen2.5-7B-Instruct", temperature: float = 0.7):
        """Initialize the Tree-of-Thoughts solver."""
        # 初始化 OpenAI 客户端及其他必要属性
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.total_tokens = 0

    def chat_with_gpt(self, prompt: str, n: int = 1, stop: str = None) -> List[str]:
        """Get completions from GPT model."""
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
        # 修改提示，引导模型按照指定格式输出，并加强对所有计算的交叉验证和自检：
        prompt = f"""You are a meticulous mathematician. Please extract key information from the problem using PDDL notation and then solve it step by step.
For every arithmetic calculation, enclose the computation in <<>> and double-check the result with a re-calculation.
Your output must follow this format exactly:
1. At the beginning, provide key extracted information using PDDL markers (e.g., PDDL: (info ...)).
2. Then, list the step-by-step reasoning with each calculation enclosed in <<>>.
   - For each step, cross-check the intermediate results with the original problem data.
   - If any ambiguous expression arises (e.g., "twice-more-dogs"), explicitly state your interpretation and verify it against the problem statement.
3. Finally, on a new line, output '####' immediately followed by the final answer.
Now, please solve the following math problem carefully.
Question: {question}
Current Thought: {current_thought}
Let's think step by step."""
        return self.chat_with_gpt(prompt, n=n_samples)
    
    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """Evaluate the likelihood that a thought process leads to the correct answer."""
        # 修改提示，要求检查格式、每个计算步骤的正确性以及是否有自检交叉验证：
        prompt = f"""You are an expert evaluator. Please check the following reasoning for compliance with the required format:
- The reasoning must start with key information extraction using PDDL markers.
- All arithmetic computations must be enclosed within <<>> and each calculation must be self-verified.
- Every intermediate step should be cross-checked with the given problem data.
- If ambiguous expressions (such as "twice-more-dogs") are present, ensure that a clear, justified interpretation is provided.
- The final answer must be on a new line starting with '####'.
Also, verify that all arithmetic is correct and consistent with the problem's constraints.
Question: {question}
Thought: {thought}

Output only the numeric rating (from 0 to 1) indicating the quality of the reasoning:"""
        response = self.chat_with_gpt(prompt)
        raw_score = response[0].strip()
        try:
            score = float(raw_score)
            score = max(0.0, min(score, 1.0))
        except:
            score = 0.0
        return score

    def select_best_thoughts(self, thoughts: List[str], scores: List[float], k: int = 2) -> List[str]:
        """Select the k best thoughts based on their scores."""
        best_thoughts = [
            thought for thought, score in 
            sorted(zip(thoughts, scores), key=lambda x: x[1], reverse=True)[:k]
        ]
        return best_thoughts

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, 
              k_best_thoughts: int = 2) -> str:
        """Solve a problem using Tree-of-Thoughts reasoning."""
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
            # 如果输出中包含最终答案，则停止搜索
            if any("####" in s[0] for s in states):
                break
        best_state = max(states, key=lambda x: x[1])
        return best_state[0]
