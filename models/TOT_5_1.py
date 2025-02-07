"""_summary_
A class to implement the Tree-of-Thoughts (ToT) solver using OpenAI's GPT model.
Attributes:
    api_key (str): The API key for accessing the OpenAI service.
    base_url (str): The base URL for the OpenAI API.
    model (str): The model name to be used for generating thoughts.
    temperature (float): The temperature setting for the GPT model.
    total_tokens (int): The total number of tokens used in the session.
    task (optional): A task instance that defines input/output, prompt templates, and answer verification.
Methods:
    __init__(api_key: str, base_url: str = "https://api.deepinfra.com/v1/openai", model: str = "Qwen/Qwen2.5-7B-Instruct", temperature: float = 0.7, task=None):
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

from typing import List, Optional
from openai import OpenAI
import re
import os
import json

class TreeOfThoughts:
    def __init__(self, 
                 api_key: str, 
                 base_url: str = "https://api.deepinfra.com/v1/openai", 
                 model: str = "Qwen/Qwen2.5-7B-Instruct", 
                 temperature: float = 0.7,
                 task: Optional[object] = None):
        """
        Initialize the Tree-of-Thoughts solver.
        
        Args:
            api_key (str): API key for OpenAI.
            base_url (str): Base URL for OpenAI API.
            model (str): Model name.
            temperature (float): Sampling temperature.
            task (optional): A task instance that provides data input/output, prompt templates, and answer verification.
        """
        # Initialize the OpenAI client and necessary attributes.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.total_tokens = 0
        self.task = task  # Optional task instance for domain-specific templates and validation.
        self.logs = []    # For logging thought branches and evaluation scores.

    def chat_with_gpt(self, prompt: str, n: int = 1, stop: str = None) -> List[str]:
        """
        Get completions from the GPT model based on the provided prompt.
        
        Args:
            prompt (str): The prompt to send to the model.
            n (int): Number of completions to generate.
            stop (str): Optional. The stop sequence.
        
        Returns:
            List[str]: A list of generated completions.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=n,
            stop=stop
        )
        # Update total tokens if available in response (depends on client implementation)
        # self.total_tokens += response.usage.total_tokens  # Uncomment if token usage available.
        return [msg.message.content.strip() for msg in response.choices]

    def generate_thoughts(self, question: str, current_thought: str = "", n_samples: int = 3) -> List[str]:
        """
        Generate multiple possible next reasoning steps given a problem and current thought.
        
        The prompt emphasizes constructing a full logical tree with detailed arithmetic validation,
        multiple reasoning branches, and a clearly marked final answer preceded by "####".
        
        Args:
            question (str): The problem statement.
            current_thought (str): The current thought or reasoning path.
            n_samples (int): Number of samples to generate.
        
        Returns:
            List[str]: A list of candidate reasoning branches.
        """
        if self.task and hasattr(self.task, "cot_prompt_wrap"):
            # Use the task's chain-of-thought prompt template if available.
            prompt = self.task.cot_prompt_wrap(question, current_thought)
        else:
            # One-shot example for TOT generation.
            prompt = f"""You are a highly skilled mathematical problem solver. Your task is to solve the problem using a Tree-of-Thoughts (ToT) approach. Do not rely solely on a single chain-of-thought. Instead, follow these steps carefully:

1. **Construct a Complete Logical Tree:**
   - Identify all basic entities, variables, quantities, and time-related factors.
   - Map out all relationships and dependencies. Re-read the problem to ensure correct interpretation of every number.
   
2. **Detailed Arithmetic and Logical Verification:**
   - For each branch, provide forward calculations, backward verifications, and cross-checks.
   - Ensure consistency in variable naming, units, and values.
   
3. **Integrate Multiple Reasoning Branches:**
   - Explore alternative paths and clearly label each branch.
   - Cross-validate different branches before converging on the final solution.
   
4. **Final Answer Clarity:**
   - Mark the final answer with "####" and include any necessary units.
   
Now, solve the following problem using the TOT approach:
Question: {question}
Current Thought (if any): {current_thought}

Provide multiple reasoning branches and ensure every step is validated.
"""
        return self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """
        Evaluate the quality of a generated reasoning process.
        
        The evaluation considers:
          - Completeness of the logical tree.
          - Accuracy of arithmetic (forward, backward, cross-checks).
          - Quality of exploring multiple branches.
          - Clarity of the final answer (marked with "####").
        
        Returns a float between 0 and 1 indicating the quality.
        
        Args:
            question (str): The problem statement.
            thought (str): The reasoning process to evaluate.
            cache (bool): Unused here but can be used to cache evaluations.
        
        Returns:
            float: A score between 0 and 1.
        """
        prompt = f"""You are a mathematical solution evaluator. Please assess the following reasoning process based on these criteria:

1. Logical Structure (0.3 points):
   - Is the logical tree fully constructed with clear entities, variables, and dependencies?
   - Are the numbers, units, and relationships consistent?
   
2. Arithmetic Validation (0.3 points):
   - Are all arithmetic steps correctly performed, with forward, backward, and cross-check validations?
   
3. TOT Branching and Reasoning (0.3 points):
   - Does the solution explore multiple reasoning branches and compare alternative paths?
   
4. Final Answer Clarity (0.1 points):
   - Is the final answer clearly marked with "####" and unambiguous?

Question: {question}
Thought Process:
{thought}

Please output a single float from 0 to 1 representing the overall quality of this reasoning process.
"""
        response = self.chat_with_gpt(prompt, n=1, stop="\n")
        try:
            score = float(response[0].strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    def select_best_thoughts(self, thoughts: List[str], scores: List[float], k: int = 2) -> List[str]:
        """
        Select the best k reasoning branches based on their evaluation scores.
        
        Args:
            thoughts (List[str]): Candidate reasoning branches.
            scores (List[float]): The associated evaluation scores.
            k (int): Number of top branches to select.
        
        Returns:
            List[str]: The selected best reasoning branches.
        """
        best_thoughts = [
            thought for thought, score in 
            sorted(zip(thoughts, scores), key=lambda x: x[1], reverse=True)[:k]
        ]
        return best_thoughts

    def validate_answer(self, question: str, candidate: str, idx: int = 0) -> bool:
        """
        Validate the candidate answer using the task's verification method, if available.
        
        Args:
            question (str): The problem statement.
            candidate (str): The candidate reasoning process.
            idx (int): An optional index for tasks that use indices (e.g., when reading data).
        
        Returns:
            bool: True if the candidate passes the test, False otherwise.
        """
        if self.task and hasattr(self.task, "test_output"):
            result = self.task.test_output(idx, candidate)
            return result.get("r", 0) == 1
        # If no task verification is provided, do a simple check: candidate must contain the final answer marker.
        return "####" in candidate

    def save_logs(self, file_path: str):
        """
        Save the search and evaluation logs as a JSON file.
        
        Args:
            file_path (str): Path to the log file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding="utf8") as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False)

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, 
              k_best_thoughts: int = 2, log_file: str = "./logs/tot_solution.json") -> str:
        """
        Solve a problem using the Tree-of-Thoughts reasoning strategy.
        
        The process:
          - Iteratively generate multiple candidate reasoning branches.
          - Evaluate each candidate and select the top-scoring ones (greedy selection).
          - Continue for a maximum of max_steps or until a branch contains a final answer marked "####".
          - Validate candidate answers using the task's verification function (if available).
          - Log the entire search process.
        
        Returns:
            str: The best reasoning process (including the final answer) according to the evaluation.
        """
        # Initialize the states: each state is a tuple (current_thought, score)
        states = [("", 0.0)]
        
        for step in range(max_steps):
            new_states = []
            step_log = {"step": step, "candidates": []}
            for (current_text, _) in states:
                generated = self.generate_thoughts(question, current_text, n_samples=n_samples_per_step)
                for branch in generated:
                    combined = (current_text + "\n" + branch).strip() if current_text else branch.strip()
                    score = self.evaluate_thought(question, combined)
                    new_states.append((combined, score))
                    step_log["candidates"].append({"branch": combined, "score": score})
                    
                    # If this candidate contains a final answer marker and is of high quality, validate it.
                    if score >= 0.9 and "####" in combined:
                        if self.validate_answer(question, combined):
                            step_log["selected"] = combined
                            self.logs.append(step_log)
                            self.save_logs(log_file)
                            return combined
            self.logs.append(step_log)
            if not new_states:
                break

            # Select best-k branches based on computed scores.
            new_states.sort(key=lambda x: x[1], reverse=True)
            states = new_states[:k_best_thoughts]

            # If any of the selected branches contains a final answer, break early.
            if any("####" in state[0] for state in states):
                break
                
        best_state = max(states, key=lambda x: x[1])
        self.save_logs(log_file)
        return best_state[0]