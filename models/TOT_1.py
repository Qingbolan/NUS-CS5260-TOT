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
