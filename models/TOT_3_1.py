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
        """Generate multiple possible next steps in reasoning with enhanced complex calculation handling."""
        prompt = f"""You are a mathematical problem solver specializing in complex relationships and multi-step calculations. Follow these strict guidelines:

    1. Relationship Tree Analysis:
    First map ALL relationships and dependencies:
    ```
    PDDL: 
    (info (base_value X) (relationship Y) (target_value Z))
    (info (condition A) (affects B) (type relative/absolute))
    ```
    Example:
    ```
    Base: $100,000 salary
    ├─ 20% savings rate
    │  └─ $20,000 annual savings
    ├─ 40% retirement need
    │  └─ $40,000 annual retirement
    ```

    2. Percentage Chain Handling:
    a) Sequential Percentages:
        - Track base value for each step
        - Identify if percentages are:
            * Additive (A + B)
            * Multiplicative (A × B)
            * Independent (separate bases)
    
    b) Multiple Base Values:
        - Identify primary base
        - Track dependent calculations
        - Verify base consistency
    
    Example:
    ```
    Betty has 30% more than Adam (50 points)
    Base: 50 points
    Increase: 30% = 0.30
    Calculation: <<50 * (1 + 0.30) = 65>> points
    Verify: 50 + (50 * 0.30) = <<50 + 15 = 65>>
    ```

    3. Time and Frequency Calculations:
    a) Period Analysis:
        - Identify base period
        - Note variations in frequency
        - Calculate total occurrences
    
    b) Multiple Conditions:
        - List all time-based conditions
        - Calculate each separately
        - Combine results correctly
    
    Example:
    ```
    Regular: 3 times × 5 days = <<3 * 5 = 15>>
    Special: 2 times × 2 days = <<2 * 2 = 4>>
    Total: <<15 + 4 = 19>>
    ```

    4. Progressive Calculations:
    a) Sequence Tracking:
        - Start with base values
        - Show each transformation
        - Verify running totals
    
    b) Intermediate Validation:
        - Check each step independently
        - Verify against previous steps
        - Cross-validate results
    
    Example:
    ```
    Base: 50 points
    Step 1: +30% = <<50 * 1.30 = 65>>
    Step 2: -30 points = <<65 - 30 = 35>>
    Step 3: ×3 = <<35 * 3 = 105>>
    ```

    5. Final Validation:
    a) Relationship Check:
        - All relationships accounted for
        - Dependencies properly handled
        - Results logically consistent
    
    b) Numeric Validation:
        - All calculations double-checked
        - Units consistent
        - Results within reasonable range
    
    c) Answer Requirements:
        - Units included if needed
        - Proper rounding applied
        - Format matches question

    6. Answer Format:
    - Include required units
    - Round as specified
    - End with #### followed by answer

    Now solve this problem:
    Question: {question}
    Current Thought: {current_thought}"""

        return self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """Enhanced evaluation with focus on complex calculation validation."""
        prompt = f"""As a mathematical solution evaluator, assess this solution against these criteria:

    1. Relationship Analysis (0.3 points):
    a) Relationship Mapping:
        - All dependencies identified
        - Base values correct
        - Relationships proper
    
    b) Percentage Handling:
        - Base values clear
        - Proper sequence
        - Correct application

    2. Calculation Process (0.3 points):
    a) Step Sequence:
        - Clear progression
        - Intermediate validation
        - Running totals accurate
    
    b) Multi-condition Handling:
        - All conditions included
        - Proper combinations
        - Results verified

    3. Validation Quality (0.3 points):
    a) Intermediate Checks:
        - Each step verified
        - Cross-validation shown
        - Results reasonable
    
    b) Final Verification:
        - All relationships checked
        - Units consistent
        - Answer format correct

    4. Documentation (0.1 points):
    - Clear step descriptions
    - Proper PDDL usage
    - #### marker present

    Question: {question}
    Thought: {thought}

    Output a single float from 0-1 representing quality score."""

        response = self.chat_with_gpt(prompt)
        try:
            score = float(response[0].strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.0

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
