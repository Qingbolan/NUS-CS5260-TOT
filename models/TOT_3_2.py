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
        """Generate multiple possible next steps in reasoning with enhanced logic chain validation."""
        prompt = f"""You are a mathematical problem solver specializing in complex logical relationships. Follow these strict guidelines:

    1. Logic Chain Analysis:
    First map ALL logical relationships and dependencies in a tree:
    ```
    PDDL: 
    (info (base_entity X) (relationship Y) (target_entity Z))
    (info (condition A) (affects B) (type relative/absolute))
    ```

    Example Age Problem:
    ```
    Base: John's current age
    ├─ Was 19 when James born
    │  └─ James is now 18
    ├─ Was 32 when youngest born
    │  └─ Youngest is now [calculate]
    ```

    2. Time and Age Relationships:
    a) Age Calculations:
        - Track reference points
        - Handle past/future/present
        - Verify age consistency
    
    b) Time Period Analysis:
        - Define start/end points
        - Calculate elapsed time
        - Account for increments
    
    Example:
    ```
    Current Time: t
    Past Event: t - 3 years
    Future Event: t + 5 years
    Verify: Past → Present → Future
    ```

    3. Quantity Relationships:
    a) Direct Relationships:
        - A is twice B
        - A is 3 more than B
        - A is half of B
    
    b) Indirect Relationships:
        - If A > B and B > C, then A > C
        - Group relationships
        - Transitive properties
    
    Example:
    ```
    James = 2 × Dora
    Dora = 12 - 3
    Therefore James = 2 × (12 - 3) = 18
    ```

    4. Sequential Logic:
    a) Order Dependencies:
        - Must happen after X
        - Cannot occur before Y
        - Simultaneous events
    
    b) Conditional Logic:
        - If X then Y
        - Either X or Y
        - Both X and Y
    
    Example:
    ```
    Given: Cake needs
    1. 20 min prep
    2. 30 min bake
    3. 120 min cool
    4. 10 min frost
    Total = 180 min
    ```

    5. Validation Steps:
    a) Logical Consistency:
        - No contradictions
        - All relationships satisfied
        - Temporal consistency
    
    b) Numeric Validation:
        - Units consistent
        - Values reasonable
        - Cross-calculations check
    
    c) Timeline Check:
        - Events properly sequenced
        - Durations realistic
        - No temporal paradoxes

    6. Answer Format:
    - State answer clearly
    - Include units if needed
    - Use #### marker

    Now solve this problem:
    Question: {question}
    Current Thought: {current_thought}"""

        return self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """Enhanced evaluation with focus on logical relationship validation."""
        prompt = f"""As a mathematical solution evaluator, assess this solution against these criteria:

    1. Logical Structure (0.3 points):
    a) Relationship Mapping:
        - Dependencies clear
        - Sequences proper
        - Logic sound
    
    b) Time/Age Handling:
        - Reference points clear
        - Time flow consistent
        - Age relationships valid

    2. Calculation Process (0.3 points):
    a) Step Sequence:
        - Clear progression
        - Intermediate validation
        - Running totals accurate
    
    b) Relationship Application:
        - Direct relationships correct
        - Indirect relationships valid
        - Cross-dependencies checked

    3. Validation Quality (0.3 points):
    a) Logic Checks:
        - No contradictions
        - All conditions met
        - Sequence valid
    
    b) Numeric Verification:
        - Calculations accurate
        - Units consistent
        - Results reasonable

    4. Documentation (0.1 points):
    - Clear explanation
    - Proper notation
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
