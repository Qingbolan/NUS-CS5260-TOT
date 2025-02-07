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
        """Generate multiple possible next steps in reasoning with enhanced validation."""
        prompt = f"""You are a mathematical problem solver with expertise in percentages and sequential calculations. Follow these strict guidelines:

    1. Percentage and Base Value Analysis:
    a) Explicitly identify the base value for each percentage:
        PDDL: (info (base_value X) (percentage Y) (target_value Z))
        Example:
        - "50% of $2400" → base = $2400, percentage = 50%
        - "20% of remaining" → base = remaining amount
    
    b) Track dependencies between calculations:
        - Is the percentage based on original or remaining amount?
        - Create dependency tree:
            Original Amount ($2400)
            ├─ 50% of $2400 = $1200 (retirement)
            └─ 20% of $2400 = $480 (car)
    
    c) Verify percentage relationships:
        - Sum of percentages ≤ 100% for mutually exclusive items
        - Cross-validate: partial amount / base = stated percentage

    2. Sequential Calculation Process:
    For each step:
    a) State the base value being used
    b) Show percentage conversion to decimal
    c) Calculate dollar/unit amount
    d) Update remaining amount if needed
    Example:
    ```
    Base: $2400
    Step 1: 50% = 0.50
    Amount: <<2400 * 0.50 = 1200>>
    Remaining: <<2400 - 1200 = 1200>>
    ```

    3. Special Case Handling:
    a) Multiple Percentages:
        - Clear which base each applies to
        - Show if percentages are:
            * Cumulative (add up)
            * Sequential (apply to remainder)
            * Independent (same base)
    
    b) Rate Problems:
        - Linear vs Compound calculations
        - Explicitly state calculation method
        - Show why chosen method is appropriate
    
    c) Group/Portion Problems:
        - Track both group and individual numbers
        - Verify portion calculations
        - Cross-check totals

    4. Validation Steps:
    After each calculation:
    a) Reality Check:
        - Is result reasonable for percentage?
        - Does remaining amount make sense?
        - Do parts sum to whole?
    
    b) Unit Verification:
        - Consistent units throughout
        - Appropriate rounding
        - Percentage vs decimal conversion
    
    c) Cross-Check Methods:
        - Calculate both ways:
            * Part = Whole × Percentage
            * Percentage = Part ÷ Whole
        - Results should match

    5. Answer Format:
    - Use decimal points for currency
    - Include units in answer
    - Round as specified
    - End with #### followed by answer

    Now solve this problem:
    Question: {question}
    Current Thought: {current_thought}"""

        return self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """Enhanced evaluation with focus on percentage and sequence handling."""
        prompt = f"""As a mathematical solution evaluator, assess this solution against these criteria:

    1. Percentage Handling (0.4 points):
    a) Base Value Identification:
        - Clear identification of base for each percentage
        - Proper tracking of changing bases
        - Explicit percentage to decimal conversions
    
    b) Dependency Management:
        - Clear sequence of operations
        - Proper base value for each calculation
        - Tracking of remaining amounts
    
    c) Relationship Validation:
        - Sum of related percentages ≤ 100%
        - Cross-validation of percentage calculations
        - Proper handling of compound effects

    2. Calculation Process (0.3 points):
    a) Step Documentation:
        - Base value stated for each step
        - Clear percentage conversions
        - Interim results shown
        - Running totals maintained
    
    b) Validation Methods:
        - Alternative calculation paths
        - Unit consistency
        - Reality checks on results

    3. Problem-Specific Handling (0.2 points):
    a) Multiple Percentages:
        - Clear base identification
        - Proper sequence
        - Cumulative vs Sequential handling
    
    b) Rate/Group Problems:
        - Appropriate method choice
        - Clear justification
        - Proper total/part relationships

    4. Format and Clarity (0.1 points):
    - PDDL notation usage
    - Clear dependency tracking
    - Proper units and rounding
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
