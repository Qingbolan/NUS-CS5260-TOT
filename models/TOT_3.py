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
from typing import List

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
        prompt = f"""You are a highly skilled mathematical problem solver. In this task, you are required not only to perform a chain-of-thought (CoT) reasoning, but to think in a Tree-of-Thoughts (TOT) fashion. This means you must:

1. **Construct a Logical Tree:**
   - Map ALL logical relationships and dependencies as a tree structure.
   - Clearly identify base entities, their relationships, and intermediate dependencies.
   - Example:
     ```
     PDDL: 
     (info (base_entity: "John's current age") (relationship: "was 19 when James was born") (target_entity: "James' age"))
     (info (condition: "James is now 18") (affects: "John's age calculation") (type: relative))
     ```

2. **Time and Sequential Reasoning:**
   - Identify temporal reference points (past, present, future).
   - Ensure proper sequence: events or age changes must be chronologically consistent.
   - Example:
     ```
     Timeline:
       - Event A at t-3 years → Event B at t → Event C at t+5 years
     ```

3. **Quantity and Numerical Relationships:**
   - Detail direct and indirect relationships (e.g., A is twice B, or A > B > C).
   - Verify all units and numerical dependencies.
   - Example:
     ```
     Calculation:
       - If James = 2 × Dora and Dora = 9, then James = 18.
     ```

4. **Branching and Conditional Logic:**
   - For each reasoning branch, propose multiple possible next steps.
   - Evaluate different potential paths before converging on the final answer.
   - Use clear branch labels (e.g., Branch 1, Branch 2, etc.).

5. **Validation and Final Answer:**
   - At each step, perform forward calculation, reverse verification, and cross-check intermediate results.
   - Clearly mark the final answer using the #### marker.
   - Include units and ensure the answer is unambiguous.

Now, solve the following problem in a TOT manner:
Question: {question}
Current Thought (if any): {current_thought}

Please provide multiple reasoning branches that explore different possible approaches.
"""
        return self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """Enhanced evaluation with focus on logical relationship validation."""
        prompt = f"""You are a mathematical solution evaluator. Evaluate the following reasoning process according to these criteria:

1. **Logical Structure (0.3 points):**
   - Is the logical tree fully constructed with all dependencies clearly mapped?
   - Are time and sequential relationships consistent and clearly identified?

2. **Calculation Process (0.3 points):**
   - Are the calculation steps clearly defined with forward, backward, and cross-check validations?
   - Are the direct and indirect numerical relationships correctly applied?

3. **TOT Branching Quality (0.3 points):**
   - Does the solution explore multiple branches effectively rather than a single chain-of-thought?
   - Are alternative reasoning paths considered and evaluated?

4. **Final Answer Documentation (0.1 points):**
   - Is the final answer clearly marked with a "####" marker?
   - Are units, precision, and notation correct?

Question: {question}
Thought Process:
{thought}

Please output a single float from 0 to 1 representing the quality of this reasoning process.
"""
        response = self.chat_with_gpt(prompt)
        try:
            score = float(response[0].strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            return 0.0

    def select_best_thoughts(self, thoughts: List[str], scores: List[float], k: int = 2) -> List[str]:
        """Select the k best thoughts based on their evaluation scores."""
        best_thoughts = [
            thought for thought, score in 
            sorted(zip(thoughts, scores), key=lambda x: x[1], reverse=True)[:k]
        ]
        return best_thoughts

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, 
              k_best_thoughts: int = 2) -> str:
        """Solve a problem using Tree-of-Thoughts reasoning."""
        states = [("", 0.0)]
        for step in range(max_steps):
            new_states = []
            for (current_text, _) in states:
                generated = self.generate_thoughts(question, current_text, n_samples=n_samples_per_step)
                for branch in generated:
                    combined = current_text + "\n" + branch if current_text else branch
                    score = self.evaluate_thought(question, combined)
                    new_states.append((combined, score))
            new_states.sort(key=lambda x: x[1], reverse=True)
            states = new_states[:k_best_thoughts]
            if any("####" in state[0] for state in states):
                break
        best_state = max(states, key=lambda x: x[1])
        return best_state[0]
