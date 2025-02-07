from typing import List, Tuple
from openai import OpenAI
import re
import logging

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
        self.thought_cache = {}  # Cache for evaluated thoughts

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
        """
        Generate multiple possible next steps in reasoning with enhanced logic chain validation.
        The prompt now requires a clearly marked final answer on a separate line using the format:
            FINAL ANSWER: <your final answer>
        """
        prompt = f"""You are a highly skilled mathematical problem solver. In this task, you are required to perform Tree-of-Thoughts (TOT) reasoning. Please:

1. **Construct a Logical Tree:**
   - Map out all logical relationships and dependencies.
   - Clearly identify base entities, their relationships, and any intermediate dependencies.

2. **Time and Sequential Reasoning:**
   - Identify temporal reference points and ensure proper sequence of events.

3. **Quantity and Numerical Relationships:**
   - Detail all numerical relationships and calculations.
   - Distinguish intermediate calculations from the final result.

4. **Branching and Conditional Logic:**
   - Explore multiple reasoning branches and alternative approaches.
   - Label branches clearly (e.g., Branch 1, Branch 2, etc.).

5. **Final Answer Extraction:**
   - On a separate line at the end, output your final answer exactly as:
       FINAL ANSWER: <your final answer>
   - Ensure the final answer is consistent with the reasoning process.

Now, solve the following problem in a TOT manner:
Question: {question}
Current Thought (if any): {current_thought}

Please provide multiple reasoning branches."""
        
        return self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """
        Evaluate the reasoning process with focus on logical structure, calculation accuracy, branching quality, and final answer extraction.
        The evaluator's response should ideally be a single float between 0 and 1.
        """
        if cache and thought in self.thought_cache:
            return self.thought_cache[thought]
        
        # If the final answer is missing, return a low score immediately.
        if "FINAL ANSWER:" not in thought:
            return 0.0
        
        prompt = f"""You are a mathematical solution evaluator. Evaluate the following reasoning process according to these criteria:

1. **Logical Structure (0.3 points):**
   - Is the logical tree complete and clear?
   - Are temporal and sequential relationships correctly identified?

2. **Calculation Process (0.3 points):**
   - Are the calculation steps clearly and correctly defined?
   - Are intermediate calculations clearly distinguished from the final answer?

3. **TOT Branching Quality (0.3 points):**
   - Does the solution explore multiple branches effectively?
   - Are alternative reasoning paths considered?

4. **Final Answer Documentation (0.1 points):**
   - Is the final answer clearly marked on a separate line using "FINAL ANSWER:"?
   - Is the final answer consistent and unambiguous?

Question: {question}
Thought Process:
{thought}

Please output a single float from 0 to 1 representing the overall quality of this reasoning process."""
        
        try:
            evaluator_response = self.chat_with_gpt(prompt)[0].strip()
            # Extract the first numeric value (integer or decimal) from the response.
            match = re.search(r"(\d+(\.\d+)?)", evaluator_response)
            if match:
                score = float(match.group(1))
            else:
                score = 0.0
            score = max(0.0, min(1.0, score))
            if cache:
                self.thought_cache[thought] = score
            return score
        except Exception as e:
            logging.error(f"Evaluation error: {e}")
            return 0.0

    def self_supervised_validation(self, thought: str) -> bool:
        """
        Perform a self-supervised check to ensure that the final answer extracted from the thought is consistent with the reasoning process.
        Returns True if validation is successful, False otherwise.
        """
        # Extract the final answer using a regular expression.
        final_answer_match = re.search(r"FINAL ANSWER:\s*(.+)", thought)
        if not final_answer_match:
            return False
        final_answer = final_answer_match.group(1).strip()
        
        # Create a validation prompt.
        validation_prompt = f"""Review the following reasoning process. Is the final answer (the line starting with "FINAL ANSWER:") fully consistent with all the calculations and logical steps provided?

Reasoning Process:
{thought}

Please reply with a single word: YES if it is consistent, or NO if there is any inconsistency."""
        try:
            validation_response = self.chat_with_gpt(validation_prompt, n=1)[0].strip().upper()
            return validation_response == "YES"
        except Exception as e:
            logging.error(f"Self-supervised validation error: {e}")
            return False

    def select_best_thoughts(self, thoughts: List[str], scores: List[float], k: int = 2) -> List[str]:
        """Select the top k thoughts based on their evaluation scores."""
        best_thoughts = [
            thought for thought, score in 
            sorted(zip(thoughts, scores), key=lambda x: x[1], reverse=True)[:k]
        ]
        return best_thoughts

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, 
              k_best_thoughts: int = 2) -> str:
        """
        Solve a problem using Tree-of-Thoughts reasoning with greedy pruning.
        Implements enhanced final answer extraction and self-supervised validation.
        """
        states = [("", 1.0)]  # Each state is a tuple: (current_thought, score)
        best_score = 0.0
        best_solution = None
        
        for step in range(max_steps):
            new_states = []
            for current_text, current_score in states:
                # Greedy pruning: skip paths with very low scores compared to the current best.
                if current_score < best_score * 0.7:
                    continue
                
                # Generate new thoughts from the current state.
                generated_thoughts = self.generate_thoughts(
                    question, current_text, n_samples=n_samples_per_step
                )
                
                for thought in generated_thoughts:
                    # Combine the current state with the newly generated thought.
                    combined = current_text + "\n" + thought if current_text else thought
                    
                    # Evaluate the new thought process.
                    score = self.evaluate_thought(question, combined)
                    
                    # Self-supervised validation: check if the final answer is consistent.
                    if "FINAL ANSWER:" in combined:
                        if not self.self_supervised_validation(combined):
                            score *= 0.9  # Penalize if self-validation fails.
                            
                    # Update the best solution if a final answer is found and the score is improved.
                    if "FINAL ANSWER:" in combined and score > best_score:
                        best_score = score
                        best_solution = combined
                    
                    new_states.append((combined, score))
            
            # Greedy selection: choose the top k_best_thoughts paths.
            new_states.sort(key=lambda x: x[1], reverse=True)
            states = new_states[:k_best_thoughts]
            
            # Early stopping if a sufficiently high scoring solution is found.
            if best_solution and best_score > 0.9:
                break
            
            # Break if no promising states remain.
            if not states or all(score < 0.3 for _, score in states):
                break
        
        # Return the best solution if found, otherwise return the highest scoring state.
        if best_solution:
            return best_solution
        return max(states, key=lambda x: x[1])[0] if states else "No valid solution found."
