from typing import List, Tuple
from openai import OpenAI

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

Please provide multiple reasoning branches that explore different possible approaches."""
        
        return self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """Enhanced evaluation with focus on logical relationship validation."""
        if cache and thought in self.thought_cache:
            return self.thought_cache[thought]

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

Please output a single float from 0 to 1 representing the quality of this reasoning process."""

        try:
            score = float(self.chat_with_gpt(prompt)[0].strip())
            score = max(0.0, min(1.0, score))
            if cache:
                self.thought_cache[thought] = score
            return score
        except Exception as e:
            return 0.0

    def select_best_thoughts(self, thoughts: List[str], scores: List[float], k: int = 2) -> List[str]:
        """Select the k best thoughts based on their evaluation scores."""
        # 贪心策略：选择得分最高的k个思维
        best_thoughts = [
            thought for thought, score in 
            sorted(zip(thoughts, scores), key=lambda x: x[1], reverse=True)[:k]
        ]
        return best_thoughts

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, 
              k_best_thoughts: int = 2) -> str:
        """Solve a problem using Tree-of-Thoughts reasoning with greedy pruning."""
        states = [("", 1.0)]  # 初始状态
        best_score = 0.0
        best_solution = None
        
        for step in range(max_steps):
            new_states = []
            for current_text, current_score in states:
                # 贪心剪枝：如果当前路径分数过低，跳过
                if current_score < best_score * 0.7:  # 动态阈值
                    continue
                    
                # 生成新的思维
                generated_thoughts = self.generate_thoughts(
                    question, current_text, n_samples=n_samples_per_step
                )
                
                for thought in generated_thoughts:
                    # 组合当前路径
                    combined = current_text + "\n" + thought if current_text else thought
                    
                    # 评估新路径
                    score = self.evaluate_thought(question, combined)
                    
                    # 更新最佳解决方案
                    if "####" in thought and score > best_score:
                        best_score = score
                        best_solution = combined
                    
                    new_states.append((combined, score))
            
            # 贪心选择最优路径
            new_states.sort(key=lambda x: x[1], reverse=True)
            states = new_states[:k_best_thoughts]
            
            # 如果找到了足够好的解决方案，提前结束
            if best_solution and best_score > 0.9:
                break
            
            # 如果所有路径都无希望，提前结束
            if not states or all(score < 0.3 for _, score in states):
                break
        
        # 返回最佳路径或最后一个状态中的最佳路径
        if best_solution:
            return best_solution
        return max(states, key=lambda x: x[1])[0] if states else "No valid solution found."