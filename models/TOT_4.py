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
        """获取 GPT 模型的回复。"""
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
        """
        生成多个可能的下一步推理，并激发 TOT（Tree-of-Thoughts）思维。
        
        要求：
          - 构建完整的逻辑树，明确列出所有基本实体、变量、单位、关系、时间及数量依赖，并对每一步的算术进行详细推导。
          - 在每个分支中要求重新阅读题干，严格核对每一步的算术和逻辑推导，确保变量命名、数值及单位一致。  
            例如：
              * 当题干写到“each of the middle 2 shelves can hold 10 books”时，必须计算为 \(2 \times 10 = 20\) 而非 10。
              * 在百分比、转换率、折旧等问题中，使用题干明确给出的模型或数据。
          - 如果题干中存在歧义或多个相同实体（如多个货架、多个苹果），必须正确应用乘法因素。
          - 多分支讨论中交叉验证不同路径，确保最终答案的一致性。
          - 最终答案前必须使用 "####" 标记，并确保答案中仅包含正确结果及必要单位。
        """
        prompt = f"""You are a highly skilled mathematical problem solver. Your task is to solve the problem using a Tree-of-Thoughts (TOT) approach. Do not rely solely on a single chain-of-thought. Instead, follow these steps carefully:

1. **Construct a Complete Logical Tree:**
   - Identify all base entities, variables, quantities, and time-related information.
   - Clearly map all relationships and dependencies (e.g., direct, indirect, conditional). Ensure that variable names and units are consistent.
   - Re-read the problem statement to ensure every number is correctly interpreted. For example:
       * When the problem says "each of the middle 2 shelves can hold 10 books," compute the total capacity for the middle shelves as 2 × 10.
       * For trading or depreciation problems, use the conversion rates or models as stated.
       * Verify if any ambiguity exists and resolve it by closely re-reading the problem.
       
2. **Perform Detailed Arithmetic and Logical Checks:**
   - For every branch, include forward calculations, reverse (backward) verifications, and cross-checks.
   - Double-check each arithmetic step, ensuring consistency across branches.
   - Clearly label each branch (e.g., Branch 1, Branch 2, etc.) and explicitly state any assumptions.

3. **Integrate Branching and Conditional Reasoning:**
   - Explore multiple reasoning paths if ambiguity exists.
   - Compare and validate alternative branches before converging on the final answer.

4. **Mark the Final Answer Clearly:**
   - Once verified, present the final answer with a "####" marker.
   - Include units if needed, and ensure the answer is unambiguous.

Now, solve the following problem using the TOT approach:
Question: {question}
Current Thought (if any): {current_thought}

Provide multiple reasoning branches and ensure every step is validated.
"""
        return self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str, cache: bool = True) -> float:
        """
        对生成的推理过程进行综合评估，重点关注：
          - 逻辑树是否构建完备，各依赖关系是否明确，变量命名、数值及单位是否一致。
          - 数值计算、算术验证（正向、逆向和交叉检查）是否准确，是否对关键数值进行了反复核对。
          - 是否充分利用了多分支 TOT 思维而非单一 CoT，是否探讨了所有可能的逻辑分支（例如：针对苹果、货架、交易、折旧等问题）。
          - 最终答案是否清晰标注（含 "####" 标记）且符合题干要求。
          
        输出一个 0 到 1 之间的浮点数作为质量得分。
        """
        prompt = f"""You are a mathematical solution evaluator. Please assess the following reasoning process based on these criteria:

1. **Logical Structure (0.3 points):**
   - Is the logical tree fully constructed with all entities, variables, and dependencies clearly mapped?
   - Are time, sequential, and quantity relationships correctly identified and re-read from the problem statement?
   - Are variable names, units, and key numbers consistent throughout the process?

2. **Calculation and Arithmetic Validation (0.3 points):**
   - Are all arithmetic steps (forward, backward, cross-check) performed accurately?
   - Is there clear evidence of double-checking numbers and units for consistency (e.g., ensuring correct multiplication for "each" items)?

3. **TOT Branching and Reasoning Quality (0.3 points):**
   - Does the solution explore multiple branches instead of a single chain-of-thought?
   - Are alternative reasoning paths evaluated and integrated consistently, especially when ambiguity exists?

4. **Final Answer Clarity (0.1 points):**
   - Is the final answer clearly marked with "####"?
   - Are units, precision, and notation correct?

Question: {question}
Thought Process:
{thought}

Please output a single float from 0 to 1 representing the overall quality of this reasoning process.
"""
        response = self.chat_with_gpt(prompt)
        try:
            score = float(response[0].strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    def select_best_thoughts(self, thoughts: List[str], scores: List[float], k: int = 2) -> List[str]:
        """根据评分选择得分最高的 k 个思路。"""
        best_thoughts = [
            thought for thought, score in 
            sorted(zip(thoughts, scores), key=lambda x: x[1], reverse=True)[:k]
        ]
        return best_thoughts

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, 
              k_best_thoughts: int = 2) -> str:
        """
        采用 Tree-of-Thoughts 推理策略解决问题：
          - 每一步扩展多个分支（n_samples_per_step），
          - 评估后保留得分最高的 k_best_thoughts 分支，
          - 持续 max_steps 步或遇到包含最终答案 ("####") 的分支时停止搜索。
        """
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
            # 如果任一分支包含 "####" 最终答案标记，则提前结束搜索
            if any("####" in state[0] for state in states):
                break
        best_state = max(states, key=lambda x: x[1])
        return best_state[0]
