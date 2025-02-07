from typing import List, Tuple, Dict
from openai import OpenAI
import re
import logging
from dataclasses import dataclass

@dataclass
class ThoughtState:
    content: str
    score: float
    step: int

class TreeOfThoughts:
    def __init__(self, api_key: str, 
                 base_url: str = "https://api.deepinfra.com/v1/openai",
                 model: str = "Qwen/Qwen2.5-7B-Instruct", 
                 temperature: float = 0.7):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.thought_cache: Dict[str, float] = {}
        self.quality_threshold = 0.5
        # 每一步的说明只包含对应步骤的格式要求，不涉及其他步骤信息
        self.step_instructions = {
            1: ("Express as a PDDL problem in concise form. "
                "Output exactly in this format:\n\n"
                "I am solving the problem, 注意要清晰明了，注意与原问题一句一句对应\n```pddl\n<your PDDL representation>\n```"),
            2: ("Present your step-by-step solution concisely. "
                "Output exactly in this format:\n\n"
                "MY solution is\n1. [First step]\n2. [Second step]\n3. [Third step]"),
            3: ("Continue any additional concise reasoning if needed. "
                "Keep your output direct and focused on current reasoning."),
            4: ("Output your final answer concisely.and also check the answer and question. "
                "Output exactly in this format:\n\n"
                "my answer is\n### [result]")
        }

    def chat_with_gpt(self, prompt: str, n: int = 1) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                n=n
            )
            return [msg.message.content for msg in response.choices]
        except Exception as e:
            logging.error(f"API call error: {e}")
            return []

    def generate_thoughts(self, question: str, current_state: ThoughtState, n_samples: int = 3) -> List[str]:
        thoughts = []
        attempts = 0
        max_attempts = n_samples
        
        # 仅包含当前步骤的说明，不涉及其他步骤格式信息
        prompt = f"""Step {current_state.step} of math solution.
Previous result: {current_state.content if current_state.content else 'None'}

Instruction: {self.step_instructions.get(current_state.step, self.step_instructions[4])}

Important: Provide only a concise and direct result for this step according to the above instruction.

Question: {question}"""

        while len(thoughts) < n_samples and attempts < max_attempts:
            responses = self.chat_with_gpt(prompt, n=1)
            if not responses:
                attempts += 1
                continue

            thought = responses[0].strip()
            combined = (f"{current_state.content.strip()}\n{thought}" 
                        if current_state.content.strip() else thought)
            score = self.evaluate_thought(question, combined, current_state.step)
            if score >= self.quality_threshold:
                thoughts.append(thought)
            attempts += 1

        return thoughts or self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str, step: int) -> float:
        if thought in self.thought_cache:
            return self.thought_cache[thought]

        criteria = {
            1: ("""1. PDDL Format (0.4):
- Correct syntax and concise definitions
2. Logical Completeness (0.6):
- Includes necessary relationships"""),
            
            2: ("""1. Step-by-Step Format (0.4):
- Follows the structure: 'MY solution is' followed by steps
2. Calculation Accuracy (0.6):
- Correct arithmetic steps"""),
            3: ("""1. Continuity (0.5):
- Connects with previous steps
2. Relevance (0.5):
- Advances the solution"""),
            4: ("""1. Final Answer Format (0.4):
- Output exactly in the format: 'my answer is' with a line '### [result]'
2. Answer Accuracy (0.6):
- Final answer is correct""")
        }

        prompt_eval = f"""Rate solution step {step} based on the following criteria:
{criteria.get(step, criteria[4])}

Question: {question}
Solution:
{thought}

Output score (0-1):"""
        try:
            response = self.chat_with_gpt(prompt_eval)[0]
            score_match = re.search(r"(\d*\.?\d+)", response)
            score = float(score_match.group(1)) if score_match else 0.0
            score = max(0.0, min(1.0, score))
            self.thought_cache[thought] = score
            return score
        except Exception as e:
            logging.error(f"Evaluation error: {e}")
            return 0.0

    def extract_final_answer(self, thought: str) -> str:
        patterns = [
            r"FINAL ANSWER:\s*(.+)",
            r"####\s*(.+)",
            r"my answer is\s*\n\s*###\s*(.+)"
        ]
        for pattern in patterns:
            if match := re.search(pattern, thought, re.IGNORECASE):
                return match.group(1).strip()
        return ""

    def validate_solution(self, thought: str) -> bool:
        if not self.extract_final_answer(thought):
            return False

        prompt = f"""Verify solution consistency:
{thought}

Answer YES/NO:"""
        try:
            return self.chat_with_gpt(prompt)[0].strip().upper() == "YES"
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return False

    def solve(self, question: str, max_steps: int = 8, n_samples_per_step: int = 3, k_best_thoughts: int = 2) -> str:
        states = [ThoughtState("", 1.0, 1)]
        best_solution = None
        best_score = 0.0

        for step in range(max_steps):
            current_step = step + 1
            new_states = []

            for state in states:
                if state.score < best_score * 0.7:
                    continue

                thoughts = self.generate_thoughts(question, state, n_samples_per_step)

                for thought in thoughts:
                    combined = (f"{state.content.strip()}\n{thought.strip()}" 
                                if state.content.strip() else thought.strip())
                    score = self.evaluate_thought(question, combined, current_step)

                    if current_step >= 5:
                        if self.extract_final_answer(combined):
                            if not self.validate_solution(combined):
                                score *= 0.8
                            if score > best_score:
                                best_score = score
                                best_solution = combined

                    new_states.append(ThoughtState(combined, score, current_step + 1))

            if not new_states:
                break

            states = sorted(new_states, key=lambda x: x.score, reverse=True)[:k_best_thoughts]

            if best_solution and best_score > 0.9:
                break

            if all(s.score < 0.3 for s in states):
                break

        return best_solution or max(states, key=lambda x: x.score).content