from typing import List, Dict, Any
import re
import logging

# Import the rich library for hierarchical printing
from rich.console import Console
from rich.tree import Tree

# Assume using the openai package to call the large model (modify as needed)
from openai import OpenAI
from utils.text_processing import extract_value


class TreeOfThoughts:
    def __init__(self, api_key: str,
                 base_url: str = "https://api.deepinfra.com/v1/openai",
                 model: str = "Qwen/Qwen2.5-7B-Instruct",
                 temperature: float = 0.7):
        """
        Initialize the TOT solver.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.thought_cache = {}  # Cache for evaluation results

    def chat_with_gpt(self, prompt: str, n: int = 1, stop: str = None) -> List[str]:
        """
        Call the large model API and return n responses.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=n,
            stop=stop
        )
        return [choice.message.content for choice in response.choices]

    def extract_final_answer(self, text: str) -> str:
        """
        Extract the final answer from the given text.
        
        Strategy:
         - Look for markers like "FINAL ANSWER:" or "####";
         - Use regex to strictly extract a pure number (allowing integers and decimals),
           while removing ellipses and other non-digit characters.
        """
        match = re.search(r"(FINAL ANSWER:|####)\s*(.+)", text)
        if match:
            answer_str = match.group(2).strip()
            answer_str = answer_str.replace("...", "").replace("…", "").strip()
            num_match = re.search(r"[-+]?\d*\.?\d+", answer_str)
            if num_match:
                return num_match.group(0)
        return ""

    def generate_thoughts(self, question: str, context: str = "", n_samples: int = 3) -> List[str]:
        """
        Generate candidate thoughts based on the question and context using a chain-of-thought strategy.
        
        参考示例中包含了正确案例与错误案例示范，帮助模型在生成推理时注意算式检查：
        
        --- 正确案例示范 ---
        Example 1:
        Question: John orders some pizzas to share with his friends. There are 20 friends in total, and John wants to make sure each can have 4 slices. Pizzas are only sold sliced into 8 portions. How many pizzas does John need to order?
        Reasoning:
        - Total slices needed = 20 * 4 = 80.
        - Pizzas needed = 80 / 8 = 10.
        FINAL ANSWER: 10

        --- 错误案例示范（计算错误示范） ---
        Example X (Error Case Demonstration):
        Question: Sally went to the seashore for vacation. Her parents gave her $10 to buy whatever she wanted. At the trinket shop, taffy was on sale for "Buy 1 pound at $3, get 1 pound 1/2 off." She scooped up 2 pounds. She also bought a mixed bag of seashells for $1.50 and 4 magnets that were $0.25 each. How much money does Sally have left?
        Candidate Reasoning (Incorrect):
        1. Determine the cost of taffy:
           - Promotional info: For 1.5 pounds, the price is $3 + $1.50 = $4.50.
           - Sally buys 2 pounds, so cost is computed as 2 * $3 = $6.
        2. Seashell cost: $1.50.
        3. Magnets cost: 4 * $0.25 = $1.
        4. Total cost = $6 + $1.50 + $1 = $8.50.
        5. Money left = $10 - $8.50 = $1.50.
        FINAL ANSWER: 1.5
        Issue:
        - The calculation for taffy cost is incorrect. It ignores that the promotional pricing should be applied as 1.5 pounds for $4.50. The correct approach is to calculate 1.5 pounds at $4.50 and the remaining 0.5 pounds at the half-price rate, which would result in a different total cost and thus a different remaining amount.
        
        For each sample, use one API call to generate the complete reasoning.
        """
        candidates = []
        reference_examples = """[Reference Examples]
Example 1:
Question: John orders some pizzas to share with his friends. There are 20 friends in total, and John wants to make sure each can have 4 slices. Pizzas are only sold sliced into 8 portions. How many pizzas does John need to order?
Reasoning:
- Total slices needed = 20 * 4 = 80.
- Pizzas needed = 80 / 8 = 10.
FINAL ANSWER: 10

Example 2:
Question: Marin and his neighbor Nancy each eat 4 apples a day. How many apples do they eat in 30 days?
Reasoning:
- Daily consumption for two people = 4 * 2 = 8.
- Over 30 days = 8 * 30 = 240.
FINAL ANSWER: 240

Example 3:
Question: Amalia, Megan, and Dior divided the home chores. Amalia mows the lawn in 4 hours, Megan takes 2 hours longer, and Dior takes over 4 hours longer than Amalia. Calculate the total time they took.
Reasoning:
- Amalia: 4 hours.
- Megan: 4 + 2 = 6 hours.
- Assume Dior takes 9 hours (minimum over 8 hours), so total = 4 + 6 + 9 = 19.
FINAL ANSWER: 19

Example 4:
Question: James needs to get more toys for his doggie shelter. Each dog needs one toy. James currently has 4 toys for 4 dogs, but there are 8 more dogs now. After buying the toys, he sees that there are twice as many more dogs than when he left, so he buys more toys. Then, when he comes back yet again, 3 dogs are gone so he no longer needs those toys. How many toys in total does James need?
Reasoning:
- Total num of toys need = Total num of dog he has
- Initial dogs: 4; additional dogs: 8 → Total dogs = 4 + 8 = 12.
- After First purchase, num of dogs = 12 + 12 * 2 = 36.
- After Second phase; 3 dog left, num of dogs = 36 - 3 = 33.
- The total of toys he need = Final num of dogs = 33
FINAL ANSWER: 33
"""
        for _ in range(n_samples):
            prompt = f"""You are a high-level problem-solving expert. Please provide a complete chain-of-thought reasoning for the following problem. Your response should:
1. Explain your understanding of the problem, breaking it into time phases if applicable.
2. Present a step-by-step solution with clear calculations.
3. Extract and check all arithmetic expressions to ensure that the calculations are consistent.
4. End with a final line strictly in the format:
   FINAL ANSWER: <pure number>
   
Original Question: {question}
Context: {context}
[Reference Examples]:
{reference_examples}
Your complete reasoning:"""
            reasoning = self.chat_with_gpt(prompt, n=1)[0].strip()
            candidates.append(reasoning)
        return candidates

    def check_calculations(self, candidate: str) -> List[str]:
        """
        Extract all arithmetic expressions in the candidate thought and execute them using Python.
        该函数查找形如 "<算式> = <结果>" 的表达式，并利用 eval 执行算式，
        如果计算结果与候选答案中的结果不一致，则返回对应的问题描述。
        """
        issues = []
        pattern = re.compile(r"(\d+(?:\s*[+\-*/]\s*\d+)+)\s*=\s*([-+]?\d+(?:\.\d+)?)")
        matches = pattern.findall(candidate)
        for expr, claimed in matches:
            try:
                computed = eval(expr)
            except Exception as e:
                issues.append(f"Error evaluating expression '{expr}': {e}")
                continue
            try:
                claimed_val = float(claimed)
            except Exception as e:
                issues.append(f"Invalid claimed result '{claimed}' for expression '{expr}'.")
                continue
            if abs(computed - claimed_val) > 1e-6:
                issues.append(f"Mismatch in expression '{expr}': computed {computed} but claimed {claimed_val}.")
        return issues

    def evaluate_thought(self, question: str, candidate: str, cache: bool = True) -> float:
        """
        Evaluate a single candidate thought.
        
        除了原有的逻辑评分之外，额外调用 check_calculations 对候选推理中的算式进行检查，
        如果存在计算错误，则降低评分（例如乘以 0.9）。
        
        此外，在评估提示中增加错误示范案例，帮助模型更好地识别算式问题。
        """
        if cache and candidate in self.thought_cache:
            return self.thought_cache[candidate]

        final_ans = extract_value(candidate)
        if not final_ans:
            score = 0.2  # 格式不符合要求
        else:
            error_demo = """For reference, consider the following error case demonstration:
            Question: Sally went to the seashore for vacation. Her parents gave her $10 to buy whatever she wanted. 
            At the trinket shop, taffy was on sale for "Buy 1 pound at $3, get 1 pound 1/2 off." She scooped up 2 pounds. 
            Candidate Reasoning (Incorrect):
            - Taffy cost computed as 2 * $3 = $6, ignoring the promotional pricing.
            - Total cost = $6 + $1.50 + $1 = $8.50.
            - Money left = $10 - $8.50 = $1.50.
            Issue: The promotional pricing should result in a different cost calculation.
            """
            eval_prompt = f"""
            You are a mathematical solution evaluator. Evaluate the following reasoning process based on these strict criteria:
            
            1. **Numerical Understanding (0.4 points):**
            - Are all numerical values in the question correctly identified and used?
            - Are the units and problem conditions properly tracked?
            - Is the sequence of events correctly interpreted?

            2. **Step-by-Step Logical Consistency (0.3 points):**
            - Is the reasoning process clearly divided into steps and are the calculations consistent at each step?

            3. **Final Answer Consistency (0.3 points):**
            - Is the final answer consistent with the logic and calculations presented throughout the reasoning?

            For reference, consider the following error case demonstration:
            {error_demo}

            Question: {question}
            Thought Process:
            {candidate}

            First, identify any numerical, arithmetic, or logical errors, then output a single float from 0 to 1.
            """
            try:
                evaluator_response = self.chat_with_gpt(eval_prompt, n=1)[0].strip()
                score_match = re.search(r"(\d+(\.\d+)?)", evaluator_response)
                if score_match:
                    score = float(score_match.group(1))
                else:
                    score = 0.2
            except Exception as e:
                logging.error(f"Evaluation error: {e}")
                score = 0.0

        # 调用计算检查函数，仅在此处进行惩罚
        calc_issues = self.check_calculations(candidate)
        if calc_issues:
            penalty_factor = 0.9 ** len(calc_issues)
            score *= penalty_factor

        if cache:
            self.thought_cache[candidate] = score
        return max(0.0, min(1.0, score))

    def self_supervised_validation(self, candidate: str) -> bool:
        """
        Self-supervised validation: Determine whether the candidate's final answer (the last line starting with "FINAL ANSWER:")
        is consistent with the preceding reasoning.
        """
        final_answer = self.extract_final_answer(candidate)
        if not final_answer:
            return False

        prompt = f"""Please check the reasoning process below and determine whether the final answer (the last line starting with "FINAL ANSWER:") is consistent with the logic and time-phase steps presented.
        Reasoning Process:
        {candidate}
        Please reply with only YES or NO."""
        try:
            response = self.chat_with_gpt(prompt, n=1)[0].strip().upper()
            return response == "YES"
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return False

    def reverse_analysis(self, question: str, candidate: str):
        """
        Reverse analysis:
        将最终候选答案带入原问题，检验其是否真正解决了问题。
        通过明确的步骤分析，验证答案的正确性和合理性。
        
        Returns:
            Tuple[bool, str]: (is_valid, response_text)
        """
        prompt = f"""You are a rigorous mathematical validator. Given a question and its candidate answer, analyze if the answer is logically sound by following these steps:

        1. First rewrite the answer as a mathematical equation:
        - Extract the final numerical answer
        - Write out the calculation formula used to reach this answer
        - List all the numbers and variables used

        2. Then validate each component:
        - Check if each number matches the original problem constraints
        - Verify if the mathematical operations are valid
        - Ensure the units are consistent
        - Confirm if the magnitude of the answer makes sense

        3. Finally identify any conflicts:
        - Look for contradictions between the numbers
        - Check if any assumptions violate the problem conditions
        - Verify if the answer satisfies all requirements

        --Example(Wrong Answer):------------
        Original Question: Kim sleeps 10pm-6am, with 23min sleepwalking (2:15-2:38am) and wakes 5min early. Minutes slept?
        Candidate Answer: 572
        <your analysis here>
        Analysis:
        - Formula used: 572 = 595 - 23 = (9h55m) - 23m = (9*60 + 55) - 23 = 10pm to 5:55am
        - Conflict found: Claims 10pm to 5:55am is 9h55m, but it's actually 7h55m
        - Therefore answer is invalid as it's based on incorrect time calculation
        </your analysis here>
        
        your response: NO
        -------------------------------------
        
        Now,your turn to analyze the following problem and candidate answer:
        Original Question: {question}
        Candidate Answer: {extract_value(candidate)}

        <your analysis here>
        your response: 

        Your response should Have YES or NO in the end."""
        
        try:
            response = self.chat_with_gpt(prompt, n=1)[0].strip().upper()
            is_valid = "YES" in response
            return is_valid, response
        except Exception as e:
            logging.error(f"Reverse analysis error: {e}")
            return False, f"Error occurred: {str(e)}"

    def refute_candidate(self, question: str, candidate: str) -> str:
        """
        Refute the candidate thought:
        详细分析答案的推理过程，找出其在问题理解、解题策略和计算过程上的缺陷。
        """
        prompt = f"""You are a critical thinking expert. First analyze the problem, then critique the solution:

    1. Problem Analysis:
    - What is being asked?
    - What information is given?
    - What information needs to be found?
    - Are there any potential traps or ambiguities?

    2. Solution Review:
    - Is the interpretation of the problem correct?
    - Is the solution strategy appropriate?
    - Are calculations accurate and justified?
    - Are there any logical gaps or contradictions?

    3. Improvement Suggestions:
    - What would be a better approach?
    - What key points were missed?
    - How could the solution be clearer?

    Original Question: {question}
    Candidate Solution: {candidate}

    Example:------------------------------------------
    Original Question: Marin and his neighbor Nancy each eat 4 apples a day. How many apples do they eat in 30 days?

    Candidate Solution:
    In one day, Marin and Nancy eat 4 + 1 = 5 apples.
    In 30 days, they eat 30 * 5 = <<30*5=150>>150 apples.
    #### 150

    Critique:
    1. Problem Analysis:
    Key Elements:
    - Two people: Marin and Nancy
    - Each person eats 4 apples per day
    - Time period: 30 days
    - Need to find: Total apples eaten by both people over 30 days
    Potential Trap:
    - May misinterpret "each eat 4 apples" as combined total

    2. Solution Review:
    ✗ Problem Interpretation:
    - Solution shows fundamental misunderstanding of "each eat 4 apples"
    - Incorrectly added 4 + 1 without any basis in the problem
    - Failed to recognize both people eat same amount

    ✗ Calculation Issues:
    - Wrong daily total: Used 5 instead of 8 (4 × 2)
    - Final answer 150 stems from initial misunderstanding
    - No explanation for where "+1" came from

    3. Improvement Suggestions:
    △ Clear Strategy:
    - First calculate daily total for both people: 4 × 2 = 8 apples/day
    - Then multiply by days: 8 × 30 = 240 apples total
    
    △ Better Presentation:
    - Show why each step is valid
    - Explain connection between steps
    - Verify answer aligns with problem conditions

    Please provide your critique following this format, with detailed analysis of both the problem and the solution:"""
        try:
            refutation = self.chat_with_gpt(prompt, n=1)[0].strip()
            return refutation
        except Exception as e:
            logging.error(f"Refutation error: {e}")
            return "Failed to obtain refutation feedback."

    def revise_candidate(self, question: str, candidate: str, refutation: str) -> str:
        """
        Revise the candidate thought based on the refutation feedback:
         - Use the feedback to comprehensively revise the candidate thought, improving its reasoning, including time-phase analysis and calculations.
         - The final line must strictly follow the format:
           FINAL ANSWER: <pure number>
        """
        prompt = f"""You are a high-level problem-solving expert. Below is a candidate thought along with refutation feedback.
        Please use the feedback to comprehensively revise the candidate thought, improving its reasoning and calculation steps (especially ensuring correct handling of time phases and arithmetic expressions). Provide the revised candidate thought. The final line must strictly follow the format:
        FINAL ANSWER: <pure number>
        Candidate Thought:
        {candidate}

        Refutation Feedback:
        {refutation}

        Please provide the revised candidate thought. If no revision is needed, you may leave it unchanged:"""
        try:
            revised = self.chat_with_gpt(prompt, n=1)[0].strip()
            return revised
        except Exception as e:
            logging.error(f"Revision error: {e}")
            return candidate

    def display_tree(self, infos: List[Dict[str, Any]], final_score: float) -> None:
        """
        Use the rich library to print the entire search process, showing each round's generated candidate thoughts,
        refutation feedback, revised results, evaluation scores, calculation issues, and selection results.
        """
        console = Console()
        root = Tree(f"[bold green]TOT Search Tree (Final Best Score: {final_score:.2f})[/bold green]")

        for info in infos:
            step = info.get('step')
            step_node = root.add(f"[bold blue]Round {step}[/bold blue]")
            prev_context = info.get('prev_context', "")
            if prev_context:
                step_node.add(f"[cyan]Current Solution:[/cyan] {prev_context}")

            gen_node = step_node.add("[magenta]Generated Candidate Thoughts[/magenta]")
            candidates = info.get('candidates', [])
            scores = info.get('scores', [])
            for idx, candidate in enumerate(candidates, start=1):
                score = scores[idx - 1] if idx - 1 < len(scores) else 0.0
                gen_node.add(f"[magenta]Candidate {idx}:[/magenta] {candidate.strip()[:200]}... (Score: {score:.2f})")

            calc_node = step_node.add("[red]Calculation Issues[/red]")
            calc_issues_list = info.get('calc_issues', [])
            for idx, issues in enumerate(calc_issues_list, start=1):
                if issues:
                    for issue in issues:
                        calc_node.add(f"[red]Candidate {idx} Issue:[/red] {issue}")
                else:
                    calc_node.add(f"[red]Candidate {idx} Issue:[/red] None")

            ref_node = step_node.add("[red]Refutation Feedback[/red]")
            for idx, ref in enumerate(info.get('refutations', []), start=1):
                ref_node.add(f"[red]Candidate {idx} Refutation:[/red] {ref.strip()[:200]}...")

            rev_node = step_node.add("[blue]Revised Candidate Thoughts[/blue]")
            for idx, candidate in enumerate(info.get('revised', []), start=1):
                rev_node.add(f"[blue]Candidate {idx} Revised:[/blue] {candidate.strip()[:200]}...")

            eval_node = step_node.add("[yellow]Evaluation Scores[/yellow]")
            for idx, score in enumerate(scores, start=1):
                eval_node.add(f"[yellow]Candidate {idx} Score: {score:.2f}[/yellow]")

            sel = info.get('selected', [])
            sel_node = step_node.add("[green]Selected Candidates for Next Round[/green]")
            for idx, candidate in enumerate(sel, start=1):
                sel_node.add(f"[green]Candidate {idx}:[/green] {candidate.strip()[:200]}...")
        console.print(root)

    def solve(self, question: str, max_steps: int = 6, n_samples_per_step: int = 3,
              k_best_thoughts: int = 2, score_threshold: float = 1) -> str:
        """
        Solve using the TOT framework in rounds. Each round consists of:
          1. Generation: Generate candidate thoughts based on the current state.
          2. Evaluation: Score all candidate thoughts (并额外检查候选中的算式是否计算正确).
          3. Selection: Choose the top candidates based on the scores.
          4. Refutation + Revision: Provide refutation feedback and generate revised candidates.
          5. Use the revised candidates as the next round’s state, iterating until a candidate exceeds the score_threshold or max_steps is reached.
          6. After obtaining a final candidate answer, perform reverse analysis by feeding the answer back into the original question.
             If the reverse analysis indicates the answer is incorrect, roll back the question and current answer into additional TOT rounds.
        """
        infos = []  # Record information for each round
        states = [{"chain": "", "score": 1.0}]
        best_solution = None
        best_score = 0.0

        for step in range(max_steps):
            print(f"====== Round {step + 1} ======")
            new_candidates = []
            candidate_scores = []
            calc_issues_all = []  # 用于记录每个候选的计算检查问题
            refutations = []
            revised_candidates = []

            # Generation phase: For each state, generate expanded candidates.
            for state in states:
                current_solution = state["chain"]
                context = f"Original Question: {question}\nCurrent Solution: {current_solution}"
                candidates = self.generate_thoughts(question, context, n_samples=n_samples_per_step)
                new_candidates.extend(candidates)
            if not new_candidates:
                break

            # Evaluation phase: Score all generated candidates.
            for cand in new_candidates:
                score = self.evaluate_thought(question, cand)
                # 如果候选中包含 FINAL ANSWER 标记，但自监督验证未通过，则额外降低评分
                if ("FINAL ANSWER:" in cand or "####" in cand) and not self.self_supervised_validation(cand):
                    score *= 0.9
                candidate_scores.append(score)
                calc_issues_all.append(self.check_calculations(cand))

            # Early stopping: Terminate if all candidate scores are lower than the current best.
            if new_candidates and candidate_scores and max(candidate_scores) < best_score:
                print("Early stopping triggered: All candidate scores are lower than the current best score. Terminating search early.")
                break

            # Selection phase: Choose the top candidates.
            idxs = sorted(range(len(new_candidates)), key=lambda i: candidate_scores[i], reverse=True)[:k_best_thoughts]
            selected = [new_candidates[i] for i in idxs]

            # Refutation phase: Provide refutation feedback for the top candidates.
            for i in idxs:
                refutation = self.refute_candidate(question, new_candidates[i])
                refutations.append(refutation)

            # Revision phase: Revise each top candidate based on refutation feedback.
            for i, ref in zip(idxs, refutations):
                cand = new_candidates[i]
                revised = self.revise_candidate(question, cand, ref)
                revised_score = self.evaluate_thought(question, revised)
                if revised_score > candidate_scores[i]:
                    candidate_scores[i] = revised_score
                    new_candidates[i] = revised
                    revised_candidates.append(revised)
                else:
                    revised_candidates.append(cand)

            # Save round information.
            infos.append({
                "step": step + 1,
                "prev_context": states[0]["chain"],
                "candidates": new_candidates,
                "calc_issues": calc_issues_all,
                "refutations": refutations,
                "revised": revised_candidates,
                "scores": candidate_scores,
                "selected": [new_candidates[i] for i in idxs]
            })

            # Update state with the selected candidates.
            states = []
            for i in idxs:
                states.append({"chain": new_candidates[i], "score": candidate_scores[i]})
                if candidate_scores[i] > best_score:
                    best_score = candidate_scores[i]
                    best_solution = new_candidates[i]
            if best_solution and best_score >= score_threshold:
                break

        # 逆向验证阶段：将最终答案带入原问题进行验证
        if best_solution:
            reverse_ok, message = self.reverse_analysis(question, best_solution)
            Console().print(f"[bold]Reverse Analysis Result:[/bold] {message}")
            extra_round = 0
            # 若逆向验证不通过，则回溯问题与当前答案，进行额外的TOT轮次修正（最多额外3轮）
            while not reverse_ok and extra_round < 3:
                print("Reverse analysis failed: The final answer appears incorrect. Rolling back into TOT for further revision.")
                new_candidates = []
                candidate_scores = []
                for state in [{"chain": best_solution, "score": best_score}]:
                    context = f"Original Question: {question}\nCurrent Answer: {state['chain']}\nPlease re-evaluate the above answer and provide a revised chain-of-thought reasoning that correctly solves the problem."
                    candidates = self.generate_thoughts(question, context, n_samples=1)
                    new_candidates.extend(candidates)
                if not new_candidates:
                    break
                for cand in new_candidates:
                    score = self.evaluate_thought(question, cand)
                    if ("FINAL ANSWER:" in cand or "####" in cand) and not self.self_supervised_validation(cand):
                        score *= 0.9
                    candidate_scores.append(score)
                idx = candidate_scores.index(max(candidate_scores))
                best_solution = new_candidates[idx]
                best_score = candidate_scores[idx]
                reverse_ok = self.reverse_analysis(question, best_solution)
                extra_round += 1
            if not reverse_ok:
                print("Reverse analysis still fails after additional rounds.")

        self.display_tree(infos, best_score)
        if best_solution:
            return best_solution
        if states:
            best_state = max(states, key=lambda s: s["score"])
            return best_state["chain"]
        return "No valid solution found."


# Example usage of the updated code
if __name__ == "__main__":
    # Replace with your own API_KEY.
    API_KEY = "your_api_key_here"
    tot_solver = TreeOfThoughts(api_key=API_KEY)
    question = ("James needs to get more toys for his doggie shelter. Each dog needs one toy. "
                "James currently has 4 toys for 4 dogs, but there are 8 more dogs now. After buying the toys, "
                "he went back to see that there are twice as many more dogs than when he left so he had to buy more toys. "
                "When James came back yet again, 3 dogs were gone so he no longer needed those toys. "
                "How many toys in total does James need?")
    solution = tot_solver.solve(
        question,
        max_steps=6,
        n_samples_per_step=3,
        k_best_thoughts=2,
        score_threshold=0.9
    )
    print("\nFinal reasoning process returned:")
    print(solution)
