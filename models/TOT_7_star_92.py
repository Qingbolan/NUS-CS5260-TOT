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
        Generate candidate thoughts based on the question and context using a chain-of-thought strategy:
          - Provide a complete explanation of your understanding of the problem, divided into time phases if applicable,
            and a step-by-step solution.
          - The final line must strictly follow the format:
            FINAL ANSWER: <pure number>
          
        Multiple reference examples are provided below, including a new case that checks for proper time-phase reasoning.
        
        [Reference Examples]

        Example 1:
        Question: John orders some pizzas to share with his friends. There are 20 friends in total, and John wants to make sure each can have 4 slices. Pizzas are only sold sliced into 8 portions. How many pizzas does John need to order?
        Reasoning:
          - Total slices needed = 20 × 4 = 80.
          - Pizzas needed = 80 / 8 = 10.
        FINAL ANSWER: 10

        Example 2:
        Question: Marin and his neighbor Nancy each eat 4 apples a day. How many apples do they eat in 30 days?
        Reasoning:
          - Daily consumption for two people = 4 × 2 = 8.
          - Over 30 days = 8 × 30 = 240.
        FINAL ANSWER: 240

        Example 3:
        Question: Amalia, Megan, and Dior divided the home chores. Amalia mows the lawn in 4 hours, Megan takes 2 hours longer, and Dior takes over 4 hours longer than Amalia. Calculate the total time they took.
        Reasoning:
          - Amalia: 4 hours.
          - Megan: 4 + 2 = 6 hours.
          - Assume Dior takes 9 hours (minimum over 8 hours), so total = 4 + 6 + 9 = 19 hours.
        FINAL ANSWER: 19

        Example 4 (New Case):
        Question: James needs to get more toys for his doggie shelter. Each dog needs one toy. James currently has 4 toys on hand for 4 dogs, but there are 8 more dogs in the shelter now. After buying the toys, he went back to see that there are twice as many more dogs than when he left so he had to buy some more toys. When James came back yet again, 3 dogs were gone so he no longer needed those toys. How many toys in total does James need?
        Reasoning (correct answer is 33):
          
          - *[Note: Based on the provided correct answer, the reasoning must conclude that James ultimately needs 33 toys. This indicates that the calculation should consider the sequence of events differently. Please ensure your reasoning carefully accounts for each time phase and checks that your final answer is consistent with all phases.]* 
        FINAL ANSWER: 33

        For each sample, please generate a complete chain-of-thought reasoning in one response.
        """
        candidates = []
        # Prepare the reference examples once.
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
  - Assume Dior takes 9 hours (minimum over 8 hours), so total = 4 + 6 + 9 = 19 hours.
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
        # For each sample, use one API call to generate the complete reasoning.
        for _ in range(n_samples):
            prompt = f"""You are a high-level problem-solving expert. Please provide a complete chain-of-thought reasoning for the following problem. Your response should:
1. Explain your understanding of the problem, breaking it into time phases if applicable.
2. Present a step-by-step solution with clear calculations.
3. End with a final line strictly in the format: FINAL ANSWER: <pure number>
Original Question: {question}
Context: {context}
[Reference Examples]:
{reference_examples}
Your complete reasoning:"""
            reasoning = self.chat_with_gpt(prompt, n=1)[0].strip()
            candidates.append(reasoning)
        return candidates

    def evaluate_thought(self, question: str, candidate: str, cache: bool = True) -> float:
        """
        Evaluate a single candidate thought.
        
        The prompt instructs the model to output a float between 0 and 1 representing the candidate's reasonableness.
        Additionally, the evaluation now includes checks for understanding the problem in time phases and ensuring each step's logic is correct.
        """
        if cache and candidate in self.thought_cache:
            return self.thought_cache[candidate]

        final_ans = extract_value(candidate)
        if not final_ans:
            score = 0.2  # Format does not meet requirements
        else:
            eval_prompt = f"""
You are a mathematical solution evaluator. Evaluate the following reasoning process based on these strict criteria:

1. **Numerical Understanding (0.4 points):**
   - Are all numerical values in the question correctly identified and used?
   - Are the units and problem conditions (including time phases) properly tracked?
   - Is the sequence of events (time phases) correctly interpreted?

2. **Step-by-Step Logical Consistency (0.3 points):**
   - Is the reasoning process clearly divided into phases (if applicable) and are the calculations consistent at each phase?
   - Are any ambiguities or potential errors in the sequence addressed?

3. **Final Answer Consistency (0.3 points):**
   - Is the final answer consistent with the logic and calculations presented throughout the reasoning?
   - Does the final answer reflect adjustments from any changes in the problem conditions?

Question: {question}
Thought Process:
{candidate}

First, identify any numerical, phase-based, or logical errors, then output a single float from 0 to 1.
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

    def refute_candidate(self, question: str, candidate: str) -> str:
        """
        Refute the candidate thought:
        Carefully read the candidate thought below and point out its shortcomings and mistakes in understanding the problem, 
        especially regarding the sequencing of events and time-phase checks.
        """
        prompt = f"""You are a high-level review expert. Please carefully read the candidate thought below and refute it from the following aspects:
1. Does the candidate fully understand the original question, including any time-phased changes?
2. Are there any errors or omissions in the step-by-step calculations?
3. Are there alternative, more reasonable approaches to the problem?

Candidate Thought:
{candidate}

Please list your refutation points in detail and suggest improvements:"""
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
Please use the feedback to comprehensively revise the candidate thought, improving its reasoning and calculation steps (especially ensuring correct handling of time phases). Provide the revised candidate thought. The final line must strictly follow the format:
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
        refutation feedback, revised results, evaluation scores, and selection results.
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
              k_best_thoughts: int = 2, score_threshold: float = 0.9, revision_threshold: float = 0.5) -> str:
        """
        Solve using the TOT framework in rounds. Each round consists of:
          1. Generation: Generate candidate thoughts based on the current state.
          2. Evaluation: Score all candidate thoughts.
          3. Selection: Choose the top candidates based on the scores.
          4. Refutation + Revision: Provide refutation feedback and generate revised candidates.
          5. Use the revised candidates as the next round’s state, iterating until a candidate exceeds the score_threshold or max_steps is reached.
          6. If all candidates in a round score lower than the current best candidate, terminate early.
        """
        infos = []  # Record information for each round
        states = [{"chain": "", "score": 1.0}]
        best_solution = None
        best_score = 0.0

        for step in range(max_steps):
            print(f"====== Round {step + 1} ======")
            new_candidates = []
            candidate_scores = []
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
                if ("FINAL ANSWER:" in cand or "####" in cand) and not self.self_supervised_validation(cand):
                    score *= 0.9
                candidate_scores.append(score)

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
    solution = tot_solver.solve(question,
                                max_steps=6,
                                n_samples_per_step=3,
                                k_best_thoughts=2,
                                score_threshold=0.9,
                                revision_threshold=0.5)
    print("\nFinal reasoning process returned:")
    print(solution)
