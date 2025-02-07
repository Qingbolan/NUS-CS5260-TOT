from typing import List, Dict, Any, Optional, Set, Tuple
import re
import logging
from dataclasses import dataclass
from rich.console import Console
from rich.tree import Tree
from openai import OpenAI


@dataclass
class ProblemContext:
    time_related: bool
    multi_step: bool
    units: List[str]
    key_numbers: List[str]
    operations: List[str]


class TreeOfThoughts:
    def __init__(self, api_key: str,
                 base_url: str = "https://api.deepinfra.com/v1/openai",
                 model: str = "Qwen/Qwen2.5-7B-Instruct",
                 temperature: float = 0.7):
        """Initialize the TOT solver."""
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.thought_cache = {}  # Cache for evaluation results

    def chat_with_gpt(self, prompt: str, n: int = 1, stop: str = None) -> List[str]:
        """Call the large model API and return n responses."""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=n,
            stop=stop
        )
        return [choice.message.content for choice in response.choices]

    def extract_final_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from the given text."""
        match = re.search(r"(FINAL ANSWER:|####)\s*(.+)", text)
        if match:
            answer_str = match.group(2).strip()
            answer_str = answer_str.replace("...", "").replace("…", "").strip()
            num_match = re.search(r"[-+]?\d*\.?\d+", answer_str)
            if num_match:
                return num_match.group(0)
        return None

    def check_calculations(self, candidate: str) -> List[str]:
        """Enhanced arithmetic expression checker."""
        issues = []
        patterns = [
            # Basic expressions: "a + b = c"
            r"(\d+(?:\s*[+\-*/]\s*\d+)+)\s*=\s*([-+]?\d+(?:\.\d+)?)",
            # Parenthesized expressions: "(a + b) * c = d"
            r"(\(?\d+(?:\s*[+\-*/]\s*\d+\)?)+)\s*=\s*([-+]?\d+(?:\.\d+)?)",
            # Chained calculations: "a * b = c, c + d = e"
            r"(\d+(?:\s*[+\-*/]\s*\d+)+)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,\s*(\2(?:\s*[+\-*/]\s*\d+)+)\s*=\s*([-+]?\d+(?:\.\d+)?)"
        ]
        
        all_numbers: Set[float] = set()
        intermediate_results: Dict[str, float] = {}
        
        for pattern in patterns:
            matches = re.finditer(pattern, candidate)
            for match in matches:
                groups = match.groups()
                for i in range(0, len(groups), 2):
                    expr, claimed = groups[i:i+2]
                    if not expr or not claimed:
                        continue
                    
                    try:
                        # Safely evaluate the expression
                        expr = expr.replace('^', '**').strip('()')
                        computed = float(eval(expr))
                        claimed_val = float(claimed)
                        
                        # Store intermediate results
                        intermediate_results[claimed] = computed
                        all_numbers.add(computed)
                        
                        # Check calculation precision
                        if abs(computed - claimed_val) > 1e-6:
                            issues.append(
                                f"Calculation error in '{expr}': computed {computed:.2f} "
                                f"but claimed {claimed_val:.2f}"
                            )
                            
                        # Check unit consistency if units are present
                        if any(unit in expr for unit in ['km', 'm', 'kg', '$']):
                            units = re.findall(r'(km|m|kg|\$)', expr)
                            if len(set(units)) > 1:
                                issues.append(f"Unit inconsistency in '{expr}'")
                                
                    except Exception as e:
                        issues.append(f"Error evaluating '{expr}': {str(e)}")
        
        # Validate final answer against intermediate calculations
        final_answer = self.extract_final_answer(candidate)
        if final_answer:
            try:
                final_val = float(final_answer)
                if final_val not in all_numbers and \
                   not any(abs(final_val - v) < 1e-6 for v in all_numbers):
                    issues.append(
                        f"Final answer {final_val} doesn't match any intermediate result"
                    )
            except ValueError:
                issues.append("Invalid final answer format")
                
        return issues

    def validate_context_consistency(self, question: str, reasoning: str) -> List[str]:
        """Validate reasoning consistency with the original question context."""
        issues = []
        
        # Extract key elements from question
        numbers_in_question = set(re.findall(r"[-+]?\d*\.?\d+", question))
        numbers_in_reasoning = set(re.findall(r"[-+]?\d*\.?\d+", reasoning))
        
        # Check if all question numbers are used
        unused_numbers = numbers_in_question - numbers_in_reasoning
        if unused_numbers:
            issues.append(f"Numbers from question not used: {', '.join(unused_numbers)}")
        
        # Check calculation step completeness
        steps = [s.strip() for s in reasoning.split('\n') if s.strip()]
        calculation_steps = [
            s for s in steps 
            if re.search(r"[-+]?\d*\.?\d+\s*[+\-*/]\s*[-+]?\d*\.?\d+", s)
        ]
        
        if len(calculation_steps) < len(numbers_in_question) - 1:
            issues.append(
                f"Found {len(calculation_steps)} calculation steps, "
                f"expected at least {len(numbers_in_question) - 1}"
            )
        
        # Check step sequence
        if not any(s.lower().startswith(('step', 'first', '1.', '1)')) for s in steps):
            issues.append("Missing explicit step sequence indicators")
            
        return issues

    def analyze_problem(self, question: str) -> ProblemContext:
        """Analyze the problem to extract key features and constraints."""
        numbers = re.findall(r"[-+]?\d*\.?\d+", question)
        units = re.findall(r"(km|m|kg|g|\$)", question)
        operations = re.findall(r"(add|subtract|multiply|divide|increase|decrease|total|sum|difference)", 
                              question.lower())
        
        return ProblemContext(
            time_related=any(word in question.lower() 
                           for word in ['day', 'week', 'month', 'year']),
            multi_step=len(numbers) > 2,
            units=units,
            key_numbers=numbers,
            operations=operations
        )

    def generate_thoughts(self, 
                         question: str, 
                         context: str = "", 
                         problem_context: Optional[ProblemContext] = None,
                         n_samples: int = 3) -> List[str]:
        """Generate candidate thoughts with enhanced validation."""
        if problem_context is None:
            problem_context = self.analyze_problem(question)
            
        checkpoints_prompt = """
        During your solution process, you MUST:
        1. List all numbers from the question and their meanings
        2. Verify each calculation with clear intermediate steps
        3. Check units consistency throughout
        4. Validate final answer against intermediate results
        
        Format each step as:
        Step X: [Purpose]
        - Given values: [list relevant numbers]
        - Calculation: <expression> = <result>
        - Unit: [unit]
        - Verification: [how this connects to previous/next step]
        """
        
        error_examples = """
        Common Error Examples:
        1. Missing Steps Error:
           Question: If 5 people each need 3 items per day for 4 days...
           Wrong: 3 * 4 = 12 items
           Correct: (5 people * 3 items) * 4 days = 60 items
        
        2. Unit Consistency Error:
           Question: Convert 2.5 km to meters...
           Wrong: 2.5 * 100 = 250 meters
           Correct: 2.5 * 1000 = 2500 meters
        
        3. Intermediate Result Error:
           Question: Calculate 15% of $80 then add $20
           Wrong: 80 * 0.15 = 12 + 20 = 32
           Correct: 80 * 0.15 = 12, 12 + 20 = 32
        """
        
        base_prompt = f"""You are a mathematical reasoning expert. Analyze and solve:

        Question: {question}
        Context: {context}
        
        Problem Features:
        - Time-related: {problem_context.time_related}
        - Multi-step: {problem_context.multi_step}
        - Units present: {', '.join(problem_context.units) if problem_context.units else 'None'}
        - Key numbers: {', '.join(problem_context.key_numbers)}
        - Operations: {', '.join(problem_context.operations)}
        
        {checkpoints_prompt}
        
        {error_examples}
        
        Provide a complete, step-by-step solution following the checkpoints above.
        End with: FINAL ANSWER: <number>"""
        
        candidates = []
        for _ in range(n_samples):
            reasoning = self.chat_with_gpt(base_prompt, n=1)[0].strip()
            candidates.append(reasoning)
            
        return candidates

    def evaluate_thought(self, 
                        question: str, 
                        candidate: str, 
                        cache: bool = True) -> Tuple[float, List[str]]:
        """Evaluate a candidate thought with detailed feedback."""
        if cache and candidate in self.thought_cache:
            return self.thought_cache[candidate]
            
        issues = []
        
        # Check calculations
        calc_issues = self.check_calculations(candidate)
        issues.extend(calc_issues)
        
        # Check context consistency
        context_issues = self.validate_context_consistency(question, candidate)
        issues.extend(context_issues)
        
        # Extract final answer
        final_answer = self.extract_final_answer(candidate)
        if not final_answer:
            issues.append("Missing or invalid final answer format")
            score = 0.2
        else:
            eval_prompt = f"""
            Evaluate this mathematical solution based on:
            
            1. Numerical Accuracy (0.4 points):
               - Calculations are precise
               - Units are consistent
               - All question values are used
            
            2. Step Completeness (0.3 points):
               - Each calculation has its own step
               - No skipped or combined steps
               - Clear progression
            
            3. Answer Verification (0.3 points):
               - Final answer matches steps
               - Units are correct
               - Format is proper
            
            Question: {question}
            Solution: {candidate}
            
            Found Issues: {'; '.join(issues) if issues else 'None'}
            
            Output a score from 0-1:"""
            
            try:
                response = self.chat_with_gpt(eval_prompt, n=1)[0].strip()
                score_match = re.search(r"(\d+(\.\d+)?)", response)
                score = float(score_match.group(1)) if score_match else 0.2
            except Exception as e:
                logging.error(f"Evaluation error: {str(e)}")
                score = 0.0

        # Adjust score based on issues
        score *= max(0.1, 1.0 - (len(issues) * 0.1))
        score = max(0.0, min(1.0, score))
        
        if cache:
            self.thought_cache[candidate] = (score, issues)
            
        return score, issues

    def solve(self, 
             question: str, 
             max_steps: int = 6,
             n_samples_per_step: int = 3,
             k_best_thoughts: int = 2,
             score_threshold: float = 0.95) -> str:
        """Solve using enhanced TOT framework with validation."""
        # Initial problem analysis
        problem_context = self.analyze_problem(question)
        
        # Pre-check for potential issues
        if not problem_context.key_numbers:
            return "Error: No numbers found in question"
            
        if len(set(problem_context.units)) > 1:
            logging.warning("Multiple units detected, ensuring conversion steps")
            
        infos = []  # Record information for each round
        states = [{"chain": "", "score": 1.0, "issues": []}]
        best_solution = None
        best_score = 0.0
        
        for step in range(max_steps):
            print(f"====== Round {step + 1} ======")
            
            # Generate and evaluate candidates
            new_candidates = []
            scores_and_issues = []
            
            for state in states:
                current_solution = state["chain"]
                context = f"Previous solution: {current_solution}" if current_solution else ""
                candidates = self.generate_thoughts(
                    question, 
                    context, 
                    problem_context,
                    n_samples=n_samples_per_step
                )
                new_candidates.extend(candidates)
                
            if not new_candidates:
                break
                
            # Evaluate all candidates
            for cand in new_candidates:
                score, issues = self.evaluate_thought(question, cand)
                scores_and_issues.append((score, issues))
                
            # Early stopping check
            max_current_score = max(score for score, _ in scores_and_issues)
            if max_current_score < best_score:
                print("Early stopping: All candidates scored lower than best")
                break
                
            # Select top candidates
            sorted_idx = sorted(
                range(len(new_candidates)),
                key=lambda i: scores_and_issues[i][0],
                reverse=True
            )[:k_best_thoughts]
            
            # Update states and track best solution
            states = []
            for idx in sorted_idx:
                score, issues = scores_and_issues[idx]
                states.append({
                    "chain": new_candidates[idx],
                    "score": score,
                    "issues": issues
                })
                if score > best_score:
                    best_score = score
                    best_solution = new_candidates[idx]

            # Save round information 
            infos.append({
                "step": step + 1,
                "prev_context": states[0]["chain"] if len(states) > 0 else "",
                "candidates": new_candidates,
                "scores_and_issues": scores_and_issues,
                "selected": [new_candidates[i] for i in sorted_idx]
            })

            # Check if we've found a good enough solution
            if best_score >= score_threshold:
                break

        # Display solution tree
        self.display_tree(infos, best_score)
        
        # Return best solution found
        if best_solution:
            return best_solution
        if states:
            return max(states, key=lambda s: s["score"])["chain"]
        return "No valid solution found."

    def display_tree(self, infos: List[Dict[str, Any]], final_score: float) -> None:
        """Display the search process using rich tree visualization."""
        console = Console()
        root = Tree(f"[bold green]TOT Search Tree (Final Score: {final_score:.2f})[/bold green]")

        for info in infos:
            step = info.get('step')
            step_node = root.add(f"[bold blue]Round {step}[/bold blue]")
            
            # Show previous context if any
            prev_context = info.get('prev_context', "")
            if prev_context:
                step_node.add(f"[cyan]Previous Solution:[/cyan] {prev_context[:200]}...")

            # Show generated candidates
            gen_node = step_node.add("[magenta]Generated Candidates[/magenta]")
            candidates = info.get('candidates', [])
            scores_and_issues = info.get('scores_and_issues', [])
            
            for idx, (candidate, (score, issues)) in enumerate(zip(candidates, scores_and_issues), 1):
                cand_node = gen_node.add(
                    f"[magenta]Candidate {idx}[/magenta] (Score: {score:.2f})"
                )
                cand_node.add(candidate[:200] + "...")
                
                if issues:
                    issues_node = cand_node.add("[red]Issues Found:[/red]")
                    for issue in issues:
                        issues_node.add(f"[red]- {issue}[/red]")

            # Show selected candidates
            sel_node = step_node.add("[green]Selected for Next Round[/green]")
            selected = info.get('selected', [])
            for idx, candidate in enumerate(selected, 1):
                sel_node.add(f"[green]Selected {idx}:[/green] {candidate[:200]}...")

        console.print(root)


def main():
    """Example usage of the improved TOT solver."""
    API_KEY = "your_api_key_here"  # Replace with your actual API key
    
    # Initialize the solver
    tot_solver = TreeOfThoughts(api_key=API_KEY)
    
    # Example question that previously had calculation errors
    question = ("It is approximately 1955 kilometers from San Diego, California to New York City, "
               "New York. If Bernice drove 325 kilometers for 4 days, how many kilometers will "
               "she still need to drive?")
    
    # Solve with enhanced validation
    solution = tot_solver.solve(
        question=question,
        max_steps=6,
        n_samples_per_step=3,
        k_best_thoughts=2,
        score_threshold=0.9
    )
    
    print("\nFinal solution:")
    print(solution)


if __name__ == "__main__":
    main()