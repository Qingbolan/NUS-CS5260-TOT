from typing import List, Dict, Any
import re
import logging

# 导入 rich 库用于分层打印
from rich.console import Console
from rich.tree import Tree

# 假设使用 openai 包调用大模型（请根据实际情况修改）
from openai import OpenAI
from utils.text_processing import extract_value


class TreeOfThoughts:
    def __init__(self, api_key: str,
                 base_url: str = "https://api.deepinfra.com/v1/openai",
                 model: str = "Qwen/Qwen2.5-7B-Instruct",
                 temperature: float = 0.7):
        """
        初始化 TOT 求解器
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.thought_cache = {}  # 用于缓存评估结果

    def chat_with_gpt(self, prompt: str, n: int = 1, stop: str = None) -> List[str]:
        """
        调用大模型接口，返回 n 个回答
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
        从给定文本中抽取最终答案。
        策略：
         - 查找形如 "FINAL ANSWER:" 或 "####" 的标记；
         - 使用正则严格提取纯数字（允许整数和小数），同时去除省略号等非数字字符。
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
        根据问题和上下文生成候选方案，此处采用链式思考（Chain-of-Thought）策略：
          1. 先让大模型详细阐述对题目的理解（问题解读部分）；
          2. 再在上述理解的基础上展开详细的解题步骤和计算过程，且最后一行严格输出格式为：
             FINAL ANSWER: <纯数字>
             
        例如参考案例：
          问题：When the water is cold Ray swims a mile in 16 minutes. When the water is warm Ray swims a mile in 2 minutes more than twice as long. How much longer does Ray take to swim 3 miles on a hot day than a cold day?
          第一问（理解）：请详细说明你对题目的理解，抓住题中的关键信息与潜在歧义。
          第二问（解题）：在你对题目理解的基础上，详细给出解题步骤和计算过程，并在最后一行输出：
          FINAL ANSWER: 54
          
        本方法将两次对话拼接后返回作为候选方案。
        """
        candidates = []
        # 固定的 one-shot 案例（可选，可增强模型回答稳定性）
        reference_example = """【参考案例】
问题：When the water is cold Ray swims a mile in 16 minutes. When the water is warm Ray swims a mile in 2 minutes more than twice as long. How much longer does Ray take to swim 3 miles on a hot day than a cold day?
第一步 - 题目理解：
观察到题目中分别描述了冷水和热水下的游泳情况，重点在于比较两种情况的游泳时间。
第二步 - 解题过程：
1. 冷水情况下：每英里的时间为 <<16>> 分钟，<<3>> 英里共 <<3 * 16 = 48>> 分钟；
2. 热水情况下：每英里的时间为 <<2 + 2 * 16>> = <<34>> 分钟，<<3>> 英里共 <<3 * 34 = 102>> 分钟；
3. 时间差为 <<102 - 48>> = <<54>> 分钟；
FINAL ANSWER: 54
"""
        # 对每个样本，采用两次提问
        for _ in range(n_samples):
            # 第一步：请求模型阐述对题目的理解
            prompt_understanding = f"""你是一个高水平的解题专家。请基于下面的原始问题，详细描述你对题目的理解，包括题目中的关键信息和可能存在的歧义。如果有必要，请举例说明。
原始问题: {question}
【参考案例】:
{reference_example}
请给出你的题目理解：
"""
            understanding = self.chat_with_gpt(prompt_understanding, n=1)[0].strip()

            # 第二步：在上述理解的基础上展开详细的解题步骤和计算过程，要求最后一行严格格式为：
            # FINAL ANSWER: <纯数字>
            prompt_solving = f"""在你对题目理解的基础上，请详细展开解题思路和计算步骤。注意：
1. 你的解题过程必须逻辑清晰、逐步展开；
2. 最后一行请严格输出格式为：FINAL ANSWER: <纯数字>
原始问题: {question}
你对题目的理解: {understanding}
请给出你的解题过程：
"""
            solving = self.chat_with_gpt(prompt_solving, n=1)[0].strip()

            # 将理解部分和解题部分拼接成最终的候选方案
            candidate = f"{understanding}\n{solving}"
            candidates.append(candidate)
        return candidates

    def evaluate_thought(self, question: str, candidate: str, cache: bool = True) -> float:
        """
        评估单个候选方案。
        在提示词中加入 one-shot 案例，指引模型按照预期要求输出评分：
          - 输出一个 0 到 1 之间的浮点数，表示候选方案的合理性
          - 参考案例见下：
                
        【参考案例】
        问题：When the water is cold Ray swims a mile in 16 minutes...
        推理过程：[参考生成过程中的详细步骤，最后一行为 "FINAL ANSWER: 54"]
        请对上述推理过程进行评分，要求只输出一个数值（例如：0.85），其中 0 表示完全不合理，1 表示非常优秀。

        【当前任务】
        问题：{question}
        推理过程：
        {candidate}

        请给出评分（只输出一个浮点数）："""
        if cache and candidate in self.thought_cache:
            return self.thought_cache[candidate]

        final_ans = extract_value(candidate)
        if not final_ans:
            score = 0.2  # 格式不符合要求
        else:
            eval_prompt = f"""
            You are a mathematical solution evaluator. Evaluate the following reasoning process based on these strict criteria:
            
            1. **Numerical Understanding (0.4 points):**
            - Are all numerical values in the question correctly identified and used?
            - Are the units (e.g., minutes, dollars, meters) properly tracked throughout?
            - Is there any confusion between different numerical quantities (e.g., mixing up prices)?
            
            2. **Scale and Multiplication Check (0.3 points):**
            - Are calculations done at the correct scale (e.g., not multiplying by 10 when unnecessary)?
            - Are all multiplication operations properly verified with intermediate steps?
            - Are rates and frequencies (e.g., per day, per month) correctly applied?
            
            3. **Context and Constraints (0.3 points):**
            - Are all contextual conditions (e.g., weekend vs weekday prices) properly considered?
            - Are time periods and frequencies correctly interpreted?
            - Is the final answer consistent with the problem's context and scale?
            
            Question: {question}
            Thought Process:
            {candidate}
            
            First identify any numerical errors or scale mistakes, then output a single float from 0 to 1.
            """
            try:
                evaluator_response = self.chat_with_gpt(eval_prompt, n=1)[0].strip()
                score_match = re.search(r"(\d+(\.\d+)?)", evaluator_response)
                if score_match:
                    score = float(score_match.group(1))
                    if score > 0.8:  # 如果评分很高，进行额外验证
                        validate_prompt = f"""
                        Double check these specific aspects of the solution:
                        1. Are there any order-of-magnitude errors (e.g. multiplying by 10 unnecessarily)?
                        2. Are all rate calculations (per day/week/month) correct?
                        3. Are contextual prices (e.g. weekend vs weekday) properly applied?
                        
                        Output VERIFIED if all checks pass, otherwise output the specific error found.
                        """
                        validation = self.chat_with_gpt(validate_prompt, n=1)[0].strip()
                        if "VERIFIED" not in validation.upper():
                            score *= 0.8  # Reduce score if validation fails
                else:
                    score = 0.2
            except Exception as e:
                logging.error(f"Evaluation error: {e}")
                score = 0.2

            if cache:
                self.thought_cache[candidate] = score
            return max(0.0, min(1.0, score))

    def self_supervised_validation(self, candidate: str) -> bool:
        """
        自监督验证：判断候选方案的最终答案是否与前文逻辑一致。
        主要关注逻辑一致性，不额外验证格式。
        """
        final_answer = self.extract_final_answer(candidate)
        if not final_answer:
            return False

        prompt = f"""请检查下面的推理过程，判断最后一行以 "FINAL ANSWER:" 开头的最终答案是否与前面的逻辑一致。
推理过程：
{candidate}
请仅回复 YES 或 NO。"""
        try:
            response = self.chat_with_gpt(prompt, n=1)[0].strip().upper()
            return response == "YES"
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return False

    def refute_candidate(self, question: str, candidate: str) -> str:
        """
        对候选方案进行驳斥：
        请仔细阅读下面的候选方案，指出该方案在题目理解和计算过程中的不足与错误，要求：
          - 分析候选方案对原始问题的理解是否全面，是否忽略了题目中的关键信息；
          - 分析其计算过程是否存在步骤遗漏或计算错误；
          - 指出是否存在其他更合理的解题角度或思路。
        返回具体且具有指导意义的驳斥意见。
        """
        prompt = f"""你是一个高水平的评审专家。请仔细阅读下面这个候选方案，并从以下几个方面进行驳斥：
        1. 该方案对原始问题的理解是否全面，是否遗漏了题目中的关键信息？
        2. 该方案的解题思路和计算过程是否存在错误或不合理之处？
        3. 是否有其他更合理或更严谨的解题角度？

        候选方案：
        {candidate}

        请详细列出你的驳斥意见，并说明改进的方向：
        """
        try:
            refutation = self.chat_with_gpt(prompt, n=1)[0].strip()
            return refutation
        except Exception as e:
            logging.error(f"Refutation error: {e}")
            return "未能获得驳斥意见"

    def revise_candidate(self, question: str, candidate: str, refutation: str) -> str:
        """
        根据驳斥意见对候选方案进行修正：
         - 请参考下面的驳斥意见，对原方案进行修正，改进其对题目的理解和计算过程，
         - 修正后的方案要求在最后一行严格输出格式为：
           FINAL ANSWER: <纯数字>
        """
        prompt = f"""你是一个高水平的解题专家。下面给出一个候选方案和针对该方案的驳斥意见，
        这些意见指出了该方案在题目理解和计算过程中的不足。请参考这些意见，对原方案进行全面修正，
        改进其解题思路和计算步骤，并给出改进后的方案。要求在最后一行严格输出格式为：
        FINAL ANSWER: <纯数字>
        候选方案：
        {candidate}

        驳斥意见：
        {refutation}

        请给出修正后的方案,如果不需要修正请保持原样：
        """
        try:
            revised = self.chat_with_gpt(prompt, n=1)[0].strip()
            return revised
        except Exception as e:
            logging.error(f"Revision error: {e}")
            return candidate

    def display_tree(self, infos: List[Dict[str, Any]], final_score: float) -> None:
        """
        利用 rich 库打印整个搜索过程，显示每轮生成的候选方案、驳斥意见、修正结果、评估得分以及选择结果。
        修改：在生成候选方案节点中直接显示对应的评分。
        """
        console = Console()
        root = Tree(f"[bold green]TOT 求解树 (最终最佳评分: {final_score:.2f})[/bold green]")

        for info in infos:
            step = info.get('step')
            step_node = root.add(f"[bold blue]第 {step} 轮[/bold blue]")
            prev_context = info.get('prev_context', "")
            if prev_context:
                step_node.add(f"[cyan]当前方案：[/cyan]{prev_context}")

            # 在生成候选方案节点中同时显示对应评分
            gen_node = step_node.add("[magenta]生成候选方案[/magenta]")
            candidates = info.get('candidates', [])
            scores = info.get('scores', [])
            for idx, candidate in enumerate(candidates, start=1):
                score = scores[idx-1] if idx-1 < len(scores) else 0.0
                gen_node.add(f"[magenta]候选 {idx}:[/magenta] {candidate.strip()[:200]}... (得分: {score:.2f})")

            ref_node = step_node.add("[red]驳斥意见[/red]")
            for idx, ref in enumerate(info.get('refutations', []), start=1):
                ref_node.add(f"[red]候选 {idx} 驳斥：[/red] {ref.strip()[:200]}...")

            rev_node = step_node.add("[blue]修正候选方案[/blue]")
            for idx, candidate in enumerate(info.get('revised', []), start=1):
                rev_node.add(f"[blue]候选 {idx} 修正后:[/blue] {candidate.strip()[:200]}...")

            eval_node = step_node.add("[yellow]评估得分[/yellow]")
            for idx, score in enumerate(scores, start=1):
                eval_node.add(f"[yellow]候选 {idx} 得分: {score:.2f}[/yellow]")

            sel = info.get('selected', [])
            sel_node = step_node.add("[green]选择进入下一轮的候选[/green]")
            for idx, candidate in enumerate(sel, start=1):
                sel_node.add(f"[green]候选 {idx}:[/green] {candidate.strip()[:200]}...")
        console.print(root)

    def solve(self, question: str, max_steps: int = 6, n_samples_per_step: int = 3,
              k_best_thoughts: int = 2, score_threshold: float = 0.9, revision_threshold: float = 0.5) -> str:
        """
        按照 TOT 框架进行分轮求解，每轮流程为：
          1. Generation：对当前状态生成 n_samples_per_step 个候选方案；
          2. Evaluation：对所有候选方案打分；
          3. 选择：从中选择评分最高的 k_best_thoughts 个作为最佳候选；
          4. Refutation+Revision：对这 k_best_thoughts 个候选进行驳斥（指出题意理解、计算过程和思路中的不足），
             并基于驳斥意见分别生成新的修正方案，再对修正方案重新评估；
          5. 将修正后的候选作为下一轮状态，循环迭代，直至候选得分超过 score_threshold 或达到 max_steps。
          6. 如果新一轮生成的候选方案的评分均低于当前已有最佳答案的得分，则提前停止搜索，直接返回最佳答案。
        注意：上下文仅记录原始问题及当前候选方案，不累计其他候选信息。
        """
        infos = []  # 记录每轮信息
        # 初始状态：初始方案为空（上下文仅为原始问题）
        states = [{"chain": "", "score": 1.0}]
        best_solution = None
        best_score = 0.0

        for step in range(max_steps):
            print(f"====== 第 {step + 1} 轮 ======")
            new_candidates = []
            candidate_scores = []
            refutations = []
            revised_candidates = []

            # Generation 阶段：对当前状态中的每个方案生成 n_samples_per_step 个扩展候选
            for state in states:
                current_solution = state["chain"]
                context = f"原始问题: {question}\n当前方案: {current_solution}"
                candidates = self.generate_thoughts(question, context, n_samples=n_samples_per_step)
                for cand in candidates:
                    new_candidates.append(cand)
            if not new_candidates:
                break

            # Evaluation 阶段：对所有生成的候选方案打分
            for cand in new_candidates:
                score = self.evaluate_thought(question, cand)
                # 如果存在最终答案但自监督验证不通过，则适当扣分
                if ("FINAL ANSWER:" in cand or "####" in cand) and not self.self_supervised_validation(cand):
                    score *= 0.9
                candidate_scores.append(score)

            # 早停判断：如果本轮生成的所有候选方案得分均低于当前最佳得分，则触发早停
            if new_candidates and candidate_scores and max(candidate_scores) < best_score:
                print("早停触发：本轮所有候选方案得分均低于当前最佳得分，提前终止搜索。")
                break

            # 选择阶段：选择评分最高的 k_best_thoughts 个候选
            idxs = sorted(range(len(new_candidates)), key=lambda i: candidate_scores[i], reverse=True)[:k_best_thoughts]
            selected = [new_candidates[i] for i in idxs]

            # Refutation 阶段：对这 k_best_thoughts 个候选进行驳斥
            for i in idxs:
                refutation = self.refute_candidate(question, new_candidates[i])
                refutations.append(refutation)

            # Revision 阶段：针对驳斥意见，对这 k_best_thoughts 个候选分别进行修正
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

            # 保存本轮信息
            infos.append({
                "step": step + 1,
                "prev_context": states[0]["chain"],
                "candidates": new_candidates,
                "refutations": refutations,
                "revised": revised_candidates,
                "scores": candidate_scores,
                "selected": [new_candidates[i] for i in idxs]
            })

            # 更新状态为选择的候选（取 k_best_thoughts 个）
            states = []
            for i in idxs:
                states.append({"chain": new_candidates[i], "score": candidate_scores[i]})
                # 根据评分更新最佳解
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


# 示例：如何使用更新后的代码
if __name__ == "__main__":
    # 请替换为你自己的 API_KEY
    API_KEY = "your_api_key_here"
    tot_solver = TreeOfThoughts(api_key=API_KEY)
    question = ("假设有一个正方形，其边长为 5，请计算其面积。"
                "\n例如对于其他题目，候选方案需要体现对题设的不同理解角度和解题思路。")
    solution = tot_solver.solve(question,
                                max_steps=6,
                                n_samples_per_step=3,
                                k_best_thoughts=2,
                                score_threshold=0.9,
                                revision_threshold=0.5)
    print("\n最终返回的推理过程：")
    print(solution)
