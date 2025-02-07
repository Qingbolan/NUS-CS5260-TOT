import os
import json
import random
import re
import logging
import sys
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 如果你已经有 from openai import OpenAI，就使用之
# 或者如果是 deepinfra，请自行引用
from openai import OpenAI  # 这里假设你已有一个 OpenAI 类可用

##############################################################################
# 1. 工具函数
##############################################################################

def clean_text(text: str) -> str:
    """对文本做基础清洗"""
    text = text.lower()
    text = re.sub(r"\$", "", text)
    text = re.sub(r"(?s).*#### ", "", text)
    text = re.sub(r"\.$", "", text)
    text = re.sub(r",", "", text)
    if not text:
        return "-1000000000"
    return text

def extract_value(text: str) -> str:
    """提取数值（只返回第一个匹配）"""
    pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = re.findall(pattern, text)
    if matches:
        for match_groups in matches[::-1]:
            for group in match_groups:
                if group:
                    return clean_text(group)
    return "-1000000000"

def load_dataset(file_path: str, sample_size: int = 5) -> List[Dict]:
    """加载数据集并随机采样"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return random.sample(data, min(sample_size, len(data)))

def evaluate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """计算准确率，比较 int 值是否匹配"""
    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        try:
            if int(pred) == int(truth):
                correct += 1
        except:
            pass
    return correct / len(predictions) if predictions else 0.0


##############################################################################
# 2. FOT 知识库：存储、检索常见错误 + 域知识
##############################################################################

class FOTKnowledgeBase:
    """
    Forest of Thoughts(FOT) 知识库，用于存储/检索【常见错误模式】以及【域知识】。
    
    更新点：
      1) analyze_error 中，先获取 LLM 的详细分析，再用“二次提示”让 LLM 返回一段“精炼版”文字，存入 analysis 字段。
      2) 新增 get_all_error_patterns() 以便后续在评估结束后做归纳。
    """
    def __init__(self, cache_file: str = "fot_cache.json"):
        self.cache_file = cache_file
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_patterns = self._load_cache()
        self.client = None

        # domain_knowledge_map 不再写死在这里；可以在下次运行 TOT 时，通过外部文件或别的函数载入

    def set_api_client(self, client):
        """设置 LLM 客户端（可在分析错误时调用）"""
        self.client = client

    def _load_cache(self) -> Dict:
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.debug(f"Loaded existing FOT cache from {self.cache_file}")
                return data
        except FileNotFoundError:
            self.logger.info(f"No existing FOT cache found, creating new one at {self.cache_file}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading FOT cache: {str(e)}")
            return {}

    def save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.error_patterns, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"FOT cache saved to {self.cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save FOT cache: {str(e)}")

    def analyze_error(self, question: str, wrong_solution: str, correct_solution: str) -> str:
        """
        调用LLM分析错误:
          1) 先让LLM给出一个较详细的分析
          2) 再让LLM精炼出“简短版”结论
        返回精炼版结论
        """
        if not self.client:
            msg = "Error: LLM client not set"
            self.logger.error(msg)
            return msg

        # 第一次提示 -> 获得详细分析
        prompt_detailed = f"""分析这道数学题中的错误:

问题: {question}
错误解法: {wrong_solution}
正确答案: {correct_solution}

请分析:
1. 错误类型(如概念理解错误、计算错误等)
2. 错误的具体原因
3. 如何避免类似错误

请用简洁的语言描述，但可以包含适当细节。"""
        self.logger.debug(f"[FOTKnowledgeBase.analyze_error] detailed prompt:\n{prompt_detailed}")
        
        try:
            resp_detailed = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt_detailed}],
                temperature=0.7
            )
            analysis_detailed = resp_detailed.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling LLM for detailed analysis: {str(e)}")
            analysis_detailed = "LLM 调用异常，无法获取详细分析"

        # 第二次提示 -> 让LLM输出更精炼、短小的版本
        prompt_concise = f"请将下述分析内容精炼成一句话，且不超过40个字：\n\n{analysis_detailed}\n"
        self.logger.debug(f"[FOTKnowledgeBase.analyze_error] concise prompt:\n{prompt_concise}")
        
        try:
            resp_concise = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt_concise}],
                temperature=0.7
            )
            analysis_concise = resp_concise.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error calling LLM for concise summary: {str(e)}")
            analysis_concise = "LLM 调用异常，无法生成精炼版摘要"

        self.logger.debug(f"[FOTKnowledgeBase.analyze_error] analysis_concise: {analysis_concise}")
        return analysis_concise

    def _generate_pattern_key(self, text: str) -> str:
        """
        根据问题文本，返回一个简单的分类 key，
        用于把问题/错误分组到同一关键词下
        """
        text_lower = text.lower()
        patterns = {
            "constant_rate_depreciation": ("constant rate", "depreciate"),
            "compound_interest": ("compound", "interest"),
            "percentage": ("percent", "rate"),
            "ratio_proportion": ("ratio", "proportion")
        }
        for key, terms in patterns.items():
            if all(term in text_lower for term in terms):
                self.logger.debug(f"Matched pattern key {key} for question: {text[:50]}")
                return key
        self.logger.debug(f"No specialized pattern matched for question: {text[:50]}, use 'general'")
        return "general"

    def add_error_pattern(self, question: str, wrong_answer: str, correct_answer: str):
        """将新错误添加到 FOT 知识库"""
        self.logger.info(f"Adding error pattern for question: {question[:60]}")
        concise_analysis = self.analyze_error(question, wrong_answer, correct_answer)
        key = self._generate_pattern_key(question)

        if key not in self.error_patterns:
            self.error_patterns[key] = []

        self.error_patterns[key].append({
            "question": question,
            "wrong_answer": wrong_answer,
            "correct_answer": correct_answer,
            "analysis": concise_analysis
        })
        self.save_cache()

    def get_related_errors(self, question: str) -> List[Dict]:
        """
        根据问题文本的分类Key，获取已有相关错误模式，
        供下游在提示词中进行警示或避免重复错误。
        """
        key = self._generate_pattern_key(question)
        related = self.error_patterns.get(key, [])
        if related:
            self.logger.debug(f"Found {len(related)} related error patterns for key={key}")
        else:
            self.logger.debug(f"No related error patterns found for key={key}")
        return related

    def get_all_error_patterns(self) -> Dict[str, List[Dict]]:
        """
        返回当前所有分类key -> [错误案例列表]
        结构如:
          {
            "constant_rate_depreciation": [
               {
                 "question": "...",
                 "wrong_answer": "...",
                 "correct_answer": "...",
                 "analysis": "..."
               },
               ...
            ],
            "compound_interest": [...],
            ...
          }
        """
        return self.error_patterns


##############################################################################
# 3. TOT：多步/多分支的思维推理
##############################################################################

class TreeOfThoughts:
    """
    主要流程:
      1) generate_thoughts: 查FOT数据库里的错误提示 + 本对象加载的domain_knowledge_map(可在外部更新)
      2) evaluate_thought: 对输出打分(0~1)
      3) 迭代多步，保留高分分支
    """
    def __init__(self, 
                 api_key: str, 
                 base_url: str = "https://api.deepinfra.com/v1/openai",
                 model: str = "Qwen/Qwen2.5-7B-Instruct", 
                 temperature: float = 0.7,
                 domain_knowledge_map: Optional[Dict[str, str]] = None):
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

        # 初始化 FOT 知识库：存储并检索常见错误
        self.fot_kb = FOTKnowledgeBase()
        self.fot_kb.set_api_client(self.client)

        # 可以从外部传入一个 domain_knowledge_map (如读取自 domain_knowledge.json)
        # 若无则留空
        if domain_knowledge_map is None:
            self.domain_knowledge_map = {}
        else:
            self.domain_knowledge_map = domain_knowledge_map

    def _generate_pattern_key(self, text: str) -> str:
        """外部使用FOT的相同匹配逻辑，以拿到domain_knowledge_map的key"""
        return self.fot_kb._generate_pattern_key(text)

    def chat_with_gpt(self, prompt: str, n: int = 1, stop: str = None) -> List[str]:
        """调用 GPT 模型生成文本"""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=n,
            stop=stop
        )
        return [msg.message.content for msg in response.choices]

    def get_domain_knowledge_for_question(self, question: str) -> str:
        """根据问题text得到key，再查看是否有domain_knowledge"""
        key = self._generate_pattern_key(question)
        return self.domain_knowledge_map.get(key, "")

    def generate_thoughts(self, question: str, current_thought: str = "", n_samples: int = 3) -> List[str]:
        """
        1) 检索已记录的错误提示
        2) 加 domain_knowledge
        3) 生成新的分支思路
        """
        related_errors = self.fot_kb.get_related_errors(question)
        error_hints = ""
        if related_errors:
            error_hints = "基于我们以往的错误案例：\n"
            for err in related_errors:
                error_hints += (
                    f"【类似问题：\n"
                    f"  - 错误解法: {err['wrong_answer']}\n"
                    f"  - 正确解法: {err['correct_answer']}\n"
                    f"  - 原因分析: {err['analysis']}\n"
                    f"】\n"
                )

        # 获取domain_knowledge
        dk = self.get_domain_knowledge_for_question(question)
        if dk:
            dk = (
                "【我们额外知道在这类问题中：\n"
                f"{dk}\n"
                "】\n"
            )

        # 拼接提示
        prompt = (
            f"{error_hints}"
            f"{dk}"
            f"请分步骤解决这道数学题，并尽量避免上述类似错误。\n\n"
            f"Question: {question}\n"
            f"Current Thought: {current_thought}\n"
            f"Now let's expand our reasoning in multiple steps.\n"
        )
        self.logger.debug(f"[generate_thoughts] Prompt:\n{prompt}")
        return self.chat_with_gpt(prompt, n=n_samples)

    def evaluate_thought(self, question: str, thought: str) -> float:
        """
        对某个思维过程做自评打分(0~1)
        """
        prompt = f"""Please provide a numeric rating (0 to 1) for the correctness of this reasoning.

Question: {question}
Thought: {thought}

Output only the numeric rating:
"""
        self.logger.debug(f"[evaluate_thought] Evaluate Prompt:\n{prompt}")
        response = self.chat_with_gpt(prompt)
        raw_score = response[0].strip()
        try:
            score = float(raw_score)
            if score < 0:
                score = 0.0
            elif score > 1:
                score = 1.0
        except:
            score = 0.0
        self.logger.debug(f"[evaluate_thought] Score={score:.4f}")
        return score

    def solve(self, question: str,
              max_steps: int = 8,
              n_samples_per_step: int = 3,
              k_best_thoughts: int = 2) -> str:
        """
        主求解函数：多步/多分支的树状思维推理
        """
        self.logger.info(f"=== TOT solve starts for question: {question[:60]} ===")
        states = [("", 0.0)]  # (思路文本, 评分)

        for step in range(max_steps):
            self.logger.info(f"--- Step {step+1}/{max_steps} ---")
            new_states = []
            for current_text, _ in states:
                # 在当前思路(分支)基础上扩展新的思路
                generated = self.generate_thoughts(question, current_thought=current_text, n_samples=n_samples_per_step)
                for g in generated:
                    combined = current_text + "\n" + g if current_text else g
                    score = self.evaluate_thought(question, combined)
                    new_states.append((combined, score))

            # 根据得分排序
            new_states.sort(key=lambda x: x[1], reverse=True)
            # 保留前 k_best_thoughts
            states = new_states[:k_best_thoughts]

            # 如果已有思路中包含 "answer" 字样，就提前结束
            if any("answer" in s[0].lower() for s in states):
                self.logger.info("Found 'answer' in a thought, stopping early.")
                break

        best_state = max(states, key=lambda x: x[1])
        self.logger.info(f"=== TOT best solution score={best_state[1]:.4f} ===\n{best_state[0]}\n")
        return best_state[0]

    def process_result(self, question: str, solution: str, correct_answer: str):
        """
        对TOT结果进行校验，若错了则更新FOT KnowledgeBase
        """
        predicted = extract_value(solution)
        true_value = extract_value(correct_answer)
        self.logger.info(f"[process_result] predicted={predicted}, correct={true_value}")
        if predicted != true_value:
            self.logger.warning("Detected mismatch -> updating FOT knowledge base with new error pattern.")
            self.fot_kb.add_error_pattern(question, solution, correct_answer)
        else:
            self.logger.info("Answer matched the ground truth, no error update needed.")


##############################################################################
# 4. 在评估结束后自动生成(或更新) domain_knowledge_map
##############################################################################

def update_domain_knowledge_from_errors(fot_kb: FOTKnowledgeBase) -> Dict[str, str]:
    """
    根据 FOT 里记录的错误数据，统计各种分类的错误频次，并自动生成一些“域知识”。
    返回一个 dict: {category_key: knowledge_str}

    注：此处仅做简单示例，可结合LLM或更复杂逻辑来生成更详细的内容。
    """
    all_errors = fot_kb.get_all_error_patterns()
    
    domain_knowledge_map = {}

    # 这里设置一些简单的“默认知识模板”，你也可以加更多复杂逻辑或LLM生成
    default_knowledge_templates = {
        "constant_rate_depreciation": (
            "多次出现此类错误：表明对“constant rate”折旧理解偏差。\n"
            "在线性折旧中，每年固定折旧额，可用：原价*(1 - r*年数)。"
        ),
        "compound_interest": (
            "多次出现此类错误：表明对复利(Compound Interest)的理解不足。\n"
            "应使用(1 + r)^n 计算增长或衰减。"
        ),
        "general": (
            "对于常见的计算问题，需要注意区分折旧、百分比、利率等概念，"
            "并逐步分解和检查。"
        ),
        # 若还有其他分类，可在此添加
    }

    # 统计每个分类的错误量
    for category_key, error_list in all_errors.items():
        if len(error_list) < 2:
            # 如果此分类错误数 <2，就不纳入domain_knowledge
            continue
        # 若有2道及以上错误，认为需要重点说明
        if category_key in default_knowledge_templates:
            domain_knowledge_map[category_key] = default_knowledge_templates[category_key]
        else:
            # 如果不在模板里，就给个笼统提示
            domain_knowledge_map[category_key] = (
                f"该类别({category_key})出现多次错误，请复查相关概念或计算方式。"
            )

    return domain_knowledge_map


##############################################################################
# 5. 评估流程
##############################################################################

def process_evaluation_item(item: Dict) -> Dict:
    """
    对单个问题执行 TOT，若错误则更新 FOT
    """
    # 你自己的 API Key
    api_key = os.getenv("DEEPINFRA_TOKEN", "6CsmsskJ9LlwYPUMXnsy2LX3u3VgfqIi")

    question = item["question"]
    true_answer = item["answer"]
    question_id = item.get("question_id", "unknown")

    logger = logging.getLogger("process_evaluation_item")
    logger.info(f"[process_evaluation_item] question_id={question_id}")

    try:
        # 初始化 TOT，此时不加载domain_knowledge_map，因为是第一次评估
        tot_solver = TreeOfThoughts(api_key=api_key)
        # 执行推理
        tot_solution = tot_solver.solve(question)
        # 校验结果
        tot_solver.process_result(question, tot_solution, true_answer)
        # 提取数字
        tot_answer = extract_value(tot_solution)

        return {
            "question_id": question_id,
            "tot_predicted": tot_answer,
            "true": true_answer
        }

    except Exception as e:
        logger.error(f"Error processing question {question_id}: {str(e)}")
        return {
            "question_id": question_id,
            "tot_predicted": "-1000000000",
            "true": true_answer
        }


def main_evaluation():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    # 1) 读入数据集
    dataset_path = "dataset/cs5260_val_random300.jsonl"
    dataset = load_dataset(dataset_path, sample_size=5)
    total_questions = len(dataset)
    print(f"Loaded {total_questions} items from dataset.")

    # 2) 并行执行 TOT
    num_processes = max(1, cpu_count() - 1)
    print(f"\nProcessing {total_questions} questions with {num_processes} processes...")

    results = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=total_questions, desc="Evaluating", ncols=100) as pbar:
            for result in pool.imap_unordered(process_evaluation_item, dataset):
                results.append(result)
                pbar.update(1)
                sys.stdout.flush()

    # 3) 计算准确率
    tot_accuracy = evaluate_accuracy(
        [r["tot_predicted"] for r in results],
        [r["true"] for r in results]
    )
    print("\n=== Validation Results ===")
    print(f"ToT Accuracy: {tot_accuracy:.2%}")

    # 4) 将评估结果存入 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nDetailed results saved to 'evaluation_results.csv'")

    # 5) 现在我们从 fot_cache.json 中收集错误信息，统计生成domain_knowledge
    fot_kb = FOTKnowledgeBase(cache_file="fot_cache.json")
    domain_map_updates = update_domain_knowledge_from_errors(fot_kb)

    # 6) 保存domain_knowledge到文件 (下次运行时可加载到TOT中)
    if domain_map_updates:
        with open("domain_knowledge.json", "w", encoding="utf-8") as f:
            json.dump(domain_map_updates, f, ensure_ascii=False, indent=2)
        print("\nUpdated domain knowledge map saved to 'domain_knowledge.json'")
    else:
        print("\nNo significant repeated errors found, domain knowledge map remains unchanged.")


if __name__ == "__main__":
    main_evaluation()
