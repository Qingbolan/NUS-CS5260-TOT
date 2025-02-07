from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List
import logging
from config import TOT_CONFIG, NUM_PROCESSES, API_KEY, BASE_URL, DEFAULT_MODEL
from utils.text_processing import extract_value
from config import TreeOfThoughts
import re

def process_single_question(item: Dict) -> Dict:
    """Process a single question using ToT."""
    try:
        question_id = item["question_id"]
        logging.info(f"Processing: {question_id}")
        
        tot_solver = TreeOfThoughts(API_KEY, BASE_URL, DEFAULT_MODEL)
        tot_solution = tot_solver.solve(
            question=item["question"],
            max_steps=TOT_CONFIG['max_depth'],
            n_samples_per_step=TOT_CONFIG['n_samples_per_step'],
            k_best_thoughts=TOT_CONFIG['k_best_thoughts']
        )
        tot_answer = extract_value(tot_solution)
        
        return {
            "question_id": question_id,
            "predicted": tot_answer,
            "solution": tot_solution
        }
    except Exception as e:
        logging.error(f"Error on {question_id}: {str(e)}")
        raise

def process_dataset_parallel(dataset: List[Dict]) -> List[Dict]:
    """使用并行处理对数据集进行处理，并按照 'question_id' 进行自然排序后返回结果。"""
    results = []
    with Pool(NUM_PROCESSES) as pool:
        with tqdm(total=len(dataset)) as pbar:
            for result in pool.imap_unordered(process_single_question, dataset):
                results.append(result)
                pbar.update(1)
    
    # 定义自然排序的 key 函数
    def natural_sort_key(s: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
    # 根据 'question_id' 对结果进行排序，确保将 'question_id' 转换为字符串进行自然排序
    results.sort(key=lambda item: natural_sort_key(str(item.get('question_id', ''))))
    
    return results