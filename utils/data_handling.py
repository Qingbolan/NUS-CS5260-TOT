import json
import random
import re
import pandas as pd
from typing import List, Dict

def load_dataset(file_path: str, sample_size: int = 20, output_file: str = "tmp/sorted_dataset.csv") -> List[Dict]:
    # 定义自然排序的 key 函数
    def natural_sort_key(s: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
    # 读取 JSONL 文件
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 随机抽取 sample_size 条数据
    sampled_data = random.sample(data, sample_size)
    
    # 将抽样数据转换为 DataFrame
    df = pd.DataFrame(sampled_data)
    
    # 检查 'question_id' 是否存在
    if 'question_id' not in df.columns:
        raise KeyError("'question_id' 列不存在于数据中")
    
    # 根据 'question_id' 进行自然排序
    df.sort_values('question_id', key=lambda col: col.map(natural_sort_key), inplace=True)
    
    # 显示排序后的前几行
    print("\nAfter sorting:")
    print(df.head())
    
    # 保存排序后的 DataFrame 到 CSV 文件
    df.to_csv(output_file, index=False)
    print(f"\nSorted file saved as: {output_file}")
    
    # 显示总行数
    print(f"Total rows: {len(df)}")
    
    # 返回排序后的数据列表，每个元素是一个字典
    return df.to_dict(orient='records')
